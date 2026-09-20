#!/usr/bin/env python3
"""
Train LightGBM on prepared IEEE dataset and save pipeline + meta + optional SHAP explainer.

Protocol (chronological, with late labels; see fraud_app.features.split_data):

  test          the newest 20% of transactions, only used for the report
  labeled data  everything older than ``--label-delay-days`` before the test period starts
                (a live system does not know the labels of more recent transactions yet)

  Stage A  fit on the oldest part of the labeled data, early-stop and pick the threshold on the
           newest part of it (the validation block), which is separated from the training rows
           by the same label-delay gap. Validation scores therefore look like test scores; a
           validation block glued to the training rows is ~0.10 F1 too optimistic.
  Stage B  refit on ALL labeled rows (the freshest data, including the validation block) with
           the number of trees found in stage A, scaled to the larger training set.

Usage:
  python3 train_ieee_lgbm.py --input data/ieee_prepared.csv --out-dir artifacts --sample 0.2
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from fraud_app.features import (
    DEFAULT_LABEL_DELAY_DAYS,
    build_preprocessor,
    category_labels,
    labeled_rows,
    model_columns,
    split_data,
)
from fraud_app.schema import CATEGORICAL_NAMES

warnings.filterwarnings("ignore")

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# Chosen on the validation block (gap-separated), never on the test split: 63 leaves with
# stronger regularisation was the best of four settings, by a margin close to the seed noise.
LGBM_PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=80,
    reg_lambda=5,
    colsample_bytree=0.4,
    subsample=0.8,
    subsample_freq=1,
    max_depth=-1,
    random_state=42,
    verbosity=-1,
)
EARLY_STOPPING_ROUNDS = 100
VAL_SCORES_FILE = "val_scores_ieee.npz"


def load_data(path):
    print("Loading:", path)
    return pd.read_csv(path)


def train_lgbm(X_train_arr, y_train, X_val_arr, y_val, params=None):
    """Stage A: fit with early stopping on the validation block."""
    params = dict(LGBM_PARAMS if params is None else params)
    print("LightGBM params:", params)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(
        X_train_arr,
        y_train,
        eval_set=[(X_val_arr, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    print(f"Early stopping: best iteration {clf.best_iteration_} of {params['n_estimators']}")
    return clf


def refit_lgbm(X_arr, y, n_estimators, params=None):
    """Stage B: fit on all labeled rows with a fixed number of trees (no validation block left)."""
    params = dict(LGBM_PARAMS if params is None else params, n_estimators=int(n_estimators))
    print(f"Refit on {len(y)} labeled rows with {params['n_estimators']} trees")
    clf = lgb.LGBMClassifier(**params)
    clf.fit(X_arr, y)
    return clf


def scale_trees(best_iteration, n_train, n_refit):
    """Trees for the refit: the stage-A optimum, scaled to the bigger training set."""
    return max(50, int(round(best_iteration * n_refit / max(n_train, 1))))


def evaluate(pipe, X_test, y_test, threshold=0.5):
    """Metrics on the held-out split. Returns (summary_dict, probs_array)."""
    probs = np.asarray(pipe.predict_proba(X_test))[:, 1].astype(float)
    y_arr = np.asarray(y_test).ravel()

    finite = np.isfinite(probs)
    if not finite.all():
        probs, y_arr = probs[finite], y_arr[finite]

    preds = (probs >= threshold).astype(int)
    summary = {
        "threshold": threshold,
        "classification_report": classification_report(
            y_arr, preds, output_dict=True, zero_division=0
        ),
        "roc_auc": float(roc_auc_score(y_arr, probs)),
        "pr_auc": float(average_precision_score(y_arr, probs)),  # average precision
        "confusion_matrix": confusion_matrix(y_arr, preds).tolist(),
        "probs_min": float(probs.min()),
        "probs_max": float(probs.max()),
        "probs_mean": float(probs.mean()),
        "n_test": int(len(y_arr)),
    }
    return summary, probs


def build_pipeline_and_train(
    df,
    target_col="is_fraud",
    out_dir="artifacts",
    sample_frac=None,
    label_delay_days=DEFAULT_LABEL_DELAY_DAYS,
):
    df = df.copy()
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # features = every prepared column except the label and the TransactionDT split column
    all_cols = [c for c in model_columns(df) if c != target_col]
    categorical = [c for c in all_cols if c in CATEGORICAL_NAMES]
    numeric = [c for c in all_cols if c not in categorical]
    print(f"Features: {len(numeric)} numeric, {len(categorical)} categorical")

    train_df, val_df, test_df = split_data(df, label_delay_days)
    labeled_df = labeled_rows(df, label_delay_days)

    def features_of(part):
        X = part[numeric + categorical].copy()
        for c in categorical:  # same labelling code as serving (fraud_app.features)
            X[c] = category_labels(X[c])
        return X

    X_train, X_val, X_test = features_of(train_df), features_of(val_df), features_of(test_df)
    y_train = train_df[target_col].astype(int)
    y_val = val_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)
    print(
        f"Rows: train={len(y_train)} val={len(y_val)} test={len(y_test)} | "
        f"labeled (refit)={len(labeled_df)} | label delay {label_delay_days:g} d | "
        f"fraud rate train={y_train.mean():.4f} val={y_val.mean():.4f} test={y_test.mean():.4f}"
    )

    # ---- Stage A: early stopping, validation scores (threshold + honest estimate)
    preproc_a = build_preprocessor(numeric, categorical, impute=False)
    X_train_t = preproc_a.fit_transform(X_train)
    X_val_t = preproc_a.transform(X_val)
    print("Stage A matrix:", X_train_t.shape)
    clf_a = train_lgbm(X_train_t, y_train, X_val_t, y_val)
    val_probs = clf_a.predict_proba(X_val_t)[:, 1]
    val_summary = {
        "roc_auc": float(roc_auc_score(y_val, val_probs)),
        "pr_auc": float(average_precision_score(y_val, val_probs)),
        "n_val": int(len(y_val)),
    }
    print(
        f"Validation (stage A model): ROC-AUC={val_summary['roc_auc']:.4f} "
        f"PR-AUC={val_summary['pr_auc']:.4f}"
    )

    # ---- Stage B: refit on every labeled row
    n_trees = scale_trees(clf_a.best_iteration_, len(y_train), len(labeled_df))
    X_lab = features_of(labeled_df)
    y_lab = labeled_df[target_col].astype(int)
    preproc = build_preprocessor(numeric, categorical, impute=False)
    X_lab_t = preproc.fit_transform(X_lab)
    clf = refit_lgbm(X_lab_t, y_lab, n_trees)

    # final pipeline: preprocessing + refit classifier
    pipe = Pipeline([("preproc", preproc), ("clf", clf)])

    eval_summary, _ = evaluate(pipe, X_test, y_test)
    eval_summary["validation_stage_a"] = val_summary

    # save artifacts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe_path = out_dir / "pipeline_ieee.joblib"
    meta_path = out_dir / "meta_ieee.joblib"
    report_path = out_dir / "train_report_ieee.json"

    joblib.dump(pipe, pipe_path)
    timed = "TransactionDT" in df.columns
    meta = {
        "features": numeric + categorical,
        "numeric": numeric,
        "categorical": categorical,
        "target": target_col,
        "split": (
            "chronological by TransactionDT, newest 20% is the test set"
            if timed
            else "random stratified 60/20/20"
        ),
        "protocol": "train | label-delay gap | validation, then refit on all labeled rows",
        "label_delay_days": float(label_delay_days) if timed else 0.0,
        "best_iteration_stage_a": int(clf_a.best_iteration_),
        "n_estimators": int(n_trees),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_refit": int(len(y_lab)),
        "n_test": int(len(y_test)),
    }
    joblib.dump(meta, meta_path)
    # the refit model has seen the validation rows, so the threshold must be chosen from the
    # stage-A scores of the validation block
    np.savez(out_dir / VAL_SCORES_FILE, y=y_val.to_numpy(), p=val_probs)

    with open(report_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("Saved pipeline to:", pipe_path)
    print("Saved meta to:", meta_path)
    print("Saved report to:", report_path)
    print(
        f"Test ROC-AUC={eval_summary['roc_auc']:.4f} PR-AUC={eval_summary['pr_auc']:.4f} "
        f"(n_test={eval_summary['n_test']})"
    )

    # SHAP (optional). A small background keeps each per-request explanation fast.
    shap_path = out_dir / "shap_explainer_ieee.joblib"
    if SHAP_AVAILABLE:
        try:
            print("Building SHAP TreeExplainer (may take time)...")
            X_bg = X_lab.sample(n=min(100, len(X_lab)), random_state=42)
            X_bg_trans = preproc.transform(X_bg)
            explainer = shap.TreeExplainer(
                clf, data=X_bg_trans, feature_perturbation="interventional"
            )
            joblib.dump(explainer, shap_path)
            print("Saved SHAP explainer to:", shap_path)
        except Exception as e:
            print("Warning: SHAP explainer build failed:", str(e))
    else:
        print("SHAP not available; skipping explainer build.")

    return pipe_path, meta_path, eval_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/ieee_prepared.csv", help="prepared csv path")
    p.add_argument("--out-dir", default="artifacts", help="where to write pipeline/meta")
    p.add_argument(
        "--sample", type=float, default=None, help="optional fraction to sample for quick runs"
    )
    p.add_argument(
        "--label-delay-days",
        type=float,
        default=DEFAULT_LABEL_DELAY_DAYS,
        help="days before the test period from which labels are treated as unknown "
        f"(default {DEFAULT_LABEL_DELAY_DAYS:g})",
    )
    args = p.parse_args()

    df = load_data(args.input)
    build_pipeline_and_train(
        df, out_dir=args.out_dir, sample_frac=args.sample, label_delay_days=args.label_delay_days
    )
    print("Done.")


if __name__ == "__main__":
    main()
