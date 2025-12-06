#!/usr/bin/env python3
"""
Train LightGBM on prepared IEEE dataset and save pipeline + meta + optional SHAP explainer.

Usage:
  python3 train_ieee_lgbm.py --input data/ieee_prepared.csv --out-dir artifacts --sample 0.2
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    import shap

    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def make_ohe():
    # compatibility for OneHotEncoder API across sklearn versions
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path):
    print("Loading:", path)
    df = pd.read_csv(path)
    return df


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value="NA")),
            ("ohe", make_ohe()),
        ]
    )
    preproc = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    return preproc


def train_lgbm(X_train_arr, y_train, X_val_arr, y_val, params=None, random_state=42):
    if params is None:
        params = dict(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=-1,
            random_state=random_state,
            verbosity=-1,
        )
    print("LightGBM params:", params)
    clf = lgb.LGBMClassifier(**params)

    eval_set = [(X_val_arr, y_val)]
    # Try modern sklearn-wrapper early_stopping param first, fallback to callbacks or simple fit
    try:
        # most friendly call (works on many installs)
        clf.fit(
            X_train_arr,
            y_train,
            eval_set=eval_set,
            eval_metric="auc",
            early_stopping_rounds=50,
            verbose=50,
        )
        print("Fitted with early_stopping_rounds parameter.")
    except TypeError as te:
        print("early_stopping_rounds not supported directly (TypeError), trying callbacks...:", te)
        try:
            # try callbacks API (different lightgbm versions)
            callbacks = []
            # build early stopping callback if available
            if hasattr(lgb, "callback") and hasattr(lgb.callback, "early_stopping"):
                callbacks.append(lgb.callback.early_stopping(50))
            # add verbose print callback if available
            if hasattr(lgb, "callback") and hasattr(lgb.callback, "print_evaluation"):
                callbacks.append(lgb.callback.print_evaluation(50))
            if callbacks:
                clf.fit(
                    X_train_arr, y_train, eval_set=eval_set, eval_metric="auc", callbacks=callbacks
                )
                print("Fitted with callbacks early stopping.")
            else:
                # callbacks api not available - fallback to fit without early stopping
                raise RuntimeError("No early stopping callbacks available")
        except Exception as e2:
            print("Callbacks early stopping failed or not available:", e2)
            print("Falling back to plain fit() without early stopping.")
            clf.fit(X_train_arr, y_train)
    except Exception as e:
        # unexpected other exceptions -> fallback to plain fit
        print("Unexpected error when trying to fit with early stopping:", e)
        print("Falling back to plain fit() without early stopping.")
        clf.fit(X_train_arr, y_train)

    return clf


def evaluate(pipe, X_test, y_test, threshold=0.5):
    """
    Robust evaluate: ensures shapes, flattens arrays and gives helpful diagnostics
    before calling sklearn metrics.
    Returns (summary_dict, probs_array)
    """
    # Получаем массив вероятностей класcа 1
    try:
        probs_raw = pipe.predict_proba(X_test)
        # иногда predict_proba может вернуть массив 1D или 2D; приводим к 1D массива положительного класса
        probs = None
        pr = np.asarray(probs_raw)
        if pr.ndim == 1:
            # если модель вернула одномерный массив — используем его как score
            probs = pr.ravel()
        elif pr.ndim == 2:
            if pr.shape[1] == 1:
                probs = pr.ravel()
            else:
                # ожидаем, что класс 1 в колонке 1
                probs = pr[:, 1].ravel()
        else:
            # неожиданный формат
            probs = pr.ravel()
    except Exception as e_pp:
        # попробуем вызвать predict_proba на numpy values (иногда ColumnTransformer/DF влияет)
        try:
            probs = np.asarray(pipe.predict_proba(X_test.values))[:, 1].ravel()
        except Exception as e2:
            raise RuntimeError(f"predict_proba failed: {e_pp}; fallback failed: {e2}")

    # Приводим y_test к numpy 1D
    y_arr = np.asarray(y_test).ravel()

    # Диагностика длин/типов
    if probs.shape[0] != y_arr.shape[0]:
        # печатаем подробную диагностику и бросаем явную ошибку
        raise RuntimeError(
            "Length mismatch between y_test and predicted probs: "
            f"len(y_test)={y_arr.shape[0]}, len(probs)={probs.shape[0]}. "
            "Investigate pipeline.predict_proba output. "
            f"probs.shape={probs.shape}, probs.dtype={probs.dtype}"
        )

    # Удаляем NaN / inf если вдруг
    mask_valid = np.isfinite(probs)
    if not np.all(mask_valid):
        # если здесь удаляем, то синхронно уменьшаем y_arr
        probs = probs[mask_valid]
        y_arr = y_arr[mask_valid]

    # final safety cast to float
    probs = probs.astype(float)

    # вычисляем метрики
    preds = (probs >= threshold).astype(int)
    report = classification_report(y_arr, preds, output_dict=True, zero_division=0)

    # roc_auc_score может упасть, если все y одинаковы — ловим исключение
    try:
        roc = float(roc_auc_score(y_arr, probs))
    except Exception as e_roc:
        roc = None
        print("[evaluate] roc_auc_score failed:", e_roc)

    try:
        prec, rec, thr = precision_recall_curve(y_arr, probs)
        pr_auc = float(auc(rec, prec))
    except Exception as e_pr:
        prec, rec, thr = None, None, None
        pr_auc = None
        print("[evaluate] precision_recall_curve failed:", e_pr)

    cm = confusion_matrix(y_arr, preds)
    summary = {
        "classification_report": report,
        "roc_auc": roc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "probs_min": float(np.min(probs)) if probs.size else None,
        "probs_max": float(np.max(probs)) if probs.size else None,
        "probs_mean": float(np.mean(probs)) if probs.size else None,
        "n_test": int(len(y_arr)),
    }
    return summary, probs

def build_pipeline_and_train(df, target_col="is_fraud", out_dir="artifacts", sample_frac=None):
    df = df.copy()
    if sample_frac is not None and 0 < sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # decide features
    all_cols = [c for c in df.columns if c != target_col]
    categorical = [
        c
        for c in ["mcc", "country_risk", "receiver_new", "device_new", "is_foreign"]
        if c in all_cols
    ]
    numeric = [c for c in all_cols if c not in categorical]

    print("Numeric features:", numeric[:10])
    print("Categorical features:", categorical)

    X = df[numeric + categorical]
    y = df[target_col].astype(int)

    # split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # IMPORTANT: make sure categorical columns are strings -> allows fill_value='NA'
    for D in (X_train, X_val, X_test):
        for c in categorical:
            if c in D.columns:
                D[c] = D[c].fillna("NA").astype(str)

    preproc = build_preprocessor(numeric, categorical)

    # Fit preprocessor on training data (categoricals are strings now)
    print("Fitting preprocessor on training data...")
    preproc.fit(X_train)

    # transform
    X_train_trans = preproc.transform(X_train)
    X_val_trans = preproc.transform(X_val)

    # train LGBM on transformed arrays (with safe early stopping fallback)
    clf = train_lgbm(X_train_trans, y_train, X_val_trans, y_val)

    # build final pipeline: preproc + trained clf
    pipe = Pipeline([("preproc", preproc), ("clf", clf)])

    # evaluate
    eval_summary, probs = evaluate(pipe, X_test, y_test)

    # save artifacts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe_path = out_dir / "pipeline_ieee.joblib"
    meta_path = out_dir / "meta_ieee.joblib"
    report_path = out_dir / "train_report_ieee.json"

    joblib.dump(pipe, pipe_path)
    meta = {
        "features": numeric + categorical,
        "target": target_col,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    joblib.dump(meta, meta_path)

    with open(report_path, "w") as f:
        json.dump(eval_summary, f, indent=2)

    print("Saved pipeline to:", pipe_path)
    print("Saved meta to:", meta_path)
    print("Saved report to:", report_path)
    print("Evaluation summary:", eval_summary)

    # SHAP (optional)
    shap_path = out_dir / "shap_explainer_ieee.joblib"
    if SHAP_AVAILABLE:
        try:
            print("Building SHAP TreeExplainer (may take time)...")
            X_bg = X_train.sample(n=min(2000, len(X_train)), random_state=42)
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
    args = p.parse_args()

    df = load_data(args.input)
    build_pipeline_and_train(df, out_dir=args.out_dir, sample_frac=args.sample)
    print("Done.")


if __name__ == "__main__":
    main()