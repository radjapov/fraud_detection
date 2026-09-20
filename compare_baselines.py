#!/usr/bin/env python3
"""
compare_baselines.py — Baseline model comparison for FraudDetect.

Same protocol as production (see train_ieee_lgbm.py): chronological split with a label delay,
fit on the training block, threshold on the gap-separated validation block, then refit on all
labeled rows and score the test period.

NOTE: Uses manual ROC-AUC and PR-AUC implementations to work around
a sklearn bug in _binary_clf_curve on Python 3.14 / numpy 2.x
(IndexError: index N is out of bounds for axis 0 with size N).

Usage:
    python3 compare_baselines.py
    python3 compare_baselines.py --sample 0.3
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from fraud_app.features import (
    DEFAULT_LABEL_DELAY_DAYS,
    build_preprocessor,
    category_labels,
    label_delay_of,
    labeled_rows,
    model_columns,
    split_data,
)
from fraud_app.schema import CATEGORICAL_NAMES

BASE_DIR     = Path(__file__).resolve().parent
DATA_PATH    = BASE_DIR / "data" / "ieee_prepared.csv"
OUT_JSON     = BASE_DIR / "artifacts" / "baseline_comparison.json"
TARGET       = "is_fraud"
RANDOM_STATE = 42


# ── Metrics (pure numpy — avoids sklearn _binary_clf_curve Python 3.14 bug) ──

def calc_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Trapezoidal ROC-AUC via cumulative TP/FP counts. No sklearn ranking."""
    order = np.argsort(y_score, kind="stable")[::-1]
    yt    = y_true[order].astype(np.float64)
    npos  = yt.sum();  nneg = len(yt) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    tpr = np.concatenate([[0.0], np.cumsum(yt)       / npos])
    fpr = np.concatenate([[0.0], np.cumsum(1.0 - yt) / nneg])
    # trapezoid: Σ Δfpr * (tpr_left + tpr_right) / 2
    return float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2.0))


def calc_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision via mean precision at each positive recall point."""
    order   = np.argsort(y_score, kind="stable")[::-1]
    yt      = y_true[order].astype(np.int32)
    npos    = int(yt.sum())
    if npos == 0:
        return 0.0
    pos_idx  = np.nonzero(yt)[0]                                 # ranks of positives
    cumhits  = np.arange(1, len(pos_idx) + 1, dtype=np.float64)  # 1,2,3,...
    prec_at  = cumhits / (pos_idx + 1).astype(np.float64)        # precision at each TP
    return float(prec_at.mean())


# ─────────────────────────────────────────────────────────────────────────────

# Boosted trees take NaN natively (missing stays a signal); the others need imputation.
NAN_NATIVE = {"LightGBM", "XGBoost"}


def make_pipe(name, estimator, numeric, categorical) -> Pipeline:
    """Same preprocessing as production (fraud_app.features), imputing only where needed."""
    pre = build_preprocessor(numeric, categorical, impute=name not in NAN_NATIVE)
    return Pipeline([("preproc", pre), ("clf", estimator)])


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, steps: int = 200) -> float:
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.01, 0.99, steps):
        f1 = f1_score(y_true, (y_score >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def get_models() -> list:
    models = []

    try:
        from lightgbm import LGBMClassifier
        models.append(("LightGBM", LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            class_weight="balanced", n_jobs=-1,
            random_state=RANDOM_STATE, verbose=-1,
        )))
    except ImportError:
        print("[warn] lightgbm not installed")

    try:
        from xgboost import XGBClassifier
        models.append(("XGBoost", XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            scale_pos_weight=27, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE, verbosity=0,
        )))
    except ImportError:
        print("[warn] xgboost not installed")

    try:
        from sklearn.ensemble import RandomForestClassifier
        models.append(("RandomForest", RandomForestClassifier(
            n_estimators=300, max_depth=20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        )))
    except ImportError:
        pass

    # lbfgs: same convex L2 problem as saga, but minutes faster on ~300 dense columns
    models.append(("LogisticRegression", LogisticRegression(
        max_iter=500, class_weight="balanced",
        solver="lbfgs", C=0.01, random_state=RANDOM_STATE,
    )))

    return models


def evaluate(name, estimator, numeric, categorical, Xtr, ytr, Xval, yval, Xlab, ylab, Xte, yte) -> dict:
    """Same protocol as production: fit on train, threshold on val, refit on all labeled rows."""
    print(f"\n[{name}] Training ...")
    t0   = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage_a = make_pipe(name, clone(estimator), numeric, categorical).fit(Xtr, ytr)
        val_p = np.asarray(stage_a.predict_proba(Xval)[:, 1], dtype=np.float64)
        thr   = best_threshold(yval, val_p)
        final = make_pipe(name, clone(estimator), numeric, categorical).fit(Xlab, ylab)
        te_p  = np.asarray(final.predict_proba(Xte)[:, 1], dtype=np.float64)
    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[{name}] Done in {elapsed}s")

    te_pred = (te_p >= thr).astype(int)
    roc = calc_roc_auc(yte, te_p)
    pr  = calc_pr_auc(yte, te_p)
    pre = float(precision_score(yte, te_pred, zero_division=0))
    rec = float(recall_score(yte, te_pred, zero_division=0))
    f1  = float(f1_score(yte, te_pred, zero_division=0))

    print(f"[{name}] ROC={roc:.4f} PR={pr:.4f} P={pre:.4f} R={rec:.4f} F1={f1:.4f} τ={thr:.3f}")
    return dict(model=name,
                roc_auc=round(roc, 4), pr_auc=round(pr, 4),
                threshold=round(thr, 3),
                precision=round(pre, 4), recall=round(rec, 4), f1=round(f1, 4),
                train_time_s=elapsed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",     default=str(DATA_PATH))
    ap.add_argument("--out-json", default=str(OUT_JSON))
    ap.add_argument("--sample",   type=float, default=None)
    ap.add_argument("--label-delay-days", type=float, default=None,
                    help="default: the delay the production model was trained with")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    if args.sample:
        df = df.sample(frac=args.sample, random_state=RANDOM_STATE).reset_index(drop=True)

    features = model_columns(df)
    categorical = [c for c in features if c in CATEGORICAL_NAMES]
    numeric = [c for c in features if c not in categorical]
    delay = args.label_delay_days
    if delay is None:
        meta_path = BASE_DIR / "artifacts" / "meta_ieee.joblib"
        delay = label_delay_of(joblib.load(meta_path)) if meta_path.exists() else DEFAULT_LABEL_DELAY_DAYS
    train_df, val_df, test_df = split_data(df, delay)  # same chronological split as training
    print(f"[split] label delay {delay:g} days")

    def xy(part):
        X = part[numeric + categorical].copy()
        for c in categorical:  # same labelling code as training / serving
            X[c] = category_labels(X[c])
        return X, part[TARGET].to_numpy(dtype=np.int32)

    Xtr, ytr = xy(train_df)
    Xval, yval = xy(val_df)
    Xlab, ylab = xy(labeled_rows(df, delay))
    Xte, yte = xy(test_df)
    print(f"[data] {len(df):,} rows | fraud rate {df[TARGET].mean():.4f} | "
          f"features: {len(numeric)} numeric + {len(categorical)} categorical")
    print(f"[split] train={len(ytr):,} val={len(yval):,} test={len(yte):,}")

    results = []
    for name, est in get_models():
        try:
            results.append(evaluate(name, est, numeric, categorical, Xtr, ytr, Xval, yval, Xlab, ylab, Xte, yte))
        except Exception as e:
            import traceback
            print(f"[{name}] FAILED: {e}")
            traceback.print_exc()

    results.sort(key=lambda r: r["roc_auc"], reverse=True)

    print("\n" + "=" * 78)
    print(f"{'Model':<22} {'ROC-AUC':>8} {'PR-AUC':>8} {'Prec':>7} {'Recall':>7} {'F1':>7} {'τ':>6} {'s':>6}")
    print("-" * 78)
    for r in results:
        print(f"{r['model']:<22} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} "
              f"{r['threshold']:>6.3f} {r['train_time_s']:>6.1f}")
    print("=" * 78)

    OUT_JSON.parent.mkdir(exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"label_delay_days": delay, "results": results}, f, indent=2)
    print(f"\n[done] → {args.out_json}")


if __name__ == "__main__":
    main()