#!/usr/bin/env python3
"""
Evaluate the IEEE LGBM pipeline on the held-out TEST split (same split as
train_ieee_lgbm.py) and export metrics to artifacts/metrics_ieee.json.

Usage:
    python3 eval_model_metrics.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from fraud_app.features import TARGET, label_delay_of, prepare_features, split_data

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ieee_prepared.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline_ieee.joblib"
META_PATH = ARTIFACTS_DIR / "meta_ieee.joblib"
THRESHOLD_JSON = ARTIFACTS_DIR / "meta_ieee_threshold.json"
METRICS_JSON = ARTIFACTS_DIR / "metrics_ieee.json"


# -------------------------
# Manual fallback metrics
# -------------------------

def manual_roc_auc(y_true, y_score):
    """Compute ROC-AUC manually when sklearn fails."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = (y_true == 1).sum()
    N = (y_true == 0).sum()
    if P == 0 or N == 0:
        return float("nan")

    tps = np.cumsum(y_true == 1)
    fps = np.cumsum(y_true == 0)

    tpr = tps / P
    fpr = fps / N

    return float(np.trapz(tpr, fpr))


def manual_pr_auc(y_true, y_score):
    """Average precision (mean precision at each true positive) when sklearn fails."""
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score), kind="stable")
    y_sorted = y_true[order]
    n_pos = int((y_sorted == 1).sum())
    if n_pos == 0:
        return float("nan")
    ranks = np.flatnonzero(y_sorted == 1) + 1  # 1-based rank of every positive
    return float(np.mean(np.arange(1, n_pos + 1) / ranks))


# -------------------------
# Threshold loader
# -------------------------

def load_threshold(default=0.35):
    """Load selected threshold from JSON."""
    if THRESHOLD_JSON.exists():
        try:
            with open(THRESHOLD_JSON, "r") as f:
                data = json.load(f)
            thr = data.get("threshold", data.get("chosen_threshold", default))
            print(f"[threshold] Loaded threshold={thr}")
            return float(thr)
        except Exception as e:
            print(f"[threshold] Failed to load threshold file: {e}")

    print(f"[threshold] Using default={default}")
    return float(default)


# -------------------------
# Main script
# -------------------------

def main():
    # Load dataset
    print(f"[data] Loading {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    if "is_fraud" not in df.columns:
        raise RuntimeError("Column 'is_fraud' missing from dataset.")

    # Load model
    print(f"[model] Loading pipeline from {PIPELINE_PATH} ...")
    pipe = joblib.load(PIPELINE_PATH)

    # Only the held-out test rows: the model never saw them, and the threshold
    # was chosen on the validation split.
    delay = label_delay_of(joblib.load(META_PATH) if META_PATH.exists() else None)
    _, _, test = split_data(df, delay)
    print(f"[data] Evaluating on test split: {len(test)} of {len(df)} rows (label delay {delay:g}d)")
    y = test[TARGET].astype(int).values
    feat_list = list(getattr(pipe, "feature_names_in_", [c for c in df.columns if c != TARGET]))
    X, _ = prepare_features(test, feat_list, pipe)

    # Predict probabilities
    print("[model] Computing predict_proba ...")
    probs = pipe.predict_proba(X)[:, 1]

    n_samples = len(y)
    fraud_rate = float(y.mean())

    # --- Global metrics ---
    try:
        roc_auc = float(roc_auc_score(y, probs))
    except Exception as e:
        print(f"[metric] ROC-AUC failed: {e}, using manual ROC-AUC.")
        roc_auc = manual_roc_auc(y, probs)

    try:
        pr_auc = float(average_precision_score(y, probs))
    except Exception as e:
        print(f"[metric] PR-AUC failed: {e}, using manual PR-AUC.")
        pr_auc = manual_pr_auc(y, probs)

    # Load classification threshold
    threshold = load_threshold()

    # Predictions at threshold
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(y, preds)
    TN, FP, FN, TP = cm.ravel()

    acc = (TP + TN) / n_samples
    precision_thr = TP / (TP + FP + 1e-9)
    recall_thr = TP / (TP + FN + 1e-9)
    f1_thr = 2 * precision_thr * recall_thr / (precision_thr + recall_thr + 1e-9)

    report = classification_report(y, preds, output_dict=True, zero_division=0)

    # --- Save JSON ---
    metrics = {
        "split": "test",
        "label_delay_days": delay,
        "samples": n_samples,
        "fraud_rate": fraud_rate,
        "probability_stats": {
            "min": float(probs.min()),
            "max": float(probs.max()),
            "mean": float(probs.mean()),
        },
        "global_metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },
        "threshold_metrics": {
            "threshold": threshold,
            "accuracy": acc,
            "precision": precision_thr,
            "recall": recall_thr,
            "f1": f1_thr,
            "confusion_matrix": {
                "TN": int(TN),
                "FP": int(FP),
                "FN": int(FN),
                "TP": int(TP),
            },
            "classification_report": report,
        },
    }

    # Validation block scored by the stage-A model (before the refit): the honest expectation.
    val_file = ARTIFACTS_DIR / "val_scores_ieee.npz"
    if val_file.exists():
        vs = np.load(val_file)
        yv, pv = vs["y"].astype(int), vs["p"].astype(float)
        vpred = (pv >= threshold).astype(int)
        vtp = int(((vpred == 1) & (yv == 1)).sum())
        vfp = int(((vpred == 1) & (yv == 0)).sum())
        vfn = int(((vpred == 0) & (yv == 1)).sum())
        vprec = vtp / max(vtp + vfp, 1)
        vrec = vtp / max(vtp + vfn, 1)
        metrics["validation"] = {
            "samples": int(len(yv)),
            "roc_auc": float(roc_auc_score(yv, pv)),
            "pr_auc": float(average_precision_score(yv, pv)),
            "threshold": threshold,
            "precision": vprec,
            "recall": vrec,
            "f1": 2 * vprec * vrec / max(vprec + vrec, 1e-12),
        }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[done] Metrics saved to {METRICS_JSON}\n")


if __name__ == "__main__":
    main()