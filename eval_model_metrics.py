#!/usr/bin/env python3
"""
Evaluate the IEEE LGBM pipeline and export metrics to artifacts/metrics_ieee.json.

Usage:
    python3 eval_model_metrics.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ieee_prepared.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
PIPELINE_PATH = ARTIFACTS_DIR / "pipeline_ieee.joblib"
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
    """Compute PR-AUC manually when sklearn fails."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(-y_score)
    y_true = y_true[order]

    P = (y_true == 1).sum()
    if P == 0:
        return float("nan")

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    for value in y_true:
        if value == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / P)

    recalls = np.asarray(recalls)
    precisions = np.asarray(precisions)

    order_r = np.argsort(recalls)
    return float(np.trapz(precisions[order_r], recalls[order_r]))


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

    y = df["is_fraud"].astype(int).values
    X = df.drop(columns=["is_fraud"])

    # Load model
    print(f"[model] Loading pipeline from {PIPELINE_PATH} ...")
    pipe = joblib.load(PIPELINE_PATH)

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
        prec, rec, _ = precision_recall_curve(y, probs)
        pr_auc = float(auc(rec, prec))
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

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[done] Metrics saved to {METRICS_JSON}\n")


if __name__ == "__main__":
    main()