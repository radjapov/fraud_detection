#!/usr/bin/env python3.14
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

PIPE = "artifacts/pipeline_ieee.joblib"
META_OUT = "artifacts/meta_ieee_threshold.json"
DATA = "data/ieee_prepared.csv"  # должен содержать столбец is_fraud (0/1)

print("Loading pipeline:", PIPE)
pipe = joblib.load(PIPE)
print("Loading sample data:", DATA)
df = pd.read_csv(DATA)
if "is_fraud" not in df.columns:
    raise SystemExit("data must contain column 'is_fraud'")

X = df.drop(columns=["is_fraud"])
y = df["is_fraud"].astype(int).values

print("Computing probabilities (this may take time)...")
probs = pipe.predict_proba(X)[:, 1]

thrs = np.linspace(0.0, 0.9, 91)
rows = []
best_f1 = (-1, None)
for t in thrs:
    preds = (probs >= t).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
    acc = accuracy_score(y, preds)
    rows.append((t, acc, prec, rec, f1))
    if f1 > best_f1[0]:
        best_f1 = (f1, t)

print("\nBest F1: %.4f at threshold %.3f\n" % (best_f1[0], best_f1[1]))
# show top thresholds by recall (but keep precision >= 0.2 as an example)
rows_sorted_by_recall = sorted(rows, key=lambda r: (r[3], r[2]), reverse=True)
for r in rows_sorted_by_recall[:8]:
    print("thr %.3f  acc %.4f  prec %.4f  rec %.4f  f1 %.4f" % r)

# Save chosen threshold (F1) to meta file
meta = {
    "chosen_threshold": float(best_f1[1]),
    "chosen_metric": "f1",
    "note": "auto-selected by eval_thresholds.py",
}
os.makedirs("artifacts", exist_ok=True)
with open(META_OUT, "w") as f:
    json.dump(meta, f, indent=2)
print("\nSaved meta with chosen threshold to", META_OUT)
