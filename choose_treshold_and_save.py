#!/usr/bin/env python3
# choose_threshold_and_save.py — finds optimal threshold for IEEE model and saves it.

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BASE = Path(__file__).resolve().parent
ART = BASE / "artifacts"
DATA = BASE / "data" / "ieee_prepared.csv"
PIPE_PATH = ART / "pipeline_ieee.joblib"
THR_JSON = ART / "meta_ieee_threshold.json"


def ensure_features(df, feat_list):
    for f in feat_list:
        if f not in df.columns:
            df[f] = 0
    return df[feat_list].fillna(0)


def main():
    print("Running choose_threshold_and_save.py")

    if not PIPE_PATH.exists():
        raise FileNotFoundError(f"No pipeline found: {PIPE_PATH}")

    pipe = joblib.load(PIPE_PATH)

    df = pd.read_csv(DATA)
    y = df["is_fraud"].astype(int).values
    X = df.drop(columns=["is_fraud"])

    feat_list = getattr(pipe, "feature_names_in_", list(X.columns))
    Xs = ensure_features(X.copy(), feat_list)

    probs = pipe.predict_proba(Xs)[:, 1]

    best_thr = 0.0
    best_f1 = -1

    for t in np.linspace(0, 1, 200):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = t

    print(f"Best F1={best_f1:.4f} at threshold={best_thr:.4f}")

    with open(THR_JSON, "w") as f:
        json.dump({"threshold": float(best_thr)}, f)

    print("Saved:", THR_JSON)


if __name__ == "__main__":
    main()
