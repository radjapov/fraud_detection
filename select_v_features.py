#!/usr/bin/env python3
"""
Choose which of the 339 Vesta ``V`` columns the model keeps.

All 339 together do not help (the model overfits them), while the 30 most useful ones do.
Importance is measured on the TRAINING block of the production split only
(``fraud_app.features.split_data``); the validation block is used for early stopping and the
test period is never touched.

Usage:
  python3 select_v_features.py [--top 30] [--label-delay-days 30]

Paste the printed list into ``V_SELECTED`` in fraud_app/schema.py, then rebuild the dataset.
Needs data/ieee/train_transaction.csv (for the V columns) and data/ieee_prepared.csv.
"""
import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from fraud_app import schema
from fraud_app.features import DEFAULT_LABEL_DELAY_DAYS, TIME_COL, split_data
from fraud_app.schema import CATEGORICAL_NAMES

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
RARE = 200  # categories with fewer training rows are treated as missing


def load_with_all_v(prepared: Path, raw: Path):
    """The prepared base features plus every V column, aligned row by row."""
    df = pd.read_csv(prepared)
    header = pd.read_csv(raw, nrows=1).columns
    v_cols = [c for c in header if c[0] == "V" and c[1:].isdigit()]
    tr = pd.read_csv(raw, usecols=["TransactionDT", "TransactionAmt"] + v_cols)
    tr = tr.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df = df.sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    if len(tr) != len(df) or not (tr["TransactionDT"].values == df[TIME_COL].values).all():
        raise SystemExit("prepared CSV and raw transactions are not row-aligned; rebuild the CSV")
    df = df.drop(columns=[c for c in df.columns if c in v_cols])  # in case some are already there
    return pd.concat([df, tr[v_cols]], axis=1), v_cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default=str(BASE_DIR / "data" / "ieee_prepared.csv"))
    ap.add_argument("--raw", default=str(BASE_DIR / "data" / "ieee" / "train_transaction.csv"))
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--label-delay-days", type=float, default=DEFAULT_LABEL_DELAY_DAYS)
    args = ap.parse_args()

    df, v_cols = load_with_all_v(Path(args.prepared), Path(args.raw))
    in_v_or_id = {f.name for f in schema.FEATURES if f.group in (schema.G_V, schema.G_ID)}
    base = [f.name for f in schema.FEATURES if f.name not in in_v_or_id and f.name in df.columns]

    train, val, _ = split_data(df, args.label_delay_days)  # test rows are dropped right here
    print(f"train {len(train)} rows | val {len(val)} rows | {len(base)} base + {len(v_cols)} V columns")

    cats = {
        c: sorted(train[c].dropna().astype(str).value_counts().loc[lambda s: s >= RARE].index)
        for c in base
        if c in CATEGORICAL_NAMES
    }

    def frame(part, cols):
        X = part[cols].copy()
        for c in cols:
            if c in cats:
                v = X[c].astype("object").map(lambda x: None if pd.isna(x) else str(x))
                X[c] = pd.Categorical(v, categories=cats[c])
        return X

    def fit(cols):
        clf = lgb.LGBMClassifier(
            n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=40,
            colsample_bytree=0.5, subsample=0.8, subsample_freq=1, random_state=1, verbosity=-1,
        )
        clf.fit(
            frame(train, cols), train.is_fraud,
            eval_set=[(frame(val, cols), val.is_fraud)], eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        return clf

    clf = fit(base + v_cols)
    gain = pd.Series(clf.booster_.feature_importance("gain"), index=base + v_cols)
    top = gain[v_cols].sort_values(ascending=False).head(args.top)
    chosen = sorted(top.index, key=lambda c: int(c[1:]))

    for name, cols in (("base only", base), (f"base + top {args.top} V", base + chosen)):
        p = fit(cols).predict_proba(frame(val, cols))[:, 1]
        print(f"validation PR-AUC, {name:16s}: {average_precision_score(val.is_fraud, p):.4f}")

    print(f"\nTop {args.top} V columns by gain (training block only):")
    print(", ".join(f"{c} ({g / gain.sum():.1%})" for c, g in top.items()))
    print("\nV_SELECTED = [" + ", ".join(f'"{c}"' for c in chosen) + "]")


if __name__ == "__main__":
    main()
