#!/usr/bin/env python3
"""
Create a simplified dataset from IEEE-CIS Fraud Detection files.

Place train_transaction.csv and train_identity.csv (optional) into:
  data/ieee/

Then run:
  python3 create_ieee_dataset.py

Output:
  data/ieee_prepared.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_csv(path, nrows=None):
    print(f"Loading {path} ...")
    return pd.read_csv(path, nrows=nrows)


def safe_merge(tr, idf):
    # merge on TransactionID if identity provided
    print("Merging transaction + identity")
    return tr.merge(idf, how="left", on="TransactionID", suffixes=("", "_id"))


def hour_from_dt(dt_series):
    # TransactionDT is seconds since some reference time — we extract hour of day by modulo day
    secs_in_day = 24 * 3600
    return ((dt_series % secs_in_day) // 3600).astype(int)


def compute_card_age_months(df):
    # approximate: for each card1 compute months since first seen in dataset
    if "card1" not in df.columns:
        return pd.Series(0.0, index=df.index)
    grp = df.groupby("card1")["TransactionDT"]
    min_dt = grp.transform("min")
    months = (df["TransactionDT"] - min_dt) / (3600 * 24 * 30)
    months = months.clip(lower=0)
    return months.fillna(0.0)


def sender_txn_24h_and_avg(df):
    # approximate sender by card1; bucket by day (TransactionDT // secs_in_day)
    if "card1" not in df.columns or "TransactionDT" not in df.columns:
        # return zeros / medians
        return pd.Series(0, index=df.index), pd.Series(df.get("TransactionAmt", 0.0)).astype(float)
    secs_in_day = 24 * 3600
    day = (df["TransactionDT"] // secs_in_day).astype(int)
    # create temporary grouping keys without modifying original df globally
    tmp = df[["card1", "TransactionID", "TransactionAmt"]].copy()
    tmp["_day"] = day
    counts = tmp.groupby(["card1", "_day"])["TransactionID"].transform("count")
    mean_amt = tmp.groupby("card1")["TransactionAmt"].transform("mean")
    counts = counts.fillna(0).astype(int)
    mean_amt = mean_amt.fillna(df["TransactionAmt"].median() if "TransactionAmt" in df.columns else 0.0)
    return counts, mean_amt


def compute_distance_km(df):
    # use dist1 then dist2 as fallback
    if "dist1" in df.columns:
        d = df["dist1"].fillna(df["dist2"] if "dist2" in df.columns else 0)
        return pd.to_numeric(d, errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def compute_ip_risk(df, v_max=50):
    # use V1..V{v_max} mean as proxy, then minmax-scale to 0..1
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return pd.Series(0.5, index=df.index)  # neutral
    v_cols = v_cols[:v_max]
    vals = df[v_cols].abs().mean(axis=1).fillna(0.0)
    scaler = MinMaxScaler()
    try:
        out = scaler.fit_transform(vals.values.reshape(-1, 1)).flatten()
    except Exception:
        out = vals.values
        out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return pd.Series(out, index=df.index)


def first_occurrence_flag(series):
    # 1 if this value was never seen before (first occurrence), else 0
    # We consider first occurrence within the whole dataframe as "new"
    filled = series.fillna("__nan__")
    # cumcount per key — first occurrence -> 0
    grp = filled.groupby(filled).cumcount()
    return (grp == 0).astype(int)


def device_new_flag(df):
    # use DeviceInfo or id_30 as device fingerprint
    if "DeviceInfo" in df.columns:
        return first_occurrence_flag(df["DeviceInfo"])
    if "id_30" in df.columns:
        return first_occurrence_flag(df["id_30"])
    return pd.Series(0, index=df.index)


def receiver_new_flag(df):
    # use addr1 as receiver proxy
    if "addr1" in df.columns:
        return first_occurrence_flag(df["addr1"])
    return pd.Series(0, index=df.index)


def is_foreign_flag(df):
    # heuristic: if dist1 large or addr1 missing -> foreign (1), else 0
    d = pd.Series(0, index=df.index)
    if "dist1" in df.columns:
        d = (pd.to_numeric(df["dist1"].fillna(0), errors="coerce") > 500).astype(int)
    if "addr1" in df.columns:
        d = d | df["addr1"].isna().astype(int)
    return d.astype(int)


def mcc_from_product(df):
    # no real MCC in IEEE; use ProductCD categorical mapped to codes
    if "ProductCD" in df.columns:
        cat = df["ProductCD"].astype(str).fillna("UNK").astype("category")
        return pd.Series(cat.cat.codes, index=df.index).astype(int)
    if "TransactionType" in df.columns:
        cat = df["TransactionType"].astype(str).fillna("UNK").astype("category")
        return pd.Series(cat.cat.codes, index=df.index).astype(int)
    return pd.Series(0, index=df.index)


def country_risk_from_addr(df):
    # heuristic: quantile-bucketize addr1 (numeric) into 0/1/2 risk
    if "addr1" in df.columns:
        a = pd.to_numeric(df["addr1"], errors="coerce").fillna(-1)
        mask_nan = a == -1
        out = pd.Series(index=a.index, dtype=int)
        non_nan = a[~mask_nan]
        if len(non_nan) > 0:
            q = pd.qcut(non_nan, q=3, labels=[0, 1, 2])
            out.loc[~mask_nan] = q.astype(int).values
        else:
            out.loc[:] = 1
        out.loc[mask_nan] = 2
        return out.fillna(1).astype(int)
    return pd.Series(1, index=df.index)


def build_features(tr, idf=None, sample_frac=None):
    df = tr.copy()
    if idf is not None:
        df = safe_merge(df, idf)

    print("Computing core features...")

    # safe defaults and basic conversions
    if "TransactionAmt" in df.columns:
        df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
    if "TransactionDT" in df.columns:
        df["TransactionDT"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(int)
    else:
        df["TransactionDT"] = 0

    df["amount"] = df.get("TransactionAmt", 0.0).astype(float)
    df["hour"] = hour_from_dt(df["TransactionDT"].fillna(0).astype(int))
    df["card_age_months"] = compute_card_age_months(df).fillna(0.0)

    counts_24h, mean_amt = sender_txn_24h_and_avg(df)
    df["sender_txn_24h"] = counts_24h.astype(int)
    df["sender_avg_amount"] = mean_amt.astype(float)

    df["distance_km"] = compute_distance_km(df)
    df["ip_risk"] = compute_ip_risk(df, v_max=50)
    df["receiver_new"] = receiver_new_flag(df).astype(int)
    df["device_new"] = device_new_flag(df).astype(int)
    df["is_foreign"] = is_foreign_flag(df).astype(int)
    df["mcc"] = mcc_from_product(df).astype(int)
    df["country_risk"] = country_risk_from_addr(df).astype(int)

    # label fallback logic
    if "isFraud" in df.columns:
        df["is_fraud"] = df["isFraud"].astype(int)
    else:
        df["is_fraud"] = df.get("fraud", pd.Series(0, index=df.index)).fillna(0).astype(int)

    feature_cols = [
        "amount",
        "hour",
        "card_age_months",
        "sender_txn_24h",
        "sender_avg_amount",
        "distance_km",
        "ip_risk",
        "receiver_new",
        "device_new",
        "is_foreign",
        "mcc",
        "country_risk",
        "is_fraud",
    ]

    # ensure all exist and correct dtypes
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    out = df[feature_cols].copy()

    out["sender_avg_amount"] = out["sender_avg_amount"].fillna(out["amount"].median() if len(out) else 0.0)
    out["distance_km"] = out["distance_km"].fillna(0.0)
    out["ip_risk"] = out["ip_risk"].fillna(0.5)
    out["card_age_months"] = out["card_age_months"].fillna(out["card_age_months"].median() if len(out) else 0.0)
    out["mcc"] = out["mcc"].fillna(0).astype(int)
    out["country_risk"] = out["country_risk"].fillna(1).astype(int)

    # optional sampling to smaller file for quick iter
    if sample_frac is not None and 0 < sample_frac < 1.0:
        out = out.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    # final dtype enforcement
    out["amount"] = out["amount"].astype(float)
    out["hour"] = out["hour"].astype(int)
    out["card_age_months"] = out["card_age_months"].astype(float)
    out["sender_txn_24h"] = out["sender_txn_24h"].astype(int)
    out["sender_avg_amount"] = out["sender_avg_amount"].astype(float)
    out["distance_km"] = out["distance_km"].astype(float)
    out["ip_risk"] = out["ip_risk"].astype(float)
    out["receiver_new"] = out["receiver_new"].astype(int)
    out["device_new"] = out["device_new"].astype(int)
    out["is_foreign"] = out["is_foreign"].astype(int)
    out["mcc"] = out["mcc"].astype(int)
    out["country_risk"] = out["country_risk"].astype(int)
    out["is_fraud"] = out["is_fraud"].astype(int)

    return out


def main(args):
    data_dir = Path(args.input_dir)
    out_path = Path(args.output)
    sample = args.sample_frac
    nrows = args.nrows

    tr_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    if not tr_path.exists():
        print("ERROR: train_transaction.csv not found in", data_dir)
        return 2

    tr = load_csv(tr_path, nrows=nrows)
    idf = load_csv(id_path, nrows=nrows) if id_path.exists() else None

    df_out = build_features(tr, idf, sample_frac=sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote prepared dataset to: {out_path}")
    print("Shape:", df_out.shape)
    print("Fraud rate:", df_out["is_fraud"].mean())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-dir",
        default="data/ieee",
        help="dir with train_transaction.csv (and optional train_identity.csv)",
    )
    p.add_argument("--output", default="data/ieee_prepared.csv", help="where to save prepared csv")
    p.add_argument(
        "--sample-frac", type=float, default=None, help="optional fraction to sample (0-1)"
    )
    p.add_argument("--nrows", type=int, default=None, help="optional number of rows to read (for testing)")
    args = p.parse_args()
    exit(main(args) or 0)