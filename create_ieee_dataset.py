# create_ieee_dataset.py
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
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_csv(path):
    print(f"Loading {path} ...")
    return pd.read_csv(path)


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
        return pd.Series(np.nan, index=df.index)
    grp = df.groupby("card1")["TransactionDT"]
    min_dt = grp.transform("min")
    months = (df["TransactionDT"] - min_dt) / (3600 * 24 * 30)
    return months.clip(lower=0)


def sender_txn_24h_and_avg(df):
    # approximate sender by card1; bucket by day (TransactionDT // secs_in_day)
    if "card1" not in df.columns:
        return pd.Series(0, index=df.index), pd.Series(np.nan, index=df.index)
    day = (df["TransactionDT"] // (24 * 3600)).astype(int)
    df["_day"] = day
    counts = df.groupby(["card1", "_day"])["TransactionID"].transform("count")
    mean_amt = df.groupby("card1")["TransactionAmt"].transform("mean")
    df.drop(columns=["_day"], inplace=True)
    return counts.fillna(0).astype(int), mean_amt


def compute_distance_km(df):
    # use dist1 then dist2 as fallback
    if "dist1" in df.columns:
        d = df["dist1"].fillna(df["dist2"] if "dist2" in df.columns else 0)
        # dist in dataset often in kilometers-ish; ensure numeric
        return pd.to_numeric(d, errors="coerce").fillna(0)
    return pd.Series(0.0, index=df.index)


def compute_ip_risk(df, v_max=50):
    # use V1..V{v_max} mean as proxy, then minmax-scale to 0..1
    v_cols = [c for c in df.columns if c.startswith("V")]
    if not v_cols:
        return pd.Series(0.5, index=df.index)  # neutral
    # pick first v_max
    v_cols = v_cols[:v_max]
    vals = df[v_cols].abs().mean(axis=1).fillna(0)
    scaler = MinMaxScaler()
    out = scaler.fit_transform(vals.values.reshape(-1, 1)).flatten()
    return pd.Series(out, index=df.index)


def first_occurrence_flag(series):
    # 1 if this value was never seen before (first occurrence), else 0
    # We consider first occurrence within the whole dataframe as "new"
    # transform: cumcount per key == 0
    if series.name is None:
        name = "col"
    else:
        name = series.name
    grp = series.fillna("__nan__")
    first = grp.groupby(grp).cumcount() == 0
    return (~first).astype(int) * 0 + first.astype(int)  # 1 for first, 0 otherwise


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
        d = (df["dist1"].fillna(0) > 500).astype(int)
    # if addr1 missing -> mark foreign
    if "addr1" in df.columns:
        d = d | df["addr1"].isna().astype(int)
    return d


def mcc_from_product(df):
    # no real MCC in IEEE; use ProductCD categorical mapped to codes
    if "ProductCD" in df.columns:
        return df["ProductCD"].astype(str).fillna("UNK").astype("category").cat.codes
    # fallback: map TransactionType if exists
    if "TransactionType" in df.columns:
        return df["TransactionType"].astype(str).fillna("UNK").astype("category").cat.codes
    # otherwise zero
    return pd.Series(0, index=df.index)


def country_risk_from_addr(df):
    # heuristic: quantile-bucketize addr1 (numeric) into 0/1/2 risk
    if "addr1" in df.columns:
        a = pd.to_numeric(df["addr1"], errors="coerce").fillna(-1)
        # very crude: nan/-1 -> high risk 2, else bucket by tertiles
        mask_nan = a == -1
        non_nan = a[~mask_nan]
        if len(non_nan) > 0:
            q = pd.qcut(non_nan, q=3, labels=[0, 1, 2])
            out = pd.Series(index=a.index, dtype=int)
            out[~mask_nan] = q.astype(int).values
        else:
            out = pd.Series(1, index=a.index)
        out[mask_nan] = 2
        return out.fillna(1).astype(int)
    return pd.Series(1, index=df.index)


def build_features(tr, idf=None, sample_frac=None):
    df = tr.copy()
    if idf is not None:
        df = safe_merge(df, idf)

    # core features
    print("Computing core features...")
    df["amount"] = df["TransactionAmt"].astype(float)
    df["hour"] = hour_from_dt(df["TransactionDT"].fillna(0).astype(int))
    df["card_age_months"] = compute_card_age_months(df).fillna(0)
    df["sender_txn_24h"], df["sender_avg_amount"] = sender_txn_24h_and_avg(df)
    df["distance_km"] = compute_distance_km(df)
    df["ip_risk"] = compute_ip_risk(df, v_max=50)
    df["receiver_new"] = receiver_new_flag(df)
    df["device_new"] = device_new_flag(df)
    df["is_foreign"] = is_foreign_flag(df)
    df["mcc"] = mcc_from_product(df)
    df["country_risk"] = country_risk_from_addr(df)

    # label
    if "isFraud" in df.columns:
        df["is_fraud"] = df["isFraud"].astype(int)
    elif "isFraud" in df.columns:
        df["is_fraud"] = df["isFraud"].astype(int)
    else:
        df["is_fraud"] = (
            df.get("isFraud", df.get("fraud", pd.Series(0, index=df.index))).fillna(0).astype(int)
        )

    # select output columns in the order we want
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

    # ensure all exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    out = df[feature_cols].copy()

    # fill missing sensible defaults
    out["sender_avg_amount"] = out["sender_avg_amount"].fillna(out["amount"].median())
    out["distance_km"] = out["distance_km"].fillna(0)
    out["ip_risk"] = out["ip_risk"].fillna(0.5)
    out["card_age_months"] = out["card_age_months"].fillna(out["card_age_months"].median())
    out["mcc"] = out["mcc"].fillna(0).astype(int)
    out["country_risk"] = out["country_risk"].fillna(1).astype(int)

    # optional sampling to smaller file for quick iter
    if sample_frac is not None and 0 < sample_frac < 1.0:
        out = out.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    return out


def main(args):
    data_dir = Path(args.input_dir)
    out_path = Path(args.output)
    sample = args.sample_frac

    tr_path = data_dir / "train_transaction.csv"
    id_path = data_dir / "train_identity.csv"

    if not tr_path.exists():
        print("ERROR: train_transaction.csv not found in", data_dir)
        return 2

    tr = load_csv(tr_path)
    idf = load_csv(id_path) if id_path.exists() else None

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
    args = p.parse_args()
    exit(main(args) or 0)
