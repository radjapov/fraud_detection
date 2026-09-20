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

from fraud_app.schema import FEATURES, ID_CATEGORICAL, ID_NUMERIC, V_SELECTED


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


class CausalGroups:
    """
    Per-row statistics over the EARLIER rows of the same group, so nothing from the future
    leaks into a time-based split. Rows are ordered by (group, time, original position);
    "earlier" means earlier in that order, so ties in time are resolved by row order.
    """

    def __init__(self, group, dt):
        group = np.asarray(group)
        dt = np.asarray(dt, dtype=np.int64)
        n = len(group)
        self.n = n
        self.order = np.lexsort((np.arange(n), dt, group))
        g_s = group[self.order]
        self.dt_s = dt[self.order]
        new_group = np.r_[True, g_s[1:] != g_s[:-1]]
        self.gid = np.cumsum(new_group) - 1
        self.start = np.flatnonzero(new_group)[self.gid]  # first sorted index of each row's group
        self.pos = np.arange(n)
        self.prior_n = self.pos - self.start  # earlier rows of the same group

    def restore(self, arr_sorted):
        out = np.empty_like(arr_sorted)
        out[self.order] = arr_sorted
        return out

    def window_count(self, seconds):
        """Rows of the group in the trailing window (dt - seconds, dt], the current one included."""
        key = self.gid.astype(np.int64) * 10**10 + self.dt_s
        left = np.searchsorted(key, key - seconds, side="right")
        return self.restore(self.pos + 1 - left)

    def prior_mean(self, values):
        """Mean of the values of EARLIER rows of the group (NaN when there are none)."""
        v = np.asarray(values, dtype=float)[self.order]
        c = np.cumsum(v)
        base = np.where(self.start > 0, c[self.start - 1], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = (c - v - base) / self.prior_n
        mean[self.prior_n == 0] = np.nan
        return self.restore(mean)


def sender_txn_24h_and_avg(df):
    """
    Causal per-card features (sender ~ card1):

      sender_txn_24h    - transactions of this card in the trailing 24h, incl. the current one
      sender_avg_amount - mean amount of the card's PREVIOUS transactions
                          (the current amount for a card's first transaction)
    """
    if "card1" not in df.columns or "TransactionDT" not in df.columns:
        # return zeros / medians
        return pd.Series(0, index=df.index), pd.Series(df.get("TransactionAmt", 0.0)).astype(float)

    amt = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    g = CausalGroups(df["card1"].fillna(-1).to_numpy(), df["TransactionDT"].to_numpy())
    mean = g.prior_mean(amt)
    mean = np.where(np.isnan(mean), amt, mean)
    return pd.Series(g.window_count(24 * 3600), index=df.index), pd.Series(mean, index=df.index)


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


RAW_NUMERIC = (
    ["card2", "card3", "card5", "addr1", "addr2"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
    + V_SELECTED
    + ID_NUMERIC
)
RAW_CATEGORICAL = (
    ["card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType"]
    + [f"M{i}" for i in range(1, 10)]
    + ID_CATEGORICAL
)


def prepare_base(tr, idf=None):
    """Merge, coerce the basics and sort chronologically (everything below relies on the order)."""
    df = tr.copy()
    if idf is not None:
        df = safe_merge(df, idf)

    if "TransactionAmt" in df.columns:
        df["TransactionAmt"] = pd.to_numeric(df["TransactionAmt"], errors="coerce").fillna(0.0)
    if "TransactionDT" in df.columns:
        df["TransactionDT"] = pd.to_numeric(df["TransactionDT"], errors="coerce").fillna(0).astype(int)
    else:
        df["TransactionDT"] = 0

    return df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def build_features(tr, idf=None, sample_frac=None):
    df = prepare_base(tr, idf)

    print("Computing core features...")

    df["amount"] = df.get("TransactionAmt", 0.0).astype(float)
    df["hour"] = hour_from_dt(df["TransactionDT"].fillna(0).astype(int))
    df["card_age_months"] = compute_card_age_months(df).fillna(0.0)

    counts_24h, mean_amt = sender_txn_24h_and_avg(df)
    df["sender_txn_24h"] = counts_24h.astype(int)
    df["sender_avg_amount"] = mean_amt.astype(float)

    df["distance_km"] = compute_distance_km(df)
    df["ip_risk"] = compute_ip_risk(df, v_max=50)
    df["is_foreign"] = is_foreign_flag(df).astype(int)
    df["mcc"] = mcc_from_product(df).astype(int)
    df["country_risk"] = country_risk_from_addr(df).astype(int)

    print("Computing extended features (raw card / email / C / D / M / V / identity)...")
    df["amount_decimal"] = (df["amount"] - np.floor(df["amount"])).round(3)
    df["dow"] = ((df["TransactionDT"] // 86400) % 7).astype(int)
    df["has_identity"] = df["id_01"].notna().astype(int) if "id_01" in df.columns else 0
    for c in RAW_NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in RAW_CATEGORICAL:
        if c not in df.columns:
            df[c] = np.nan

    # label fallback logic
    if "isFraud" in df.columns:
        df["is_fraud"] = df["isFraud"].astype(int)
    else:
        df["is_fraud"] = df.get("fraud", pd.Series(0, index=df.index)).fillna(0).astype(int)

    feature_cols = [f.name for f in FEATURES] + [
        "is_fraud",
        "TransactionDT",  # kept for the time-based split; NOT a model feature
    ]
    out = df[feature_cols].copy()

    # legacy demo features have no missing values by construction
    out["sender_avg_amount"] = out["sender_avg_amount"].fillna(out["amount"].median() if len(out) else 0.0)
    out["distance_km"] = out["distance_km"].fillna(0.0)
    out["ip_risk"] = out["ip_risk"].fillna(0.5)
    out["card_age_months"] = out["card_age_months"].fillna(out["card_age_months"].median() if len(out) else 0.0)
    out["mcc"] = out["mcc"].fillna(0).astype(int)
    out["country_risk"] = out["country_risk"].fillna(1).astype(int)

    # optional sampling to smaller file for quick iter
    if sample_frac is not None and 0 < sample_frac < 1.0:
        out = out.sample(frac=sample_frac, random_state=42)
        out = out.sort_values("TransactionDT", kind="stable").reset_index(drop=True)

    # final dtype enforcement for the integer-valued columns (NaN-able ones stay float)
    for c in [
        "hour", "dow", "sender_txn_24h", "is_foreign", "mcc", "country_risk", "is_fraud",
        "TransactionDT", "has_identity",
    ]:
        out[c] = out[c].astype(int)
    for c in ["amount", "card_age_months", "sender_avg_amount", "distance_km", "ip_risk"]:
        out[c] = out[c].astype(float)

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