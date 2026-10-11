#!/usr/bin/env python3
"""
How much does the evaluation protocol change the reported quality of the SAME model?

The production LightGBM (parameters and tree count from train_ieee_lgbm.py / meta_ieee.joblib)
is trained and scored under four protocols:

  resubstitution     fit on all rows, score the same rows (training data included)
  random_80_20       stratified random split, 80% train / 20% test
  time_80_20         oldest 80% train, newest 20% test
  time_80_20_delay   as above, but only rows whose label is known when the test period starts
                     (30-day label delay), as in the service

  python -m experiments.run_protocols --out ../runs/protocols
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from experiments.common import DEFAULT_DATA, _x, dump_json, feature_lists, load_prepared, recall_at_fpr
from fraud_app.features import DEFAULT_LABEL_DELAY_DAYS, TARGET, TIME_COL, build_preprocessor, labeled_rows
from train_ieee_lgbm import LGBM_PARAMS

ROOT = Path(__file__).resolve().parent.parent


def run(train, test, num, cat, n_trees, seed):
    pre = build_preprocessor(num, cat, impute=False)
    Xtr = pre.fit_transform(_x(train, num, cat)).astype(np.float32)
    Xte = pre.transform(_x(test, num, cat)).astype(np.float32)
    clf = lgb.LGBMClassifier(**dict(LGBM_PARAMS, n_estimators=n_trees, random_state=seed))
    t0 = time.perf_counter()
    clf.fit(Xtr, train[TARGET].to_numpy())
    p = clf.predict_proba(Xte)[:, 1]
    y = test[TARGET].to_numpy()
    return dict(n_train=len(train), n_test=len(test), test_fraud_rate=float(y.mean()),
                roc_auc=float(roc_auc_score(y, p)), pr_auc=float(average_precision_score(y, p)),
                recall_at_1pct_fpr=recall_at_fpr(y, p), fit_seconds=time.perf_counter() - t0)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs/protocols")
    a = ap.parse_args()
    out = Path(a.out)
    df = load_prepared(a.data).sort_values(TIME_COL, kind="stable").reset_index(drop=True)
    num, cat = feature_lists(df)
    meta_path = ROOT / "artifacts" / "meta_ieee.joblib"
    n_trees = int(joblib.load(meta_path)["n_estimators"]) if meta_path.exists() else 500
    j = int(len(df) * 0.8)
    results = []
    for seed in a.seeds:
        tr_r, te_r = train_test_split(df, test_size=0.2, random_state=seed, stratify=df[TARGET])
        protocols = {
            "resubstitution": (df, df),
            "random_80_20": (tr_r, te_r),
            "time_80_20": (df.iloc[:j], df.iloc[j:]),
            "time_80_20_delay": (labeled_rows(df, DEFAULT_LABEL_DELAY_DAYS), df.iloc[j:]),
        }
        for name, (tr, te) in protocols.items():
            r = dict(protocol=name, seed=seed, n_trees=n_trees, **run(tr, te, num, cat, n_trees, seed))
            results.append(r)
            print(time.strftime("%H:%M:%S"), name, seed,
                  f"PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} R@1%FPR={r['recall_at_1pct_fpr']:.4f}",
                  flush=True)
            dump_json(results, out / "protocols.json")
    summary = {}
    for name in ["resubstitution", "random_80_20", "time_80_20", "time_80_20_delay"]:
        rs = [r for r in results if r["protocol"] == name]
        summary[name] = {k: dict(mean=float(np.mean([r[k] for r in rs])), std=float(np.std([r[k] for r in rs], ddof=1)))
                         for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr"]}
    dump_json(summary, out / "summary.json")
    for k, v in summary.items():
        print(f"{k:<20} PR-AUC {v['pr_auc']['mean']:.4f}  ROC-AUC {v['roc_auc']['mean']:.4f}  "
              f"R@1%FPR {v['recall_at_1pct_fpr']['mean']:.4f}")


if __name__ == "__main__":
    main()
