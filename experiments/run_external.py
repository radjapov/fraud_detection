#!/usr/bin/env python3
"""
Run a third-party open-source IEEE-CIS pipeline under the evaluation protocol of this study.

External pipeline: github.com/Joumanayoussef/ieee-cis-fraud-detection (feature engineering and
LightGBM / XGBoost / logistic-regression models in src/features.py and src/train.py). The
repository is not copied into this project (it carries no licence); it is cloned separately and
imported from --repo. Its code is used unchanged. What changes is the data split:

  * full data (the repository itself uses a 50% stratified sample),
  * the chronological split with 30-day label delay of this study (same rows as experiments/*):
      train / validation / test = fraud_app.features.split_data(..., 30),
  * the repository's own recipe on top of it: fit features on train, early stopping and the F1
    threshold on validation, one pass over the test block.

  git clone --depth 1 https://github.com/Joumanayoussef/ieee-cis-fraud-detection.git external_repo
  pip install imbalanced-learn
  python -m experiments.run_external --repo external_repo --raw data/ieee --out runs/external
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import bootstrap_ci, dump_json, scores
from fraud_app.features import DEFAULT_LABEL_DELAY_DAYS, TARGET, TIME_COL, split_data


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="clone of the external repository")
    ap.add_argument("--raw", default="data/ieee", help="folder with train_transaction.csv and train_identity.csv")
    ap.add_argument("--out", default="runs/external")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N transactions (smoke test)")
    ap.add_argument("--delay", type=float, default=DEFAULT_LABEL_DELAY_DAYS)
    a = ap.parse_args()
    repo, raw, out = Path(a.repo).resolve(), Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = repo / "src"
    ext_data = load_module("ext_data", src / "data.py")
    ext_feat = load_module("ext_features", src / "features.py")
    ext_train = load_module("ext_train", src / "train.py")

    t0 = time.time()
    txn = pd.read_csv(raw / "train_transaction.csv", nrows=a.nrows,
                      dtype={"TransactionID": np.int32, "isFraud": np.int8, "TransactionDT": np.int32})
    idn = pd.read_csv(raw / "train_identity.csv", dtype={"TransactionID": np.int32})
    df = ext_data._downcast(txn).merge(ext_data._downcast(idn), on="TransactionID", how="left")
    del txn, idn
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    print(f"merged {len(df):,} rows x {df.shape[1]} columns, fraud {df['isFraud'].mean():.4%}", flush=True)

    # same row sets as the other experiments: split a thin frame with this study's own function
    thin = pd.DataFrame({TIME_COL: df["TransactionDT"].to_numpy(), TARGET: df["isFraud"].to_numpy()}, index=df.index)
    tr_i, va_i, te_i = (p.index for p in split_data(thin, a.delay))
    sizes = {k: dict(rows=int(len(i)), fraud_rate=round(float(df.loc[i, "isFraud"].mean()), 5))
             for k, i in dict(train=tr_i, val=va_i, test=te_i).items()}
    print("split:", sizes, flush=True)
    train, val, test = df.loc[tr_i].copy(), df.loc[va_i].copy(), df.loc[te_i].copy()
    del df

    X_tr, y_tr, art = ext_feat.fit_transform(train)
    X_va, y_va = ext_feat.transform(val, art)
    X_te, y_te = ext_feat.transform(test, art)
    print(f"features used by the external pipeline: {X_tr.shape[1]}", flush=True)

    metrics, store = ext_train.train_all(X_tr, y_tr, X_va, y_va, X_te, y_te)

    # re-score with this study's metric code (recall at 1% FPR, bootstrap CI) from the stored probabilities
    names = {"LightGBM (no weight)": "lgb_base", "LightGBM (scale_pos_weight)": "lgb_weighted",
             "XGBoost (scale_pos_weight)": "xgb", "LogisticRegression (balanced)": "lr", "LR + SMOTE": "lr_smote"}
    y = store["y_test"].astype(int)
    rows = []
    for m in metrics:
        p = np.asarray(store["test_probs"][names[m["model"]]])
        r = scores(y, p, m["threshold"])
        lo, hi = bootstrap_ci(y, p, n=1000, seed=0, metric="pr_auc")
        r.update(model=m["model"], pr_auc_ci=[lo, hi], best_iteration=m.get("best_iteration"))
        rows.append(r)
        np.save(out / f"probs_{names[m['model']]}.npy", p)
        print(f"{m['model']:<32} PR-AUC {r['pr_auc']:.4f} [{lo:.4f}, {hi:.4f}]  ROC {r['roc_auc']:.4f}  "
              f"R@1%FPR {r['recall_at_1pct_fpr']:.4f}", flush=True)
    dump_json(dict(source="github.com/Joumanayoussef/ieee-cis-fraud-detection", split=sizes,
                   n_features=int(X_tr.shape[1]), label_delay_days=a.delay, nrows=a.nrows,
                   seconds=time.time() - t0, models=rows), out / "external.json")
    print("wrote", out / "external.json")


if __name__ == "__main__":
    main()
