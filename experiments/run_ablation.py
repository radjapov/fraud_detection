#!/usr/bin/env python3
"""
Ablation of drifting features.

Section 5.1 explains the gap between random and chronological evaluation by drift. The analysis of
the test period found features that move far outside their training range (card_age_months, the
time-delta fields D1, D4, D10, D11). If they are what makes the future hard, removing them should
change the chronological score; if it does not, the explanation needs another feature set.

LightGBM is refit for every feature set with the hyperparameters found for the full feature set
(runs/models/best/LightGBM.json), under the same two-stage fit and protocol as run_models.py
(stage A: early stopping on validation; stage B: refit on all labeled rows; test scored once).
Hyperparameters are not tuned again for the smaller sets, which favours the full set slightly.

Feature sets
  full                     all 130 features
  no_card_age              without card_age_months
  no_named_drifters        without card_age_months, D1, D4, D10, D11
  no_D_family              without D1..D15
  no_V_family              without the selected V features
  no_drift_top5 / top10    without the 5 / 10 numeric features whose distribution moved most
                           between the training and the validation block (Kolmogorov-Smirnov
                           statistic). The test block is not used to choose them.
  python -m experiments.run_ablation --seeds 0 1 2 --out runs/ablation
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments import models as M
from experiments.common import (DEFAULT_DATA, Data, dump_json, load_prepared, paired_bootstrap_delta,
                                recall_at_fpr)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def drift_ranking(data: Data):
    """Numeric features ordered by KS distance between the training and the validation block."""
    rows = []
    for f in data.num:
        a, b = data.X["train"][f].dropna().to_numpy(), data.X["val"][f].dropna().to_numpy()
        if len(a) > 50 and len(b) > 50 and (np.ptp(a) > 0 or np.ptp(b) > 0):
            rows.append((f, float(ks_2samp(a, b).statistic)))
    return sorted(rows, key=lambda r: -r[1])


def feature_sets(columns, ranking):
    cols = set(columns)
    D = [c for c in columns if re.fullmatch(r"D\d+", c)]
    V = [c for c in columns if re.fullmatch(r"V\d+", c)]
    named = [c for c in ["card_age_months", "D1", "D4", "D10", "D11"] if c in cols]
    return {
        "full": [],
        "no_card_age": [c for c in ["card_age_months"] if c in cols],
        "no_named_drifters": named,
        "no_D_family": D,
        "no_V_family": V,
        "no_drift_top5": [f for f, _ in ranking[:5]],
        "no_drift_top10": [f for f, _ in ranking[:10]],
    }


def run_variant(df, drop, params, seeds, out: Path, name):
    data = Data(df.drop(columns=drop))
    n_feat = len(data.num) + len(data.cat)
    res = []
    for seed in seeds:
        path = out / f"{name}_seed{seed}.json"
        npz = out / f"{name}_seed{seed}_test.npz"
        if path.exists() and npz.exists():
            res.append(json.loads(path.read_text()))
            continue
        Xtr, Xva, _ = data.matrices("A", False)
        m_a, it, _ = M.fit_stage_a("LightGBM", params, seed, Xtr, data.y["train"], Xva, data.y["val"])
        del m_a
        Xlab, Xte, _ = data.matrices("B", False)
        m_b, n_trees, sec = M.fit_stage_b("LightGBM", params, seed, it, len(data.y["train"]), Xlab, data.y["labeled"])
        p, y = M.proba(m_b, Xte), data.y["test"]
        r = dict(set=name, seed=seed, n_features=n_feat, dropped=list(drop), stage_a_iterations=it, refit_trees=n_trees,
                 refit_seconds=sec, pr_auc=float(average_precision_score(y, p)), roc_auc=float(roc_auc_score(y, p)),
                 recall_at_1pct_fpr=recall_at_fpr(y, p))
        dump_json(r, path)
        np.savez_compressed(npz, p=p, y=y)
        res.append(r)
        log(f"[{name}] seed {seed}: {n_feat} features, PR-AUC={r['pr_auc']:.4f} ROC={r['roc_auc']:.4f} "
            f"R@1%FPR={r['recall_at_1pct_fpr']:.4f}")
        data.drop_stage("B")
    del data
    gc.collect()
    return res


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--best", default="runs/models/best/LightGBM.json")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--sets", nargs="+", default=None)
    ap.add_argument("--out", default="runs/ablation")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(a.best).read_text())["params"]
    df = load_prepared(a.data, a.sample)
    full = Data(df)
    ranking = drift_ranking(full)
    del full
    gc.collect()
    dump_json([dict(feature=f, ks_train_vs_validation=s) for f, s in ranking], out / "drift_ranking.json")
    log("most drifting numeric features (train vs validation):", [(f, round(s, 3)) for f, s in ranking[:10]])
    sets = feature_sets([c for c in df.columns if c not in ("is_fraud", "TransactionDT")], ranking)
    summary = {}
    for name, drop in sets.items():
        if a.sets and name not in a.sets:
            continue
        summary[name] = run_variant(df, drop, params, a.seeds, out, name)
    # report: mean +- std over seeds, paired bootstrap delta against the full set (first seed)
    rows = []
    for name, rs in summary.items():
        row = dict(set=name, n_features=rs[0]["n_features"], dropped=rs[0]["dropped"])
        for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr"]:
            v = np.array([r[k] for r in rs])
            row[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        rows.append(row)
    if "full" in summary:
        z0 = np.load(out / f"full_seed{a.seeds[0]}_test.npz")
        for row in rows:
            if row["set"] != "full":
                z = np.load(out / f"{row['set']}_seed{a.seeds[0]}_test.npz")
                row["pr_auc_delta_vs_full"] = paired_bootstrap_delta(z0["y"], z["p"], z0["p"])
    dump_json(rows, out / "summary.json")
    print(f"\n{'feature set':<20}{'n':>5}{'PR-AUC':>16}{'ROC-AUC':>16}{'R@1%FPR':>16}{'Δ PR-AUC vs full [95% CI]':>34}")
    for r in rows:
        f = lambda k: f"{r[k]['mean']:.4f}±{r[k]['std']:.4f}"
        d = r.get("pr_auc_delta_vs_full")
        ds = f"{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]" if d else ""
        print(f"{r['set']:<20}{r['n_features']:>5}{f('pr_auc'):>16}{f('roc_auc'):>16}{f('recall_at_1pct_fpr'):>16}{ds:>34}")


if __name__ == "__main__":
    main()
