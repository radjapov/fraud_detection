#!/usr/bin/env python3
"""
Calibration and alert-capacity metrics of the compared models, from the test scores that
run_models.py saved (runs/models/final/<model>_seed<k>_test.npz, runs/lrq/final/...).

Calibration (are the probabilities probabilities?)
  Brier score, log loss, mean predicted probability against the observed fraud rate,
  expected calibration error over 20 equal-frequency bins, calibration slope and intercept
  (logistic regression of the outcome on logit(p); slope 1 and intercept 0 = calibrated)

Alert capacity (what an investigation team gets for a fixed number of alerts)
  precision and recall among the top 0.1%, 0.5%, 1%, 2% and 5% of test transactions by score,
  and the lift of precision over the base rate

The models were trained with class weights tuned for PR-AUC, so their raw probabilities are not
expected to be calibrated; the point is to measure how far off they are on the test period, since
decision thresholds and SHAP values in probability units rely on the scale.

  python -m experiments.run_calibration --runs runs --out runs/calibration
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from experiments.common import dump_json
from experiments.make_figures import C, GRAY, INK, INK2, LABEL, MODEL_COLOR, MODEL_ORDER  # noqa: F401  (sets plot style)

FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05)
EPS = 1e-6


def ece_quantile(y, p, bins=20):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(y) * abs(p[c].mean() - y[c].mean()) for c in chunks))


def reliability(y, p, bins=10):
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return [(float(p[c].mean()), float(y[c].mean())) for c in chunks]


def calibration_line(y, p):
    z = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def top_fraction(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    idx = np.argpartition(-p, k - 1)[:k]
    tp = int(y[idx].sum())
    return dict(alerts=k, precision=tp / k, recall=tp / max(int(y.sum()), 1), lift=(tp / k) / float(y.mean()))


def metrics(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    slope, intercept = calibration_line(y, p)
    return dict(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p)), mean_predicted=float(p.mean()),
                observed_rate=float(y.mean()), ece20=ece_quantile(y, p), slope=slope, intercept=intercept,
                top={str(f): top_fraction(y, p, f) for f in FRACTIONS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/calibration")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, first = [], {}
    for name in MODEL_ORDER:
        d = runs / ("lrq" if name == "LogisticRegressionQ" else "models") / "final"
        per = []
        for s in a.seeds:
            f = d / f"{name}_seed{s}_test.npz"
            if f.exists():
                z = np.load(f)
                y, p = z["y"].astype(int), z["p"].astype(float)
                per.append(metrics(y, p))
                if s == a.seeds[0]:
                    first[name] = (y, p)
        if not per:
            continue
        avg = lambda get: float(np.mean([get(m) for m in per]))
        row = dict(model=name, n_seeds=len(per),
                   brier=avg(lambda m: m["brier"]), log_loss=avg(lambda m: m["log_loss"]),
                   mean_predicted=avg(lambda m: m["mean_predicted"]), observed_rate=per[0]["observed_rate"],
                   ece20=avg(lambda m: m["ece20"]), slope=avg(lambda m: m["slope"]), intercept=avg(lambda m: m["intercept"]))
        for f in FRACTIONS:
            for k in ("alerts", "precision", "recall", "lift"):
                row[f"top{f:g}_{k}"] = avg(lambda m: m["top"][str(f)][k])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "calibration_topk.csv", index=False)
    dump_json(rows, out / "summary.json")

    pd.set_option("display.width", 200)
    print("\nCalibration on the test period (mean over seeds)")
    print(df[["model", "brier", "log_loss", "mean_predicted", "observed_rate", "ece20", "slope", "intercept"]]
          .round(4).to_string(index=False))
    print("\nPrecision / recall among the top fraction of transactions by score")
    cols = ["model"] + [f"top{f:g}_{k}" for f in FRACTIONS for k in ("precision", "recall")]
    print(df[cols].round(3).to_string(index=False))

    # figure: reliability curves + recall as a function of alert volume
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax1.plot([0, 1], [0, 1], color=GRAY, lw=1, ls="--")
    for name, (y, p) in first.items():
        pts = np.array(reliability(y, np.clip(p, EPS, 1 - EPS)))
        ax1.plot(pts[:, 0], pts[:, 1], marker="o", ms=3, lw=1.4, color=MODEL_COLOR[name], label=LABEL[name])
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel("mean predicted probability (decile of score)"); ax1.set_ylabel("observed fraud rate")
    ax1.set_title("Reliability on the test period", loc="left", color=INK, fontsize=10)
    ax1.legend(fontsize=7.5, loc="upper left")
    fr = np.array([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1])
    for name, (y, p) in first.items():
        ax2.plot(fr * 100, [top_fraction(y, p, f)["recall"] for f in fr], marker="o", ms=3, lw=1.4,
                 color=MODEL_COLOR[name], label=LABEL[name])
    ax2.set_xscale("log")
    ax2.set_xticks([0.1, 0.5, 1, 5, 10]); ax2.set_xticklabels(["0.1%", "0.5%", "1%", "5%", "10%"])
    ax2.set_xlabel("share of transactions flagged (highest scores)"); ax2.set_ylabel("share of fraud caught (recall)")
    ax2.set_title("Recall for a fixed alert volume", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig7_calibration.png")
    print("\nwrote", out)


if __name__ == "__main__":
    main()
