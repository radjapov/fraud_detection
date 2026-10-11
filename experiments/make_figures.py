#!/usr/bin/env python3
"""
Figures (PNG, 300 dpi) and tables (CSV + Markdown) for the paper, from the files in runs/.

  python -m experiments.make_figures --runs runs --out runs/paper
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

# validated categorical palette (fixed order) + neutral ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
GRAY, INK, INK2, GRID = "#9a9993", "#0b0b0b", "#52514e", "#e5e4e0"
LABEL = {"LightGBM": "LightGBM", "XGBoost": "XGBoost", "CatBoost": "CatBoost",
         "RandomForest": "Random Forest", "LogisticRegression": "Logistic Regression",
         "LogisticRegressionQ": "Logistic Regression (quantile)"}
MODEL_ORDER = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogisticRegressionQ", "LogisticRegression"]
MODEL_COLOR = dict(zip(MODEL_ORDER, C))
TREES = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogisticRegression"]
PROTO = [("resubstitution", "Scored on all data\n(training rows included)"),
         ("random_80_20", "Random 80/20 split"),
         ("time_80_20", "Chronological 80/20"),
         ("time_80_20_delay", "Chronological 80/20\n+ 30-day label delay")]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
})


def load(p):
    return json.loads(Path(p).read_text())


def save(fig, out, name):
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def fig_protocols(runs, out):
    s = load(runs / "protocols" / "summary.json")
    names = [lbl for k, lbl in PROTO][::-1]
    pr = [s[k]["pr_auc"]["mean"] for k, _ in PROTO][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    bars = ax.barh(names, pr, color=C[0], height=0.55)
    for b, v in zip(bars, pr):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC on the evaluated rows (LightGBM, mean of 3 seeds)")
    ax.grid(axis="y", visible=False)
    ax.set_title("The same model scored under four evaluation protocols", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig1_protocols")
    rows = [dict(protocol=lbl.replace("\n", " "), pr_auc=s[k]["pr_auc"]["mean"], roc_auc=s[k]["roc_auc"]["mean"],
                 recall_at_1pct_fpr=s[k]["recall_at_1pct_fpr"]["mean"]) for k, lbl in PROTO]
    return pd.DataFrame(rows)


def model_rows(runs):
    rows = {r["model"]: r for r in load(runs / "models" / "summary.json")}
    q = runs / "lrq" / "summary.json"
    if q.exists():
        for r in load(q):
            rows[r["model"]] = r
    return rows


def fig_models(runs, out):
    rows = model_rows(runs)
    # models run separately (e.g. the quantile LR) get their paired delta against LightGBM here
    from experiments.common import paired_bootstrap_delta
    ref = runs / "models" / "final" / "LightGBM_seed0_test.npz"
    for m, r in rows.items():
        if m != "LightGBM" and "pr_auc_delta_vs_LightGBM" not in r and ref.exists():
            f = runs / ("lrq" if m == "LogisticRegressionQ" else "models") / "final" / f"{m}_seed0_test.npz"
            if f.exists():
                a, b = np.load(f), np.load(ref)
                assert (a["y"] == b["y"]).all()
                r["pr_auc_delta_vs_LightGBM"] = paired_bootstrap_delta(a["y"], a["p"], b["p"])
    order = sorted(rows, key=lambda m: rows[m]["pr_auc"]["mean"])
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    for i, m in enumerate(order):
        r = rows[m]
        lo, hi = r["pr_auc_ci95_seed0"]
        ax.plot([lo, hi], [i, i], color=C[0], lw=2, solid_capstyle="round")
        ax.plot(r["pr_auc"]["mean"], i, "o", ms=8, color=C[0], mec="#fcfcfb", mew=1.5)
        ax.text(hi + 0.01, i, f"{r['pr_auc']['mean']:.3f}", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(order)), [LABEL[m] for m in order])
    ax.set_xlim(0.1, 0.6)
    ax.set_xlabel("Test PR-AUC: mean of 5 seeds (dot), bootstrap 95% CI (line)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Model comparison on the test period", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig2_models")
    t = []
    for m in sorted(rows, key=lambda m: -rows[m]["pr_auc"]["mean"]):
        r = rows[m]
        d = r.get("pr_auc_delta_vs_LightGBM")
        t.append(dict(model=LABEL[m], pr_auc=f"{r['pr_auc']['mean']:.3f} ± {r['pr_auc']['std']:.3f}",
                      pr_auc_ci95=f"[{r['pr_auc_ci95_seed0'][0]:.3f}, {r['pr_auc_ci95_seed0'][1]:.3f}]",
                      roc_auc=f"{r['roc_auc']['mean']:.3f}", recall_at_1pct_fpr=f"{r['recall_at_1pct_fpr']['mean']:.3f}",
                      f1=f"{r['f1']['mean']:.3f}",
                      delta_pr_vs_lightgbm=(f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]" if d else "—"),
                      refit_s=f"{r['refit_seconds']:.1f}", ms_per_row=f"{r['single_row_median_ms']:.2f}"))
    return pd.DataFrame(t)


def fig_pr_curves(runs, out):
    styles = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (1, 1))]
    fig, ax = plt.subplots(figsize=(5.2, 3.9))
    base = None
    for (m, ls) in zip(MODEL_ORDER, styles):
        f = runs / ("lrq" if m == "LogisticRegressionQ" else "models") / "final" / f"{m}_seed0_test.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p, r, _ = precision_recall_curve(z["y"], z["p"])
        ax.plot(r, p, lw=2, ls=ls, color=MODEL_COLOR[m], label=LABEL[m])
        base = z["y"].mean()
    ax.axhline(base, color=GRAY, lw=1, ls="--")
    ax.text(0.02, base + 0.015, f"fraud rate {base:.3f} (no-skill line)", ha="left", color=INK2, fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Precision-recall on the test period (seed 0)", loc="left", color=INK, fontsize=10)
    save(fig, out, "fig3_pr_curves")


METHODS = [("SHAP", C[0], "o", "-"), ("Permutation", C[1], "s", "--"), ("LIME", C[2], "^", "-."), ("Random", GRAY, "D", ":")]


def fig_fidelity(runs, out):
    ex = {m: load(runs / "explain" / f"{m}.json") for m in TREES if (runs / "explain" / f"{m}.json").exists()}
    ks = [1, 3, 5, 10]
    fig, axes = plt.subplots(1, len(ex), figsize=(9.6, 2.7), sharey=True)
    for ax, (m, e) in zip(np.atleast_1d(axes), ex.items()):
        for name, col, mk, ls in METHODS:
            ax.plot(ks, [e["fidelity"][name][f"top{k}"] for k in ks], color=col, marker=mk, ms=5, lw=2, ls=ls, label=name)
        ax.set_title(LABEL[m], fontsize=9, color=INK)
        ax.set_xticks(ks)
        ax.set_xlabel("features removed (k)")
    np.atleast_1d(axes)[0].set_ylabel("mean |Δ fraud probability|")
    np.atleast_1d(axes)[0].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Deletion test: removing an explanation's top-k features vs k random ones", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    save(fig, out, "fig4_fidelity")
    rows = []
    for m, e in ex.items():
        rows.append(dict(model=LABEL[m],
                         fidelity_top5_shap=e["fidelity"]["SHAP"]["top5"], fidelity_top5_lime=e["fidelity"]["LIME"]["top5"],
                         fidelity_top5_permutation=e["fidelity"]["Permutation"]["top5"], fidelity_top5_random=e["fidelity"]["Random"]["top5"],
                         stability_shap=e["stability"]["near_identical_input"]["SHAP"]["spearman"],
                         stability_lime=e["stability"]["near_identical_input"]["LIME"]["spearman"],
                         lime_across_seeds=e["stability"]["lime_across_seeds"]["spearman"],
                         shap_lime_rank_corr=e["agreement"]["local_shap_lime_spearman"],
                         shap_lime_top5_overlap=e["agreement"]["local_shap_lime_top5"],
                         shap_perm_global=e["agreement"]["global_shap_perm"],
                         shap_ms_per_row=e["cost"]["shap_ms_per_row"], lime_ms_per_row=e["cost"]["lime_ms_per_row"],
                         permutation_total_s=e["cost"]["permutation_total_s"]))
    return pd.DataFrame(rows).round(3), ex


def fig_stability(ex, out):
    models = list(ex)
    series = [("SHAP, near-identical input", lambda e: e["stability"]["near_identical_input"]["SHAP"]["spearman"], C[0]),
              ("LIME, near-identical input", lambda e: e["stability"]["near_identical_input"]["LIME"]["spearman"], C[1]),
              ("LIME, same input, new seed", lambda e: e["stability"]["lime_across_seeds"]["spearman"], C[2])]
    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, (name, f, col) in enumerate(series):
        vals = [f(ex[m]) for m in models]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.03, color=col, label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x, [LABEL[m].replace(" ", "\n", 1) for m in models])
    ax.set_ylim(0, 1.1); ax.set_ylabel("rank correlation of attributions")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=7.5, ncol=3, loc="upper left", bbox_to_anchor=(0, 1.13))
    save(fig, out, "fig5_stability")


def fig_latency(runs, out):
    L = load(runs / "latency" / "latency.json")
    r = L["results"]
    items = [("POST /api/predict", r["predict"]),
             ("/api/shap: in the web process", r["shap_inprocess"]),
             ("/api/shap: long-lived worker", r["shap_pool"]),
             ("/api/shap: new process per request", r["shap_worker"])][::-1]
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    med = [v["median_ms"] for _, v in items]
    p95 = [v["p95_ms"] for _, v in items]
    bars = ax.barh([n for n, _ in items], med, color=C[0], height=0.55)
    for b, m, p in zip(bars, med, p95):
        ax.text(m * 1.12, b.get_y() + b.get_height() / 2, f"{m:,.0f} ms (p95 {p:,.0f})", va="center", fontsize=8.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(5, 20000)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel("median response time over HTTP, ms (log scale)")
    ax.grid(axis="y", visible=False)
    env = L["environment"]
    ax.set_title(f"Service response time ({env['machine']}, {env['cpu_count']} cores, Python {env['python']})",
                 loc="left", color=INK, fontsize=10)
    save(fig, out, "fig6_latency")
    return pd.DataFrame([dict(endpoint=n, median_ms=round(v["median_ms"], 1), p95_ms=round(v["p95_ms"], 1), n=v["n"],
                              cold_start_ms=round(v.get("cold_start_ms", float("nan")), 0)) for n, v in items[::-1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/paper")
    a = ap.parse_args()
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    tables["table1_models"] = fig_models(runs, out)
    tables["table2_protocols"] = fig_protocols(runs, out).round(3)
    fig_pr_curves(runs, out)
    tables["table3_explanations"], ex = fig_fidelity(runs, out)
    fig_stability(ex, out)
    tables["table4_latency"] = fig_latency(runs, out)
    md = []
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
        md += [f"## {name}", "", df.to_markdown(index=False), ""]
    (out / "tables.md").write_text("\n".join(md))
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
