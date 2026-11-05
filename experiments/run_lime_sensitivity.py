#!/usr/bin/env python3
"""
How much do the LIME conclusions depend on LIME's settings?

experiments/run_explain.py uses LIME with 2000 perturbation samples, no discretisation, sampling
around the instance and the library's kernel width. A reviewer can ask whether the poor fidelity
and stability of LIME come from those choices. This script repeats the LIME part of the comparison
for a grid of settings, on exactly the same explained rows as run_explain.py (same seed and the
same draw order), and measures per setting:

  fidelity   probability change when the top-k explained features are replaced by background values
  stability  LIME against itself across random seeds; against a near-identical input
  agreement  per-row Spearman and top-5 overlap with SHAP; with the study setting
  cost       milliseconds per explanation

Settings: the study configuration ("study"), 500 and 10000 samples, kernel width x0.25 and x4,
sampling from the training distribution instead of around the instance, and the library defaults
(quartile discretisation, 5000 samples, sampling from the training distribution).

  python -m experiments.run_lime_sensitivity --models LightGBM --out runs/lime_sensitivity
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np

from experiments import run_explain as RE
from experiments.common import DEFAULT_DATA, Data, dump_json, load_prepared

CONFIGS = {
    "study": dict(num_samples=2000, discretize=False, around=True, kernel=1.0),
    "samples_500": dict(num_samples=500, discretize=False, around=True, kernel=1.0),
    "samples_10000": dict(num_samples=10000, discretize=False, around=True, kernel=1.0),
    "kernel_x0.25": dict(num_samples=2000, discretize=False, around=True, kernel=0.25),
    "kernel_x4": dict(num_samples=2000, discretize=False, around=True, kernel=4.0),
    "sample_from_training_data": dict(num_samples=2000, discretize=False, around=False, kernel=1.0),
    "library_defaults": dict(num_samples=5000, discretize=True, around=False, kernel=1.0),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def lime_explainer(bg, cat_cols, cfg, seed):
    from lime.lime_tabular import LimeTabularExplainer
    kernel = np.sqrt(bg.shape[1]) * 0.75 * cfg["kernel"]  # the library default is sqrt(n_cols) * 0.75
    return LimeTabularExplainer(bg, categorical_features=cat_cols, mode="classification",
                                discretize_continuous=cfg["discretize"], sample_around_instance=cfg["around"],
                                kernel_width=kernel, random_state=seed)


def lime_values(f, X, explainer, num_samples):
    def pf(Z):
        p = f(Z)
        return np.c_[1 - p, p]
    out = np.zeros_like(X, dtype=float)
    for i in range(X.shape[0]):
        e = explainer.explain_instance(X[i], pf, num_features=X.shape[1], num_samples=num_samples, labels=(1,))
        for j, w in e.as_map()[1]:
            out[i, j] = w
    return out


def run_model(name, pipe, data: Data, n_rows, n_stab, out: Path, configs):
    pre = pipe.named_steps["preproc"]
    sources = data.num + data.cat
    owner = RE.column_owner(pre, sources)
    Xlab = pre.transform(data.X["labeled"]).astype(np.float32)
    Xte = pre.transform(data.X["test"]).astype(np.float32)
    yte = data.y["test"]
    # the same random draws, in the same order, as run_explain.run_model -> the same explained rows
    rng = np.random.default_rng(0)
    bg = Xlab[rng.choice(len(Xlab), min(1000, len(Xlab)), replace=False)]
    if not np.isfinite(bg).all():
        bg = np.nan_to_num(bg, nan=0.0)
    baseline = np.nanmedian(Xlab[rng.choice(len(Xlab), min(20000, len(Xlab)), replace=False)], axis=0)
    cat_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("cat__")]
    num_cols = np.array([n.startswith("num__") for n in pre.get_feature_names_out()])
    E = RE.Explainers(name, pipe, bg, cat_cols, 0)
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    idx = np.r_[rng.choice(pos, n_rows // 2, replace=False), rng.choice(neg, n_rows - n_rows // 2, replace=False)]
    X = np.nan_to_num(Xte[idx], nan=0.0)
    f = E.f
    n_src = len(sources)

    S = RE.fold(E.shap_values(X), owner, n_src)
    rrng = np.random.default_rng(123)
    ref = {"SHAP": np.argsort(-np.abs(S), axis=1),
           "Random": np.array([rrng.permutation(n_src) for _ in range(len(X))])}
    reference = {m: {f"top{k}": RE.deletion_drop(f, X, r, owner, baseline, k) for k in RE.K_LIST} for m, r in ref.items()}
    log(f"[{name}] reference fidelity top5: SHAP={reference['SHAP']['top5']:.4f} Random={reference['Random']['top5']:.4f}")

    noise_rng = np.random.default_rng(7)
    Xn = X[:n_stab].copy()
    Xn = Xn + noise_rng.normal(0, 0.01, Xn.shape) * num_cols

    study_L = None
    results = {}
    for cname in configs:
        cfg = CONFIGS[cname]
        path = out / f"{name}_{cname}.json"
        npz = out / f"{name}_{cname}_attr.npz"
        if path.exists() and npz.exists():
            results[cname] = __import__("json").loads(path.read_text())
            if cname == "study":
                study_L = np.load(npz)["lime"]
            log(f"[{name}] {cname}: done already")
            continue
        t0 = time.perf_counter()
        L = RE.fold(lime_values(f, X, lime_explainer(bg, cat_cols, cfg, 0), cfg["num_samples"]), owner, n_src)
        ms = (time.perf_counter() - t0) / len(X) * 1000
        fid = {f"top{k}": RE.deletion_drop(f, X, np.argsort(-np.abs(L), axis=1), owner, baseline, k) for k in RE.K_LIST}
        runs = [RE.fold(lime_values(f, X[:n_stab], lime_explainer(bg, cat_cols, cfg, s), cfg["num_samples"]), owner, n_src)
                for s in range(1, 4)]
        seeds = dict(spearman=float(np.mean([RE.rowwise_spearman(runs[0], r) for r in runs[1:]])),
                     top5=float(np.mean([RE.top_overlap(runs[0], r) for r in runs[1:]])))
        Ln = RE.fold(lime_values(f, Xn, lime_explainer(bg, cat_cols, cfg, 0), cfg["num_samples"]), owner, n_src)
        near = dict(spearman=RE.rowwise_spearman(L[:n_stab], Ln), top5=RE.top_overlap(L[:n_stab], Ln))
        if cname == "study":
            study_L = L
        res = dict(model=name, config=cname, settings=cfg, n_rows=len(X), n_stability_rows=n_stab, ms_per_row=ms,
                   fidelity=fid, stability_across_seeds=seeds, stability_near_identical_input=near,
                   agreement_with_shap=dict(spearman=RE.rowwise_spearman(S, L), top5=RE.top_overlap(S, L)),
                   agreement_with_study_setting=(None if study_L is None or cname == "study" else
                                                  dict(spearman=RE.rowwise_spearman(study_L, L),
                                                       top5=RE.top_overlap(study_L, L))),
                   reference_fidelity=reference)
        dump_json(res, path)
        np.savez_compressed(npz, lime=L, shap=S, rows=idx)
        results[cname] = res
        log(f"[{name}] {cname}: fidelity top5={fid['top5']:.4f}  seeds rho={seeds['spearman']:.3f}  "
            f"SHAP rho={res['agreement_with_shap']['spearman']:.3f}  {ms:.0f} ms/row")

    print(f"\n{name}: LIME settings (fidelity = mean |change in probability| after removing top-k features)")
    print(f"{'setting':<28}{'top1':>8}{'top5':>8}{'top10':>8}{'seeds rho':>11}{'near rho':>10}{'SHAP rho':>10}{'SHAP top5':>10}{'ms/row':>9}")
    for c, r in results.items():
        print(f"{c:<28}{r['fidelity']['top1']:>8.4f}{r['fidelity']['top5']:>8.4f}{r['fidelity']['top10']:>8.4f}"
              f"{r['stability_across_seeds']['spearman']:>11.3f}{r['stability_near_identical_input']['spearman']:>10.3f}"
              f"{r['agreement_with_shap']['spearman']:>10.3f}{r['agreement_with_shap']['top5']:>10.3f}{r['ms_per_row']:>9.0f}")
    print(f"{'(SHAP, for reference)':<28}{reference['SHAP']['top1']:>8.4f}{reference['SHAP']['top5']:>8.4f}{reference['SHAP']['top10']:>8.4f}")
    print(f"{'(random features)':<28}{reference['Random']['top1']:>8.4f}{reference['Random']['top5']:>8.4f}{reference['Random']['top10']:>8.4f}")
    dump_json(dict(model=name, results=results), out / f"{name}_summary.json")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--models", nargs="+", default=["LightGBM"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--pipelines", default="runs/models/pipelines")
    ap.add_argument("--rows", type=int, default=200, help="must equal run_explain's --rows to explain the same rows")
    ap.add_argument("--stability-rows", type=int, default=50)
    ap.add_argument("--out", default="runs/lime_sensitivity")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Data(load_prepared(a.data, a.sample))
    configs = ["study"] + [c for c in a.configs if c != "study"] if "study" in a.configs else a.configs
    for name in a.models:
        run_model(name, joblib.load(Path(a.pipelines) / f"{name}.joblib"), data, a.rows, a.stability_rows, out, configs)


if __name__ == "__main__":
    main()
