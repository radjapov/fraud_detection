#!/usr/bin/env python3
"""
Comparison of explanation methods on the trained pipelines (runs/models/pipelines/*.joblib).

Methods
  SHAP         TreeExplainer for tree ensembles, LinearExplainer for logistic regression
  LIME         LimeTabularExplainer in the model's input space (one-hot columns as categorical)
  Permutation  global permutation importance on the test period (PR-AUC drop); global only

All attributions are folded back from the model columns (scaled numerics + one-hot) to the
130 input features, as the service does (fraud_app.explain): one-hot parts are summed.

Measures
  fidelity     probability change when an explanation's top-k features are replaced by their
               background (training) values, against k random features  (deletion test)
  stability    LIME vs itself across random seeds; every method vs a near-identical input
  agreement    per-row Spearman / top-5 overlap between SHAP and LIME; global rank agreement of
               mean |SHAP|, mean |LIME| and permutation importance
  cost         milliseconds per explanation

  python -m experiments.run_explain --models LightGBM XGBoost CatBoost RandomForest LogisticRegression
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr

from experiments.common import DEFAULT_DATA, Data, dump_json, load_prepared

K_LIST = (1, 3, 5, 10)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def column_owner(pre, sources):
    """Index of the input feature that owns each model column."""
    owners = []
    for name in pre.get_feature_names_out():
        rest = name.split("__", 1)[-1]
        f = max((s for s in sources if rest == s or rest.startswith(s + "_")), key=len)
        owners.append(sources.index(f))
    return np.asarray(owners)


def fold(attr_cols, owner, n_src):
    """(rows, model columns) -> (rows, input features) by summing the columns of each feature."""
    out = np.zeros((attr_cols.shape[0], n_src))
    np.add.at(out.T, owner, attr_cols.T)
    return out


class Explainers:
    def __init__(self, name, pipe, X_bg, cat_cols, seed=0):
        self.name, self.clf = name, pipe.named_steps["clf"]
        self.X_bg = X_bg
        self.f = lambda X: np.asarray(self.clf.predict_proba(np.asarray(X, dtype=np.float32)))[:, 1]
        import shap
        if name == "LogisticRegression":
            self.shap = shap.LinearExplainer(self.clf, X_bg)
        else:
            self.shap = shap.TreeExplainer(self.clf, data=X_bg[:100], feature_perturbation="interventional",
                                           model_output="probability")
        from lime.lime_tabular import LimeTabularExplainer
        self.lime_args = dict(training_data=X_bg, categorical_features=cat_cols, mode="classification",
                              discretize_continuous=False, sample_around_instance=True)
        self._lime = {}
        self._lime_cls = LimeTabularExplainer

    def shap_values(self, X):
        sv = self.shap.shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        if self.name == "LogisticRegression":  # LinearExplainer explains the log-odds
            pass
        return sv

    def lime_values(self, X, seed=0, num_samples=2000):
        if seed not in self._lime:
            self._lime[seed] = self._lime_cls(random_state=seed, **self.lime_args)
        ex = self._lime[seed]
        n_cols = X.shape[1]
        out = np.zeros_like(X, dtype=float)
        pf = lambda Z: np.c_[1 - self.f(Z), self.f(Z)]
        for i in range(X.shape[0]):
            e = ex.explain_instance(X[i], pf, num_features=n_cols, num_samples=num_samples, labels=(1,))
            for j, w in e.as_map()[1]:
                out[i, j] = w
        return out


def deletion_drop(f, X, rank_src, owner, baseline, k):
    """Mean |p(x) - p(x with the top-k input features set to background values)|."""
    Xd = X.copy()
    for i in range(X.shape[0]):
        for s in rank_src[i, :k]:
            cols = owner == s
            Xd[i, cols] = baseline[cols]
    return float(np.mean(np.abs(f(X) - f(Xd))))


def top_overlap(a, b, k=5):
    ta = np.argsort(-np.abs(a), axis=1)[:, :k]
    tb = np.argsort(-np.abs(b), axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ta, tb)]))


def rowwise_spearman(a, b):
    v = [spearmanr(np.abs(x), np.abs(y)).statistic for x, y in zip(a, b)]
    return float(np.nanmean(v))


def run_model(name, pipe, data: Data, n_rows, n_stab, seed, out: Path):
    pre = pipe.named_steps["preproc"]
    sources = data.num + data.cat
    owner = column_owner(pre, sources)
    Xlab = pre.transform(data.X["labeled"]).astype(np.float32)
    Xte = pre.transform(data.X["test"]).astype(np.float32)
    yte = data.y["test"]
    rng = np.random.default_rng(seed)
    bg = Xlab[rng.choice(len(Xlab), min(1000, len(Xlab)), replace=False)]
    if not np.isfinite(bg).all():  # LIME / LinearExplainer need finite background; NaN = "missing" for trees
        bg = np.nan_to_num(bg, nan=0.0)
    baseline = np.nanmedian(Xlab[rng.choice(len(Xlab), min(20000, len(Xlab)), replace=False)], axis=0)
    cat_cols = [i for i, n in enumerate(pre.get_feature_names_out()) if n.startswith("cat__")]
    E = Explainers(name, pipe, bg, cat_cols, seed)

    # explained rows: half fraud, half legitimate, from the test period
    pos, neg = np.flatnonzero(yte == 1), np.flatnonzero(yte == 0)
    idx = np.r_[rng.choice(pos, n_rows // 2, replace=False), rng.choice(neg, n_rows - n_rows // 2, replace=False)]
    # LIME cannot take NaN; every method explains the same input, with missing values at the
    # scaled mean (0), so the methods are compared on identical rows
    X = np.nan_to_num(Xte[idx], nan=0.0)
    Xf = X

    t0 = time.perf_counter(); s_cols = E.shap_values(Xf); t_shap = (time.perf_counter() - t0) / len(X) * 1000
    t0 = time.perf_counter(); l_cols = E.lime_values(X, seed=seed); t_lime = (time.perf_counter() - t0) / len(X) * 1000
    S, L = fold(s_cols, owner, len(sources)), fold(l_cols, owner, len(sources))
    log(f"[{name}] SHAP {t_shap:.1f} ms/row, LIME {t_lime:.0f} ms/row")

    # permutation importance (global), on a test subsample, scored by PR-AUC on the input features
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import average_precision_score
    sub = rng.choice(len(Xte), min(20000, len(Xte)), replace=False)
    t0 = time.perf_counter()
    perm = np.zeros(len(sources))
    base_ap = average_precision_score(yte[sub], E.f(Xte[sub]))
    for s in range(len(sources)):
        cols = owner == s
        drops = []
        for r in range(3):
            Xp = Xte[sub].copy()
            Xp[:, cols] = Xp[rng.permutation(len(sub))][:, cols]
            drops.append(base_ap - average_precision_score(yte[sub], E.f(Xp)))
        perm[s] = np.mean(drops)
    t_perm = time.perf_counter() - t0

    # fidelity: deletion of top-k by each method vs random k
    f = E.f
    rank = {"SHAP": np.argsort(-np.abs(S), axis=1), "LIME": np.argsort(-np.abs(L), axis=1),
            "Permutation": np.tile(np.argsort(-perm), (len(X), 1)),
            "Random": np.array([rng.permutation(len(sources)) for _ in range(len(X))])}
    fidelity = {m: {f"top{k}": deletion_drop(f, X, r, owner, baseline, k) for k in K_LIST} for m, r in rank.items()}

    # stability: LIME across seeds, every local method on a near-identical input
    Xs = np.nan_to_num(X[:n_stab], nan=0.0)
    lime_runs = [fold(E.lime_values(Xs, seed=s), owner, len(sources)) for s in range(1, 4)]
    lime_seed = dict(spearman=float(np.mean([rowwise_spearman(lime_runs[0], r) for r in lime_runs[1:]])),
                     top5=float(np.mean([top_overlap(lime_runs[0], r) for r in lime_runs[1:]])))
    num_cols = np.array([n.startswith("num__") for n in pre.get_feature_names_out()])
    Xn = X[:n_stab].copy()
    noise = rng.normal(0, 0.01, Xn.shape) * num_cols  # 0.01 std of a scaled numeric feature
    Xn = Xn + np.where(np.isnan(Xn), 0, noise)
    S_n = fold(E.shap_values(np.nan_to_num(Xn, nan=0.0) if name == "LogisticRegression" else Xn), owner, len(sources))
    L_n = fold(E.lime_values(np.nan_to_num(Xn, nan=0.0), seed=seed), owner, len(sources))
    perturb = {"SHAP": dict(spearman=rowwise_spearman(S[:n_stab], S_n), top5=top_overlap(S[:n_stab], S_n)),
               "LIME": dict(spearman=rowwise_spearman(L[:n_stab], L_n), top5=top_overlap(L[:n_stab], L_n))}

    g_shap, g_lime = np.abs(S).mean(0), np.abs(L).mean(0)
    agreement = dict(
        local_shap_lime_spearman=rowwise_spearman(S, L), local_shap_lime_top5=top_overlap(S, L),
        global_shap_lime=float(spearmanr(g_shap, g_lime).statistic),
        global_shap_perm=float(spearmanr(g_shap, perm).statistic),
        global_lime_perm=float(spearmanr(g_lime, perm).statistic),
    )
    top = lambda v: [sources[i] for i in np.argsort(-v)[:10]]
    res = dict(model=name, n_rows=len(X), n_stability_rows=n_stab,
               cost=dict(shap_ms_per_row=t_shap, lime_ms_per_row=t_lime, permutation_total_s=t_perm),
               fidelity=fidelity, stability=dict(lime_across_seeds=lime_seed, near_identical_input=perturb),
               agreement=agreement, top10=dict(shap=top(g_shap), lime=top(g_lime), permutation=top(perm)))
    dump_json(res, out / f"{name}.json")
    np.savez_compressed(out / f"{name}_attr.npz", shap=S, lime=L, perm=perm, rows=idx)
    log(f"[{name}] fidelity top5 SHAP={fidelity['SHAP']['top5']:.4f} LIME={fidelity['LIME']['top5']:.4f} "
        f"Perm={fidelity['Permutation']['top5']:.4f} Random={fidelity['Random']['top5']:.4f} | "
        f"SHAP~LIME rho={agreement['local_shap_lime_spearman']:.3f}")
    return res


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--models", nargs="+", default=["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogisticRegression"])
    ap.add_argument("--pipelines", default="runs/models/pipelines")
    ap.add_argument("--rows", type=int, default=200)
    ap.add_argument("--stability-rows", type=int, default=50)
    ap.add_argument("--out", default="runs/explain")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    data = Data(load_prepared(a.data, a.sample))
    for name in a.models:
        if (out / f"{name}.json").exists():
            log(f"[{name}] done already"); continue
        p = Path(a.pipelines) / f"{name}.joblib"
        if not p.exists():
            log(f"[{name}] no pipeline at {p}"); continue
        run_model(name, joblib.load(p), data, a.rows, a.stability_rows, 0, out)


if __name__ == "__main__":
    main()
