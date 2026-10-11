#!/usr/bin/env python3
"""
Model comparison for the NTCC paper.

  tune    Optuna (TPE, same number of trials for every model), objective = PR-AUC on the
          gap-separated validation block, stage-A fit only. Resumable (SQLite storage).
  final   best parameters x seeds: stage A (threshold on validation) + stage B refit on all
          labeled rows, scored on the test period. Saves test scores per seed and the seed-0
          pipeline for the explanation experiments. Resumable (skips finished model/seed pairs).
  report  mean +- std over seeds, bootstrap CIs and paired deltas against LightGBM.

  python -m experiments.run_models --stage all --trials 25 --seeds 0 1 2 3 4
"""
from __future__ import annotations

import argparse
import warnings
import json
import time
from pathlib import Path

import joblib
import numpy as np
import optuna
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from experiments import models as M
from experiments.common import (
    DEFAULT_DATA, Data, best_f1_threshold, bootstrap_ci, dump_json, load_prepared,
    paired_bootstrap_delta, scores, single_row_latency_ms,
)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def tune(data: Data, names, trials, out: Path):
    storage = f"sqlite:///{out / 'optuna.db'}"
    for name in names:
        Xtr, Xva, _ = data.matrices("A", M.NEEDS_IMPUTE[name])
        ytr, yva = data.y["train"], data.y["val"]
        study = optuna.create_study(study_name=name, storage=storage, direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=0), load_if_exists=True)
        done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        left = trials - done
        log(f"[tune] {name}: {done} done, {max(left, 0)} to go")

        def objective(trial):
            params = M.suggest(name, trial)
            m, best, sec = M.fit_stage_a(name, params, 0, Xtr, ytr, Xva, yva)
            ap = float(average_precision_score(yva, M.proba(m, Xva)))
            trial.set_user_attr("best_iteration", best)
            trial.set_user_attr("fit_seconds", sec)
            log(f"[tune] {name} #{trial.number} val PR-AUC={ap:.4f} it={best} {sec:.0f}s")
            return ap

        if left > 0:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=left, gc_after_trial=True)
        dump_json(dict(params=study.best_params, val_pr_auc=study.best_value,
                       best_iteration=study.best_trial.user_attrs.get("best_iteration"),
                       n_trials=len(study.trials)), out / "best" / f"{name}.json")


def final(data: Data, names, seeds, out: Path):
    for name in names:
        best = json.loads((out / "best" / f"{name}.json").read_text())
        params = best["params"]
        imp = M.NEEDS_IMPUTE[name]
        for seed in seeds:
            res_path = out / "final" / f"{name}_seed{seed}.json"
            if res_path.exists():
                continue
            Xtr, Xva, _ = data.matrices("A", imp)
            m_a, it, sec_a = M.fit_stage_a(name, params, seed, Xtr, data.y["train"], Xva, data.y["val"])
            p_val = M.proba(m_a, Xva)
            thr = best_f1_threshold(data.y["val"], p_val)
            val = scores(data.y["val"], p_val, thr)
            del m_a
            Xlab, Xte, pre = data.matrices("B", imp)
            m_b, n_trees, sec_b = M.fit_stage_b(name, params, seed, it, len(data.y["train"]),
                                                Xlab, data.y["labeled"])
            t0 = time.perf_counter()
            p_te = M.proba(m_b, Xte)
            batch_s = time.perf_counter() - t0
            te = scores(data.y["test"], p_te, thr)
            # one row at a time, single thread: what a scoring request costs (thread pools only add overhead here)
            for attr in ("n_jobs", "thread_count"):
                if attr in m_b.get_params():
                    m_b.set_params(**{attr: 1})
            lat = single_row_latency_ms(lambda X: m_b.predict_proba(X), Xte)
            np.savez_compressed(out / "final" / f"{name}_seed{seed}_test.npz", p=p_te, y=data.y["test"])
            if seed == seeds[0]:
                joblib.dump(Pipeline([("preproc", pre), ("clf", m_b)]),
                            out / "pipelines" / f"{name}.joblib")
            res = dict(model=name, seed=seed, params=params, stage_a_iterations=it,
                       refit_trees=n_trees, fit_seconds_stage_a=sec_a, fit_seconds_refit=sec_b,
                       test_predict_rows_per_s=len(p_te) / batch_s, single_row=lat,
                       validation=val, test=te)
            dump_json(res, res_path)
            log(f"[final] {name} seed {seed}: test PR-AUC={te['pr_auc']:.4f} ROC={te['roc_auc']:.4f} "
                f"R@1%FPR={te['recall_at_1pct_fpr']:.4f} F1={te['f1']:.4f}")
        data.drop_stage("B")


def report(names, seeds, out: Path, ref="LightGBM"):
    rows, preds = [], {}
    for name in names:
        rs = [json.loads((out / "final" / f"{name}_seed{s}.json").read_text()) for s in seeds
              if (out / "final" / f"{name}_seed{s}.json").exists()]
        if not rs:
            continue
        z = np.load(out / "final" / f"{name}_seed{seeds[0]}_test.npz")
        preds[name], y = z["p"], z["y"]
        agg = {}
        for k in ["pr_auc", "roc_auc", "recall_at_1pct_fpr", "precision", "recall", "f1"]:
            v = np.array([r["test"][k] for r in rs])
            agg[k] = dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        agg["pr_auc_ci95_seed0"] = bootstrap_ci(y, preds[name], metric="pr_auc")
        agg["roc_auc_ci95_seed0"] = bootstrap_ci(y, preds[name], metric="roc_auc")
        agg["single_row_median_ms"] = float(np.median([r["single_row"]["median_ms"] for r in rs]))
        agg["single_row_p95_ms"] = float(np.median([r["single_row"]["p95_ms"] for r in rs]))
        agg["refit_seconds"] = float(np.median([r["fit_seconds_refit"] for r in rs]))
        agg["n_seeds"] = len(rs)
        rows.append(dict(model=name, **agg))
    if ref in preds:
        for r in rows:
            if r["model"] != ref:
                r[f"pr_auc_delta_vs_{ref}"] = paired_bootstrap_delta(y, preds[r["model"]], preds[ref])
    rows.sort(key=lambda r: -r["pr_auc"]["mean"])
    dump_json(rows, out / "summary.json")
    print(f"\n{'model':<20}{'PR-AUC':>16}{'ROC-AUC':>16}{'R@1%FPR':>16}{'F1':>16}{'ms/row':>9}")
    for r in rows:
        f = lambda k: f"{r[k]['mean']:.4f}±{r[k]['std']:.4f}"
        print(f"{r['model']:<20}{f('pr_auc'):>16}{f('roc_auc'):>16}{f('recall_at_1pct_fpr'):>16}"
              f"{f('f1'):>16}{r['single_row_median_ms']:>9.2f}")


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tune", "final", "report", "all"], default="all")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--sample", type=float, default=None)
    ap.add_argument("--models", nargs="+", default=M.MODELS)
    ap.add_argument("--trials", type=int, default=25)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="runs/models")
    a = ap.parse_args()
    out = Path(a.out)
    for d in ["best", "final", "pipelines"]:
        (out / d).mkdir(parents=True, exist_ok=True)
    data = None
    if a.stage != "report":
        log("loading", a.data, "sample", a.sample)
        data = Data(load_prepared(a.data, a.sample))
        log("split", data.sizes(), "features", len(data.num), "+", len(data.cat))
        dump_json(dict(sizes=data.sizes(), label_delay_days=data.delay, sample=a.sample,
                       n_numeric=len(data.num), n_categorical=len(data.cat)), out / "setup.json")
    if a.stage in ("tune", "all"):
        tune(data, a.models, a.trials, out)
        data.drop_stage("A")
    if a.stage in ("final", "all"):
        final(data, a.models, a.seeds, out)
    if a.stage in ("report", "all"):
        report(a.models, a.seeds, out)


if __name__ == "__main__":
    main()
