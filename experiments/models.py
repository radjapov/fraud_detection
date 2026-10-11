"""
The five compared classifiers: search space, construction and the two-stage fit used by the
service (stage A: fit on train, early-stop / threshold on validation; stage B: refit on all
labeled rows). Every model gets the same features, the same split and the same tuning budget.
"""
from __future__ import annotations

import time
import warnings

import numpy as np

EARLY_STOP = 100
MAX_TREES = 3000
BOOSTING = {"LightGBM", "XGBoost", "CatBoost"}
# boosted trees take NaN natively; the other two need median imputation
NEEDS_IMPUTE = {"LogisticRegression": True, "LogisticRegressionQ": True, "RandomForest": True,
                "XGBoost": False, "CatBoost": False, "LightGBM": False}
MODELS = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "LightGBM"]


def suggest(name, trial) -> dict:
    s = trial
    if name in ("LogisticRegression", "LogisticRegressionQ"):
        return dict(C=s.suggest_float("C", 1e-4, 10.0, log=True))
    if name == "RandomForest":
        return dict(
            n_estimators=s.suggest_int("n_estimators", 100, 400, step=50),
            max_depth=s.suggest_int("max_depth", 8, 30),
            min_samples_leaf=s.suggest_int("min_samples_leaf", 1, 50, log=True),
            max_features=s.suggest_categorical("max_features", ["sqrt", 0.1, 0.3]),
        )
    common = dict(
        learning_rate=s.suggest_float("learning_rate", 0.02, 0.2, log=True),
        subsample=s.suggest_float("subsample", 0.5, 1.0),
        reg_lambda=s.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight=s.suggest_float("scale_pos_weight", 1.0, 30.0, log=True),
    )
    if name == "LightGBM":
        return dict(common,
                    num_leaves=s.suggest_int("num_leaves", 15, 255, log=True),
                    min_child_samples=s.suggest_int("min_child_samples", 10, 300, log=True),
                    colsample_bytree=s.suggest_float("colsample_bytree", 0.2, 1.0))
    if name == "XGBoost":
        return dict(common,
                    max_depth=s.suggest_int("max_depth", 3, 10),
                    min_child_weight=s.suggest_float("min_child_weight", 1.0, 50.0, log=True),
                    colsample_bytree=s.suggest_float("colsample_bytree", 0.2, 1.0))
    if name == "CatBoost":
        p = dict(common,
                 depth=s.suggest_int("depth", 4, 10),
                 rsm=s.suggest_float("rsm", 0.2, 1.0))
        p["l2_leaf_reg"] = p.pop("reg_lambda")
        return p
    raise ValueError(name)


def build(name, params: dict, seed: int, n_trees: int | None = None):
    p = dict(params)
    if name == "LogisticRegression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs",
                                  random_state=seed, **p)
    if name == "LogisticRegressionQ":
        # same model behind a quantile transform: heavy-tailed counts (C1..C14 etc.) drift upward in
        # later months, and a linear model on standardised values extrapolates them to p = 1
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import QuantileTransformer
        return Pipeline([
            ("qt", QuantileTransformer(n_quantiles=1000, output_distribution="normal",
                                       subsample=200_000, random_state=seed)),
            ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs",
                                      random_state=seed, **p)),
        ])
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(class_weight="balanced_subsample", n_jobs=-1,
                                      random_state=seed, **p)
    n = n_trees or MAX_TREES
    if name == "LightGBM":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=n, subsample_freq=1, random_state=seed,
                                  verbosity=-1, n_jobs=-1, **p)
    if name == "XGBoost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=n, tree_method="hist", eval_metric="aucpr",
                                 random_state=seed, n_jobs=-1, verbosity=0,
                                 early_stopping_rounds=None if n_trees else EARLY_STOP, **p)
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        # subsample needs Bernoulli bootstrap on CPU
        return CatBoostClassifier(iterations=n, bootstrap_type="Bernoulli", eval_metric="PRAUC",
                                  random_seed=seed, verbose=0, thread_count=-1,
                                  allow_writing_files=False, **p)
    raise ValueError(name)


def fit_stage_a(name, params, seed, Xtr, ytr, Xva, yva):
    """Returns (model, best_iteration or None, fit_seconds)."""
    m = build(name, params, seed)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if name == "LightGBM":
            import lightgbm as lgb
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="average_precision",
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            best = int(m.best_iteration_ or m.n_estimators)
        elif name == "XGBoost":
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            best = int(m.best_iteration + 1)
        elif name == "CatBoost":
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=EARLY_STOP, use_best_model=True)
            best = int(m.get_best_iteration() + 1)
        else:
            m.fit(Xtr, ytr)
            best = None
    return m, best, time.perf_counter() - t0


def fit_stage_b(name, params, seed, best_iter, n_train, Xlab, ylab):
    """Refit on all labeled rows; boosted models get the stage-A tree count scaled to the data."""
    n_trees = None
    if name in BOOSTING:
        n_trees = max(50, int(round(best_iter * len(ylab) / max(n_train, 1))))
    m = build(name, params, seed, n_trees=n_trees)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(Xlab, ylab)
    return m, n_trees, time.perf_counter() - t0


def proba(m, X) -> np.ndarray:
    return np.asarray(m.predict_proba(X))[:, 1].astype(np.float64)
