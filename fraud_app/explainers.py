"""
fraud_app.explainers

One interface for explaining a single transaction with any model of the registry:

    explain(entry, X_prepared, method)  ->  {"method", "model", "space", "base_value",
                                             "contributions": [{"feature", "contribution", "value"}],
                                             "ms"}

  shap   TreeExplainer (interventional, probability output) for tree ensembles; LinearExplainer
         for logistic regression (log-odds), after the quantile step when the model has one
  lime   LimeTabularExplainer in the model's input space; one-hot columns are categorical;
         reseeded per call, so the same request gives the same answer

Attributions are computed on the model's columns (scaled numerics + one-hot) and summed back to
the input features, as in fraud_app.explain. Explainers are built once per model and cached.
Unlike /api/shap (production model, process-isolated modes), these run in the web process.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd

METHODS = ("shap", "lime")
LIME_SAMPLES = 2000
_BUILD_LOCK = threading.Lock()


def _owners(pre: Any, sources: List[str]) -> np.ndarray:
    owners = []
    for name in pre.get_feature_names_out():
        rest = name.split("__", 1)[-1]
        f = max((s for s in sources if rest == s or rest.startswith(s + "_")), key=len, default=None)
        if f is None:
            raise RuntimeError(f"Cannot map column {name!r} to an input feature")
        owners.append(sources.index(f))
    return np.asarray(owners)


def _split(pipe: Any):
    """(preprocessor, final estimator, optional transform between them)"""
    pre, clf = pipe.steps[0][1], pipe.steps[-1][1]
    inner = None
    if hasattr(clf, "steps"):  # e.g. quantile transform + logistic regression
        inner, clf = clf[:-1], clf.steps[-1][1]
    return pre, clf, inner


def _is_linear(clf: Any) -> bool:
    return type(clf).__name__ in ("LogisticRegression", "SGDClassifier", "LinearSVC")


class _Built:
    def __init__(self, method: str, entry: Any, background: pd.DataFrame, sources: List[str]):
        import shap  # noqa: F401  (import errors surface here, not at module import)

        self.method = method
        pipe = entry.pipe
        self.pre, self.clf, self.inner = _split(pipe)
        self.sources = sources
        self.owner = _owners(self.pre, sources)
        bg = np.asarray(self.pre.transform(background[sources]), dtype=np.float64)
        if self.inner is not None:
            bg = np.asarray(self.inner.transform(np.nan_to_num(bg)), dtype=np.float64)
        self.linear = _is_linear(self.clf)
        if method == "shap":
            import shap

            if self.linear:
                self.ex = shap.LinearExplainer(self.clf, np.nan_to_num(bg))
                self.space = "log-odds"
            else:
                self.ex = shap.TreeExplainer(self.clf, data=bg[:100], feature_perturbation="interventional",
                                             model_output="probability")
                self.space = "probability"
        else:
            from lime.lime_tabular import LimeTabularExplainer

            cat_cols = [i for i, n in enumerate(self.pre.get_feature_names_out()) if n.startswith("cat__")]
            self.ex = LimeTabularExplainer(np.nan_to_num(bg), categorical_features=cat_cols, mode="classification",
                                           discretize_continuous=False, sample_around_instance=True, random_state=0)
            self.space = "local linear weight"

    def columns(self, X_prepared: pd.DataFrame) -> np.ndarray:
        t = np.asarray(self.pre.transform(X_prepared[self.sources]), dtype=np.float64)
        if self.inner is not None:
            t = np.asarray(self.inner.transform(np.nan_to_num(t)), dtype=np.float64)
        return t

    def attributions(self, t: np.ndarray):
        if self.method == "shap":
            x = np.nan_to_num(t) if self.linear else t
            sv = self.ex.shap_values(x)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)
            if sv.ndim == 3:
                sv = sv[..., 1]
            base = np.asarray(self.ex.expected_value).ravel()
            return sv[0], float(base[-1]) if base.size else None
        from sklearn.utils import check_random_state

        self.ex.random_state = check_random_state(0)  # same request -> same explanation
        f = lambda Z: self.clf.predict_proba(np.asarray(Z))
        e = self.ex.explain_instance(np.nan_to_num(t[0]), f, num_features=t.shape[1],
                                     num_samples=LIME_SAMPLES, labels=(1,))
        w = np.zeros(t.shape[1])
        for j, v in e.as_map()[1]:
            w[j] = v
        return w, float(e.intercept[1])


def _built(entry: Any, method: str, background_fn, sources: List[str]) -> _Built:
    if method not in entry.explainers:
        with _BUILD_LOCK:
            if method not in entry.explainers:
                entry.explainers[method] = _Built(method, entry, background_fn(), sources)
    return entry.explainers[method]


def explain(entry: Any, X_prepared: pd.DataFrame, method: str, background_fn) -> Dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {METHODS}")
    sources = list(X_prepared.columns)
    b = _built(entry, method, background_fn, sources)
    t0 = time.perf_counter()
    cols, base = b.attributions(b.columns(X_prepared.iloc[[0]]))
    per_feature = np.zeros(len(sources))
    np.add.at(per_feature, b.owner, cols)
    ms = (time.perf_counter() - t0) * 1000
    row = X_prepared.iloc[0]

    def shown(v):
        if v is None or (isinstance(v, str) and v == "NA"):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return v
        return None if np.isnan(f) else f

    contribs = [{"feature": s, "contribution": float(c), "value": shown(row[s])}
                for s, c in zip(sources, per_feature)]
    contribs.sort(key=lambda d: -abs(d["contribution"]))
    return {"method": method, "model": entry.name, "space": b.space, "base_value": base,
            "contributions": contribs, "ms": ms}
