"""
fraud_app.registry

The models the service can score with, by name:

  production       the pipeline in artifacts/ (what /api/predict and /api/shap use)
  <experiment>     every pipeline saved by experiments/run_models.py, e.g. runs/models/pipelines/
                   XGBoost.joblib, with the threshold chosen on its validation block and its test
                   metrics from that run's summary.json

Pipelines are loaded on first use (a Random Forest pipeline is several hundred MB).
Directories are taken from FRAUD_MODEL_DIRS (os.pathsep-separated), default runs/models and
runs/lrq under the project root.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent
PRODUCTION = "production"

LABELS = {
    PRODUCTION: "Production (LightGBM)",
    "LightGBM": "LightGBM",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "RandomForest": "Random Forest",
    "LogisticRegression": "Logistic regression",
    "LogisticRegressionQ": "Logistic regression (quantile)",
}


def model_dirs() -> List[Path]:
    env = os.environ.get("FRAUD_MODEL_DIRS")
    if env:
        return [Path(p) for p in env.split(os.pathsep) if p]
    return [ROOT_DIR / "runs" / "models", ROOT_DIR / "runs" / "lrq"]


class ModelEntry:
    def __init__(self, name: str, path: Optional[Path], threshold: float, metrics: Dict[str, Any],
                 pipe: Any = None):
        self.name = name
        self.label = LABELS.get(name, name)
        self.path = path
        self.threshold = float(threshold)
        self.metrics = metrics
        self._pipe = pipe
        self._lock = threading.Lock()
        self.explainers: Dict[str, Any] = {}  # method -> explainer, built by fraud_app.explainers

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def pipe(self) -> Any:
        if self._pipe is None:
            with self._lock:
                if self._pipe is None:
                    self._pipe = joblib.load(self.path)
        return self._pipe

    def describe(self) -> Dict[str, Any]:
        return {"name": self.name, "label": self.label, "threshold": self.threshold,
                "metrics": self.metrics, "loaded": self.loaded}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _metrics_from_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    def mean(k):
        v = row.get(k)
        return v.get("mean") if isinstance(v, dict) else v
    return {
        "pr_auc": mean("pr_auc"), "roc_auc": mean("roc_auc"),
        "recall_at_1pct_fpr": mean("recall_at_1pct_fpr"), "f1": mean("f1"),
        "pr_auc_ci95": row.get("pr_auc_ci95_seed0"),
        "single_row_ms": row.get("single_row_median_ms"), "refit_seconds": row.get("refit_seconds"),
        "n_seeds": row.get("n_seeds"),
    }


def discover(dirs: Optional[List[Path]] = None) -> Dict[str, ModelEntry]:
    """Experiment pipelines found on disk (not loaded yet), in a stable order."""
    found: Dict[str, ModelEntry] = {}
    for d in dirs if dirs is not None else model_dirs():
        summary = _read_json(d / "summary.json") or []
        by_name = {r.get("model"): r for r in summary if isinstance(r, dict)}
        for p in sorted((d / "pipelines").glob("*.joblib")):
            name = p.stem
            final = _read_json(d / "final" / f"{name}_seed0.json") or {}
            thr = (final.get("validation") or {}).get("threshold", 0.5)
            metrics = _metrics_from_summary(by_name[name]) if name in by_name else {}
            found[name] = ModelEntry(name, p, thr, metrics)
    order = ["LightGBM", "XGBoost", "CatBoost", "RandomForest", "LogisticRegressionQ", "LogisticRegression"]
    return dict(sorted(found.items(), key=lambda kv: order.index(kv[0]) if kv[0] in order else 99))


_REGISTRY: Optional[Dict[str, ModelEntry]] = None
_REG_LOCK = threading.Lock()


def registry() -> Dict[str, ModelEntry]:
    """production first (if loaded by services), then the experiment pipelines."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REG_LOCK:
            if _REGISTRY is None:
                from . import services

                reg: Dict[str, ModelEntry] = {}
                if services.PIPE is not None:
                    m = services.load_metrics() or {}
                    g = m.get("global_metrics") or {}
                    reg[PRODUCTION] = ModelEntry(PRODUCTION, services.PIPELINE_FILE, services.get_threshold(),
                                                 {"pr_auc": g.get("pr_auc"), "roc_auc": g.get("roc_auc")},
                                                 pipe=services.PIPE)
                reg.update(discover())
                _REGISTRY = reg
    return _REGISTRY


def reset() -> None:
    """Forget discovered models (tests, or after new pipelines were trained)."""
    global _REGISTRY
    _REGISTRY = None


def get(name: str) -> ModelEntry:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown model {name!r}; available: {', '.join(reg)}")
    return reg[name]
