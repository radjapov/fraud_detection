"""
fraud_app.compare

API for comparing models and explanation methods on one transaction:

  GET  /api/models        models of the registry with threshold and test metrics
  POST /api/compare       {"row": {...}, "models": [...]?, "explain": "shap"|"lime"|"none", "top": 5}
                          -> per model: probability, decision at its own threshold, top features,
                             prediction and explanation time
  POST /api/explain       {"row": {...}, "model": "LightGBM", "method": "shap"|"lime", "top": 10}
  GET  /api/experiments   the result tables of the NTCC experiments (runs/paper/*.csv), if present
"""

from __future__ import annotations

import csv
import time
import traceback
from typing import Any, Dict, List

import pandas as pd
from flask import Blueprint, jsonify, request
from werkzeug.exceptions import BadRequest

from . import registry, services
from .explainers import METHODS, explain
from .features import prepare_features

bp_compare = Blueprint("compare", __name__)
MAX_TOP = 30


def _row_from(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Body must be a JSON object")
    row = payload.get("row")
    if not isinstance(row, dict) or not row:
        raise ValueError("'row' must be a non-empty object of feature values")
    return row


def _top(payload: Dict[str, Any], default: int) -> int:
    try:
        return max(1, min(MAX_TOP, int(payload.get("top", default))))
    except (TypeError, ValueError):
        raise ValueError("'top' must be an integer")


def _prepared(entry: Any, row: Dict[str, Any]) -> pd.DataFrame:
    X, _ = prepare_features(pd.DataFrame([row]), services._get_feature_list_from_meta(), entry.pipe)
    return X


def _score(entry: Any, X: pd.DataFrame) -> Dict[str, Any]:
    t0 = time.perf_counter()
    p = float(entry.pipe.predict_proba(X)[:, 1][0])
    ms = (time.perf_counter() - t0) * 1000
    return {"probability": p, "fraud": p >= entry.threshold, "threshold": entry.threshold, "predict_ms": ms}


def _explain(entry: Any, X: pd.DataFrame, method: str, top: int) -> Dict[str, Any]:
    out = explain(entry, X, method, background_fn=lambda: services.background_frame(entry.pipe))
    out["contributions"] = out["contributions"][:top]
    return out


@bp_compare.route("/models", methods=["GET"])
def api_models():
    return jsonify({"models": [e.describe() for e in registry.registry().values()],
                    "methods": list(METHODS)})


@bp_compare.route("/compare", methods=["POST"])
def api_compare():
    try:
        payload = request.get_json(force=True)
        row = _row_from(payload)
        method = payload.get("explain", "shap")
        if method not in METHODS + ("none",):
            raise ValueError(f"'explain' must be one of {METHODS + ('none',)}")
        top = _top(payload, 5)
        reg = registry.registry()
        names: List[str] = payload.get("models") or list(reg)
        unknown = [n for n in names if n not in reg]
        if unknown:
            raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    except BadRequest:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    results = []
    for name in names:
        entry = reg[name]
        item: Dict[str, Any] = {"model": name, "label": entry.label, "metrics": entry.metrics}
        try:
            X = _prepared(entry, row)
            item.update(_score(entry, X))
            if method != "none":
                ex = _explain(entry, X, method, top)
                item.update(explanation=ex["contributions"], explain_ms=ex["ms"], space=ex["space"])
        except Exception as e:  # one broken model must not hide the others
            print(f"[api_compare] {name}:", traceback.format_exc())
            item["error"] = f"{type(e).__name__}: {e}"
        results.append(item)
    return jsonify(services.json_safe({"method": method, "results": results}))


@bp_compare.route("/explain", methods=["POST"])
def api_explain():
    try:
        payload = request.get_json(force=True)
        row = _row_from(payload)
        method = payload.get("method", "shap")
        if method not in METHODS:
            raise ValueError(f"'method' must be one of {METHODS}")
        entry = registry.get(payload.get("model", registry.PRODUCTION))
        top = _top(payload, 10)
    except BadRequest:
        return jsonify({"error": "Request body must be valid JSON"}), 400
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e).strip("'\"")}), 400
    try:
        X = _prepared(entry, row)
        out = _explain(entry, X, method, top)
        out.update(_score(entry, X))
        return jsonify(services.json_safe(out))
    except Exception:
        print("[api_explain] error:", traceback.format_exc())
        return jsonify({"error": "Explanation error"}), 500


@bp_compare.route("/experiments", methods=["GET"])
def api_experiments():
    paper = registry.ROOT_DIR / "runs" / "paper"
    tables = {}
    for name in ("table1_models", "table2_protocols", "table3_explanations", "table4_latency"):
        f = paper / f"{name}.csv"
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            # columns as a list: a JSON object would lose the column order
            tables[name] = {"columns": rows[0] if rows else [], "rows": rows[1:]}
    return jsonify({"tables": tables, "available": bool(tables)})
