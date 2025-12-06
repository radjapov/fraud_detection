"""
fraud_app.api

Blueprint with REST API:
  GET  /api/random?fraud=0|1
  GET  /api/examples
  POST /api/predict
  POST /api/shap
  GET  /api/history
  GET/POST /api/threshold
  GET  /api/metrics
"""

from __future__ import annotations

import traceback
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from flask import Blueprint, jsonify, request

from .services import (
    ensure_features_df,
    compute_probs_from_pipeline,
    save_history_entry,
    load_history,
    compute_shap_for_df_via_worker,
    load_metrics,
    load_random_row,
    get_threshold,
    set_threshold,
    load_artifacts,
)

bp_api = Blueprint("api", __name__, url_prefix="/api")

# Load artifacts once on import (pipeline, meta, shap explainer, threshold)
load_artifacts()


@bp_api.route("/examples", methods=["GET"])
def api_examples():
    """Simple example payload for the UI form."""
    example = {
        "amount": 50.0,
        "hour": 12,
        "card_age_months": 10,
        "sender_txn_24h": 2,
        "sender_avg_amount": 80.0,
        "distance_km": 5.0,
        "ip_risk": 0.02,
        "mcc": 4.0,
        "country_risk": 0,
        "receiver_new": 0,
        "device_new": 0,
        "is_foreign": 0,
    }
    return jsonify({"examples": [example]})


@bp_api.route("/random", methods=["GET"])
def api_random():
    """Random row from prepared dataset. Optional ?fraud=1 or 0."""
    fraud_param = request.args.get("fraud")
    fraud_val = None
    if fraud_param is not None:
        fraud_val = 1 if fraud_param in ("1", "true", "True") else 0

    try:
        row = load_random_row(fraud_val)
        return jsonify({"row": row})
    except Exception as e:
        print("[api_random] error:", e)
        return jsonify({"error": str(e)}), 500


@bp_api.route("/predict", methods=["POST"])
def api_predict():
    """Main prediction endpoint."""
    try:
        payload = request.get_json(force=True)

        if isinstance(payload, dict):
            rows: List[Dict[str, Any]] = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            return jsonify({"error": "Payload must be dict or list"}), 400

        raw_df = pd.DataFrame(rows)
        safe_df, feat_list = ensure_features_df(raw_df.copy())

        probs = compute_probs_from_pipeline(safe_df)
        thr = get_threshold()
        labels = [1 if p >= thr else 0 for p in probs]

        result = {
            "probs": [float(p) for p in probs],
            "labels": labels,
            "threshold": float(thr),
        }

        # save history (best-effort)
        try:
            entry = {
                "ts": datetime.utcnow().isoformat(),
                "input": raw_df.to_dict(orient="records"),
                "result": result,
            }
            save_history_entry(entry)
        except Exception as e_hist:
            print("[api_predict] history save failed:", e_hist)

        return jsonify(result)
    except Exception:
        tb = traceback.format_exc()
        print("[api_predict] error:", tb)
        return jsonify({"error": "Predict error", "traceback": tb}), 500


@bp_api.route("/shap", methods=["POST"])
def api_shap():
    """
    SHAP explanation for a single example (or first row if list).
    Computed via worker process to avoid crashes inside main app.
    """
    try:
        payload = request.get_json(force=True)

        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list) and payload:
            rows = payload[:1]  # take only first row for explanation
        else:
            return jsonify({"error": "Payload must be dict or non-empty list"}), 400

        raw_df = pd.DataFrame(rows)
        safe_df, feat_list = ensure_features_df(raw_df.copy())

        shap_data = compute_shap_for_df_via_worker(safe_df)
        return jsonify(shap_data)
    except Exception:
        tb = traceback.format_exc()
        print("[api_shap] error:", tb)
        return jsonify({"error": "SHAP runtime error", "traceback": tb}), 500


@bp_api.route("/history", methods=["GET"])
def api_history():
    """Return last N predictions from history.json."""
    try:
        hist = load_history()
        return jsonify(hist)
    except Exception as e:
        print("[api_history] error:", e)
        return jsonify({"error": str(e)}), 500


@bp_api.route("/threshold", methods=["GET", "POST"])
def api_threshold():
    """Get or set decision threshold τ."""
    if request.method == "GET":
        return jsonify({"threshold": float(get_threshold())})

    try:
        payload = request.get_json(force=True)
        t_raw = payload.get("threshold")
        new_thr = set_threshold(float(t_raw))
        return jsonify({"threshold": float(new_thr)})
    except Exception as e:
        print("[api_threshold] error:", e)
        return jsonify({"error": str(e)}), 400


@bp_api.route("/metrics", methods=["GET"])
def api_metrics():
    """Return precomputed global model metrics from artifacts/metrics_ieee.json."""
    try:
        data = load_metrics()
        if data is None:
            return (
                jsonify(
                    {
                        "error": "metrics file not found. "
                        "Run eval_model_metrics.py to generate artifacts/metrics_ieee.json."
                    }
                ),
                404,
            )
        return jsonify(data)
    except Exception as e:
        print("[api_metrics] error:", e)
        return jsonify({"error": str(e)}), 500