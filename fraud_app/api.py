"""
fraud_app.api

Blueprint с REST API:
  GET  /api/random?fraud=0|1
  POST /api/predict
  POST /api/shap
  GET  /api/history
  GET/POST /api/threshold
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List

import pandas as pd
from flask import Blueprint, jsonify, request

from .services import (
    compute_probs,
    compute_shap_for_df,
    ensure_features_df,
    get_history,
    get_threshold,
    load_artifacts,
    load_random_row,
    make_history_entry,
    save_history_entry,
    set_threshold,
)

bp_api = Blueprint("api", __name__, url_prefix="/api")


# загружаем артефакты при импортe модуля (если ещё не загружены)
load_artifacts()


@bp_api.route("/examples", methods=["GET"])
def api_examples():
    """Простой пример для заполнения формы."""
    example = {
        "amount": 50.0,
        "hour": 12,
        "card_age_months": 10,
        "sender_txn_24h": 2,
        "sender_avg_amount": 80.0,
        "distance_km": 5.0,
        "ip_risk": 0.02,
        "mcc": 4.0,
        "country_risk": 0.0,
        "receiver_new": 0.0,
        "device_new": 0.0,
        "is_foreign": 0.0,
    }
    return jsonify({"examples": [example]})


@bp_api.route("/random", methods=["GET"])
def api_random():
    """Рандомный пример из датасета. ?fraud=1 или 0."""
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
    """Основной эндпоинт предикта."""
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

        probs = compute_probs(safe_df)
        thr = get_threshold()
        labels = [1 if p >= thr else 0 for p in probs]

        result = {
            "probs": [float(p) for p in probs],
            "labels": labels,
            "threshold": thr,
        }

        # пишем историю (по желанию можно выключить)
        try:
            entry = make_history_entry(raw_df, result)
            save_history_entry(entry)
        except Exception as e_hist:
            print("[api_predict] history save failed:", e_hist)

        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        print("[api_predict] error:", tb)
        return jsonify({"error": "Predict error", "traceback": tb}), 500


@bp_api.route("/shap", methods=["POST"])
def api_shap():
    """Эндпоинт для одного примера, считает SHAP в том же процессе."""
    try:
        payload = request.get_json(force=True)

        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            return jsonify({"error": "Payload must be dict or list"}), 400

        raw_df = pd.DataFrame(rows)
        safe_df, feat_list = ensure_features_df(raw_df.copy())
        shap_data = compute_shap_for_df(safe_df)

        return jsonify(shap_data)
    except Exception as e:
        tb = traceback.format_exc()
        print("[api_shap] error:", tb)
        return jsonify({"error": "SHAP runtime error", "traceback": tb}), 500


@bp_api.route("/history", methods=["GET"])
def api_history():
    return jsonify(get_history())


@bp_api.route("/threshold", methods=["GET", "POST"])
def api_threshold():
    if request.method == "GET":
        return jsonify({"threshold": get_threshold()})

    try:
        payload = request.get_json(force=True)
        t_raw = payload.get("threshold")
        new_thr = set_threshold(float(t_raw))
        return jsonify({"threshold": new_thr})
    except Exception as e:
        return jsonify({"error": str(e)}), 400