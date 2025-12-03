#!/usr/bin/env python3
# app_ui.py — исправленная/робастная версия для IEEE pipeline + SHAP
# Положите рядом: artifacts/pipeline_ieee.joblib, artifacts/meta_ieee.joblib, artifacts/shap_explainer_ieee.joblib

import json
import os
import subprocess
import sys
import traceback
import typing as t
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

# ==== Конфиг путей ====
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
HISTORY_FILE = ARTIFACTS_DIR / "history.json"

PIPELINE_FILE = ARTIFACTS_DIR / "pipeline_ieee.joblib"
META_FILE = ARTIFACTS_DIR / "meta_ieee.joblib"
SHAP_FILE = ARTIFACTS_DIR / "shap_explainer_ieee.joblib"
THRESH_JSON = ARTIFACTS_DIR / "meta_ieee_threshold.json"

# ==== Flask ====
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config['JSON_SORT_KEYS'] = False

# Global holders
PIPE = None
META = None
SHAP_EXPLAINER = None
THRESHOLD = None


# === Helpers ===
def safe_print(*a, **kw):
    print(*a, **kw)


def load_artifacts():
    """Load pipeline, meta, shap and threshold if present."""
    global PIPE, META, SHAP_EXPLAINER, THRESHOLD
    safe_print("Loading artifacts from:", ARTIFACTS_DIR)
    # pipeline
    if PIPELINE_FILE.exists():
        try:
            PIPE = joblib.load(PIPELINE_FILE)
            safe_print(f"[load_artifacts] Loaded pipeline: {PIPELINE_FILE} type: {type(PIPE)}")
        except Exception as e:
            safe_print("[load_artifacts] Failed to load pipeline:", e)
            PIPE = None
    else:
        safe_print("[load_artifacts] pipeline file not found:", PIPELINE_FILE)

    # meta
    if META_FILE.exists():
        try:
            META = joblib.load(META_FILE)
            safe_print(f"[load_artifacts] Loaded meta: {META_FILE}")
        except Exception as e:
            safe_print("[load_artifacts] Failed to load meta:", e)
            META = None
    else:
        META = None

    # threshold (json)
    if THRESH_JSON.exists():
        try:
            with open(THRESH_JSON, "r", encoding="utf8") as f:
                jd = json.load(f)
                THRESHOLD = float(jd.get("threshold", jd.get("chosen_threshold", 0.5)))
                safe_print(f"[load_artifacts] Loaded threshold from json: {THRESHOLD}")
        except Exception as e:
            safe_print("[load_artifacts] Cannot load threshold json:", e)
            THRESHOLD = None

    # shap
    if SHAP_FILE.exists():
        try:
            SHAP_EXPLAINER = joblib.load(SHAP_FILE)
            safe_print(f"[load_artifacts] Loaded SHAP explainer: {SHAP_FILE}")
        except Exception as e:
            safe_print("[load_artifacts] Failed to load SHAP explainer:", e)
            SHAP_EXPLAINER = None
    else:
        SHAP_EXPLAINER = None


def save_history_entry(entry: dict):
    """Append to history.json (keeps max 200). Uses timezone-aware timestamp if caller didn't provide."""
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        hist = []
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf8") as f:
                    hist = json.load(f)
            except Exception:
                hist = []
        # ensure ts exists and is ISO timezone-aware
        if "ts" not in entry:
            entry["ts"] = datetime.now(timezone.utc).isoformat()
        hist.insert(0, entry)
        hist = hist[:200]
        with open(HISTORY_FILE, "w", encoding="utf8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        safe_print("[save_history_entry] failed:", e)


def load_history() -> list:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        safe_print("[load_history] failed:", e)
        return []


def numpy_to_native(x):
    """Convert numpy scalars/arrays to python native types for JSON."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64, np.int8, np.int16)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    # pandas types
    try:
        import pandas as pd

        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        if isinstance(x, pd.Series):
            return x.to_list()
    except Exception:
        pass
    return x


def _get_feature_list_from_meta_or_pipe():
    """Determine expected features order from META or pipeline (best effort)."""
    global META, PIPE
    feat_list = None
    if isinstance(META, dict):
        # common keys used in our flows
        for k in ("features", "feat_names", "feat_names_order", "feature_names"):
            if k in META:
                try:
                    feat_list = list(META[k])
                    return feat_list
                except Exception:
                    pass
    # try pipeline attribute
    if PIPE is not None:
        # sklearn pipeline: find final estimator or attribute feature_names_in_
        # feature_names_in_ might be on pipeline or on final estimator
        if hasattr(PIPE, "feature_names_in_"):
            try:
                return list(getattr(PIPE, "feature_names_in_"))
            except Exception:
                pass
        # pipeline object: check last step
        try:
            # sklearn Pipeline has attribute named_steps or steps
            steps = getattr(PIPE, "named_steps", None) or getattr(PIPE, "steps", None)
            if steps:
                # take last estimator
                last = steps[-1] if isinstance(steps, list) else list(steps.items())[-1]
                est = last[1] if isinstance(last, tuple) else last
                if hasattr(est, "feature_names_in_"):
                    return list(getattr(est, "feature_names_in_"))
        except Exception:
            pass
    return None


def ensure_features_df(X_df: pd.DataFrame) -> t.Tuple[pd.DataFrame, list]:
    """
    Ensure df has required features in correct order and numeric dtype.
    META is expected to contain feature list in key 'features' or 'feat_names'.
    Missing features are added with zeros.
    Returns dataframe with columns ordered as model expects and the feature list.
    """
    if META is None and PIPE is None:
        raise RuntimeError("Neither META nor PIPE is loaded to infer features")

    # prefer META list then pipeline feature names
    feat_list = _get_feature_list_from_meta_or_pipe()
    if feat_list is None:
        # fallback to X_df columns as-is
        feat_list = list(X_df.columns)

    # Build DataFrame copy to avoid mutating caller
    X_work = X_df.copy()

    # Add missing features with default 0
    missing = [f for f in feat_list if f not in X_work.columns]
    if missing:
        safe_print("[ensure_features_df] Adding missing features:", missing)
        for m in missing:
            X_work[m] = 0

    # Also, sometimes pipeline expects both raw categorical columns AND one-hot columns.
    # If pipeline has attribute 'feature_names_in_' we trust that exact set. Otherwise we'll
    # reorder to feat_list and cast to float where possible.
    ordered = []
    for f in feat_list:
        if f in X_work.columns:
            ordered.append(f)
        else:
            # as fallback, create zero column
            X_work[f] = 0
            ordered.append(f)

    # Reorder and coerce numeric types where possible.
    X_safe = X_work[ordered].copy()
    for c in X_safe.columns:
        # if dtype is object, try convert to numeric; if fails, leave as-is and fillna later
        try:
            X_safe[c] = pd.to_numeric(X_safe[c], errors='coerce')
        except Exception:
            X_safe[c] = X_safe[c]
    # Fill NaNs with 0 for numeric and empty string for object columns
    for c in X_safe.columns:
        if pd.api.types.is_numeric_dtype(X_safe[c]):
            X_safe[c] = X_safe[c].fillna(0.0).astype(np.float64)
        else:
            # keep object but convert to str and replace NaN
            X_safe[c] = X_safe[c].fillna("").astype(str)

    return X_safe, ordered


def compute_probs_from_pipeline(X_safe: pd.DataFrame) -> t.List[float]:
    """
    X_safe: pandas.DataFrame with features (may be numeric or object depending on pipeline)
    returns list of float probabilities (class 1).
    """
    if PIPE is None:
        raise RuntimeError("Pipeline not loaded")
    # Typical pipeline: sklearn Pipeline with predict_proba
    try:
        if hasattr(PIPE, "predict_proba"):
            probs = PIPE.predict_proba(X_safe)
            # predict_proba may return shape (n,2) or (n,k) - take second column if binary
            probs = np.asarray(probs)
            if probs.ndim == 1:
                # some models return single-prob; treat as probability of class 1
                out = probs
            elif probs.shape[1] >= 2:
                out = probs[:, 1]
            else:
                out = probs[:, 0]
            return [float(x) for x in out]
        # fallback: dict-like wrapper
        if isinstance(PIPE, dict):
            # common patterns: {'model': clf} or {'pipeline': pipe}
            for key in ("model", "clf", "pipeline"):
                if key in PIPE and hasattr(PIPE[key], "predict_proba"):
                    probs = PIPE[key].predict_proba(X_safe)
                    probs = np.asarray(probs)
                    if probs.ndim == 1:
                        out = probs
                    elif probs.shape[1] >= 2:
                        out = probs[:, 1]
                    else:
                        out = probs[:, 0]
                    return [float(x) for x in out]
        # try attribute .model (some wrappers)
        if hasattr(PIPE, "model") and hasattr(getattr(PIPE, "model"), "predict_proba"):
            probs = PIPE.model.predict_proba(X_safe)
            probs = np.asarray(probs)
            if probs.ndim == 1:
                out = probs
            elif probs.shape[1] >= 2:
                out = probs[:, 1]
            else:
                out = probs[:, 0]
            return [float(x) for x in out]
    except Exception as e:
        raise RuntimeError(f"pipeline predict_proba failed: {e}")

    raise RuntimeError("No usable predict_proba in pipeline")


# ==== Flask routes ====
@app.route("/")
def index():
    # render index.html in templates
    return render_template("index.html")


@app.route("/api/examples", methods=["GET"])
def api_examples():
    examples = []
    if isinstance(META, dict) and META.get("examples"):
        examples = META.get("examples")
    if not examples:
        examples = [
            {
                "amount": 20.5,
                "hour": 13,
                "card_age_months": 12,
                "sender_txn_24h": 1,
                "sender_avg_amount": 30.0,
                "distance_km": 2.0,
                "ip_risk": 0.0,
                "mcc": 5411,
                "country_risk": 0,
                "receiver_new": 0,
                "device_new": 0,
                "is_foreign": 0,
            }
        ]
    return jsonify({"examples": examples})


@app.route("/api/random", methods=["GET"])
def api_random():
    fraud = request.args.get("fraud", None)
    try:
        synth_path = BASE_DIR / "data" / "ieee_prepared.csv"
        if synth_path.exists():
            df = pd.read_csv(synth_path, nrows=20000)
            if fraud is None:
                row = df.sample(n=1).iloc[0].to_dict()
            else:
                target = 1 if str(fraud) in ("1", "true", "True") else 0
                s = df[df.get("is_fraud", pd.Series(dtype=int)) == target]
                if s.shape[0] == 0:
                    row = df.sample(n=1).iloc[0].to_dict()
                else:
                    row = s.sample(n=1).iloc[0].to_dict()
            row.pop("is_fraud", None)
            for k, v in list(row.items()):
                row[k] = numpy_to_native(v)
            return jsonify({"row": row})
    except Exception as e:
        safe_print("[api_random] failed to sample real dataset:", e)
    import random

    row = {
        "amount": round(random.uniform(1, 500), 2),
        "hour": random.randint(0, 23),
        "card_age_months": random.randint(0, 60),
        "sender_txn_24h": random.randint(0, 10),
        "sender_avg_amount": round(random.uniform(10, 200), 2),
        "distance_km": round(random.uniform(0, 500), 2),
        "ip_risk": round(random.random(), 6),
        "mcc": random.choice([4812, 5411, 5912, 6011, 7995]),
        "country_risk": random.choice([0, 1, 2]),
        "receiver_new": random.choice([0, 1]),
        "device_new": random.choice([0, 1]),
        "is_foreign": random.choice([0, 1]),
    }
    return jsonify({"row": row})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            return jsonify({"error": "payload must be dict or list"}), 400

        df = pd.DataFrame(rows)
        X_safe, feat_list = ensure_features_df(df)
        probs = compute_probs_from_pipeline(X_safe)
        thr = (
            THRESHOLD
            if THRESHOLD is not None
            else (
                float(META.get("chosen_threshold"))
                if isinstance(META, dict) and META.get("chosen_threshold") is not None
                else 0.5
            )
        )
        labels = [1 if p >= thr else 0 for p in probs]
        result = {"probs": [float(p) for p in probs], "labels": labels, "threshold": float(thr)}
        try:
            hist_entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "input": df.to_dict(orient="records"),
                "result": result,
            }
            save_history_entry(hist_entry)
        except Exception as e:
            safe_print("[api_predict] history save failed:", e)
        return jsonify(result)
    except Exception as e:
        tb = traceback.format_exc()
        safe_print("Predict error:", tb)
        return jsonify({"error": "Predict error", "trace": str(e), "traceback": tb}), 500


def compute_shap_for_df_via_worker(df, artifacts_dir=str(ARTIFACTS_DIR), timeout=12):
    """
    Запускает внешнюю программу shap_worker.py в отдельном процессе.
    df: pandas.DataFrame single-row (or multi-row) — будет сериализован в JSON (только первые N колонок)
    timeout: seconds
    """
    # keep payload small: only first row is needed for UI
    rows = df.to_dict(orient="records")
    payload = {"rows": rows, "artifacts_dir": artifacts_dir}
    # call worker
    worker_path = Path(__file__).resolve().parent / "shap_worker.py"
    if not worker_path.exists():
        raise RuntimeError(f"shap_worker.py not found at {worker_path}")

    cmd = [sys.executable, str(worker_path)]
    try:
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload).encode("utf8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as te:
        raise RuntimeError(f"SHAP worker timeout ({timeout}s): {te}")
    # read stdout
    try:
        out_text = proc.stdout.decode("utf8").strip()
        if not out_text:
            # include stderr
            err = proc.stderr.decode("utf8", errors="ignore")
            raise RuntimeError(f"SHAP worker produced no stdout. Stderr: {err}")
        res = json.loads(out_text)
        if not res.get("ok"):
            raise RuntimeError(f"SHAP worker error: {res.get('error')} | trace: {res.get('trace')}")
        return res.get("result")
    except json.JSONDecodeError as je:
        err = proc.stderr.decode("utf8", errors="ignore")
        raise RuntimeError(
            f"Cannot parse SHAP worker output as JSON. stdout: {out_text!r}, stderr: {err}"
        )


@app.route("/api/shap", methods=["POST"])
def api_shap():
    # new API using worker
    try:
        if SHAP_EXPLAINER is None:
            return jsonify({"error": "SHAP explainer not available (or disabled)"}), 400
        payload = request.get_json(force=True)
        if isinstance(payload, dict):
            rows = [payload]
        elif isinstance(payload, list):
            rows = payload
        else:
            return jsonify({"error": "payload must be dict or list"}), 400
        df = pd.DataFrame(rows)
        X_safe, feat_list = ensure_features_df(df)
        try:
            # worker handles safety & timeouts
            res = compute_shap_for_df_via_worker(
                X_safe, artifacts_dir=str(ARTIFACTS_DIR), timeout=12
            )
            return jsonify(res)
        except Exception as e:
            tb = traceback.format_exc()
            print("[api_shap] worker failed:", tb)
            return jsonify({"error": "SHAP worker failed", "trace": str(e)}), 500
    except Exception as e:
        tb = traceback.format_exc()
        print("SHAP error outer:", tb)
        return jsonify({"error": "SHAP error", "trace": str(e), "traceback": tb}), 500


@app.route("/api/history", methods=["GET", "POST"])
def api_history():
    if request.method == "GET":
        return jsonify(load_history())
    else:
        try:
            payload = request.get_json(force=True)
            entry = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "input": payload.get("input") if isinstance(payload, dict) else payload,
                "result": payload.get("result") if isinstance(payload, dict) else None,
            }
            save_history_entry(entry)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"error": str(e)}), 400


@app.route("/api/threshold", methods=["GET", "POST"])
def api_threshold():
    global THRESHOLD
    if request.method == "GET":
        return jsonify({"threshold": THRESHOLD})
    try:
        payload = request.get_json(force=True)
        t = float(payload.get("threshold"))
        THRESHOLD = float(t)
        try:
            with open(THRESH_JSON, "w", encoding="utf8") as f:
                json.dump({"threshold": THRESHOLD}, f)
        except Exception as e:
            safe_print("[api_threshold] save error:", e)
        return jsonify({"threshold": THRESHOLD})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/static/<path:p>")
def static_proxy(p):
    return send_from_directory("static", p)


# ==== startup ====
if __name__ == "__main__":
    load_artifacts()
    # sanity defaults
    if THRESHOLD is None:
        if isinstance(META, dict) and META.get("chosen_threshold") is not None:
            THRESHOLD = float(META.get("chosen_threshold"))
        else:
            THRESHOLD = 0.5
    safe_print(f"Starting app on http://0.0.0.0:5001 (threshold={THRESHOLD})")
    app.run(host="0.0.0.0", port=5001, debug=False)
