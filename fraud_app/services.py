"""
fraud_app.services

All logic for:
  - loading artifacts (pipeline, meta, SHAP, thresholds)
  - feature handling
  - prediction
  - SHAP explainability (via worker or inline)
  - history storage
  - metrics loading

Used by fraud_app.api.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# ==== paths ====

# project root: .../fraud_detection
ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
DATA_DIR = ROOT_DIR / "data"
METRICS_JSON = ARTIFACTS_DIR / "metrics_ieee.json"

PIPELINE_FILE = ARTIFACTS_DIR / "pipeline_ieee.joblib"
META_FILE = ARTIFACTS_DIR / "meta_ieee.joblib"
SHAP_FILE = ARTIFACTS_DIR / "shap_explainer_ieee.joblib"
THRESH_JSON = ARTIFACTS_DIR / "meta_ieee_threshold.json"
HISTORY_FILE = ARTIFACTS_DIR / "history.json"
DATASET_FILE = DATA_DIR / "ieee_prepared.csv"

# where SHAP worker script lives
SHAP_WORKER_FILE = ROOT_DIR / "fraud_app" / "shap_worker.py"
SHAP_INPUT_JSON = ARTIFACTS_DIR / "shap_input.json"

# ==== globals ====

PIPE: Optional[Any] = None
META: Optional[Dict[str, Any]] = None
SHAP_EXPLAINER: Any = None  # kept for inline SHAP (optional)
THRESHOLD: Optional[float] = None


# ==== utils ====


def numpy_to_native(x: Any) -> Any:
    """Convert numpy scalars/arrays to Python native types for JSON."""
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


# ==== artifacts loading ====


def load_artifacts() -> Dict[str, bool]:
    """
    Load pipeline, meta, SHAP explainer and threshold.
    Called once on app startup.
    """
    global PIPE, META, SHAP_EXPLAINER, THRESHOLD

    summary = {"pipeline": False, "meta": False, "shap": False, "threshold": False}

    # pipeline
    if PIPELINE_FILE.exists():
        try:
            PIPE = joblib.load(PIPELINE_FILE)
            print(f"[services] Loaded pipeline: {PIPELINE_FILE} type={type(PIPE)}")
            summary["pipeline"] = True
        except Exception as e:
            print("[services] Failed to load pipeline:", e)
            PIPE = None
    else:
        print("[services] Pipeline file not found:", PIPELINE_FILE)

    # meta
    if META_FILE.exists():
        try:
            META = joblib.load(META_FILE)
            print(f"[services] Loaded meta: {META_FILE}")
            summary["meta"] = True
        except Exception as e:
            print("[services] Failed to load meta:", e)
            META = None
    else:
        print("[services] Meta file not found:", META_FILE)

    # threshold json
    if THRESH_JSON.exists():
        try:
            with open(THRESH_JSON, "r", encoding="utf8") as f:
                jd = json.load(f)
            t = jd.get("threshold", jd.get("chosen_threshold"))
            if t is not None:
                THRESHOLD = float(t)
                summary["threshold"] = True
                print(f"[services] Loaded threshold from json: {THRESHOLD}")
        except Exception as e:
            print("[services] Failed to load threshold json:", e)

    # SHAP explainer (for inline mode, not used by worker)
    if SHAP_FILE.exists():
        try:
            SHAP_EXPLAINER = joblib.load(SHAP_FILE)
            print(f"[services] Loaded SHAP explainer: {SHAP_FILE}")
            summary["shap"] = True
        except Exception as e:
            print("[services] Failed to load SHAP explainer:", e)
            SHAP_EXPLAINER = None
    else:
        print("[services] SHAP file not found:", SHAP_FILE)

    # fallback threshold
    if THRESHOLD is None:
        if isinstance(META, dict) and META.get("chosen_threshold") is not None:
            THRESHOLD = float(META["chosen_threshold"])
        else:
            THRESHOLD = 0.5
        print(f"[services] Using fallback threshold={THRESHOLD}")

    print("[services] Load summary:", summary)
    return summary


# ==== metrics ====


def load_metrics() -> Optional[Dict[str, Any]]:
    """Load precomputed model metrics from artifacts/metrics_ieee.json."""
    if not METRICS_JSON.exists():
        print(f"[metrics] File not found: {METRICS_JSON}")
        return None
    try:
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("[metrics] Loaded metrics from", METRICS_JSON)
        return data
    except Exception as e:
        print("[metrics] Failed to load metrics:", e)
        return None


# ==== threshold ====


def get_threshold() -> float:
    global THRESHOLD
    if THRESHOLD is None:
        THRESHOLD = 0.5
    return float(THRESHOLD)


def set_threshold(value: float) -> float:
    """Save threshold both in memory and in JSON file."""
    global THRESHOLD
    THRESHOLD = float(value)
    THRESH_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESH_JSON, "w", encoding="utf8") as f:
        json.dump({"threshold": THRESHOLD}, f)
    print(f"[services] Threshold updated to {THRESHOLD}")
    return THRESHOLD


# ==== history ====


def save_history_entry(entry: Dict[str, Any]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        hist: List[Dict[str, Any]] = []
        if HISTORY_FILE.exists():
            with open(HISTORY_FILE, "r", encoding="utf8") as f:
                hist = json.load(f)
        hist.insert(0, entry)
        hist = hist[:200]
        with open(HISTORY_FILE, "w", encoding="utf8") as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[services] save_history_entry failed:", e)


def load_history() -> List[Dict[str, Any]]:
    """Return stored history entries (last N predictions)."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        print("[services] load_history failed:", e)
        return []


# Backward compat alias (if somewhere you still use get_history)
def get_history() -> List[Dict[str, Any]]:
    return load_history()


# ==== features & pipeline ====


def _get_feature_list_from_meta() -> List[str]:
    """Get feature list from META or from pipeline.feature_names_in_."""
    global META, PIPE

    if isinstance(META, dict):
        for key in ("features", "feat_names", "feat_names_order"):
            if key in META and META[key] is not None:
                return list(META[key])

    if PIPE is not None and hasattr(PIPE, "feature_names_in_"):
        return list(PIPE.feature_names_in_)

    raise RuntimeError("META not loaded and pipeline has no feature_names_in_")


def ensure_features_df(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Bring arbitrary input df to the model's expected feature set and order.
    Missing columns are added with zeros; everything is cast to float64.
    """
    feat_list = _get_feature_list_from_meta()

    missing = [f for f in feat_list if f not in df.columns]
    if missing:
        print("[services.ensure_features_df] Adding missing features:", missing)
        for m in missing:
            df[m] = 0

    X_safe = df[feat_list].copy()

    for c in X_safe.columns:
        try:
            X_safe[c] = pd.to_numeric(X_safe[c], errors="coerce").astype(np.float64)
        except Exception:
            X_safe[c] = pd.to_numeric(X_safe[c].fillna(0), errors="coerce").astype(np.float64)

    X_safe = X_safe.fillna(0.0)
    return X_safe, feat_list


def compute_probs_from_pipeline(X_safe: pd.DataFrame) -> List[float]:
    """
    Compute fraud probabilities via the loaded pipeline (class 1).
    """
    global PIPE
    if PIPE is None:
        raise RuntimeError("Pipeline not loaded")

    if hasattr(PIPE, "predict_proba"):
        probs = PIPE.predict_proba(X_safe)[:, 1]
    elif isinstance(PIPE, dict) and "model" in PIPE and hasattr(PIPE["model"], "predict_proba"):
        probs = PIPE["model"].predict_proba(X_safe)[:, 1]
    else:
        raise RuntimeError("No usable predict_proba in pipeline")

    return [float(p) for p in np.asarray(probs)]


# backward compat alias (if some scripts still call compute_probs)
def compute_probs(X_safe: pd.DataFrame) -> List[float]:
    return compute_probs_from_pipeline(X_safe)


# ==== random row from dataset ====


def load_random_row(fraud: Optional[int]) -> Dict[str, Any]:
    """
    Sample a random row from data/ieee_prepared.csv.
    If fraud == 0/1 — filter by is_fraud, otherwise use full dataset.
    """
    if not DATASET_FILE.exists():
        raise RuntimeError(f"Dataset not found: {DATASET_FILE}")

    df = pd.read_csv(DATASET_FILE)
    if "is_fraud" not in df.columns:
        raise RuntimeError("Dataset has no 'is_fraud' column")

    if fraud in (0, 1):
        subset = df[df["is_fraud"] == fraud]
        if subset.empty:
            subset = df
    else:
        subset = df

    row = subset.sample(n=1, random_state=None).iloc[0].to_dict()
    row.pop("is_fraud", None)

    for k, v in list(row.items()):
        row[k] = numpy_to_native(v)
    return row


# ==== SHAP inline (no worker, optional) ====


def compute_shap_for_df(df_safe: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute SHAP values in-process for a single row (df_safe already
    prepared by ensure_features_df). Uses check_additivity=False to
    avoid fragile internal SHAP checks.

    Not used by blueprint now (we use worker), but kept as a helper.
    """
    global SHAP_EXPLAINER
    if SHAP_EXPLAINER is None:
        raise RuntimeError("SHAP explainer not loaded")

    vals = SHAP_EXPLAINER(df_safe.values, check_additivity=False)

    base_value = None
    if hasattr(vals, "base_values"):
        base_value = vals.base_values
    elif hasattr(vals, "expected_value"):
        base_value = vals.expected_value

    # normalize to numpy array
    if isinstance(vals, (list, tuple)):
        shap_values = np.array(vals[0]) if len(vals) == 1 else np.array(vals)
    elif hasattr(vals, "values"):
        shap_values = np.asarray(vals.values)
    else:
        shap_values = np.asarray(vals)

    if shap_values.ndim == 3:
        # (n_classes, n_samples, n_features)
        if shap_values.shape[0] >= 2:
            sv = shap_values[1][0]
        else:
            sv = shap_values[0][0]
    elif shap_values.ndim == 2:
        sv = shap_values[0]
    elif shap_values.ndim == 1:
        sv = shap_values
    else:
        raise RuntimeError(f"Unexpected SHAP values shape: {shap_values.shape}")

    feat_names = list(df_safe.columns)
    out: Dict[str, Any] = {"base_value": numpy_to_native(base_value), "shap": []}

    for i, feat in enumerate(feat_names):
        try:
            v = df_safe.iloc[0, i]
            v = numpy_to_native(v)
        except Exception:
            v = None

        s_val = sv[i] if i < len(sv) else None
        try:
            s_val = float(s_val)
        except Exception:
            s_val = None

        out["shap"].append(
            {
                "feature": feat,
                "shap": s_val,
                "value": v,
            }
        )

    return out


# ==== SHAP via worker process ====


def compute_shap_for_df_via_worker(df_safe: pd.DataFrame, timeout: int = 15) -> Dict[str, Any]:
    """
    Compute SHAP using a separate worker process (shap_worker.py),
    to avoid crashes inside the main Flask process.

    Contract with shap_worker.py:
      - stdin: JSON {"row": {feature: value, ...}}
      - stdout: pure JSON: {"base_value": ..., "shap": [...]}
      - logs печатаются в stderr.
    """
    if not SHAP_WORKER_FILE.exists():
        raise RuntimeError(f"SHAP worker script not found: {SHAP_WORKER_FILE}")

    # берём первую строку
    row_dict = df_safe.iloc[0].to_dict()
    payload = {"row": {k: numpy_to_native(v) for k, v in row_dict.items()}}

    try:
        proc = subprocess.Popen(
            [sys.executable, str(SHAP_WORKER_FILE)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        out, err = proc.communicate(
            input=json.dumps(payload, ensure_ascii=False),
            timeout=timeout,
        )

        if proc.returncode != 0:
            raise RuntimeError(f"SHAP worker failed (code {proc.returncode}): {err}")

        if not out.strip():
            raise RuntimeError(f"SHAP worker produced no output. Stderr: {err}")

        # здесь stdout должен быть чистым JSON
        data = json.loads(out)
        return data

    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("SHAP worker timeout")
    except Exception as e:
        raise RuntimeError(f"SHAP worker error: {e}")


# ==== helper to build history entry ====


def make_history_entry(raw_df: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": datetime.utcnow().isoformat(),
        "input": raw_df.to_dict(orient="records"),
        "result": result,
    }
