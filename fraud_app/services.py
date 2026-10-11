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
import math
import os
import select
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from . import schema
from .explain import explain_row
from .features import categorical_features, handles_missing, prepare_features, split_data

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

# How /api/shap computes an explanation (FRAUD_SHAP_MODE):
#   worker     a new Python process per request (process-isolated; pays start-up + loading)
#   pool       one long-lived worker process, started once (process-isolated, loads once)
#   inprocess  in the web process itself (fastest; no isolation from native crashes)
SHAP_MODES = ("worker", "pool", "inprocess")
SHAP_MODE = os.environ.get("FRAUD_SHAP_MODE", "worker")

# ==== globals ====

PIPE: Optional[Any] = None
META: Optional[Dict[str, Any]] = None
SHAP_EXPLAINER: Any = None  # kept for inline SHAP (optional)
THRESHOLD: Optional[float] = None


# ==== utils ====


def numpy_to_native(x: Any) -> Any:
    """Convert numpy scalars/arrays to Python native types for JSON (NaN/inf -> None)."""
    if isinstance(x, np.ndarray):
        return [numpy_to_native(v) for v in x.tolist()]
    if isinstance(x, np.generic):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def json_safe(obj: Any) -> Any:
    """Recursively make a structure valid JSON: NaN/inf become null (browsers reject NaN)."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return numpy_to_native(obj)


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
        hist.insert(0, json_safe(entry))
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
            return json_safe(json.load(f))  # entries written by older versions may hold NaN
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
    Columns the caller did not send are treated as missing (NaN / "NA"), which the model
    handles natively. See fraud_app.features for the exact typing rules.
    """
    return prepare_features(df, _get_feature_list_from_meta(), PIPE)


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


_DEMO_FRAME: Optional[pd.DataFrame] = None


_DATASET: Optional[pd.DataFrame] = None
_BACKGROUND: Optional[pd.DataFrame] = None


def _dataset() -> pd.DataFrame:
    """The prepared dataset, read once (the CSV is large)."""
    global _DATASET
    if _DATASET is None:
        if not DATASET_FILE.exists():
            raise RuntimeError(f"Dataset not found: {DATASET_FILE}")
        df = pd.read_csv(DATASET_FILE)
        if "is_fraud" not in df.columns:
            raise RuntimeError("Dataset has no 'is_fraud' column")
        _DATASET = df
    return _DATASET


def _demo_frame() -> pd.DataFrame:
    """
    Rows for the "Random normal / Random fraud" buttons: the chronological TEST split, i.e.
    transactions the model never saw.
    """
    global _DEMO_FRAME
    if _DEMO_FRAME is None:
        _, _, test = split_data(_dataset())
        _DEMO_FRAME = test.drop(columns=["TransactionDT"], errors="ignore").reset_index(drop=True)
    return _DEMO_FRAME


def background_frame(pipe: Any, n: int = 1000) -> pd.DataFrame:
    """
    Reference transactions for the explainers: a fixed sample of the rows whose labels are known
    before the test period (what the models were refit on), prepared for ``pipe``.
    """
    global _BACKGROUND
    if _BACKGROUND is None:
        from .features import labeled_rows

        meta_delay = float((META or {}).get("label_delay_days", 0.0)) if isinstance(META, dict) else 0.0
        lab = labeled_rows(_dataset(), meta_delay)
        _BACKGROUND = lab.sample(n=min(n, len(lab)), random_state=0).reset_index(drop=True)
    X, _ = prepare_features(_BACKGROUND, _get_feature_list_from_meta(), pipe)
    return X


def load_random_row_labeled(fraud: Optional[int]) -> Tuple[Dict[str, Any], Optional[int]]:
    """
    Sample a random held-out transaction; returns (row, true label).
    If fraud == 0/1 - filter by is_fraud, otherwise use all rows.
    Missing values are returned as None (JSON null).
    """
    df = _demo_frame()

    if fraud in (0, 1):
        subset = df[df["is_fraud"] == fraud]
        if subset.empty:
            subset = df
    else:
        subset = df

    row = subset.sample(n=1, random_state=None).iloc[0].to_dict()
    label = row.pop("is_fraud", None)
    out = {k: numpy_to_native(v) for k, v in row.items()}
    return out, (None if label is None else int(label))


def load_random_row(fraud: Optional[int]) -> Dict[str, Any]:
    """A random held-out transaction without its label (see load_random_row_labeled)."""
    return load_random_row_labeled(fraud)[0]


# ==== form schema for the UI ====


def _category_options() -> Dict[str, List[str]]:
    """
    Category values worth offering for each one-hot encoded column: the ones the fitted pipeline
    keeps as their own category. Rare values are pooled into one "infrequent" bucket by the
    encoder, so listing hundreds of them (device models, screen sizes) would only bloat the form.
    """
    if PIPE is None or not categorical_features(PIPE):
        return {}
    for name, trans, cols in PIPE.steps[0][1].transformers_:
        if name == "cat":
            ohe = trans.named_steps["ohe"]
            infrequent = getattr(ohe, "infrequent_categories_", None) or [None] * len(cols)
            options: Dict[str, List[str]] = {}
            for col, cats, rare in zip(cols, ohe.categories_, infrequent):
                rare_set = {str(v) for v in rare} if rare is not None else set()
                known = [str(v) for v in cats if str(v) != "NA"]  # "NA" == leave the field empty
                options[col] = [v for v in known if v not in rare_set] or known
            return options
    return {}


def build_schema() -> Dict[str, Any]:
    """Grouped description of the model's input features, used to build the UI form."""
    feats = _get_feature_list_from_meta()
    cats = set(categorical_features(PIPE)) if PIPE is not None else set()
    options = _category_options()

    groups: Dict[str, List[Dict[str, Any]]] = {}
    for name in feats:
        spec = schema.describe(name)
        item: Dict[str, Any] = {
            "name": name,
            "label": spec.label,
            "help": spec.help,
            "kind": schema.CATEGORICAL if name in cats else schema.NUMERIC,
        }
        if name in cats:
            item["options"] = [
                {"value": v, "label": spec.option_labels.get(v, v)} for v in options.get(name, [])
            ]
        groups.setdefault(spec.group, []).append(item)

    ordered = sorted(groups, key=schema.group_index)
    return {
        "n_features": len(feats),
        "groups": [{"name": g, "features": groups[g]} for g in ordered],
    }


# ==== SHAP inline (no worker, optional) ====


def compute_shap_for_df(df_safe: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute SHAP values in-process for a single row (df_safe already
    prepared by ensure_features_df).

    Not used by blueprint now (we use worker), but kept as a helper.
    """
    if SHAP_EXPLAINER is None or PIPE is None:
        raise RuntimeError("SHAP explainer or pipeline not loaded")
    model = getattr(SHAP_EXPLAINER, "model", None)
    if model is not None and not hasattr(model, "threshold_types"):  # LightGBM >= 4 (as in the worker)
        model.threshold_types = None
    return explain_row(PIPE, SHAP_EXPLAINER, df_safe)


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
            [sys.executable, "-m", "fraud_app.shap_worker"],
            cwd=str(ROOT_DIR),
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


# ==== SHAP via one long-lived worker process ====


class PersistentShapWorker:
    """
    Keeps one ``python -m fraud_app.shap_worker --serve`` process and talks to it line by line.
    A crash in the native code kills only the worker; the next request starts a new one.
    """

    def __init__(self, start_timeout: float = 120.0):
        self.proc: Optional[subprocess.Popen] = None
        self.lock = threading.Lock()
        self.start_timeout = start_timeout

    def _readline(self, timeout: float) -> str:
        assert self.proc is not None and self.proc.stdout is not None
        ready, _, _ = select.select([self.proc.stdout], [], [], timeout)
        if not ready:
            raise TimeoutError("SHAP worker did not answer in time")
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("SHAP worker exited")
        return line

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "fraud_app.shap_worker", "--serve"],
            cwd=str(ROOT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # worker logs go to the server log
            text=True,
            bufsize=1,
        )
        if not json.loads(self._readline(self.start_timeout)).get("ready"):
            raise RuntimeError("SHAP worker did not start")

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def explain(self, row: Dict[str, Any], timeout: float = 15.0) -> Dict[str, Any]:
        with self.lock:
            if not self.alive():
                self._start()
            assert self.proc is not None and self.proc.stdin is not None
            try:
                self.proc.stdin.write(json.dumps({"row": row}, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                data = json.loads(self._readline(timeout))
            except Exception:
                self.close()  # a hung or dead worker is replaced on the next request
                raise
        if "error" in data:
            raise RuntimeError(f"SHAP worker error: {data['error']}")
        return data

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
            finally:
                self.proc = None


SHAP_POOL = PersistentShapWorker()


def compute_shap(df_safe: pd.DataFrame, mode: Optional[str] = None) -> Dict[str, Any]:
    """SHAP for the first row of ``df_safe`` in the configured mode (see SHAP_MODE)."""
    mode = mode or SHAP_MODE
    if mode == "inprocess":
        return compute_shap_for_df(df_safe.iloc[[0]])
    if mode == "pool":
        row = {k: numpy_to_native(v) for k, v in df_safe.iloc[0].to_dict().items()}
        return SHAP_POOL.explain(row)
    return compute_shap_for_df_via_worker(df_safe)


# ==== helper to build history entry ====


def make_history_entry(raw_df: pd.DataFrame, result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ts": datetime.utcnow().isoformat(),
        "input": raw_df.to_dict(orient="records"),
        "result": result,
    }
