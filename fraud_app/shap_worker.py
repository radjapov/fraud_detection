#!/usr/bin/env python3
"""
Standalone SHAP worker.

Reads JSON from stdin:

  {"row": {"feature1": value1, ...}}

or

  {"rows": [ { ... }, ... ]}

Loads SHAP explainer from artifacts/shap_explainer_ieee.joblib
and prints JSON to stdout:

  {
    "base_value": ...,
    "shap": [
      {"feature": "amount", "shap": 0.12, "value": 100.0},
      ...
    ]
  }

Все логи идут в stderr, чтобы stdout был чистым JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

SHAP_FILE = ARTIFACTS_DIR / "shap_explainer_ieee.joblib"
SHAP_INPUT_JSON = ARTIFACTS_DIR / "shap_input.json"


def log(*args):
    print("[worker]", *args, file=sys.stderr, flush=True)


def numpy_to_native(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    if not SHAP_FILE.exists():
        log("SHAP explainer not found:", SHAP_FILE)
        sys.exit(2)

    try:
        explainer = joblib.load(SHAP_FILE)
        log("Loaded SHAP explainer from", SHAP_FILE)
    except Exception as e:
        log("Failed to load SHAP explainer:", e)
        sys.exit(2)

    # читаем payload из stdin
    try:
        payload = json.load(sys.stdin)
    except Exception as e:
        log("Failed to read JSON from stdin:", e)
        sys.exit(2)

    row = payload.get("row")
    if row is None:
        rows = payload.get("rows") or []
        row = rows[0] if rows else None

    if row is None:
        log("No 'row' in payload")
        sys.exit(2)

    # порядок фичей сохраняем такой же, как в df.columns
    feat_names = list(row.keys())
    values = [float(row[f]) for f in feat_names]
    X = np.array([values], dtype=float)

    try:
        vals = explainer(X, check_additivity=False)
    except Exception as e:
        log("Explainer call failed:", e)
        sys.exit(2)

    base_value = None
    if hasattr(vals, "base_values"):
        base_value = vals.base_values
    elif hasattr(vals, "expected_value"):
        base_value = vals.expected_value

    if hasattr(vals, "values"):
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
        log(f"Unexpected SHAP values shape: {shap_values.shape}")
        sys.exit(2)

    # собираем ответ
    if isinstance(base_value, np.ndarray):
        base_native = numpy_to_native(base_value[0])
    else:
        base_native = numpy_to_native(base_value)

    out = {
        "base_value": base_native,
        "shap": [],
    }

    for i, feat in enumerate(feat_names):
        s_val = float(sv[i]) if i < len(sv) else None
        out["shap"].append(
            {
                "feature": feat,
                "shap": s_val,
                "value": numpy_to_native(row[feat]),
            }
        )

    json.dump(out, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
