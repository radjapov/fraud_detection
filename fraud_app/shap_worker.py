#!/usr/bin/env python3
"""
Standalone SHAP worker. Run as a module from the project root:

    python -m fraud_app.shap_worker

Reads JSON from stdin:

  {"row": {"feature1": value1, ...}}

or

  {"rows": [ { ... }, ... ]}

Loads the pipeline and the SHAP explainer from artifacts/ and prints JSON to stdout:

  {
    "base_value": ...,
    "shap": [
      {"feature": "amount", "shap": 0.12, "value": 100.0},
      ...
    ]
  }

SHAP is computed on the preprocessed matrix and folded back to the input features
(see fraud_app.explain).

Все логи идут в stderr, чтобы stdout был чистым JSON.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from fraud_app.explain import explain_row
from fraud_app.features import prepare_features

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"

PIPELINE_FILE = ARTIFACTS_DIR / "pipeline_ieee.joblib"
SHAP_FILE = ARTIFACTS_DIR / "shap_explainer_ieee.joblib"


def log(*args):
    print("[worker]", *args, file=sys.stderr, flush=True)


def main():
    for path in (PIPELINE_FILE, SHAP_FILE):
        if not path.exists():
            log("artifact not found:", path)
            sys.exit(2)

    try:
        pipe = joblib.load(PIPELINE_FILE)
        explainer = joblib.load(SHAP_FILE)
        log("Loaded pipeline and SHAP explainer from", ARTIFACTS_DIR)

        # Monkey-patch fix for LightGBM >= 4.0 'threshold_types' error
        if hasattr(explainer, "model") and not hasattr(explainer.model, "threshold_types"):
            explainer.model.threshold_types = None
    except Exception as e:
        log("Failed to load artifacts:", e)
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

    try:
        X, _ = prepare_features(pd.DataFrame([row]), list(row.keys()), pipe)
        out = explain_row(pipe, explainer, X)
    except Exception as e:
        log("SHAP computation failed:", e)
        sys.exit(2)

    json.dump(out, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
