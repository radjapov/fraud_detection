#!/usr/bin/env python3
"""
shap_worker.py

Читает JSON из stdin:
{
  "row": { ... фичи ... },
  "artifacts_dir": "/путь/к/artifacts"   # опционально
}

Считает SHAP по сохранённому explainer'у и печатает JSON в stdout:
{
  "base_value": ...,
  "shap": [
    {"feature": "amount", "shap": 0.01, "value": 123.45},
    ...
  ]
}
"""

import json
import sys
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def numpy_to_native(x):
    if isinstance(x, (np.floating, np.float32, np.float64)):
        return float(x)
    if isinstance(x, (np.integer, np.int32, np.int64)):
        return int(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print("No input on stdin", file=sys.stderr)
            sys.exit(1)

        payload = json.loads(raw)
        row = payload.get("row")
        if row is None:
            print("No 'row' in payload", file=sys.stderr)
            sys.exit(1)

        artifacts_dir = payload.get("artifacts_dir")
        if artifacts_dir is None:
            # корень проекта = два уровня вверх от этого файла
            artifacts_dir = Path(__file__).resolve().parent / "artifacts"
        else:
            artifacts_dir = Path(artifacts_dir)

        print(f"[worker] artifacts_dir={artifacts_dir}", file=sys.stderr)

        shap_path = artifacts_dir / "shap_explainer_ieee.joblib"
        meta_path = artifacts_dir / "meta_ieee.joblib"

        if not shap_path.exists():
            print(f"[worker] SHAP explainer not found: {shap_path}", file=sys.stderr)
            sys.exit(2)

        # грузим explainer
        explainer = joblib.load(shap_path)

        # делаем DataFrame из одной строки
        df = pd.DataFrame([row])

        # просто даём explainer'у сырые фичи - он был обучен на тех же колонках
        try:
            vals = explainer(df)
        except Exception:
            vals = explainer(df.values)

        base_value = None
        shap_values = None

        if hasattr(vals, "base_values"):
            base_value = vals.base_values
        elif hasattr(vals, "expected_value"):
            base_value = vals.expected_value

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
            raise RuntimeError(f"Unexpected shap_values shape: {shap_values.shape}")

        features = list(df.columns)
        out = {
            "base_value": numpy_to_native(base_value),
            "shap": [],
        }

        for i, feat in enumerate(features):
            try:
                v = df.iloc[0, i]
            except Exception:
                v = None
            v = numpy_to_native(v)

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

        sys.stdout.write(json.dumps(out))
        sys.stdout.flush()
    except Exception as e:
        print("WORKER ERROR:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(2)


if __name__ == "__main__":
    main()