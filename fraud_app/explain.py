"""
fraud_app.explain

SHAP for a single row, computed in the space the LightGBM model was trained in
(after the pipeline's preprocessor) and mapped back to the 12 input features.

The explainer is a TreeExplainer over the classifier only, so it expects the
preprocessed matrix (scaled numerics + one-hot). One-hot columns of the same
source feature are summed: SHAP values are additive, so the sum is that
feature's contribution.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def _to_native(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def _positive_class_values(vals: Any) -> np.ndarray:
    arr = np.asarray(vals.values if hasattr(vals, "values") else vals)
    if arr.ndim == 3:  # (n_samples, n_features, n_classes) or (n_classes, n_samples, n_features)
        raise RuntimeError(f"Unexpected multi-class SHAP shape: {arr.shape}")
    if arr.ndim == 2:
        return arr[0]
    if arr.ndim == 1:
        return arr
    raise RuntimeError(f"Unexpected SHAP values shape: {arr.shape}")


def explain_row(pipe: Any, explainer: Any, X_prepared: pd.DataFrame) -> Dict[str, Any]:
    """
    ``X_prepared`` comes from features.prepare_features (1 row).
    Returns {"base_value": float, "shap": [{"feature", "shap", "value"}, ...]}.
    """
    preproc, _clf = pipe.steps[0][1], pipe.steps[-1][1]
    T = preproc.transform(X_prepared.iloc[[0]])
    T = np.asarray(T, dtype=float)

    vals = explainer(T, check_additivity=False)
    sv = _positive_class_values(vals)

    out_names = list(preproc.get_feature_names_out())
    if len(out_names) != len(sv):
        raise RuntimeError(
            f"SHAP explainer has {len(sv)} features but the pipeline produces "
            f"{len(out_names)}; rebuild shap_explainer_ieee.joblib with train_ieee_lgbm.py"
        )

    src_features = list(X_prepared.columns)
    shap_by_feature = {f: 0.0 for f in src_features}
    for name, s in zip(out_names, sv):
        # "num__amount" / "cat__mcc_4": strip the branch prefix, then match the
        # longest source feature that the remainder equals or starts with "<feat>_".
        rest = name.split("__", 1)[-1]
        owner = max(
            (f for f in src_features if rest == f or rest.startswith(f + "_")),
            key=len,
            default=None,
        )
        if owner is None:
            raise RuntimeError(f"Cannot map SHAP column {name!r} to an input feature")
        shap_by_feature[owner] += float(s)

    base = getattr(vals, "base_values", None)
    if base is None:
        base = getattr(explainer, "expected_value", None)
    base = np.asarray(base).ravel()
    base_value = float(base[0]) if base.size else None

    row = X_prepared.iloc[0]

    def _display_value(v: Any) -> Any:
        # categorical columns are strings for the encoder; show them as numbers when they are
        # numeric codes, and show missing values (NaN / "NA") as null
        if v is None or (isinstance(v, str) and v == "NA"):
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return _to_native(v)
        return None if np.isnan(f) else f

    shap_list: List[Dict[str, Any]] = [
        {"feature": f, "shap": shap_by_feature[f], "value": _display_value(row[f])}
        for f in src_features
    ]
    return {"base_value": base_value, "shap": shap_list}
