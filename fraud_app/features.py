"""
fraud_app.features

Single place that turns raw input rows into what the trained pipeline expects, and that
builds the preprocessing itself, so training, serving and evaluation cannot drift apart.

Rules (each was a real bug once):
  * categorical columns are STRINGS ("4", "gmail.com", "NA"), produced by ``category_labels``
    both when training and when serving. The OneHotEncoder learns those exact strings, so
    ``4.0`` or ``4`` would otherwise be an unseen category and silently encode as zeros.
  * missing numeric values stay NaN (LightGBM handles them natively) instead of being
    turned into fake zeros; missing categories become the explicit label "NA".
"""

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "is_fraud"
TIME_COL = "TransactionDT"  # kept in the prepared CSV for splitting; never a model feature
SPLIT_SEED = 42
MISSING_LABEL = "NA"

# ---------------------------------------------------------------- pipeline introspection


def _preproc(pipe: Any) -> Any:
    steps = getattr(pipe, "steps", None)
    return steps[0][1] if steps else None


def categorical_features(pipe: Any) -> List[str]:
    """Columns the pipeline one-hot encodes ('cat' branch of its ColumnTransformer)."""
    preproc = _preproc(pipe)
    if not isinstance(preproc, ColumnTransformer):
        return []
    for name, _, cols in preproc.transformers:
        if name == "cat":
            return list(cols)
    return []


def handles_missing(pipe: Any) -> bool:
    """True for pipelines built by ``build_preprocessor`` (they are NaN-safe end to end)."""
    return isinstance(_preproc(pipe), ColumnTransformer)


# ---------------------------------------------------------------- category labels


def _label_one(v: Any) -> str:
    if v is None or v is pd.NA:
        return MISSING_LABEL
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return MISSING_LABEL
        return str(int(v)) if float(v).is_integer() else str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    text = str(v).strip()
    if text == "" or text.lower() in ("nan", "none", "null"):
        return MISSING_LABEL
    try:  # numeric-looking strings ("4", "4.0") share the label of the number
        f = float(text)
        return str(int(f)) if f.is_integer() else text
    except ValueError:
        return text


def category_labels(s: pd.Series) -> pd.Series:
    """Map values to the string labels the OneHotEncoder sees (NaN -> "NA", 4.0 -> "4")."""
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        num = pd.to_numeric(s, errors="coerce").astype("float64")
        out = pd.Series(MISSING_LABEL, index=s.index, dtype=object)
        ok = num.notna()
        whole = ok & (num % 1 == 0)
        out[whole] = num[whole].astype("int64").astype(str)
        frac = ok & ~whole
        out[frac] = num[frac].astype(str)
        return out
    return s.map(_label_one)


# ---------------------------------------------------------------- serving / eval input


def prepare_features(
    df: pd.DataFrame, feat_list: Sequence[str], pipe: Any
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return a copy of ``df`` restricted to ``feat_list`` in that order:
      numeric      -> float64, NaN kept (0 only for pipelines that cannot take NaN)
      categorical  -> string labels (see ``category_labels``)
    Columns absent from ``df`` are treated as missing.
    """
    feat_list = list(feat_list)
    cats = set(categorical_features(pipe))
    keep_nan = handles_missing(pipe)

    X = pd.DataFrame(index=df.index)
    for f in feat_list:
        col = df[f] if f in df.columns else pd.Series(np.nan, index=df.index, dtype=object)
        if f in cats:
            X[f] = category_labels(col)
        else:
            num = pd.to_numeric(col, errors="coerce").astype(np.float64)
            X[f] = num if keep_nan else num.fillna(0.0)
    return X, feat_list


# ---------------------------------------------------------------- preprocessing

# Categories rarer than this in the training data share one "infrequent" bucket, so a domain
# seen 3 times cannot become a fingerprint, and unseen domains at serving time land there too.
MIN_CATEGORY_FREQUENCY = 500


def make_ohe(min_frequency: int = MIN_CATEGORY_FREQUENCY) -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=min_frequency, sparse_output=False
        )
    except TypeError:  # old scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], impute: bool = False
) -> ColumnTransformer:
    """
    ``impute=False`` (LightGBM / trees): NaN passes through untouched, keeping "missing" as
    a signal. ``impute=True`` (linear models, random forest): median imputation.
    """
    num_steps: list = []
    if impute:
        num_steps.append(("imp", SimpleImputer(strategy="median", keep_empty_features=True)))
    num_steps.append(("sc", StandardScaler()))

    cat_pipe = Pipeline(
        [
            ("imp", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL)),
            ("ohe", make_ohe()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), list(numeric)),
            ("cat", cat_pipe, list(categorical)),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


# ---------------------------------------------------------------- splitting


def model_columns(df: pd.DataFrame) -> List[str]:
    """Feature columns of a prepared dataset (everything except label and time)."""
    return [c for c in df.columns if c not in (TARGET, TIME_COL)]


SECONDS_PER_DAY = 86400

# Chargebacks are reported weeks after a transaction (IEEE-CIS only calls a transaction legit
# after 120 quiet days), so a model deployed at time T knows the label of a transaction only
# once it is old enough. Scripts train and evaluate with this assumption unless told otherwise.
DEFAULT_LABEL_DELAY_DAYS = 30


def label_delay_of(meta: Any) -> float:
    """Label delay a model was trained with (0 for artifacts that predate the setting)."""
    try:
        return float((meta or {}).get("label_delay_days", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _labeled_and_test(
    df: pd.DataFrame, label_delay_days: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Newest 20% of rows = test; the older rows whose label is known when the test starts."""
    ordered = df.sort_values(TIME_COL, kind="stable")
    j = int(len(ordered) * 0.8)
    older, test = ordered.iloc[:j], ordered.iloc[j:]
    if label_delay_days > 0 and len(test):
        cutoff = test[TIME_COL].iloc[0] - label_delay_days * SECONDS_PER_DAY
        older = older[older[TIME_COL] <= cutoff]
    return older, test


def split_data(
    df: pd.DataFrame, label_delay_days: float = 0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Train / validation / test split used by training and by every evaluation script.

    With a ``TransactionDT`` column the split is chronological: the newest 20% of rows is the
    test set ("the future"). Everything the model may learn from is the older data; its newest
    25% is the validation block (early stopping, threshold, honest estimate), the rest trains.

    ``label_delay_days`` models late labels. At deployment time T (the start of the test period)
    a label is only known once the transaction is that many days old, so learning data is limited
    to ``TransactionDT <= T - delay``. The validation block is then separated from the training
    rows by the same gap: it has to look like the test set does to a model trained before it,
    otherwise validation scores are far too optimistic (the model has just seen the same
    cards, in the same week) and the threshold picked on them does not transfer.

    Without a time column (e.g. the dummy dataset) it falls back to a stratified random split.
    """
    if TIME_COL in df.columns:
        older, test = _labeled_and_test(df, label_delay_days)
        i = int(len(older) * 0.75)
        val, train = older.iloc[i:], older.iloc[:i]
        if label_delay_days > 0 and len(val):
            val_start = val[TIME_COL].iloc[0]
            train = older[older[TIME_COL] <= val_start - label_delay_days * SECONDS_PER_DAY]
        return train, val, test

    y = df[TARGET].astype(int)
    temp, test = train_test_split(df, test_size=0.2, random_state=SPLIT_SEED, stratify=y)
    train, val = train_test_split(
        temp, test_size=0.25, random_state=SPLIT_SEED, stratify=temp[TARGET].astype(int)
    )
    return train, val, test


def labeled_rows(df: pd.DataFrame, label_delay_days: float = 0) -> pd.DataFrame:
    """
    Every row whose label is known at deployment time (train + the gap + validation): the data
    the final model is refit on once early stopping and the threshold have been settled.
    """
    if TIME_COL in df.columns:
        return _labeled_and_test(df, label_delay_days)[0]
    train, val, _ = split_data(df)
    return pd.concat([train, val])
