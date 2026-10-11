"""
Shared pieces for the NTCC experiments: data loading, the chronological split with label delay
(the same one the service is trained with, see fraud_app.features), preprocessing and metrics.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from fraud_app.features import (
    DEFAULT_LABEL_DELAY_DAYS,
    TARGET,
    build_preprocessor,
    category_labels,
    labeled_rows,
    model_columns,
    split_data,
)
from fraud_app.schema import CATEGORICAL_NAMES

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "ieee_prepared.csv"


def load_prepared(path=DEFAULT_DATA, sample: float | None = None, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(path)
    if sample and 0 < sample < 1:
        df = df.sample(frac=sample, random_state=seed).reset_index(drop=True)
    return df


def feature_lists(df: pd.DataFrame):
    feats = [c for c in model_columns(df) if c != TARGET]
    cat = [c for c in feats if c in CATEGORICAL_NAMES]
    num = [c for c in feats if c not in cat]
    return num, cat


def _x(part: pd.DataFrame, num, cat) -> pd.DataFrame:
    X = part[num + cat].copy()
    for c in cat:
        X[c] = category_labels(X[c])
    return X


class Data:
    """Raw frames for train / validation / labeled (= refit) / test, plus fitted matrices."""

    def __init__(self, df: pd.DataFrame, label_delay_days: float = DEFAULT_LABEL_DELAY_DAYS):
        self.num, self.cat = feature_lists(df)
        tr, va, te = split_data(df, label_delay_days)
        lab = labeled_rows(df, label_delay_days)
        self.delay = label_delay_days
        self.X = {k: _x(p, self.num, self.cat) for k, p in
                  dict(train=tr, val=va, labeled=lab, test=te).items()}
        self.y = {k: p[TARGET].to_numpy(dtype=np.int8) for k, p in
                  dict(train=tr, val=va, labeled=lab, test=te).items()}
        self._cache: Dict[Tuple[str, bool], tuple] = {}

    def sizes(self) -> dict:
        return {k: dict(rows=int(len(v)), fraud_rate=round(float(v.mean()), 5)) for k, v in self.y.items()}

    def matrices(self, stage: str, impute: bool):
        """
        stage "A": preprocessor fit on train -> (Xtrain, Xval)
        stage "B": preprocessor fit on labeled rows -> (Xlabeled, Xtest, preprocessor)
        float32 to keep memory down; cached because every trial reuses them.
        """
        key = (stage, impute)
        if key not in self._cache:
            pre = build_preprocessor(self.num, self.cat, impute=impute)
            if stage == "A":
                a = pre.fit_transform(self.X["train"]).astype(np.float32)
                b = pre.transform(self.X["val"]).astype(np.float32)
            else:
                a = pre.fit_transform(self.X["labeled"]).astype(np.float32)
                b = pre.transform(self.X["test"]).astype(np.float32)
            self._cache[key] = (a, b, pre)
        return self._cache[key]

    def drop_stage(self, stage: str):
        for k in [k for k in self._cache if k[0] == stage]:
            del self._cache[k]


# ----------------------------------------------------------------------------- metrics


def best_f1_threshold(y, p) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]) if len(thr) else 0.5


def recall_at_fpr(y, p, max_fpr=0.01) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def scores(y, p, threshold: float) -> dict:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(
        roc_auc=float(roc_auc_score(y, p)),
        pr_auc=float(average_precision_score(y, p)),
        recall_at_1pct_fpr=recall_at_fpr(y, p, 0.01),
        threshold=float(threshold),
        precision=prec, recall=rec, f1=f1,
        tp=tp, fp=fp, fn=fn,
    )


def bootstrap_ci(y, p, n=1000, seed=0, metric="pr_auc", alpha=0.05):
    """Percentile CI of a ranking metric over test rows resampled with replacement."""
    rng = np.random.default_rng(seed)
    f = average_precision_score if metric == "pr_auc" else roc_auc_score
    m = len(y)
    vals = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        vals[i] = f(y[idx], p[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_bootstrap_delta(y, p_a, p_b, n=1000, seed=0, metric="pr_auc", alpha=0.05):
    """CI of metric(a) - metric(b) on the same resampled rows; excludes 0 => a real difference."""
    rng = np.random.default_rng(seed)
    f = average_precision_score if metric == "pr_auc" else roc_auc_score
    m = len(y)
    d = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, m, m)
        d[i] = f(y[idx], p_a[idx]) - f(y[idx], p_b[idx])
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(d)), float(lo), float(hi)


def single_row_latency_ms(predict_proba, X, n=300, seed=0) -> dict:
    """Wall time of predict_proba on one preprocessed row at a time (median / p95, ms)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(X), n)
    predict_proba(X[idx[:5]])  # warm-up
    t = []
    for i in idx:
        t0 = time.perf_counter()
        predict_proba(X[i:i + 1])
        t.append((time.perf_counter() - t0) * 1000)
    t = np.asarray(t)
    return dict(median_ms=float(np.median(t)), p95_ms=float(np.quantile(t, 0.95)))


def dump_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
