"""Label-delay split, the two-stage training protocol and the threshold search."""

import joblib
import numpy as np
import pandas as pd
import pytest

import choose_treshold_and_save as chooser
import train_ieee_lgbm as trainer
from fraud_app.features import (
    DEFAULT_LABEL_DELAY_DAYS,
    label_delay_of,
    labeled_rows,
    split_data,
)

DAY = 86400


def _timed_frame(n_days=100, per_day=10, seed=0):
    rng = np.random.default_rng(seed)
    n = n_days * per_day
    return pd.DataFrame(
        {
            "amount": rng.random(n) * 100,
            "is_fraud": (rng.random(n) < 0.1).astype(int),
            "TransactionDT": np.repeat(np.arange(n_days) * DAY, per_day) + rng.integers(0, DAY, n),
        }
    )


def test_label_delay_leaves_a_gap_before_the_test_period():
    df = _timed_frame()
    _, _, test0 = split_data(df, 0)
    train, val, test = split_data(df, 10)

    assert test.index.equals(test0.index)  # the test period does not depend on the delay
    test_start = test["TransactionDT"].min()
    # nothing the model learns from is younger than test_start - 10 days
    assert max(train["TransactionDT"].max(), val["TransactionDT"].max()) <= test_start - 10 * DAY
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert len(train) + len(val) < len(df) - len(test)  # the recent rows were really dropped


def test_validation_is_separated_from_training_by_the_same_gap():
    # a validation block glued to the training rows gives scores that are far too optimistic
    df = _timed_frame()
    train, val, _ = split_data(df, 10)
    assert val["TransactionDT"].min() - train["TransactionDT"].max() >= 10 * DAY
    train0, val0, _ = split_data(df, 0)
    assert val0["TransactionDT"].min() >= train0["TransactionDT"].max()  # contiguous without delay


def test_labeled_rows_cover_train_gap_and_validation():
    df = _timed_frame()
    train, val, test = split_data(df, 10)
    labeled = labeled_rows(df, 10)
    assert set(train.index) | set(val.index) <= set(labeled.index)
    assert len(labeled) > len(train) + len(val)  # the rows inside the train/val gap are labeled too
    assert not set(labeled.index) & set(test.index)
    assert labeled["TransactionDT"].max() <= test["TransactionDT"].min() - 10 * DAY


def test_no_delay_matches_the_plain_chronological_split():
    df = _timed_frame()
    train, val, test = split_data(df, 0)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    assert train["TransactionDT"].max() <= val["TransactionDT"].min()
    assert val["TransactionDT"].max() <= test["TransactionDT"].min()
    assert len(labeled_rows(df, 0)) == 800


def test_delay_is_ignored_without_a_time_column():
    df = _timed_frame().drop(columns="TransactionDT")
    a, b, c = split_data(df, 30)
    assert (len(a), len(b), len(c)) == (600, 200, 200)
    assert len(labeled_rows(df, 30)) == 800


def test_label_delay_of_meta():
    assert label_delay_of({"label_delay_days": 30}) == 30.0
    assert label_delay_of({}) == 0.0 and label_delay_of(None) == 0.0  # artifacts from before
    assert label_delay_of({"label_delay_days": "n/a"}) == 0.0


def test_default_delay_is_a_real_gap():
    assert DEFAULT_LABEL_DELAY_DAYS > 0


def test_scale_trees_follows_the_data_size():
    assert trainer.scale_trees(250, 200_000, 400_000) == 500
    assert trainer.scale_trees(250, 200_000, 200_000) == 250
    assert trainer.scale_trees(1, 200_000, 1_000) == 50  # never fewer than 50 trees


def test_best_threshold_maximises_f1():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    p = np.array([0.05, 0.1, 0.2, 0.55, 0.4, 0.6, 0.7, 0.9])
    thr, f1 = chooser.best_threshold(y, p)
    assert 0.2 < thr <= 0.4 and f1 == pytest.approx(8 / 9)  # flags all 4 frauds + one legit one


def _synthetic_training_frame():
    df = _timed_frame(n_days=120, per_day=40)
    df["mcc"] = np.random.default_rng(1).integers(0, 3, len(df))
    df["P_emaildomain"] = np.where(df["amount"] > 50, "gmail.com", None)
    df["C1"] = df["amount"] + np.random.default_rng(2).normal(size=len(df))
    df.loc[::7, "C1"] = np.nan
    df["is_fraud"] = (df["amount"] + 20 * np.random.default_rng(3).random(len(df)) > 105).astype(int)
    return df


def test_training_is_two_stage_and_records_the_protocol(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "SHAP_AVAILABLE", False)
    df = _synthetic_training_frame()

    _, meta_path, _ = trainer.build_pipeline_and_train(df, out_dir=tmp_path, label_delay_days=14)

    meta = joblib.load(meta_path)
    train, val, test = split_data(df, 14)
    assert meta["label_delay_days"] == 14.0
    assert (meta["n_train"], meta["n_val"], meta["n_test"]) == (len(train), len(val), len(test))
    assert meta["n_refit"] == len(labeled_rows(df, 14)) > meta["n_train"] + meta["n_val"]
    assert meta["n_estimators"] == trainer.scale_trees(
        meta["best_iteration_stage_a"], meta["n_train"], meta["n_refit"]
    )
    assert "TransactionDT" not in meta["features"] and "is_fraud" not in meta["features"]
    assert "P_emaildomain" in meta["categorical"] and "C1" in meta["numeric"]

    # the final model is the refit one: it has exactly the scaled number of trees
    pipe = joblib.load(tmp_path / "pipeline_ieee.joblib")
    assert pipe.steps[-1][1].booster_.num_trees() == meta["n_estimators"]
    assert pipe.predict_proba(df[meta["features"]].head(5))[:, 1].shape == (5,)  # NaN inputs are fine

    # the threshold is chosen from stage-A scores of the validation block (in-sample otherwise)
    scores = np.load(tmp_path / trainer.VAL_SCORES_FILE)
    assert len(scores["y"]) == len(scores["p"]) == len(val)
    assert (scores["y"] == val["is_fraud"].to_numpy()).all()
