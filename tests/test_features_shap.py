"""Regression tests for the serving path: categorical labels, missing values, SHAP mapping."""

import numpy as np
import pandas as pd
import pytest

from fraud_app import services
from fraud_app.features import (
    categorical_features,
    category_labels,
    model_columns,
    prepare_features,
    split_data,
)
from fraud_app.schema import CATEGORICAL_NAMES

needs_model = pytest.mark.skipif(
    not (services.PIPELINE_FILE.exists() and services.DATASET_FILE.exists()),
    reason="needs artifacts/ and data/",
)


@pytest.fixture(scope="module")
def loaded():
    services.load_artifacts()
    if not categorical_features(services.PIPE):
        pytest.skip("dummy pipeline: no one-hot categorical branch")
    return services.PIPE


def _sample_rows(n=300):
    df = pd.read_csv(services.DATASET_FILE, nrows=30000)
    return df.sample(n, random_state=0).drop(columns=["is_fraud", "TransactionDT"], errors="ignore")


def test_category_labels_agree_across_dtypes():
    ints = category_labels(pd.Series([4, 0, 2]))
    floats = category_labels(pd.Series([4.0, 0.0, 2.0]))
    objects = category_labels(pd.Series(["4", "0.0", 2.0], dtype=object))
    assert ints.tolist() == floats.tolist() == objects.tolist() == ["4", "0", "2"]  # never "4.0"


def test_category_labels_missing_and_text():
    s = pd.Series(["gmail.com", None, np.nan, "  ", "NaN", " visa "], dtype=object)
    assert category_labels(s).tolist() == ["gmail.com", "NA", "NA", "NA", "NA", "visa"]
    assert category_labels(pd.Series([1.0, np.nan])).tolist() == ["1", "NA"]


@needs_model
def test_prepare_features_keeps_missing_as_missing(loaded):
    feats = ["amount", "D1", "mcc", "P_emaildomain"]
    X, _ = prepare_features(pd.DataFrame([{"amount": 10, "mcc": 4.0}]), feats, loaded)
    assert X.loc[0, "mcc"] == "4"  # not "4.0": the encoder only knows '0'..'4'
    assert X.loc[0, "P_emaildomain"] == "NA"  # absent categorical -> explicit "NA"
    assert np.isnan(X.loc[0, "D1"])  # absent numeric stays NaN, not a fake 0
    assert X["amount"].dtype == np.float64


@needs_model
def test_service_matches_training_input_path(loaded):
    """JSON-style input (floats) must score exactly like the CSV-typed training input."""
    raw = _sample_rows()
    reference = raw.copy()  # native dtypes: ints for the numeric-coded categoricals
    for c in categorical_features(loaded):
        # deliberately NOT category_labels(): this is how training encoded categories, so the
        # test stays independent of the function under test
        reference[c] = reference[c].fillna("NA").astype(str)

    served = raw.copy()  # what a JSON client sends: 4.0 instead of 4
    for c in ("mcc", "country_risk", "is_foreign"):
        served[c] = served[c].astype(float)
    X, _ = services.ensure_features_df(served)

    np.testing.assert_allclose(
        loaded.predict_proba(X)[:, 1], loaded.predict_proba(reference)[:, 1]
    )


@needs_model
def test_shap_sums_to_model_logit_with_missing_values(loaded):
    if services.SHAP_EXPLAINER is None:
        pytest.skip("no SHAP explainer artifact")
    rows = _sample_rows(400)
    raw = rows.loc[[rows.isna().sum(axis=1).idxmax()]]  # the row with the most missing fields
    assert raw.isna().sum(axis=1).iloc[0] > 5
    X, feats = services.ensure_features_df(raw.copy())

    out = services.compute_shap_for_df(X)

    assert [i["feature"] for i in out["shap"]] == feats  # source features, not one-hot columns
    reconstructed = out["base_value"] + sum(i["shap"] for i in out["shap"])
    clf = loaded.steps[-1][1]
    expected = clf.predict(loaded.steps[0][1].transform(X), raw_score=True)[0]
    assert reconstructed == pytest.approx(expected, abs=1e-4)


@needs_model
def test_shap_worker_matches_inline(loaded):
    if services.SHAP_EXPLAINER is None:
        pytest.skip("no SHAP explainer artifact")
    rows = _sample_rows(400)
    X, _ = services.ensure_features_df(rows.loc[[rows.isna().sum(axis=1).idxmax()]])
    inline = services.compute_shap_for_df(X)
    worker = services.compute_shap_for_df_via_worker(X, timeout=120)
    np.testing.assert_allclose(
        [i["shap"] for i in inline["shap"]], [i["shap"] for i in worker["shap"]], atol=1e-8
    )
    # missing inputs come back as null, not NaN (NaN is not valid JSON)
    assert any(i["value"] is None for i in worker["shap"])


@needs_model
def test_schema_matches_the_model(loaded):
    schema = services.build_schema()
    listed = [f["name"] for g in schema["groups"] for f in g["features"]]
    assert sorted(listed) == sorted(services._get_feature_list_from_meta())
    assert schema["n_features"] == len(listed)

    cats = set(categorical_features(loaded))
    for g in schema["groups"]:
        for f in g["features"]:
            assert (f["kind"] == "categorical") == (f["name"] in cats)
            if f["kind"] == "categorical":
                values = [o["value"] for o in f["options"]]
                assert values and "NA" not in values  # "NA" is the empty selection


def test_split_is_chronological_when_time_column_present():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.random(1000),
            "is_fraud": (rng.random(1000) < 0.1).astype(int),
            "TransactionDT": rng.permutation(1000),  # shuffled on purpose
        }
    )
    train, val, test = split_data(df)
    assert (len(train), len(val), len(test)) == (600, 200, 200)
    # strictly in the past -> present -> future, no overlap
    assert train["TransactionDT"].max() < val["TransactionDT"].min()
    assert val["TransactionDT"].max() < test["TransactionDT"].min()
    assert "TransactionDT" not in model_columns(df) and "is_fraud" not in model_columns(df)


def test_split_falls_back_to_stratified_random_without_time_column():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.random(1000), "is_fraud": (rng.random(1000) < 0.1).astype(int)})
    train, val, test = split_data(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not (
        set(train.index) & set(val.index)
        or set(train.index) & set(test.index)
        or set(val.index) & set(test.index)
    )
    assert (len(train), len(val), len(test)) == (600, 200, 200)


def test_causal_sender_features_ignore_the_future():
    from create_ieee_dataset import sender_txn_24h_and_avg

    rng = np.random.default_rng(1)
    n = 1500
    df = pd.DataFrame(
        {
            "card1": rng.integers(0, 30, n),
            "TransactionDT": np.sort(rng.integers(0, 86400 * 10, n)),
            "TransactionAmt": rng.random(n) * 100,
            "TransactionID": np.arange(n),
        }
    )
    cnt, mean = sender_txn_24h_and_avg(df)

    future_edited = df.copy()
    future_edited.loc[1000:, "TransactionAmt"] *= 7
    future_edited.loc[1000:, "card1"] = future_edited.loc[1000:, "card1"].values[::-1]
    cnt2, mean2 = sender_txn_24h_and_avg(future_edited)

    assert (cnt[:1000] == cnt2[:1000]).all()
    np.testing.assert_allclose(mean[:1000], mean2[:1000])
    assert (cnt >= 1).all()


def test_schema_covers_every_categorical_the_pipeline_expects():
    # the trainer takes its categorical list from the schema: a typo here would silently
    # turn a categorical column into a numeric one
    for name in ("mcc", "card4", "card6", "P_emaildomain", "R_emaildomain", "DeviceType", "M4"):
        assert name in CATEGORICAL_NAMES
