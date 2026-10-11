"""Model registry, explainer interface and the comparison API (/api/models, /api/compare, /api/explain)."""

import json

import numpy as np
import pytest

from fraud_app import registry, services
from fraud_app.app import create_app

needs_data = pytest.mark.skipif(
    not (services.PIPELINE_FILE.exists() and services.DATASET_FILE.exists()),
    reason="needs artifacts/ and data/",
)
needs_experiments = pytest.mark.skipif(
    not (registry.ROOT_DIR / "runs" / "models" / "pipelines" / "LightGBM.joblib").exists(),
    reason="needs runs/models/pipelines (python -m experiments.run_models)",
)


def test_discover_reads_threshold_and_metrics(tmp_path):
    (tmp_path / "pipelines").mkdir()
    (tmp_path / "final").mkdir()
    (tmp_path / "pipelines" / "XGBoost.joblib").write_bytes(b"not loaded during discovery")
    (tmp_path / "final" / "XGBoost_seed0.json").write_text(json.dumps({"validation": {"threshold": 0.42}}))
    (tmp_path / "summary.json").write_text(json.dumps([{
        "model": "XGBoost", "pr_auc": {"mean": 0.51, "std": 0.003}, "roc_auc": {"mean": 0.9, "std": 0.0},
        "recall_at_1pct_fpr": {"mean": 0.41, "std": 0}, "f1": {"mean": 0.49, "std": 0},
        "pr_auc_ci95_seed0": [0.49, 0.52], "single_row_median_ms": 0.1, "refit_seconds": 18.0, "n_seeds": 5}]))
    found = registry.discover([tmp_path])
    e = found["XGBoost"]
    assert e.threshold == 0.42 and e.metrics["pr_auc"] == 0.51 and e.label == "XGBoost"
    assert not e.loaded  # pipelines are loaded lazily


def test_discover_without_results_defaults(tmp_path):
    (tmp_path / "pipelines").mkdir()
    (tmp_path / "pipelines" / "Mystery.joblib").write_bytes(b"")
    e = registry.discover([tmp_path])["Mystery"]
    assert e.threshold == 0.5 and e.metrics == {}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(scope="module")
def row(client):
    d = client.get("/api/random?fraud=1").get_json()
    assert d["label"] == 1 and "is_fraud" not in d["row"]
    return d["row"]


@needs_data
def test_models_lists_production_first(client):
    d = client.get("/api/models").get_json()
    names = [m["name"] for m in d["models"]]
    assert names[0] == registry.PRODUCTION and set(d["methods"]) == {"shap", "lime"}


@needs_data
def test_compare_scores_every_model(client, row):
    d = client.post("/api/compare", json={"row": row, "explain": "none"}).get_json()
    assert len(d["results"]) == len(registry.registry())
    for r in d["results"]:
        assert "error" not in r, r
        assert 0.0 <= r["probability"] <= 1.0
        assert r["fraud"] == (r["probability"] >= r["threshold"])


@needs_data
def test_compare_rejects_bad_input(client, row):
    assert client.post("/api/compare", json={"row": {}}).status_code == 400
    assert client.post("/api/compare", json={"row": row, "explain": "magic"}).status_code == 400
    assert client.post("/api/compare", json={"row": row, "models": ["NoSuchModel"]}).status_code == 400
    assert client.post("/api/explain", json={"row": row, "model": "NoSuchModel"}).status_code == 400


@needs_data
@needs_experiments
def test_shap_adds_up_to_the_probability(client, row):
    d = client.post("/api/explain", json={"row": row, "model": "LightGBM", "method": "shap", "top": 30}).get_json()
    full = client.post("/api/compare", json={"row": row, "models": ["LightGBM"], "explain": "shap", "top": 30}).get_json()
    assert d["space"] == "probability"
    # all 130 contributions (not only the top 30) must add up: ask the explainer directly
    from fraud_app.explainers import explain
    from fraud_app.features import prepare_features
    import pandas as pd

    e = registry.get("LightGBM")
    X, _ = prepare_features(pd.DataFrame([row]), services._get_feature_list_from_meta(), e.pipe)
    out = explain(e, X, "shap", lambda: services.background_frame(e.pipe))
    total = out["base_value"] + sum(c["contribution"] for c in out["contributions"])
    assert total == pytest.approx(d["probability"], abs=1e-3)
    assert len(out["contributions"]) == len(services._get_feature_list_from_meta())
    assert full["results"][0]["explanation"][0]["feature"] == d["contributions"][0]["feature"]


@needs_data
@needs_experiments
def test_lime_is_reproducible_per_request(client, row):
    body = {"row": row, "model": "LightGBM", "method": "lime", "top": 10}
    a = client.post("/api/explain", json=body).get_json()
    b = client.post("/api/explain", json=body).get_json()
    assert [c["feature"] for c in a["contributions"]] == [c["feature"] for c in b["contributions"]]
    assert np.allclose([c["contribution"] for c in a["contributions"]], [c["contribution"] for c in b["contributions"]])


@needs_data
def test_experiments_tables_keep_column_order(client):
    d = client.get("/api/experiments").get_json()
    if not d["available"]:
        pytest.skip("no runs/paper tables")
    t = d["tables"]["table1_models"]
    assert t["columns"][0] == "model" and all(len(r) == len(t["columns"]) for r in t["rows"])
