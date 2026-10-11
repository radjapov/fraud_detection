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


@needs_data
def test_production_threshold_follows_the_service(client, tmp_path, monkeypatch):
    # the registry is cached, but POST /api/threshold changes the threshold at runtime
    monkeypatch.setattr(services, "THRESH_JSON", tmp_path / "threshold.json")
    monkeypatch.setattr(services, "THRESHOLD", services.THRESHOLD)  # restored on teardown
    client.get("/api/models")  # builds the registry with the current threshold
    assert client.post("/api/threshold", json={"threshold": 0.77}).status_code == 200
    models = {m["name"]: m for m in client.get("/api/models").get_json()["models"]}
    assert models[registry.PRODUCTION]["threshold"] == pytest.approx(0.77)


@needs_data
def test_compare_decides_with_the_current_production_threshold(client, row, tmp_path, monkeypatch):
    monkeypatch.setattr(services, "THRESH_JSON", tmp_path / "threshold.json")
    monkeypatch.setattr(services, "THRESHOLD", services.THRESHOLD)
    client.get("/api/models")
    assert client.post("/api/threshold", json={"threshold": 0.01}).status_code == 200
    d = client.post("/api/compare", json={"row": row, "models": [registry.PRODUCTION], "explain": "none"}).get_json()
    (r,) = d["results"]
    assert r["threshold"] == pytest.approx(0.01) and r["fraud"] == (r["probability"] >= 0.01)


@needs_data
def test_compare_and_explain_validate_model_names(client, row):
    for bad in (5, "production", [["x"]], [1], {"a": 1}):
        r = client.post("/api/compare", json={"row": row, "models": bad, "explain": "none"})
        assert r.status_code == 400, bad
    assert client.post("/api/explain", json={"row": row, "model": ["production"]}).status_code == 400
    assert client.post("/api/explain", json={"row": row, "model": 7}).status_code == 400


@needs_data
def test_compare_scores_each_model_once(client, row):
    body = {"row": row, "models": [registry.PRODUCTION] * 20, "explain": "none"}
    d = client.post("/api/compare", json=body).get_json()
    assert [r["model"] for r in d["results"]] == [registry.PRODUCTION]


@needs_data
@needs_experiments
def test_lime_gives_the_same_answer_under_concurrent_requests(row):
    # the LIME explainer is shared and reseeded per call: concurrent requests must not mix seeds
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd

    from fraud_app.explainers import explain
    from fraud_app.features import prepare_features

    e = registry.get("LightGBM")
    X, _ = prepare_features(pd.DataFrame([row]), services._get_feature_list_from_meta(), e.pipe)

    def background():
        return services.background_frame(e.pipe)

    def run(_):
        return [(c["feature"], round(c["contribution"], 9)) for c in explain(e, X, "lime", background)["contributions"]]

    run(0)  # build the explainer once
    with ThreadPoolExecutor(4) as pool:
        runs = list(pool.map(run, range(12)))
    assert all(r == runs[0] for r in runs[1:])
