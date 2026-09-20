import json
from pathlib import Path
from unittest.mock import patch

import pytest

from fraud_app import services
from fraud_app.app import create_app

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    not (services.PIPELINE_FILE.exists() and services.DATASET_FILE.exists()),
    reason="needs artifacts/ and data/ (train the model or run generate_dummy_artifacts.py)",
)


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Never touch the real threshold / history files from tests.
    monkeypatch.setattr(services, "THRESH_JSON", tmp_path / "threshold.json")
    monkeypatch.setattr(services, "HISTORY_FILE", tmp_path / "history.json")
    monkeypatch.setattr(services, "THRESHOLD", services.THRESHOLD)  # restored on teardown
    monkeypatch.delenv("FRAUD_ADMIN_TOKEN", raising=False)

    app = create_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_api_random(client):
    rv = client.get('/api/random')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'row' in data
    assert isinstance(data['row'], dict)
    assert 'TransactionDT' not in data['row'] and 'is_fraud' not in data['row']


def test_api_random_is_strict_json(client):
    # Missing values must be null: a bare NaN is not JSON and breaks response.json() in browsers.
    for _ in range(5):
        body = client.get('/api/random?fraud=1').get_data(as_text=True)
        assert 'NaN' not in body
        json.loads(body, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_api_predict(client):
    # Get a random row first to use as input
    rv = client.get('/api/random')
    row = rv.get_json()['row']

    # Predict
    rv = client.post('/api/predict', json=row)
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'probs' in data
    assert 'labels' in data
    assert 'threshold' in data
    assert all(0.0 <= p <= 1.0 for p in data['probs'])


def test_api_predict_with_only_a_few_fields(client):
    # the model treats everything that is not sent as missing
    rv = client.post('/api/predict', json={"amount": 50.0, "mcc": 4})
    assert rv.status_code == 200
    assert 0.0 <= rv.get_json()['probs'][0] <= 1.0


def test_history_stays_valid_json_with_null_inputs(client):
    row = client.get('/api/random').get_json()['row']
    row[next(iter(row))] = None  # real demo rows carry missing values; make sure this one does
    assert client.post('/api/predict', json=row).status_code == 200

    body = client.get('/api/history').get_data(as_text=True)
    assert 'NaN' not in body
    assert json.loads(body)[0]['result']['probs']


def test_api_predict_does_not_leak_traceback(client):
    rv = client.post('/api/predict', data='not json', content_type='application/json')
    assert rv.status_code >= 400
    assert 'traceback' not in rv.get_json()


def test_api_schema(client):
    rv = client.get('/api/schema')
    assert rv.status_code == 200
    data = rv.get_json()
    names = [f['name'] for g in data['groups'] for f in g['features']]
    assert data['n_features'] == len(names) == len(set(names))
    assert 'amount' in names


def test_api_metrics(client):
    rv = client.get('/api/metrics')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'roc_auc' in data['global_metrics']


def test_api_threshold(client):
    # Get
    rv = client.get('/api/threshold')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'threshold' in data

    # Set
    new_thr = 0.8
    rv = client.post('/api/threshold', json={'threshold': new_thr})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['threshold'] == new_thr

    # Verify Get
    rv = client.get('/api/threshold')
    assert rv.get_json()['threshold'] == new_thr


def test_api_threshold_rejects_out_of_range(client):
    assert client.post('/api/threshold', json={'threshold': 7}).status_code == 400


def test_api_threshold_is_closed_to_remote_clients(client):
    remote = {"REMOTE_ADDR": "203.0.113.9"}
    rv = client.post('/api/threshold', json={'threshold': 0.4}, environ_overrides=remote)
    assert rv.status_code == 403
    # reading stays open
    assert client.get('/api/threshold', environ_overrides=remote).status_code == 200


def test_api_threshold_admin_token(client, monkeypatch):
    monkeypatch.setenv("FRAUD_ADMIN_TOKEN", "s3cret")
    remote = {"REMOTE_ADDR": "203.0.113.9"}
    assert client.post('/api/threshold', json={'threshold': 0.4}, environ_overrides=remote).status_code == 403
    bad = client.post('/api/threshold', json={'threshold': 0.4}, headers={"X-Admin-Token": "nope"})
    assert bad.status_code == 403
    ok = client.post(
        '/api/threshold',
        json={'threshold': 0.4},
        headers={"X-Admin-Token": "s3cret"},
        environ_overrides=remote,
    )
    assert ok.status_code == 200 and ok.get_json()['threshold'] == 0.4


@patch('fraud_app.api.compute_shap_for_df_via_worker')
def test_api_shap(mock_worker, client):
    mock_worker.return_value = {
        "base_value": 0.5,
        "shap": [{"feature": "f1", "shap": 0.1, "value": 1.0}],
    }

    rv = client.get('/api/random')
    row = rv.get_json()['row']

    rv = client.post('/api/shap', json=row)
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'base_value' in data
    assert 'shap' in data
