import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import joblib
import pytest

from fraud_app.app import create_app


@pytest.fixture
def client():
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


def test_api_metrics(client):
    rv = client.get('/api/metrics')
    # It might be 200 or 404 depending on artifacts, but with generate_dummy_artifacts it should be 200
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'roc_auc' in data


def test_api_threshold(client):
    # Get
    rv = client.get('/api/threshold')
    assert rv.status_code == 200
    data = rv.get_json()
    assert 'threshold' in data
    old_thr = data['threshold']

    # Set
    new_thr = 0.8
    rv = client.post('/api/threshold', json={'threshold': new_thr})
    assert rv.status_code == 200
    data = rv.get_json()
    assert data['threshold'] == new_thr

    # Verify Get
    rv = client.get('/api/threshold')
    assert rv.get_json()['threshold'] == new_thr


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
