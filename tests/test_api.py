"""API tests using FastAPI's TestClient with a tiny model saved to tmp."""
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from heart_mlops.api import create_app


@pytest.fixture
def client(tmp_path, fitted_model):
    model_path = tmp_path / "model.joblib"
    joblib.dump(fitted_model, model_path)
    (tmp_path / "model_card.json").write_text(json.dumps(
        {"model_name": "test_lr", "trained_at_utc": "test", "decision_threshold": 0.5}))
    with TestClient(create_app(model_path)) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_health_reports_missing_model(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as c:
        assert c.get("/health").status_code == 503
        assert c.post("/predict", json={}).status_code == 422


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["model_name"] == "test_lr"
    assert r.headers["x-request-id"] == body["request_id"]


def test_predict_rejects_out_of_range_values(client, sample_patient):
    bad = {**sample_patient, "age": 250}
    assert client.post("/predict", json=bad).status_code == 422


def test_predict_rejects_invalid_category(client, sample_patient):
    bad = {**sample_patient, "thal": 5}
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_predict(client, sample_patient):
    r = client.post("/predict/batch", json={"patients": [sample_patient] * 3})
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_metrics_exposes_counters(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "model_predictions_total" in text
    assert 'endpoint="/predict"' in text


def test_model_info(client):
    assert client.get("/model-info").json()["model_name"] == "test_lr"
