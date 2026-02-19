"""
Tests for the MLOps model serving API.

Uses httpx AsyncClient with FastAPI's TestClient for endpoint testing.
Covers prediction, batch prediction, A/B routing, health checks,
and error handling scenarios.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture(scope="module")
def trained_model():
    """Train a simple classifier for testing."""
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=3,
        n_redundant=1, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def model_path(trained_model):
    """Save trained model to a temporary file."""
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "champion.joblib")
    joblib.dump(trained_model, path)
    return path


@pytest.fixture(scope="module")
def challenger_path(trained_model):
    """Save a challenger model to a temporary file."""
    X, y = make_classification(
        n_samples=200, n_features=4, n_informative=3,
        n_redundant=1, random_state=123,
    )
    challenger = RandomForestClassifier(n_estimators=15, random_state=123)
    challenger.fit(X, y)

    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "challenger.joblib")
    joblib.dump(challenger, path)
    return path


@pytest.fixture(scope="module")
def client(model_path, challenger_path):
    """Create a test client with a loaded model."""
    os.environ["MODEL_PATH"] = model_path
    os.environ["CHALLENGER_MODEL_PATH"] = challenger_path
    os.environ["CONFIG_PATH"] = "configs/config.yaml"

    from src.model_server import app, model_manager

    model_manager.load_model(model_path, version="test_v1")
    model_manager.load_challenger(challenger_path, version="test_v2")

    with TestClient(app) as test_client:
        yield test_client


class TestHealthEndpoint:
    """Tests for the /model/health endpoint."""

    def test_health_check_returns_200(self, client):
        response = client.get("/model/health")
        assert response.status_code == 200

    def test_health_check_model_loaded(self, client):
        response = client.get("/model/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"

    def test_health_check_has_version(self, client):
        response = client.get("/model/health")
        data = response.json()
        assert data["model_version"] == "test_v1"

    def test_health_check_has_uptime(self, client):
        response = client.get("/model/health")
        data = response.json()
        assert data["uptime_seconds"] >= 0


class TestModelInfoEndpoint:
    """Tests for the /model/info endpoint."""

    def test_model_info_returns_200(self, client):
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_has_version(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert data["model_version"] == "test_v1"

    def test_model_info_has_features(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert data["num_features"] >= 0

    def test_model_info_has_model_type(self, client):
        response = client.get("/model/info")
        data = response.json()
        assert data["model_type"] == "RandomForestClassifier"


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            }
        })
        assert response.status_code == 200

    def test_predict_returns_prediction(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            }
        })
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]

    def test_predict_returns_probabilities(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            }
        })
        data = response.json()
        assert "probability" in data
        assert len(data["probability"]) == 2
        assert abs(sum(data["probability"]) - 1.0) < 1e-6

    def test_predict_returns_model_version(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            }
        })
        data = response.json()
        assert data["model_version"] == "test_v1"

    def test_predict_returns_latency(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            }
        })
        data = response.json()
        assert data["latency_ms"] >= 0

    def test_predict_with_request_id(self, client):
        response = client.post("/predict", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            },
            "request_id": "test-req-001",
        })
        data = response.json()
        assert data["request_id"] == "test-req-001"

    def test_predict_missing_features(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict_returns_200(self, client):
        response = client.post("/predict/batch", json={
            "instances": [
                {"feature_0": 0.5, "feature_1": -0.3, "feature_2": 1.2, "feature_3": 0.8},
                {"feature_0": -0.1, "feature_1": 0.9, "feature_2": -0.5, "feature_3": 0.2},
                {"feature_0": 1.5, "feature_1": 0.3, "feature_2": 0.7, "feature_3": -1.1},
            ]
        })
        assert response.status_code == 200

    def test_batch_predict_correct_count(self, client):
        instances = [
            {"feature_0": float(i), "feature_1": 0.0, "feature_2": 0.0, "feature_3": 0.0}
            for i in range(5)
        ]
        response = client.post("/predict/batch", json={"instances": instances})
        data = response.json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_predict_has_probabilities(self, client):
        response = client.post("/predict/batch", json={
            "instances": [
                {"feature_0": 0.5, "feature_1": -0.3, "feature_2": 1.2, "feature_3": 0.8},
            ]
        })
        data = response.json()
        assert data["probabilities"] is not None
        assert len(data["probabilities"]) == 1

    def test_batch_predict_empty_list(self, client):
        response = client.post("/predict/batch", json={"instances": []})
        assert response.status_code == 422


class TestABPredictEndpoint:
    """Tests for the /predict/ab endpoint."""

    def test_ab_predict_returns_200(self, client):
        response = client.post("/predict/ab", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            },
            "user_id": "user_001",
        })
        assert response.status_code == 200

    def test_ab_predict_has_variant(self, client):
        response = client.post("/predict/ab", json={
            "features": {
                "feature_0": 0.5, "feature_1": -0.3,
                "feature_2": 1.2, "feature_3": 0.8,
            },
            "user_id": "user_001",
        })
        data = response.json()
        assert data["variant"] in ["champion", "challenger"]

    def test_ab_predict_consistent_routing(self, client):
        """Same user ID should always get the same variant."""
        variants = set()
        for _ in range(10):
            response = client.post("/predict/ab", json={
                "features": {
                    "feature_0": 0.5, "feature_1": -0.3,
                    "feature_2": 1.2, "feature_3": 0.8,
                },
                "user_id": "consistent_user_42",
            })
            data = response.json()
            variants.add(data["variant"])
        assert len(variants) == 1

    def test_ab_predict_distributes_traffic(self, client):
        """Different user IDs should distribute across variants."""
        variants = set()
        for i in range(100):
            response = client.post("/predict/ab", json={
                "features": {
                    "feature_0": 0.5, "feature_1": -0.3,
                    "feature_2": 1.2, "feature_3": 0.8,
                },
                "user_id": f"user_{i:04d}",
            })
            data = response.json()
            variants.add(data["variant"])
        assert len(variants) >= 1


class TestMetricsEndpoint:
    """Tests for the /metrics endpoint."""

    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_returns_prometheus_format(self, client):
        response = client.get("/metrics")
        assert "model_predictions_total" in response.text or "http_requests_total" in response.text

    def test_metrics_content_type(self, client):
        response = client.get("/metrics")
        assert "text/plain" in response.headers.get("content-type", "")


class TestErrorHandling:
    """Tests for error handling and edge cases."""

    def test_invalid_endpoint_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_invalid_method_returns_405(self, client):
        response = client.get("/predict")
        assert response.status_code == 405

    def test_invalid_json_returns_422(self, client):
        response = client.post(
            "/predict",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
