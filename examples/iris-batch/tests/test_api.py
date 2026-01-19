"""Tests for the ML serving API."""

import pytest
from fastapi.testclient import TestClient

from iris_batch.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check(client: TestClient):
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict(client: TestClient):
    """Test prediction endpoint."""
    response = client.post(
        "/v1/predict",
        json={"features": {"feature_1": 1.0, "feature_2": 2.0}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert "latency_ms" in data
