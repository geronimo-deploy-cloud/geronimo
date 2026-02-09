"""Pytest configuration and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Generator, Any
from unittest.mock import Mock, MagicMock, patch

import pandas as pd
import pytest


# =============================================================================
# Directory and Path Fixtures
# =============================================================================

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# DataFrame Fixtures
# =============================================================================

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Sample DataFrame for testing."""
    return pd.DataFrame({
        "age": [25, 35, 45, 55, 65],
        "income": [50000, 75000, 100000, 125000, 150000],
        "segment": ["A", "B", "A", "C", "B"],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    })


@pytest.fixture
def iris_df() -> pd.DataFrame:
    """Iris-like DataFrame for testing."""
    return pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7, 5.0, 5.4],
        "sepal_width": [3.5, 3.0, 3.2, 3.6, 3.9],
        "petal_length": [1.4, 1.4, 1.3, 1.4, 1.7],
        "petal_width": [0.2, 0.2, 0.2, 0.2, 0.4],
        "target": [0, 0, 0, 0, 0],
    })


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def keys_file(temp_dir: Path) -> Path:
    """Create a temporary keys file."""
    keys_path = temp_dir / "keys.json"
    keys_path.write_text(json.dumps({"keys": {}}))
    return keys_path


@pytest.fixture
def geronimo_config(temp_dir: Path) -> Path:
    """Create a sample geronimo.yaml config."""
    config_path = temp_dir / "geronimo.yaml"
    config_path.write_text("""
project:
  name: test-project
  version: "1.0.0"

model:
  type: realtime
  framework: sklearn
  artifact_path: models/model.joblib
""")
    return config_path


# =============================================================================
# Mock HTTP Client Fixtures
# =============================================================================

@pytest.fixture
def mock_http_client() -> MagicMock:
    """Mock httpx.Client for API testing.
    
    Delegates to deploy_testing_fixtures for implementation consistency.
    
    Example:
        def test_api_call(mock_http_client):
            mock_http_client.post.return_value.json.return_value = {"id": "123"}
            # ... test logic ...
    """
    from geronimo.deploy_testing_fixtures import create_mock_http_client
    return create_mock_http_client()


@pytest.fixture
def mock_http_response() -> MagicMock:
    """Create a mock HTTP response.
    
    Example:
        def test_with_response(mock_http_response):
            mock_http_response.json.return_value = {"data": "value"}
            mock_http_response.status_code = 201
    """
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {}
    response.text = ""
    response.content = b""
    response.raise_for_status = MagicMock()
    return response


# =============================================================================
# Artifact Store Fixtures
# =============================================================================

@pytest.fixture
def temp_artifact_store(temp_dir: Path):
    """Temporary local artifact store for testing.
    
    Example:
        def test_artifact_save(temp_artifact_store):
            uri = temp_artifact_store.save("model", my_model)
            loaded = temp_artifact_store.get("model")
    """
    from geronimo.artifacts import ArtifactStore
    
    store = ArtifactStore(
        project="test-project",
        version="1.0.0",
        backend="local",
        base_path=str(temp_dir)
    )
    return store


@pytest.fixture
def mock_cloud_artifact_backend() -> MagicMock:
    """Mock cloud artifact backend for testing.
    
    Example:
        def test_cloud_save(mock_cloud_artifact_backend):
            mock_cloud_artifact_backend.save.return_value = "s3://bucket/path"
    """
    backend = MagicMock()
    backend.save.return_value = "s3://test-bucket/artifacts/test.pkl"
    backend.load.return_value = {"mock": "data"}
    backend.list.return_value = ["s3://test-bucket/artifacts/test.pkl"]
    backend.delete.return_value = None
    return backend


# =============================================================================
# Cloud Client Fixtures
# =============================================================================

@pytest.fixture
def mock_cloud_client() -> MagicMock:
    """Mock GeronimoCloudClient for testing.
    
    Delegates to deploy_testing_fixtures for implementation consistency.
    
    Example:
        def test_deploy(mock_cloud_client):
            mock_cloud_client.deploy_project.return_value = {"id": "deploy-123"}
    """
    from geronimo.deploy_testing_fixtures import create_mock_cloud_client
    return create_mock_cloud_client()


# =============================================================================
# Database Fixtures
# =============================================================================

@pytest.fixture
def mock_db_connection() -> MagicMock:
    """Mock database connection for testing.
    
    Example:
        def test_query(mock_db_connection, sample_df):
            mock_db_connection.execute.return_value = sample_df
    """
    conn = MagicMock()
    conn._connection = MagicMock()
    conn.execute.return_value = pd.DataFrame()
    conn.connect.return_value = None
    conn.close.return_value = None
    
    # Context manager support
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    
    return conn

