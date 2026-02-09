"""Tests for geronimo.deploy_testing_fixtures utilities.

Verifies that test fixtures work correctly and can be used
by deploy-cloud integration tests.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestCreateTestProject:
    """Tests for create_test_project fixture."""

    def test_creates_project_structure(self):
        """Test that create_test_project creates valid project structure."""
        from geronimo.deploy_testing_fixtures import create_test_project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = create_test_project(
                name="test-fixture",
                temp_dir=Path(tmpdir),
            )

            # Verify project structure
            assert project_path.exists()
            assert (project_path / "geronimo.yaml").exists()
            assert (project_path / "src").exists()
            assert (project_path / "src" / "test_fixture").exists()
            assert (project_path / "src" / "test_fixture" / "sdk").exists()

    def test_creates_batch_template(self):
        """Test batch template creation."""
        from geronimo.deploy_testing_fixtures import create_test_project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = create_test_project(
                name="batch-test",
                temp_dir=Path(tmpdir),
                template="batch",
            )

            # Verify batch-specific structure
            assert project_path.exists()
            pkg_dir = project_path / "src" / "batch_test"
            assert (pkg_dir / "flow.py").exists()

    def test_creates_realtime_template(self):
        """Test realtime template creation."""
        from geronimo.deploy_testing_fixtures import create_test_project

        with tempfile.TemporaryDirectory() as tmpdir:
            project_path = create_test_project(
                name="realtime-test",
                temp_dir=Path(tmpdir),
                template="realtime",
            )

            # Verify realtime-specific structure
            assert project_path.exists()
            pkg_dir = project_path / "src" / "realtime_test"
            assert (pkg_dir / "app.py").exists()


class TestTestModel:
    """Tests for TestModel fixture."""

    def test_model_can_train(self):
        """Test that TestModel can train successfully."""
        pytest.importorskip("sklearn")
        from geronimo.deploy_testing_fixtures import TestModel

        model = TestModel()
        metrics = model.train()

        assert "accuracy" in metrics
        assert "n_samples" in metrics
        assert "n_features" in metrics
        assert metrics["accuracy"] > 0.9  # Iris should be easy
        assert metrics["n_samples"] == 150  # Iris dataset size
        assert model.is_fitted

    def test_model_can_predict(self):
        """Test that TestModel can make predictions."""
        pytest.importorskip("sklearn")
        import numpy as np
        from geronimo.deploy_testing_fixtures import TestModel

        model = TestModel()
        model.train()

        # Single prediction
        X = np.array([[5.1, 3.5, 1.4, 0.2]])
        predictions = model.predict(X)

        assert len(predictions) == 1
        assert predictions[0] in [0, 1, 2]  # Valid class

    def test_model_can_save_and_load(self):
        """Test that TestModel can save and load from ArtifactStore."""
        pytest.importorskip("sklearn")
        import tempfile
        import numpy as np
        from geronimo.deploy_testing_fixtures import TestModel
        from geronimo.artifacts import ArtifactStore

        model = TestModel()
        model.train()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            store = ArtifactStore(
                project="test-model",
                version="1.0.0",
                backend="local",
                base_path=tmpdir,
            )
            paths = model.save(store)
            assert len(paths) == 2  # estimator + features

            # Load into new model
            new_model = TestModel()
            new_model.load(store)
            assert new_model.is_fitted

            # Verify predictions match
            X = np.array([[5.1, 3.5, 1.4, 0.2]])
            original_pred = model.predict(X)
            loaded_pred = new_model.predict(X)
            assert original_pred[0] == loaded_pred[0]


class TestMockUtilities:
    """Tests for mock utility functions."""

    def test_create_mock_cloud_client(self):
        """Test mock cloud client has expected attributes."""
        from geronimo.deploy_testing_fixtures import create_mock_cloud_client

        client = create_mock_cloud_client()

        assert client.api_url == "https://api.test.geronimo.cloud"
        assert client.token == "test-token"

        # Test default return values
        result = client.deploy_project("test", {}, Path("test.zip"))
        assert result["id"] == "deploy-123"
        assert result["status"] == "pending"

        result = client.sync_keys([])
        assert result["synced"] == 1

    def test_create_mock_http_client(self):
        """Test mock HTTP client has expected behavior."""
        from geronimo.deploy_testing_fixtures import create_mock_http_client

        client = create_mock_http_client()

        # Test default behavior
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json() == {}

        # Test context manager
        with client as c:
            resp = c.post("/test")
            assert resp.status_code == 200
