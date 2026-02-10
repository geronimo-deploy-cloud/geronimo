"""Tests for MLFlowArtifactBackend."""

import os
import pickle
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from geronimo.artifacts.protocol import ArtifactBackend

# Mock mlflow before importing module related to it
mock_mlflow = MagicMock()
sys.modules["mlflow"] = mock_mlflow
sys.modules["mlflow.tracking"] = MagicMock()
import mlflow

from geronimo.artifacts.mlflow_backend import MLFlowArtifactBackend


class TestMLFlowArtifactBackend:
    """Tests for MLFlowArtifactBackend."""

    @pytest.fixture
    def mock_mlflow(self):
        """Reset mlflow mock for each test."""
        mlflow.reset_mock()
        # Mock specific return values if needed
        mlflow.get_experiment_by_name.return_value = None
        mlflow.create_experiment.return_value = "exp_id_123"
        return mlflow

    def test_initialization_creates_experiment(self, mock_mlflow):
        """Test that initialization sets up experiment."""
        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")

        mock_mlflow.get_experiment_by_name.assert_called_with("test-project")
        mock_mlflow.create_experiment.assert_called_with("test-project")
        
        assert backend.project == "test-project"
        assert backend.version == "1.0.0"

    def test_initialization_uses_existing_experiment(self, mock_mlflow):
        """Test initialization with existing experiment."""
        mock_exp = MagicMock()
        mock_exp.experiment_id = "exp_id_456"
        mock_mlflow.get_experiment_by_name.return_value = mock_exp

        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")

        mock_mlflow.get_experiment_by_name.assert_called_with("test-project")
        mock_mlflow.create_experiment.assert_not_called()
        assert backend._experiment_id == "exp_id_456"

    def test_save_artifact(self, mock_mlflow, tmp_path):
        """Test saving an artifact."""
        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")
        
        # Setup empty search result so it creates a new run
        mock_mlflow.search_runs.return_value = []
        
        # Setup mock run context manager
        mock_run = MagicMock()
        mock_run.info.run_id = "run_123"
        # Since it's used as context manager: with start_run() as run:
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

        data = {"key": "value"}
        metadata = {"artifact_type": "dict", "tags": {"env": "prod"}}
        
        # Prevent file deletion so we can verify content
        with patch("geronimo.artifacts.mlflow_backend.os.unlink"):
            # Execute
            uri = backend.save("test_artifact", data, metadata)

        # Verify
        mock_mlflow.start_run.assert_called()
        assert uri.startswith("runs:/run_123/test_artifact/")
        
        # Check that log_artifact was called
        mock_mlflow.log_artifact.assert_called()
        args, _ = mock_mlflow.log_artifact.call_args
        saved_path = args[0]
        args, _ = mock_mlflow.log_artifact.call_args
        saved_path = args[0]
        assert saved_path.endswith(".pkl")
        
        # Verify serialization
        with open(saved_path, "rb") as f:
            assert pickle.load(f) == data

        # Verify calls to log_param
        mock_mlflow.log_param.assert_any_call("test_artifact_artifact_type", "dict")
        
    def test_load_artifact(self, mock_mlflow, tmp_path):
        """Test loading an artifact."""
        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")
        
        # Setup mock run search
        mock_runs = MagicMock()
        # Mock dataframe-like behavior for runs
        mock_runs.__len__.return_value = 1
        mock_runs.iloc = [
            {"run_id": "found_run_id", "artifact_uri": "s3://bucket/path"}
        ]
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Setup mock download
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        pkl_file = artifact_dir / "test_artifact.pkl"
        with open(pkl_file, "wb") as f:
            pickle.dump({"loaded": "data"}, f)
            
        mock_client = MagicMock()
        mock_client.download_artifacts.return_value = str(artifact_dir)
        
        # Patch MlflowClient to return our mock
        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            data = backend.load("test_artifact")
            
        assert data == {"loaded": "data"}
        mock_client.download_artifacts.assert_called_with("found_run_id", "test_artifact")

    def test_list_artifacts(self, mock_mlflow):
        """Test listing artifacts."""
        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")
        
        # Setup mock matching run
        mock_runs = MagicMock()
        mock_runs.__len__.return_value = 1
        mock_runs.iloc = [{"run_id": "run_abc"}]
        mock_mlflow.search_runs.return_value = mock_runs
        
        # Setup mock list_artifacts
        mock_client = MagicMock()
        artifact1 = MagicMock()
        artifact1.path = "model"
        artifact2 = MagicMock()
        artifact2.path = "encoder"
        
        mock_client.list_artifacts.return_value = [artifact1, artifact2]

        with patch("mlflow.tracking.MlflowClient", return_value=mock_client):
            artifacts = backend.list()
            
        assert len(artifacts) == 2
        assert "runs:/run_abc/model" in artifacts
        assert "runs:/run_abc/encoder" in artifacts

    def test_delete_artifact(self, mock_mlflow):
        """Test deleting artifact (not implemented in MLflow backend usually, but check behavior)."""
        backend = MLFlowArtifactBackend(project="test-project", version="1.0.0")
        
        # Since standard mlflow doesn't support deleting single artifact easily via easy API,
        # we might implement it as a pass or log warning. 
        # For now, let's assuming it just attempts it via client if we implemented it that way,
        # or does nothing but shouldn't crash.
        
        backend.delete("some_artifact")
        # Just ensure no exception

    def test_artifact_store_integration(self, mock_mlflow):
        """Test integration with ArtifactStore."""
        from geronimo.artifacts import ArtifactStore
        
        store = ArtifactStore(
            project="test-project",
            version="1.0.0",
            backend="mlflow",
        )
        
        assert store.backend == "mlflow"
        assert isinstance(store._backend_impl, MLFlowArtifactBackend)
        assert store._backend_impl.project == "test-project"

