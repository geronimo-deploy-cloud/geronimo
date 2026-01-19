"""Tests for geronimo.artifacts module."""

import json
from pathlib import Path

import pytest

from geronimo.artifacts import ArtifactStore


class TestArtifactStore:
    """Tests for ArtifactStore class."""

    def test_local_store_creation(self, temp_dir: Path):
        """Test creating a local artifact store."""
        store = ArtifactStore(
            project="test-project",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        assert store.project == "test-project"
        assert store.version == "1.0.0"

    def test_save_and_load_artifact(self, temp_dir: Path):
        """Test saving and loading an artifact."""
        store = ArtifactStore(
            project="test",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        
        # Save a simple object
        test_data = {"key": "value", "number": 42}
        store.save("test-artifact", test_data)
        
        # Load it back
        loaded = store.get("test-artifact")
        assert loaded == test_data

    def test_save_with_tags(self, temp_dir: Path):
        """Test saving artifact with tags."""
        store = ArtifactStore(
            project="test",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        
        store.save(
            "model",
            {"weights": [1, 2, 3]},
            artifact_type="model",
            tags={"accuracy": "0.95"},
        )
        
        # Verify artifact exists
        loaded = store.get("model")
        assert loaded["weights"] == [1, 2, 3]

    def test_list_artifacts(self, temp_dir: Path):
        """Test listing artifacts."""
        store = ArtifactStore(
            project="test",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        
        store.save("artifact1", {"a": 1})
        store.save("artifact2", {"b": 2})
        
        artifacts = store.list()
        assert len(artifacts) >= 2

    def test_artifact_versioning(self, temp_dir: Path):
        """Test different versions are separate."""
        store_v1 = ArtifactStore(
            project="test",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        store_v2 = ArtifactStore(
            project="test",
            version="2.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        
        store_v1.save("model", {"version": 1})
        store_v2.save("model", {"version": 2})
        
        assert store_v1.get("model")["version"] == 1
        assert store_v2.get("model")["version"] == 2

    def test_artifact_not_found(self, temp_dir: Path):
        """Test error when artifact not found."""
        store = ArtifactStore(
            project="test",
            version="1.0.0",
            backend="local",
            base_path=str(temp_dir),
        )
        
        with pytest.raises(KeyError):
            store.get("nonexistent")
