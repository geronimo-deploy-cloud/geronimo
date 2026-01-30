"""Tests for pipeline artifact integration and template imports."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import pandas as pd

from geronimo.batch import BatchPipeline, Schedule


class TestPipelineArtifactIntegration:
    """Tests for BatchPipeline ArtifactStore integration."""

    def test_save_results_default_file_based(self, temp_dir: Path):
        """Test default file-based saving still works."""
        class TestPipeline(BatchPipeline):
            def run(self):
                return pd.DataFrame({"x": [1, 2, 3]})
        
        pipeline = TestPipeline()
        # Use dict results which will be saved as JSON
        results = {"status": "ok", "count": 10}
        
        output_path = str(temp_dir / "test_output.json")
        path = pipeline.save_results(results, output_path=output_path)
        
        assert Path(path).exists()
        import json
        loaded = json.loads(Path(path).read_text())
        assert loaded["status"] == "ok"
        assert loaded["count"] == 10

    def test_save_results_json_fallback(self, temp_dir: Path):
        """Test JSON fallback for non-DataFrame results."""
        class TestPipeline(BatchPipeline):
            def run(self):
                return {"status": "complete"}
        
        pipeline = TestPipeline()
        results = {"key": "value", "count": 42}
        
        output_path = str(temp_dir / "test_output.json")
        path = pipeline.save_results(results, output_path=output_path)
        
        assert Path(path).exists()
        assert path.endswith(".json")

    @patch("geronimo.artifacts.ArtifactStore")
    def test_save_results_with_artifact_store(self, mock_store_class, temp_dir: Path):
        """Test saving via ArtifactStore when requested."""
        mock_store = MagicMock()
        mock_store.save.return_value = "s3://bucket/path/results.pkl"
        mock_store_class.return_value = mock_store
        
        class TestPipeline(BatchPipeline):
            artifact_project = "test-project"
            artifact_version = "1.0.0"
            
            def run(self):
                return {"status": "complete"}
        
        pipeline = TestPipeline()
        results = pd.DataFrame({"x": [1, 2, 3]})
        
        path = pipeline.save_results(
            results, 
            use_artifact_store=True,
            artifact_name="my_results"
        )
        
        # ArtifactStore.save should have been called
        mock_store.save.assert_called_once()
        call_args = mock_store.save.call_args
        assert call_args[0][0] == "my_results"
        assert call_args[1]["artifact_type"] == "pipeline_result"

    def test_save_results_artifact_store_default_name(self, temp_dir: Path):
        """Test artifact name defaults to pipeline class name."""
        class CustomPipeline(BatchPipeline):
            def run(self):
                return {}
        
        pipeline = CustomPipeline()
        
        with patch("geronimo.artifacts.ArtifactStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store.save.return_value = "path"
            mock_store_class.return_value = mock_store
            
            pipeline.save_results({"data": 1}, use_artifact_store=True)
            
            # Should use class name in artifact name
            call_args = mock_store.save.call_args
            assert "CustomPipeline" in call_args[0][0]


class TestTemplateImports:
    """Tests for template import handling."""

    def test_template_imports_raise_helpful_error(self):
        """Test that template imports raise helpful ImportError."""
        # The template should raise ImportError when imported
        # because geronimo.api.models.schemas doesn't exist
        with pytest.raises(ImportError) as exc_info:
            # Need to reload to trigger fresh import
            import importlib
            import sys
            
            # Remove from cache if present
            if "geronimo.templates.agent.server" in sys.modules:
                del sys.modules["geronimo.templates.agent.server"]
            
            from geronimo.templates.agent import server
        
        # Should have helpful error message
        error_msg = str(exc_info.value)
        assert "template" in error_msg.lower() or "generate" in error_msg.lower()
