"""Tests for geronimo.config module."""

from pathlib import Path

import pytest

from geronimo.config.loader import load_config, save_config
from geronimo.config.schema import (
    GeronimoConfig,
    ProjectConfig,
    ModelConfig,
    ModelType,
    MLFramework,
)


class TestGeronimoConfig:
    """Tests for GeronimoConfig schema."""

    def test_basic_config(self):
        """Test creating a basic config."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="models/model.joblib",
            ),
        )
        assert config.project.name == "test"
        assert config.model.type == ModelType.REALTIME

    def test_config_defaults(self):
        """Test config default values."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
        )
        # Runtime should have defaults
        assert config.runtime.python_version == "3.11"


class TestConfigLoader:
    """Tests for config loading and saving."""

    def test_save_and_load_config(self, temp_dir: Path):
        """Test saving and loading config."""
        config = GeronimoConfig(
            project=ProjectConfig(
                name="test-project",
                version="1.0.0",
                description="Test",
            ),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="models/model.joblib",
            ),
        )
        
        config_path = temp_dir / "geronimo.yaml"
        save_config(config, config_path)
        
        assert config_path.exists()
        
        loaded = load_config(config_path)
        assert loaded.project.name == "test-project"
        assert loaded.model.framework == MLFramework.SKLEARN

    def test_load_minimal_config(self, temp_dir: Path):
        """Test loading minimal YAML config."""
        config_path = temp_dir / "geronimo.yaml"
        config_path.write_text("""
project:
  name: minimal
  version: "1.0.0"

model:
  type: realtime
  framework: sklearn
  artifact_path: model.joblib
""")
        
        config = load_config(config_path)
        assert config.project.name == "minimal"

    def test_load_nonexistent_config(self, temp_dir: Path):
        """Test error on missing config file."""
        with pytest.raises(FileNotFoundError):
            load_config(temp_dir / "nonexistent.yaml")

    def test_load_invalid_yaml(self, temp_dir: Path):
        """Test error on invalid YAML."""
        config_path = temp_dir / "bad.yaml"
        config_path.write_text("invalid: yaml: syntax:")
        
        with pytest.raises(Exception):  # YAML parse error
            load_config(config_path)
