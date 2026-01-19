"""Pytest configuration and fixtures."""

import json
import tempfile
from pathlib import Path
from typing import Generator

import pandas as pd
import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


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
