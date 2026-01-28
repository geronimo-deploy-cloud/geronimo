"""Geronimo Artifact Store.

Provides versioned storage for ML artifacts (models, encoders, etc.).
"""

from geronimo.artifacts.store import ArtifactStore
from geronimo.artifacts.protocol import ArtifactBackend

# Optional MLflow backend
try:
    from geronimo.artifacts.mlflow_backend import MLflowArtifactStore
except ImportError:
    MLflowArtifactStore = None

__all__ = ["ArtifactStore", "ArtifactBackend", "MLflowArtifactStore"]

