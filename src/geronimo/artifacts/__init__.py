"""Geronimo Artifact Store.

Provides versioned storage for ML artifacts (models, encoders, etc.).
"""

from geronimo.artifacts.store import ArtifactStore
from geronimo.artifacts.protocol import ArtifactBackend
from geronimo.artifacts.local_backend import LocalArtifactBackend
from geronimo.artifacts.s3_backend import S3ArtifactBackend
from geronimo.artifacts.gdc_backend import GeronimoDeployCloudArtifactBackend

# Optional MLflow backend
try:
    from geronimo.artifacts.mlflow_backend import MLflowArtifactStore
except ImportError:
    MLflowArtifactStore = None

__all__ = [
    "ArtifactStore",
    "ArtifactBackend",
    "LocalArtifactBackend",
    "S3ArtifactBackend",
    "GeronimoDeployCloudArtifactBackend",
    "MLflowArtifactStore",
]

