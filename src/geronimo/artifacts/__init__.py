"""Geronimo Artifact Store.

The artifacts module provides a unified interface for storing and retrieving
machine learning artifacts such as trained models, encoders, and other binary blobs.
It supports versioned storage to ensure reproducibility and traceability.

Key components:
- ArtifactStore: The main entry point for saving and loading artifacts.
- ArtifactBackend: Protocol that defines the storage backend interface.

Supported backends:
- LocalArtifactBackend: Stores artifacts on the local filesystem.
- S3ArtifactBackend: Stores artifacts in an AWS S3 bucket.
- GeronimoDeployCloudArtifactBackend: Managed artifact storage via Geronimo Deploy Cloud.
- MLflowArtifactStore: Optional integration for tracking artifacts with MLflow.
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


__docformat__ = "google"
