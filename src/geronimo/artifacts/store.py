"""ArtifactStore for versioned ML artifact management."""

import os
from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel

from geronimo.artifacts.protocol import ArtifactBackend


class ArtifactMetadata(BaseModel):
    """Metadata for a stored artifact."""

    name: str
    version: str
    artifact_type: str
    created_at: datetime
    size_bytes: int
    checksum: Optional[str] = None
    tags: dict[str, str] = {}


class ArtifactStore:
    """Versioned storage for ML artifacts.

    Supports local filesystem, S3, and Geronimo Cloud backends.
    Reads defaults from ~/.geronimo/config.yaml (run `geronimo config show`).

    Example:
        ```python
        from geronimo.artifacts import ArtifactStore

        # Save during training (uses global config defaults)
        store = ArtifactStore(project="credit-risk", version="1.2.0")
        store.save("model", trained_model)
        store.save("encoder", fitted_encoder)

        # Load in production
        store = ArtifactStore.load(project="credit-risk", version="1.2.0")
        model = store.get("model")
        encoder = store.get("encoder")
        ```
    
    Configuration:
        Run `geronimo config init` to set up your preferred backend.
        Run `geronimo config show` to view current settings.
    """

    def __init__(
        self,
        project: str,
        version: str,
        backend: Optional[Union[Literal["local", "s3", "gdc"], ArtifactBackend]] = None,
        base_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ):
        """Initialize artifact store.

        Args:
            project: Project name.
            version: Version string (e.g., "1.2.0").
            backend: Storage backend ("local", "s3", "gdc") or custom ArtifactBackend instance.
                     Defaults to value from ~/.geronimo/config.yaml.
            base_path: Base path for local storage.
                       Defaults to ~/.geronimo/artifacts.
            s3_bucket: S3 bucket for s3 backend.
                       Defaults to value from config or GERONIMO_ARTIFACT_BUCKET env var.
        """
        # Load user config for defaults
        from geronimo.config.user_config import load_user_config
        user_config = load_user_config()
        
        self.project = project
        self.version = version
        
        # Handle custom backend instance
        if isinstance(backend, ArtifactBackend):
            self.backend = "custom"
            self._backend_impl = backend
        else:
            self.backend = backend or user_config.artifacts.backend
            
            # Resolve s3_bucket with fallback chain: param -> config -> env -> default
            self.s3_bucket = (
                s3_bucket 
                or user_config.artifacts.s3_bucket 
                or os.getenv("GERONIMO_ARTIFACT_BUCKET", "ml-artifacts")
            )
            
            # Resolve base_path
            self.base_path = base_path or os.path.expanduser(user_config.artifacts.base_path)
            
            # Create backend instance using factory
            self._backend_impl = self._create_backend()

        self._metadata: dict[str, ArtifactMetadata] = {}

    def _create_backend(self) -> ArtifactBackend:
        """Factory method to create backend instance.
        
        Returns:
            ArtifactBackend implementation based on self.backend type.
        """
        if self.backend == "local":
            from geronimo.artifacts.local_backend import LocalArtifactBackend
            return LocalArtifactBackend(
                project=self.project,
                version=self.version,
                base_path=self.base_path,
            )
        elif self.backend == "s3":
            from geronimo.artifacts.s3_backend import S3ArtifactBackend
            return S3ArtifactBackend(
                project=self.project,
                version=self.version,
                bucket=self.s3_bucket,
            )
        elif self.backend == "gdc":
            from geronimo.artifacts.gdc_backend import GeronimoDeployCloudArtifactBackend
            return GeronimoDeployCloudArtifactBackend(
                project=self.project,
                version=self.version,
            )
        else:
            raise ValueError(f"Unknown backend type: {self.backend}")

    @classmethod
    def load(
        cls,
        project: str,
        version: str,
        backend: Optional[Literal["local", "s3", "gdc"]] = None,
        **kwargs,
    ) -> "ArtifactStore":
        """Load existing artifact store.

        Args:
            project: Project name.
            version: Version string.
            backend: Storage backend. Defaults to value from config.
            **kwargs: Additional backend options.

        Returns:
            ArtifactStore instance with loaded metadata.
        """
        store = cls(project=project, version=version, backend=backend, **kwargs)
        store._load_metadata()
        return store

    def save(
        self,
        name: str,
        artifact: Any,
        artifact_type: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Save an artifact.

        Args:
            name: Artifact name (e.g., "model", "encoder").
            artifact: Python object to serialize.
            artifact_type: Optional type hint (auto-detected if not provided).
            tags: Optional metadata tags.

        Returns:
            Path or URI where artifact was saved.
        """
        artifact_type = artifact_type or type(artifact).__name__
        tags = tags or {}
        
        meta_dict = {
            "artifact_type": artifact_type,
            "tags": tags,
        }

        # Delegate to backend
        uri = self._backend_impl.save(name, artifact, meta_dict)
        
        # Keep local metadata cache in sync
        self._metadata[name] = ArtifactMetadata(
            name=name,
            version=self.version,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            size_bytes=0,  # Backend may have actual size
            tags=tags,
        )
        
        return uri

    def get(self, name: str) -> Any:
        """Load an artifact by name.

        Args:
            name: Artifact name.

        Returns:
            Deserialized artifact.

        Raises:
            KeyError: If artifact not found.
        """
        return self._backend_impl.load(name)

    def list(self) -> list[ArtifactMetadata]:
        """List all artifacts in this store.

        Returns:
            List of artifact metadata.
        """
        self._load_metadata()
        return list(self._metadata.values())
        
    def delete(self, name: str) -> None:
        """Delete an artifact.
        
        Args:
           name: Artifact name to delete
        """
        self._backend_impl.delete(name)
        if name in self._metadata:
            del self._metadata[name]

    def _load_metadata(self) -> None:
        """Load metadata from backend."""
        # Try to get metadata from backend if it supports it
        if hasattr(self._backend_impl, 'get_metadata_index'):
            data = self._backend_impl.get_metadata_index()
            self._metadata = {
                k: ArtifactMetadata.model_validate(v) for k, v in data.items()
            }

    def __repr__(self) -> str:
        return f"ArtifactStore({self.project}@{self.version}, backend={self.backend})"
