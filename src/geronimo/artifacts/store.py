"""ArtifactStore for versioned ML artifact management."""

import json
import os
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel


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
        backend: Optional[Literal["local", "s3", "cloud"]] = None,
        base_path: Optional[str] = None,
        s3_bucket: Optional[str] = None,
    ):
        """Initialize artifact store.

        Args:
            project: Project name.
            version: Version string (e.g., "1.2.0").
            backend: Storage backend ("local", "s3", or "cloud").
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
        self.backend = backend or user_config.artifacts.backend
        
        # Resolve s3_bucket with fallback chain: param -> config -> env -> default
        self.s3_bucket = (
            s3_bucket 
            or user_config.artifacts.s3_bucket 
            or os.getenv("GERONIMO_ARTIFACT_BUCKET", "ml-artifacts")
        )

        if self.backend == "local":
            self.base_path = Path(
                base_path 
                or os.path.expanduser(user_config.artifacts.base_path)
            )
            self.artifact_path = self.base_path / project / version
            self.artifact_path.mkdir(parents=True, exist_ok=True)
            self._backend_impl = None  # Use internal local logic for now
        elif self.backend == "cloud":
            # Cloud backend uses Geronimo Cloud API
            from geronimo.artifacts.cloud_backend import GeronimoCloudArtifactBackend
            self._backend_impl = GeronimoCloudArtifactBackend(
                project=self.project,
                version=self.version
            )
        else:
            # S3 backend
            self._backend_impl = None  # Use internal s3 logic for now
            self.base_path = None
            self.artifact_path = None

        self._metadata: dict[str, ArtifactMetadata] = {}

    @classmethod
    def load(
        cls,
        project: str,
        version: str,
        backend: Optional[Literal["local", "s3", "cloud"]] = None,
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
        
        # Calculate size/meta for protocol if needed, but the backends might do it themselves.
        # For now, we construct minimal metadata dict to pass to backend if strictly following protocol,
        # but our custom backend `save` takes `metadata`.
        
        meta_dict = {
            "artifact_type": artifact_type,
            "tags": tags,
        }

        if self.backend == "cloud" and self._backend_impl:
            # Delegate to cloud backend
            uri = self._backend_impl.save(name, artifact, meta_dict)
            
            # Store metadata locally in instance provided we can get size etc?
            # The cloud backend returns URI.
            # We might want to fetch metadata back? Or construct what we can.
            # For now, let's keep _metadata in sync for list() usage if we can.
            # But cloud backend manages its own state possibly?
            # Re-fetching metadata or estimating it.
            # Cloud backend save flow didn't return metadata, just URI.
            # We create a placeholder metadata object.
            self._metadata[name] = ArtifactMetadata(
                name=name,
                version=self.version,
                artifact_type=artifact_type,
                created_at=datetime.utcnow(),
                size_bytes=0, # Unknown without extra call or return from save
                tags=tags,
            )
            return uri
            
        elif self.backend == "local":
            return self._save_local(name, artifact, artifact_type, tags)
        else:
            return self._save_s3(name, artifact, artifact_type, tags)

    def get(self, name: str) -> Any:
        """Load an artifact by name.

        Args:
            name: Artifact name.

        Returns:
            Deserialized artifact.

        Raises:
            KeyError: If artifact not found.
        """
        if self.backend == "cloud" and self._backend_impl:
            # We can pass name directly if relying on context in backend
            return self._backend_impl.load(name)
        elif self.backend == "local":
            return self._load_local(name)
        else:
            return self._load_s3(name)

    def list(self) -> list[ArtifactMetadata]:
        """List all artifacts in this store.

        Returns:
            List of artifact metadata.
        """
        if self.backend == "cloud" and self._backend_impl:
            # Reload metadata from cloud
            self._load_metadata()
        return list(self._metadata.values())
        
    def delete(self, name: str) -> None:
        """Delete an artifact.
        
        Args:
           name: Artifact name to delete
        """
        if self.backend == "cloud" and self._backend_impl:
             self._backend_impl.delete(name)
             if name in self._metadata:
                 del self._metadata[name]
        elif self.backend == "local":
            raise NotImplementedError("Local delete not implemented")
        else:
             raise NotImplementedError("S3 delete not implemented")

    def _save_local(
        self, name: str, artifact: Any, artifact_type: str, tags: dict
    ) -> str:
        """Save artifact to local filesystem."""
        artifact_file = self.artifact_path / f"{name}.pkl"

        with open(artifact_file, "wb") as f:
            pickle.dump(artifact, f)

        metadata = ArtifactMetadata(
            name=name,
            version=self.version,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            size_bytes=artifact_file.stat().st_size,
            tags=tags,
        )
        self._metadata[name] = metadata
        self._save_metadata()

        return str(artifact_file)

    def _load_local(self, name: str) -> Any:
        """Load artifact from local filesystem."""
        artifact_file = self.artifact_path / f"{name}.pkl"
        if not artifact_file.exists():
            raise KeyError(f"Artifact not found: {name}")

        with open(artifact_file, "rb") as f:
            return pickle.load(f)

    def _save_s3(
        self, name: str, artifact: Any, artifact_type: str, tags: dict
    ) -> str:
        """Save artifact to S3."""
        import boto3

        s3 = boto3.client("s3")
        key = f"{self.project}/{self.version}/{name}.pkl"

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            temp_path = f.name

        try:
            s3.upload_file(temp_path, self.s3_bucket, key)
            size = os.path.getsize(temp_path)
        finally:
            os.unlink(temp_path)

        metadata = ArtifactMetadata(
            name=name,
            version=self.version,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            size_bytes=size,
            tags=tags,
        )
        self._metadata[name] = metadata
        self._save_metadata()

        return f"s3://{self.s3_bucket}/{key}"

    def _load_s3(self, name: str) -> Any:
        """Load artifact from S3."""
        import boto3

        s3 = boto3.client("s3")
        key = f"{self.project}/{self.version}/{name}.pkl"

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            s3.download_file(self.s3_bucket, key, f.name)
            temp_path = f.name

        try:
            with open(temp_path, "rb") as f:
                return pickle.load(f)
        finally:
            os.unlink(temp_path)

    def _save_metadata(self) -> None:
        """Save metadata index."""
        if self.backend == "local":
            meta_file = self.artifact_path / "metadata.json"
            data = {k: v.model_dump(mode="json") for k, v in self._metadata.items()}
            meta_file.write_text(json.dumps(data, indent=2, default=str))
        elif self.backend == "cloud":
            pass # Cloud backend manages its own metadata
        else:
            import boto3

            s3 = boto3.client("s3")
            key = f"{self.project}/{self.version}/metadata.json"
            data = {k: v.model_dump(mode="json") for k, v in self._metadata.items()}
            s3.put_object(
                Bucket=self.s3_bucket,
                Key=key,
                Body=json.dumps(data, indent=2, default=str),
            )

    def _load_metadata(self) -> None:
        """Load metadata index."""
        if self.backend == "local":
            meta_file = self.artifact_path / "metadata.json"
            if meta_file.exists():
                data = json.loads(meta_file.read_text())
                self._metadata = {
                    k: ArtifactMetadata.model_validate(v) for k, v in data.items()
                }
        elif self.backend == "cloud":
            if self._backend_impl:
                # We can't easily sync full metadata object from list() URIs directly
                # assuming the backend list returns URIs.
                # Ideally, the backend list would return metadata objects or we fetch them.
                # For now, let's just leave empty or try to list?
                # The task said use existing API which supports list.
                # If we want to populate _metadata, we'd need more info from backend.
                pass
        else:
            import boto3

            s3 = boto3.client("s3")
            key = f"{self.project}/{self.version}/metadata.json"
            try:
                response = s3.get_object(Bucket=self.s3_bucket, Key=key)
                data = json.loads(response["Body"].read())
                self._metadata = {
                    k: ArtifactMetadata.model_validate(v) for k, v in data.items()
                }
            except s3.exceptions.NoSuchKey:
                pass

    def __repr__(self) -> str:
        return f"ArtifactStore({self.project}@{self.version}, backend={self.backend})"
