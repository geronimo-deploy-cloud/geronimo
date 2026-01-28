"""Artifact storage backend protocol for custom implementations."""

from typing import Any, Protocol, runtime_checkable
from datetime import datetime


@runtime_checkable
class ArtifactBackend(Protocol):
    """Protocol for custom artifact storage backends.
    
    Implement this protocol to use your own storage system
    (internal object store, custom S3, etc.) with Geronimo.
    
    Example:
        class MyCompanyStore(ArtifactBackend):
            def save(self, name, artifact, metadata):
                return self._internal_api.upload(artifact)
            
            def load(self, uri):
                return self._internal_api.download(uri)
            
            def list(self, prefix):
                return self._internal_api.list_objects(prefix)
            
            def delete(self, uri):
                self._internal_api.delete(uri)
    
    Usage:
        from geronimo.artifacts import ArtifactStore
        
        custom_backend = MyCompanyStore(config=my_config)
        store = ArtifactStore(
            project="my-model",
            version="1.0.0",
            backend=custom_backend,
        )
    """
    
    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        """Save an artifact.
        
        Args:
            name: Artifact name (e.g., "model", "encoder")
            artifact: Serialized artifact bytes
            metadata: Artifact metadata dict
        
        Returns:
            URI or path where artifact was saved
        """
        ...
    
    def load(self, uri: str) -> Any:
        """Load an artifact by URI.
        
        Args:
            uri: URI returned from save()
        
        Returns:
            Deserialized artifact
        """
        ...
    
    def list(self, prefix: str) -> list[str]:
        """List artifacts matching prefix.
        
        Args:
            prefix: Prefix to filter artifacts
        
        Returns:
            List of artifact URIs
        """
        ...
    
    def delete(self, uri: str) -> None:
        """Delete an artifact.
        
        Args:
            uri: URI of artifact to delete
        """
        ...
