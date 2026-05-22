"""Local filesystem artifact backend."""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from geronimo.artifacts.protocol import ArtifactBackend

logger = logging.getLogger(__name__)


class LocalArtifactBackend(ArtifactBackend):
    """Artifact backend for local filesystem storage.
    
    Stores artifacts as pickle files with JSON metadata index.
    """

    project: str
    """The project name."""

    version: str
    """The version string."""

    base_path: Path
    """Base directory for artifacts."""

    artifact_path: Path
    """Directory for the specific project version."""

    _metadata_file: Path
    """Path to the metadata index file."""
    
    def __init__(
        self,
        project: str,
        version: str,
        base_path: Optional[str] = None,
    ):
        """Initialize local backend.
        
        Args:
            project: Project name.
            version: Version string.
            base_path: Base directory for artifacts. 
                       Defaults to ~/.geronimo/artifacts.
        """
        self.project = project
        self.version = version
        self.base_path = Path(base_path or os.path.expanduser("~/.geronimo/artifacts"))
        self.artifact_path = self.base_path / project / version
        self.artifact_path.mkdir(parents=True, exist_ok=True)
        self._metadata_file = self.artifact_path / "metadata.json"
    
    def save(self, name: str, artifact: Any, metadata: dict) -> tuple[str, int]:
        """Save an artifact to local filesystem.
        
        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            metadata: Artifact metadata dict.
            
        Returns:
            Tuple of (Path to saved artifact file, file size in bytes).
        """
        artifact_file = self.artifact_path / f"{name}.pkl"
        
        with open(artifact_file, "wb") as f:
            pickle.dump(artifact, f)
        
        logger.debug(f"Saved artifact '{name}' to {artifact_file}")
        
        size_bytes = artifact_file.stat().st_size
        
        # Update metadata index
        self._save_to_metadata_index(name, size_bytes, metadata)
        
        return str(artifact_file), size_bytes
    
    def load(self, uri: str) -> Any:
        """Load an artifact by name or path.
        
        Args:
            uri: Artifact name or file path.
            
        Returns:
            Deserialized artifact.
            
        Raises:
            KeyError: If artifact not found.
        """
        # If path, use directly; otherwise treat as name
        if os.path.isabs(uri) or os.path.exists(uri):
            artifact_file = Path(uri)
        else:
            artifact_file = self.artifact_path / f"{uri}.pkl"
        
        if not artifact_file.exists():
            raise KeyError(f"Artifact not found: {uri}")
        
        logger.debug(f"Loading artifact from {artifact_file}")
        
        with open(artifact_file, "rb") as f:
            return pickle.load(f)
    
    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List artifacts in the store.
        
        Args:
            prefix: Optional name prefix to filter.
            
        Returns:
            List of artifact file paths.
        """
        paths = []
        for pkl_file in self.artifact_path.glob("*.pkl"):
            name = pkl_file.stem
            if prefix is None or name.startswith(prefix):
                paths.append(str(pkl_file))
        return paths
    
    def delete(self, uri: str) -> None:
        """Delete an artifact.
        
        Args:
            uri: Artifact name or file path.
        """
        # If path, use directly; otherwise treat as name
        if os.path.isabs(uri) or os.path.exists(uri):
            artifact_file = Path(uri)
            name = artifact_file.stem
        else:
            name = uri
            artifact_file = self.artifact_path / f"{uri}.pkl"
        
        if artifact_file.exists():
            artifact_file.unlink()
            logger.info(f"Deleted artifact '{name}'")
            self._remove_from_metadata_index(name)
        else:
            logger.warning(f"Artifact not found for deletion: {uri}")
    
    def _save_to_metadata_index(self, name: str, size_bytes: int, metadata: dict) -> None:
        """Add artifact to metadata index."""
        index = self._load_metadata_index()
        
        index[name] = {
            "name": name,
            "version": self.version,
            "artifact_type": metadata.get("artifact_type", "unknown"),
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": size_bytes,
            "tags": metadata.get("tags", {}),
        }
        
        self._metadata_file.write_text(json.dumps(index, indent=2))
    
    def _remove_from_metadata_index(self, name: str) -> None:
        """Remove artifact from metadata index."""
        index = self._load_metadata_index()
        if name in index:
            del index[name]
            self._metadata_file.write_text(json.dumps(index, indent=2))
    
    def _load_metadata_index(self) -> dict:
        """Load metadata index from file."""
        if self._metadata_file.exists():
            return json.loads(self._metadata_file.read_text())
        return {}
    
    def get_metadata_index(self) -> dict:
        """Get the full metadata index for this store.
        
        Returns:
            Dict mapping artifact names to metadata dicts.
        """
        return self._load_metadata_index()
