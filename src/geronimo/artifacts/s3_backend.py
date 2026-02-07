"""S3 artifact backend."""

import json
import logging
import os
import pickle
import tempfile
from datetime import datetime
from typing import Any, Optional

from geronimo.artifacts.protocol import ArtifactBackend

logger = logging.getLogger(__name__)


class S3ArtifactBackend(ArtifactBackend):
    """Artifact backend for S3 storage.
    
    Stores artifacts as pickle files in S3 with JSON metadata index.
    """
    
    def __init__(
        self,
        project: str,
        version: str,
        bucket: Optional[str] = None,
    ):
        """Initialize S3 backend.
        
        Args:
            project: Project name.
            version: Version string.
            bucket: S3 bucket name. Defaults to GERONIMO_ARTIFACT_BUCKET env var.
        """
        self.project = project
        self.version = version
        self.bucket = bucket or os.getenv("GERONIMO_ARTIFACT_BUCKET", "ml-artifacts")
        self._prefix = f"{project}/{version}"
    
    def _get_s3_client(self):
        """Get boto3 S3 client (lazy import)."""
        import boto3
        return boto3.client("s3")
    
    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        """Save an artifact to S3.
        
        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            metadata: Artifact metadata dict.
            
        Returns:
            S3 URI of saved artifact.
        """
        s3 = self._get_s3_client()
        key = f"{self._prefix}/{name}.pkl"
        
        # Serialize to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            temp_path = f.name
        
        try:
            s3.upload_file(temp_path, self.bucket, key)
            size_bytes = os.path.getsize(temp_path)
            logger.debug(f"Uploaded artifact '{name}' to s3://{self.bucket}/{key}")
        finally:
            os.unlink(temp_path)
        
        # Update metadata index in S3
        self._save_to_metadata_index(name, size_bytes, metadata)
        
        return f"s3://{self.bucket}/{key}"
    
    def load(self, uri: str) -> Any:
        """Load an artifact by name or S3 URI.
        
        Args:
            uri: Artifact name or S3 URI.
            
        Returns:
            Deserialized artifact.
            
        Raises:
            KeyError: If artifact not found.
        """
        s3 = self._get_s3_client()
        
        # Parse URI or construct from name
        if uri.startswith("s3://"):
            # Parse s3://bucket/path/to/artifact.pkl
            path = uri.replace("s3://", "")
            parts = path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.bucket
            key = f"{self._prefix}/{uri}.pkl"
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name
        
        try:
            s3.download_file(bucket, key, temp_path)
            logger.debug(f"Downloaded artifact from s3://{bucket}/{key}")
            
            with open(temp_path, "rb") as f:
                return pickle.load(f)
        except s3.exceptions.NoSuchKey:
            raise KeyError(f"Artifact not found: {uri}")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List artifacts in the store.
        
        Args:
            prefix: Optional name prefix to filter.
            
        Returns:
            List of S3 URIs.
        """
        s3 = self._get_s3_client()
        
        search_prefix = self._prefix
        if prefix:
            search_prefix = f"{search_prefix}/{prefix}"
        
        uris = []
        paginator = s3.get_paginator("list_objects_v2")
        
        for page in paginator.paginate(Bucket=self.bucket, Prefix=search_prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".pkl"):
                    uris.append(f"s3://{self.bucket}/{key}")
        
        return uris
    
    def delete(self, uri: str) -> None:
        """Delete an artifact.
        
        Args:
            uri: Artifact name or S3 URI.
        """
        s3 = self._get_s3_client()
        
        # Parse URI or construct from name
        if uri.startswith("s3://"):
            path = uri.replace("s3://", "")
            parts = path.split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ""
            # Extract name from key for metadata
            name = key.split("/")[-1].replace(".pkl", "")
        else:
            name = uri
            bucket = self.bucket
            key = f"{self._prefix}/{uri}.pkl"
        
        try:
            s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"Deleted artifact from s3://{bucket}/{key}")
            self._remove_from_metadata_index(name)
        except Exception as e:
            logger.warning(f"Failed to delete artifact: {e}")
    
    def _save_to_metadata_index(self, name: str, size_bytes: int, metadata: dict) -> None:
        """Add artifact to metadata index in S3."""
        index = self._load_metadata_index()
        
        index[name] = {
            "name": name,
            "version": self.version,
            "artifact_type": metadata.get("artifact_type", "unknown"),
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": size_bytes,
            "tags": metadata.get("tags", {}),
        }
        
        s3 = self._get_s3_client()
        key = f"{self._prefix}/metadata.json"
        s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(index, indent=2),
            ContentType="application/json",
        )
    
    def _remove_from_metadata_index(self, name: str) -> None:
        """Remove artifact from metadata index."""
        index = self._load_metadata_index()
        if name in index:
            del index[name]
            s3 = self._get_s3_client()
            key = f"{self._prefix}/metadata.json"
            s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(index, indent=2),
                ContentType="application/json",
            )
    
    def _load_metadata_index(self) -> dict:
        """Load metadata index from S3."""
        s3 = self._get_s3_client()
        key = f"{self._prefix}/metadata.json"
        
        try:
            response = s3.get_object(Bucket=self.bucket, Key=key)
            return json.loads(response["Body"].read())
        except Exception:
            return {}
    
    def get_metadata_index(self) -> dict:
        """Get the full metadata index for this store.
        
        Returns:
            Dict mapping artifact names to metadata dicts.
        """
        return self._load_metadata_index()
