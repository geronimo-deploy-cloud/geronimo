"""Geronimo Cloud Artifact Backend."""

import os
import pickle
import tempfile
import re
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from geronimo.artifacts.protocol import ArtifactBackend
from geronimo.cloud.client import GeronimoCloudClient


class GeronimoCloudArtifactBackend(ArtifactBackend):
    """Artifact backend for Geronimo Cloud.

    Stores artifacts using the Geronimo Cloud API.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        version: Optional[str] = None,
        client: Optional[GeronimoCloudClient] = None,
    ):
        """Initialize the cloud backend.

        Args:
            project: Project name (optional context).
            version: Project version (optional context).
            client: Optional pre-configured client.
        """
        self.project = project
        self.version = version
        self.client = client or GeronimoCloudClient()

    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        """Save an artifact to the cloud.

        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            metadata: Artifact metadata.

        Returns:
            S3 URI of the saved artifact.
        """
        # 1. Serialize artifact
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            temp_path = f.name
            size_bytes = f.tell()

        try:
            # 2. Get upload URL
            with httpx.Client(
                base_url=self.client.api_url, headers=self.client.headers
            ) as http:
                resp = http.post(
                    "/artifacts/cloud-save",
                    json={
                        "project": self.project,
                        "version": self.version,
                        "name": name,
                        "size_bytes": size_bytes,
                        "metadata": metadata,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                upload_url = data["upload_url"]
                artifact_id = data["id"]
                s3_uri = data["s3_uri"]

            # 3. Upload content
            with open(temp_path, "rb") as f:
                with httpx.Client() as http:
                    http.put(upload_url, content=f).raise_for_status()

            # 4. Confirm upload
            with httpx.Client(
                base_url=self.client.api_url, headers=self.client.headers
            ) as http:
                http.post(
                    f"/artifacts/{artifact_id}/confirm",
                    json={"size_bytes": size_bytes},
                ).raise_for_status()

            return s3_uri

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load(self, uri: str) -> Any:
        """Load an artifact from the cloud.

        Args:
            uri: S3 URI or artifact name.

        Returns:
            Deserialized artifact.
        """
        # Determine params from URI or context
        if uri.startswith("s3://"):
            # Try to parse properties from s3 uri:
            # s3://{bucket}/{user_id}/{project}/{version}/{name}
            parts = uri.replace("s3://", "").split("/")
            if len(parts) >= 5:
                # Assuming standard structure
                project = parts[2]
                version = parts[3]
                name = parts[4].replace(".pkl", "")
            else:
                 # If parsing failed, fallback to context if available
                 if not (self.project and self.version):
                     raise ValueError(f"Cannot parse URI '{uri}' and no context provided.")
                 # Treat uri as name if it doesn't look like a valid path,
                 # but here it starts with s3:// so likely we should just fail or retry?
                 # Actually, let's treat it as if we just need to call cloud-load with context if parsing fails?
                 # But sticking to plan: fallback means using s3_uri to extract.
                 # If extraction fails, we might just try using the full URI as name? Unlikely unique.
                 raise ValueError(f"Invalid cloud artifact URI: {uri}")
        else:
            # Treat as name if context exists
            if not (self.project and self.version):
                raise ValueError("Project and version context required when loading by name.")
            name = uri
            project = self.project
            version = self.version

        # 1. Get download URL
        with httpx.Client(
            base_url=self.client.api_url, headers=self.client.headers
        ) as http:
            resp = http.post(
                "/artifacts/cloud-load",
                json={
                    "project": project,
                    "version": version,
                    "name": name,
                },
            )
            resp.raise_for_status()
            download_url = resp.json()["download_url"]

        # 2. Download and deserialize
        with httpx.Client() as http:
            resp = http.get(download_url)
            resp.raise_for_status()
            return pickle.loads(resp.content)

    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List artifacts.

        Args:
            prefix: Optional name prefix.

        Returns:
            List of artifact URIs.
        """
        params = {}
        if self.project:
            params["project"] = self.project
        if self.version:
            params["version"] = self.version
        if prefix:
            params["prefix"] = prefix

        with httpx.Client(
            base_url=self.client.api_url, headers=self.client.headers
        ) as http:
            resp = http.get("/artifacts", params=params)
            resp.raise_for_status()
            artifacts = resp.json().get("artifacts", [])
            return [a["s3_uri"] for a in artifacts]

    def delete(self, uri: str) -> None:
        """Delete an artifact.

        Args:
            uri: S3 URI of the artifact.
        """
        # We need the ID to delete.
        # Option 1: Parse ID from specific URI/metadata if we had it.
        # Option 2: Search by properties derived from URI.

        # Let's search by properties
        if not uri.startswith("s3://"):
             # If passed as name
            if not (self.project and self.version):
                 raise ValueError("Context required to delete by name")
            name = uri
            project = self.project
            version = self.version
        else:
             parts = uri.replace("s3://", "").split("/")
             if len(parts) >= 5:
                project = parts[2]
                version = parts[3]
                name = parts[4].replace(".pkl", "")
             else:
                 raise ValueError(f"Cannot parse URI for deletion: {uri}")

        # Find the artifact ID
        with httpx.Client(
            base_url=self.client.api_url, headers=self.client.headers
        ) as http:
            # List to find ID
            resp = http.get(
                "/artifacts",
                params={"project": project, "version": version, "name": name},
            )
            resp.raise_for_status()
            matches = resp.json().get("artifacts", [])
            
            if not matches:
                 # Already gone or doesn't exist
                 return
            
            # Delete match(es)
            for artifact in matches:
                 http.delete(f"/artifacts/{artifact['id']}").raise_for_status()
