"""Geronimo Deploy Cloud Artifact Backend."""

import logging
import os
import pickle
import tempfile
from typing import Any, Optional

from geronimo.artifacts.protocol import ArtifactBackend
from geronimo.deploy_cloud.client import GeronimoCloudClient
from geronimo.deploy_cloud.http_utils import api_client, transfer_client, TRANSFER_TIMEOUT

logger = logging.getLogger(__name__)


class GeronimoDeployCloudArtifactBackend(ArtifactBackend):
    """Artifact backend for Geronimo Deploy Cloud.

    Stores artifacts using the Geronimo Deploy Cloud API with support for
    cross-user access via namespaces.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        version: Optional[str] = None,
        namespace: Optional[str] = None,
        client: Optional[GeronimoCloudClient] = None,
    ):
        """Initialize the cloud backend.

        Args:
            project: Project name (optional context).
            version: Project version (optional context).
            namespace: Optional namespace for cross-user access.
            client: Optional pre-configured client.
        """
        self.project = project
        self.version = version
        self.namespace = namespace
        self.client = client or GeronimoCloudClient()

    def _check_auth(self) -> None:
        """Check if client is authenticated."""
        if not self.client.token:
            raise RuntimeError(
                "Not authenticated. Run 'geronimo auth login' first."
            )

    def _parse_artifact_uri(self, uri: str) -> tuple[str, str, str]:
        """Parse S3 URI or artifact name to extract (project, version, name).
        
        Args:
            uri: S3 URI (s3://bucket/user/project/version/name.pkl) or artifact name
        
        Returns:
            Tuple of (project, version, name)
        
        Raises:
            ValueError: If URI format is invalid or context missing
        """
        if uri.startswith("s3://"):
            # Parse s3://{bucket}/{user_id}/{project}/{version}/{name}.pkl
            parts = uri.replace("s3://", "").split("/")
            if len(parts) >= 5:
                return parts[2], parts[3], parts[4].replace(".pkl", "")
            else:
                raise ValueError(
                    f"Invalid S3 URI format: {uri}. "
                    f"Expected: s3://bucket/user/project/version/name"
                )
        else:
            # Treat as artifact name, use context
            if not (self.project and self.version):
                raise ValueError(
                    "Project and version context required when using artifact name. "
                    f"Got: project={self.project}, version={self.version}"
                )
            return self.project, self.version, uri

    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        """Save an artifact to the cloud.

        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            metadata: Artifact metadata.

        Returns:
            S3 URI of the saved artifact.

        Raises:
            RuntimeError: If not authenticated.
            httpx.HTTPError: If API request fails.
        """
        self._check_auth()

        # 1. Serialize artifact to get actual size
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            f.flush()
            temp_path = f.name
            size_bytes = os.path.getsize(temp_path)

        try:
            # 2. Get upload URL from cloud API
            with api_client(
                self.client.api_url, self.client.headers, operation=f"save/{name}"
            ) as http:
                logger.debug(f"Requesting upload URL for artifact '{name}'")
                
                payload = {
                    "project": self.project,
                    "version": self.version,
                    "name": name,
                    "size_bytes": size_bytes,
                    "metadata": metadata,
                }
                if self.namespace:
                    payload["namespace"] = self.namespace

                resp = http.post("/v1/artifacts/cloud-save", json=payload)
                resp.raise_for_status()
                data = resp.json()
                upload_url = data["upload_url"]
                artifact_id = data["id"]
                s3_uri = data["s3_uri"]

                logger.info(f"Uploading artifact '{name}' to cloud (ID: {artifact_id})")

            # 3. Upload content to S3
            with open(temp_path, "rb") as f:
                with transfer_client(operation=f"upload/{name}") as http:
                    resp = http.put(upload_url, content=f)
                    resp.raise_for_status()

            # 4. Confirm upload
            with api_client(
                self.client.api_url, self.client.headers, operation=f"confirm/{name}"
            ) as http:
                resp = http.post(
                    f"/v1/artifacts/{artifact_id}/confirm",
                    json={"size_bytes": size_bytes},
                )
                resp.raise_for_status()

            logger.info(f"Successfully saved artifact '{name}' to {s3_uri}")
            return s3_uri

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def load(self, uri: str) -> Any:
        """Load an artifact from the cloud.

        Args:
            uri: S3 URI (s3://bucket/user/project/version/name) or artifact name.

        Returns:
            Deserialized artifact.

        Raises:
            ValueError: If URI cannot be parsed or context is missing.
            RuntimeError: If not authenticated.
            httpx.HTTPError: If API request fails.
        """
        self._check_auth()

        project, version, name = self._parse_artifact_uri(uri)

        # 1. Get download URL
        with api_client(
            self.client.api_url, self.client.headers, operation=f"load/{name}"
        ) as http:
            logger.debug(f"Requesting download URL for '{project}/{version}/{name}'")
            
            payload = {
                "project": project,
                "version": version,
                "name": name,
            }
            if self.namespace:
                payload["namespace"] = self.namespace

            resp = http.post("/v1/artifacts/cloud-load", json=payload)
            resp.raise_for_status()
            download_url = resp.json()["download_url"]

        # 2. Download and deserialize
        logger.info(f"Loading artifact '{name}' from cloud")
        with transfer_client(operation=f"download/{name}") as http:
            resp = http.get(download_url)
            resp.raise_for_status()
            return pickle.loads(resp.content)

    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List artifacts.

        Args:
            prefix: Optional name prefix to filter results.

        Returns:
            List of artifact S3 URIs.

        Raises:
            RuntimeError: If not authenticated.
            httpx.HTTPError: If API request fails.
        """
        self._check_auth()

        params = {}
        if self.project:
            params["project"] = self.project
        if self.version:
            params["version"] = self.version
        if self.namespace:
            params["namespace"] = self.namespace
        if prefix:
            params["prefix"] = prefix

        with api_client(
            self.client.api_url, self.client.headers, operation="list"
        ) as http:
            logger.debug(f"Listing artifacts with params: {params}")
            resp = http.get("/v1/artifacts", params=params)
            resp.raise_for_status()
            artifacts = resp.json().get("artifacts", [])
            return [a["s3_uri"] for a in artifacts]

    def delete(self, uri: str) -> None:
        """Delete an artifact.

        Args:
            uri: S3 URI or artifact name.

        Raises:
            ValueError: If URI cannot be parsed or context is missing.
            RuntimeError: If not authenticated.
            httpx.HTTPError: If API request fails.
        """
        self._check_auth()

        project, version, name = self._parse_artifact_uri(uri)

        with api_client(
            self.client.api_url, self.client.headers, operation=f"delete/{name}"
        ) as http:
            # Find artifact ID by searching
            search_params = {
                "project": project,
                "version": version,
                "name": name
            }
            if self.namespace:
                search_params["namespace"] = self.namespace

            logger.debug(f"Searching for artifact to delete: {search_params}")
            resp = http.get("/v1/artifacts", params=search_params)
            resp.raise_for_status()
            matches = resp.json().get("artifacts", [])

            if not matches:
                logger.warning(f"Artifact '{name}' not found, skipping delete")
                return

            # Delete all matches
            for artifact in matches:
                artifact_id = artifact["id"]
                logger.info(f"Deleting artifact '{name}' (ID: {artifact_id})")
                resp = http.delete(f"/v1/artifacts/{artifact_id}")
                resp.raise_for_status()
