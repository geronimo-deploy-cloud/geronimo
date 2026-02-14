"""Client for Geronimo Cloud API."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from geronimo import __version__
from geronimo.deploy_cloud.http_utils import api_client, transfer_client


class GeronimoCloudClient:
    """Client for interacting with Geronimo Cloud API."""

    DEFAULT_API_URL = "https://api.geronimo.dev/v1"
    """Default base URL for the API."""

    api_url: str
    """The API URL for Geronimo Cloud."""

    token: Optional[str]
    """The authentication token."""

    headers: Dict[str, str]
    """HTTP headers for API requests."""

    def __init__(self, api_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize the cloud client.
        
        Args:
            api_url: Optional API URL override.
            token: Optional auth token. If not provided, tries to load from credentials file.
        """
        self.api_url = api_url or os.getenv("GERONIMO_API_URL", self.DEFAULT_API_URL)
        self.token = token or self._load_token()
        
        self.headers = {
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "User-Agent": f"geronimo-cli/{__version__}",
            "Content-Type": "application/json",
        }
        
    def _load_token(self) -> Optional[str]:
        """Load token from credentials file."""
        creds_path = Path.home() / ".geronimo" / "credentials"
        if creds_path.exists():
            try:
                data = json.loads(creds_path.read_text())
                return data.get("token")
            except Exception:
                return None
        return None

    def login(self, token: str) -> Dict[str, Any]:
        """Verify token and save credentials."""
        # Validate token with API
        with api_client(self.api_url, {"Authorization": f"Bearer {token}"}, operation="login") as client:
            response = client.get("/auth/verify")
            response.raise_for_status()
            user_data = response.json()
            
        # Save to file
        creds_dir = Path.home() / ".geronimo"
        creds_dir.mkdir(parents=True, exist_ok=True)
        (creds_dir / "credentials").write_text(json.dumps({"token": token}))
        
        # Update current instance
        self.token = token
        self.headers["Authorization"] = f"Bearer {token}"
        
        return user_data

    def deploy_project(self, project_name: str, config: Dict[str, Any], zip_path: Path) -> Dict[str, Any]:
        """Deploy a project to the cloud.
        
        Args:
            project_name: Name of the project.
            config: Full deployment configuration.
            zip_path: Path to the zipped project artifacts.
        """
        if not self.token:
            raise RuntimeError("Not authenticated. Run 'geronimo auth login' first.")

        # 1. Create deployment record
        with api_client(self.api_url, self.headers, operation="create_deployment") as client:
            resp = client.post("/deployments", json={
                "project": project_name,
                "config": config
            })
            resp.raise_for_status()
            deployment = resp.json()
            upload_url = deployment["upload_url"]
            deployment_id = deployment["id"]

        # 2. Upload artifacts
        with open(zip_path, "rb") as f:
            with transfer_client(operation="upload_deployment") as client:
                client.put(upload_url, content=f)
            
        # 3. Trigger build/deploy
        with api_client(self.api_url, self.headers, operation="start_deployment") as client:
            resp = client.post(f"/deployments/{deployment_id}/start")
            resp.raise_for_status()
            return resp.json()

    def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        with api_client(self.api_url, self.headers, operation="get_status") as client:
            resp = client.get(f"/deployments/{deployment_id}")
            resp.raise_for_status()
            return resp.json()

    def get_logs(self, deployment_id: str) -> str:
        """Get build/runtime logs."""
        with api_client(self.api_url, self.headers, operation="get_logs") as client:
            resp = client.get(f"/deployments/{deployment_id}/logs")
            resp.raise_for_status()
            return resp.text

    def teardown(self, deployment_id: str) -> Dict[str, Any]:
        """Teardown a deployment."""
        with api_client(self.api_url, self.headers, operation="teardown") as client:
            resp = client.delete(f"/deployments/{deployment_id}")
            resp.raise_for_status()
            return resp.json()

    def sync_keys(self, keys: list[Dict[str, Any]]) -> Dict[str, Any]:
        """Sync local API keys to Geronimo Cloud.
        
        Args:
            keys: List of key dictionaries from APIKey.to_dict().
            
        Returns:
            Response with synced/skipped counts.
            
        Raises:
            RuntimeError: If not authenticated.
            httpx.HTTPStatusError: If API request fails.
        """
        if not self.token:
            raise RuntimeError("Not authenticated. Run 'geronimo auth login' first.")
        
        with api_client(self.api_url, self.headers, operation="sync_keys") as client:
            resp = client.post("/inference-keys/sync", json={"keys": keys})
            resp.raise_for_status()
            return resp.json()
