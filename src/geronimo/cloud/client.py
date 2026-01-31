"""Client for Geronimo Cloud API."""

import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from geronimo import __version__


class GeronimoCloudClient:
    """Client for interacting with Geronimo Cloud API."""

    DEFAULT_API_URL = "https://api.geronimo.dev/v1"

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
        with httpx.Client(base_url=self.api_url) as client:
            response = client.get(
                "/auth/verify",
                headers={"Authorization": f"Bearer {token}"}
            )
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
        with httpx.Client(base_url=self.api_url, headers=self.headers) as client:
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
            httpx.put(upload_url, content=f)
            
        # 3. Trigger build/deploy
        with httpx.Client(base_url=self.api_url, headers=self.headers) as client:
            resp = client.post(f"/deployments/{deployment_id}/start")
            resp.raise_for_status()
            return resp.json()

    def get_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment status."""
        with httpx.Client(base_url=self.api_url, headers=self.headers) as client:
            resp = client.get(f"/deployments/{deployment_id}")
            resp.raise_for_status()
            return resp.json()

    def get_logs(self, deployment_id: str) -> str:
        """Get build/runtime logs."""
        with httpx.Client(base_url=self.api_url, headers=self.headers) as client:
            resp = client.get(f"/deployments/{deployment_id}/logs")
            resp.raise_for_status()
            return resp.text

    def teardown(self, deployment_id: str) -> Dict[str, Any]:
        """Teardown a deployment."""
        with httpx.Client(base_url=self.api_url, headers=self.headers) as client:
            resp = client.delete(f"/deployments/{deployment_id}")
            resp.raise_for_status()
            return resp.json()
