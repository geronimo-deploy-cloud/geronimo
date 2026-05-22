"""Cloud deployment target implementation."""

import os
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

from geronimo.deploy.config import DeploymentConfig
from geronimo.deploy_cloud.client import GeronimoCloudClient


class GeronimoCloudTarget:
    """Deployment target for Geronimo Cloud."""

    config: DeploymentConfig
    """The deployment configuration."""

    client: GeronimoCloudClient
    """Client for communicating with the cloud API."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = GeronimoCloudClient()

    def deploy(self, component: Optional[str] = None, wait: bool = True) -> Dict[str, Any]:
        """Deploy to Geronimo Cloud.
        
        Args:
            component: Optional component filter (ignored for now, full project deployed).
            wait: Whether to wait for deployment completion.
        
        Returns:
            Deployment result with status and URLs.
        """
        print(f"Deploying project '{self.config.project}' to Geronimo Cloud...")

        # 1. Package Project
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = Path(temp_dir) / "project.zip"
            
            # Custom zip bundling to exclude unnecessary files
            def _should_include(path_str: str) -> bool:
                excludes = [".venv", "__pycache__", ".git", "models", ".geronimo"]
                if any(ex in path_str.split(os.sep) for ex in excludes):
                    return False
                if path_str.endswith(".pyc"):
                    return False
                return True

            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk("."):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Remove leading ./
                        arcname = os.path.relpath(file_path, ".")
                        if _should_include(arcname):
                            zf.write(file_path, arcname)
            
            # 2. Upload and Deploy
            try:
                result = self.client.deploy_project(
                    project_name=self.config.project,
                    config=self.config.model_dump(),  # Serialize config
                    zip_path=zip_path
                )
                
                deployment_id = result["id"]
                print(f"Deployment started: {deployment_id}")
                
                # 3. Poll for completion (if requested)
                if wait:
                    return self._wait_for_deployment(deployment_id)
                else:
                    return result
                
            except Exception as e:
                raise RuntimeError(f"Cloud deployment failed: {e}")

    def _wait_for_deployment(self, deployment_id: str, max_wait_seconds: int = 600) -> Dict[str, Any]:
        """Wait for deployment to complete."""
        print("Waiting for deployment to complete...")
        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait_seconds:
                raise RuntimeError(f"Deployment {deployment_id} timed out after {max_wait_seconds} seconds")
                
            status = self.client.get_status(deployment_id)
            state = status.get("status")
            
            if state == "active":
                return status
            elif state in ("failed", "error", "cancelled"):
                raise RuntimeError(f"Deployment failed/cancelled: {status.get('error', state)}")
            
            time.sleep(5)

    def destroy(self) -> Dict[str, Any]:
        """Destroy cloud resources."""
        print(f"Teardown requested for project '{self.config.project}'")
        try:
            deployment_id = self.client.get_active_deployment(self.config.project)
            if not deployment_id:
                raise RuntimeError(f"No active cloud deployment found for project '{self.config.project}'")
                
            return self.client.teardown(deployment_id)
        except Exception as e:
            # Raise a clear message specifically for CLI handler
            raise RuntimeError(f"Cloud teardown failed: {e}")

    def logs(self, deployment_id: str) -> str:
        """Get logs."""
        return self.client.get_logs(deployment_id)
