"""Cloud deployment target implementation."""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional

from geronimo.deploy.config import DeploymentConfig
from geronimo.cloud.client import GeronimoCloudClient


class GeronimoCloudTarget:
    """Deployment target for Geronimo Cloud."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.client = GeronimoCloudClient()

    def deploy(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Deploy to Geronimo Cloud.
        
        Args:
            component: Optional component filter (ignored for now, full project deployed).
        
        Returns:
            Deployment result with status and URLs.
        """
        print(f"Deploying project '{self.config.project}' to Geronimo Cloud...")

        # 1. Package Project
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = shutil.make_archive(
                base_name=f"{temp_dir}/project",
                format="zip",
                root_dir=".",  # Assuming running from project root
                base_dir="."   
            )
            
            # 2. Upload and Deploy
            try:
                result = self.client.deploy_project(
                    project_name=self.config.project,
                    config=self.config.model_dump(),  # Serialize config
                    zip_path=Path(archive_path)
                )
                
                deployment_id = result["id"]
                print(f"Deployment started: {deployment_id}")
                
                # 3. Poll for completion (simple implementation)
                return self._wait_for_deployment(deployment_id)
                
            except Exception as e:
                raise RuntimeError(f"Cloud deployment failed: {e}")

    def _wait_for_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """Wait for deployment to complete."""
        print("Waiting for deployment to complete...")
        while True:
            status = self.client.get_status(deployment_id)
            state = status.get("status")
            
            if state == "active":
                return status
            elif state in ("failed", "error"):
                raise RuntimeError(f"Deployment failed: {status.get('error')}")
            
            time.sleep(5)

    def destroy(self) -> Dict[str, Any]:
        """Destroy cloud resources."""
        # In a real scenario, we'd need to know which deployment ID to destroy.
        # For now, we might need to look it up or assume the user provides it.
        # This is a simplification.
        print(f"Teardown requested for project '{self.config.project}'")
        # TODO: Implement lookup of active deployment for this project/stack
        raise NotImplementedError("Teardown not yet fully implemented for cloud target without ID lookup.")

    def logs(self, deployment_id: str) -> str:
        """Get logs."""
        return self.client.get_logs(deployment_id)
