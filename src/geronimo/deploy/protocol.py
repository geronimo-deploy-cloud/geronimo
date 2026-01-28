"""Deployment target protocol for custom infrastructure."""

from typing import Any, Protocol, runtime_checkable
from enum import Enum
from pydantic import BaseModel


class DeploymentStatus(str, Enum):
    """Status of a deployment."""
    
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"


class DeploymentInfo(BaseModel):
    """Information about a deployment."""
    
    deployment_id: str
    status: DeploymentStatus
    endpoint_url: str | None = None
    message: str | None = None
    resources: dict[str, Any] = {}


@runtime_checkable
class DeploymentTarget(Protocol):
    """Protocol for custom deployment targets.
    
    Implement this protocol to deploy Geronimo models to your own
    infrastructure (internal K8s, custom ECS, on-prem servers, etc.).
    
    Example:
        class InternalK8sTarget(DeploymentTarget):
            def __init__(self, cluster: str, namespace: str):
                self.cluster = cluster
                self.namespace = namespace
            
            def deploy(self, config, artifacts) -> DeploymentInfo:
                # Use Helm, kubectl, or K8s API
                return DeploymentInfo(
                    deployment_id=f"{self.namespace}/{config.project}",
                    status=DeploymentStatus.RUNNING,
                    endpoint_url=f"http://{config.project}.{self.namespace}.svc",
                )
            
            def status(self, deployment_id) -> DeploymentInfo:
                # Query K8s for pod status
                ...
            
            def teardown(self, deployment_id) -> None:
                # Delete K8s resources
                ...
    
    Usage in geronimo.yaml:
        deployment:
          target: custom
          custom_target:
            class: my_company.deploy.InternalK8sTarget
            config:
              cluster: production-eks
              namespace: ml-models
    """
    
    def deploy(self, config: Any, artifacts: dict[str, str]) -> DeploymentInfo:
        """Deploy a model.
        
        Args:
            config: DeploymentConfig or GeronimoConfig
            artifacts: Dict of artifact name -> URI
        
        Returns:
            DeploymentInfo with endpoint URL and status
        """
        ...
    
    def status(self, deployment_id: str) -> DeploymentInfo:
        """Check deployment status.
        
        Args:
            deployment_id: ID returned from deploy()
        
        Returns:
            Current deployment status
        """
        ...
    
    def teardown(self, deployment_id: str) -> None:
        """Remove a deployment.
        
        Args:
            deployment_id: ID of deployment to remove
        """
        ...
    
    def logs(self, deployment_id: str, lines: int = 100) -> list[str]:
        """Get deployment logs.
        
        Args:
            deployment_id: ID of deployment
            lines: Number of log lines to retrieve
        
        Returns:
            List of log lines
        """
        ...
