"""Deployment configuration."""

from typing import Literal, Optional
from pydantic import BaseModel


class ArtifactStorageConfig(BaseModel):
    """Configuration for artifact storage infrastructure."""
    
    bucket_prefix: str = "geronimo-artifacts"
    retention_days: int = 90
    versioning: bool = True


class ServingConfig(BaseModel):
    """Configuration for serving infrastructure."""
    
    cpu: int = 1
    memory: int = 2048  # MB
    min_replicas: int = 1
    max_replicas: int = 10
    port: int = 8000


class BatchConfig(BaseModel):
    """Configuration for batch pipeline infrastructure."""
    
    schedule: str = "0 6 * * *"  # Cron expression
    timeout_minutes: int = 60


class DeploymentConfig(BaseModel):
    """Complete deployment configuration.
    
    Example:
        config = DeploymentConfig(
            project="iris-classifier",
            target="aws",
            region="us-west-2",
        )
    """
    
    project: str
    version: str = "1.0.0"
    target: Literal["aws", "gcp", "azure"] = "aws"
    region: str = "us-east-1"
    
    # Component configs
    artifacts: ArtifactStorageConfig = ArtifactStorageConfig()
    serving: Optional[ServingConfig] = None
    batch: Optional[BatchConfig] = None
    
    # Pulumi settings
    stack_name: str = "dev"
