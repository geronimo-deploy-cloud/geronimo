"""Geronimo Deploy Module.

This module provides the necessary tooling to deploy infrastructure and applications
to cloud providers using Infrastructure as Code (IaC). It primarily integrates with
Pulumi's Automation API to programmatically manage cloud resources.

It allows you to:
- Define infrastructure resources (S3 buckets, EC2 instances, SageMaker endpoints)
- Deploy entire stacks with a single Python command
- Manage deployment state and targeted environments (dev/stage/prod)
- Destroy resources when no longer needed

Note: Pulumi is an optional dependency. Install it with `pip install geronimo[pulumi]`.
"""

from geronimo.deploy.config import DeploymentConfig
from geronimo.deploy.targets import deploy, destroy, get_available_targets
from geronimo.deploy.protocol import DeploymentTarget, DeploymentInfo, DeploymentStatus

__all__ = [
    "DeploymentConfig",
    "deploy",
    "destroy",
    "get_available_targets",
    "DeploymentTarget",
    "DeploymentInfo",
    "DeploymentStatus",
]
