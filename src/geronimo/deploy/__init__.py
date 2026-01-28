"""Geronimo Deploy Module.

Provides infrastructure deployment using Pulumi Automation API.
Pulumi is an optional dependency - install with: pip install geronimo[pulumi]
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
