"""Deployment targets with runtime Pulumi detection."""

from typing import Callable
from geronimo.deploy.config import DeploymentConfig


class PulumiNotInstalledError(Exception):
    """Raised when Pulumi is required but not installed."""
    
    def __init__(self):
        super().__init__(
            "Pulumi is not installed. Install with: pip install geronimo[pulumi]\n"
            "Or generate static IaC files: geronimo generate --target terraform"
        )


def _check_pulumi_available() -> bool:
    """Check if Pulumi is available."""
    try:
        import pulumi  # noqa: F401
        from pulumi import automation  # noqa: F401
        return True
    except ImportError:
        return False


def get_available_targets() -> list[str]:
    """Get list of available deployment targets.
    
    Returns:
        List of target names (aws, gcp, azure)
    """
    return ["aws", "gcp", "azure"]


def deploy(config: DeploymentConfig, component: str | None = None) -> dict:
    """Deploy infrastructure using Pulumi.
    
    Args:
        config: Deployment configuration
        component: Optional component to deploy (artifacts, serving, batch)
                   If None, deploys all configured components
    
    Returns:
        Dict with deployment outputs (URLs, resource IDs, etc.)
    
    Raises:
        PulumiNotInstalledError: If Pulumi is not installed
    """
    if not _check_pulumi_available():
        raise PulumiNotInstalledError()
    
    # Import Pulumi providers dynamically
    if config.target == "aws":
        from geronimo.deploy.providers.aws import deploy_aws
        return deploy_aws(config, component)
    elif config.target == "gcp":
        from geronimo.deploy.providers.gcp import deploy_gcp
        return deploy_gcp(config, component)
    elif config.target == "azure":
        from geronimo.deploy.providers.azure import deploy_azure
        return deploy_azure(config, component)
    else:
        raise ValueError(f"Unknown target: {config.target}")


def destroy(config: DeploymentConfig) -> dict:
    """Destroy deployed infrastructure.
    
    Args:
        config: Deployment configuration
    
    Returns:
        Dict with destruction summary
    
    Raises:
        PulumiNotInstalledError: If Pulumi is not installed
    """
    if not _check_pulumi_available():
        raise PulumiNotInstalledError()
    
    if config.target == "aws":
        from geronimo.deploy.providers.aws import destroy_aws
        return destroy_aws(config)
    elif config.target == "gcp":
        from geronimo.deploy.providers.gcp import destroy_gcp
        return destroy_gcp(config)
    elif config.target == "azure":
        from geronimo.deploy.providers.azure import destroy_azure
        return destroy_azure(config)
    else:
        raise ValueError(f"Unknown target: {config.target}")

