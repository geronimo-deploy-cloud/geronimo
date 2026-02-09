"""Test fixtures for Geronimo projects.

Provides utilities for creating test projects and mock clients
that can be used by both OSS tests and deploy-cloud integration tests.
"""

import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock


def create_test_project(
    name: str = "test-project",
    temp_dir: Optional[Path] = None,
    template: str = "realtime",
    framework: str = "sklearn",
) -> Path:
    """Create a test project using Geronimo's ProjectGenerator.
    
    Uses the same generation logic as `geronimo init`, ensuring
    test projects match production project structure.
    
    Args:
        name: Project name (default: "test-project")
        temp_dir: Directory to create project in. If None, creates temp directory.
        template: Project template - "realtime", "batch", or "both"
        framework: ML framework - "sklearn", "pytorch", "tensorflow"
        
    Returns:
        Path to the created project directory
        
    Example:
        >>> from geronimo.testing import create_test_project
        >>> project_path = create_test_project("my-test")
        >>> assert (project_path / "geronimo.yaml").exists()
    """
    from geronimo.generators.project import ProjectGenerator
    
    output_dir = temp_dir or Path(tempfile.mkdtemp())
    
    generator = ProjectGenerator(
        project_name=name,
        framework=framework,
        output_dir=str(output_dir),
        template=template,
    )
    
    return generator.generate()


def create_mock_cloud_client() -> MagicMock:
    """Create a pre-configured mock GeronimoCloudClient.
    
    Returns a MagicMock with standard return values matching
    the deploy-cloud API expectations.
    
    Returns:
        Configured MagicMock for GeronimoCloudClient
        
    Example:
        >>> client = create_mock_cloud_client()
        >>> result = client.deploy_project("test", {}, Path("test.zip"))
        >>> assert result["id"] == "deploy-123"
    """
    client = MagicMock()
    client.api_url = "https://api.test.geronimo.cloud"
    client.token = "test-token"
    client.headers = {"Authorization": "Bearer test-token"}
    
    # Standard deployment flow responses
    client.login.return_value = {
        "email": "test@example.com",
        "org": "test-org",
    }
    
    client.deploy_project.return_value = {
        "id": "deploy-123",
        "status": "pending",
        "upload_url": "https://storage.geronimo.cloud/upload/deploy-123",
    }
    
    client.get_status.return_value = {
        "id": "deploy-123",
        "status": "running",
        "endpoint_url": "https://deploy-123.geronimo.cloud",
    }
    
    client.sync_keys.return_value = {
        "synced": 1,
        "skipped": 0,
    }
    
    client.teardown.return_value = {
        "id": "deploy-123",
        "status": "terminated",
    }
    
    return client


def create_mock_http_client() -> MagicMock:
    """Create a pre-configured mock httpx.Client.
    
    Returns a MagicMock with standard HTTP response behavior
    for testing API interactions.
    
    Returns:
        Configured MagicMock for httpx.Client
        
    Example:
        >>> client = create_mock_http_client()
        >>> client.post.return_value.json.return_value = {"id": "123"}
        >>> resp = client.post("/api/test")
        >>> assert resp.json()["id"] == "123"
    """
    client = MagicMock()
    
    # Configure default response behavior
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {}
    response.text = ""
    response.content = b""
    response.raise_for_status = MagicMock()
    
    client.get.return_value = response
    client.post.return_value = response
    client.put.return_value = response
    client.delete.return_value = response
    
    # Context manager support
    client.__enter__ = MagicMock(return_value=client)
    client.__exit__ = MagicMock(return_value=False)
    
    return client
