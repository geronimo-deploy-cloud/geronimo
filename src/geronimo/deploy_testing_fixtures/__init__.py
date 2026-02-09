"""Geronimo Deploy Testing Fixtures.

Provides reusable test fixtures and utilities for testing Geronimo projects
and integrating with deploy-cloud integration tests.

Example:
    from geronimo.deploy_testing_fixtures import create_test_project, TestModel
    
    # Create a test project using ProjectGenerator
    project_path = create_test_project("my-test-project")
    
    # Use a minimal model for integration testing
    model = TestModel()
    metrics = model.train()
    model.save(store)
"""

from geronimo.deploy_testing_fixtures.fixtures import (
    create_test_project,
    create_mock_cloud_client,
    create_mock_http_client,
)
from geronimo.deploy_testing_fixtures.models import TestModel, TestFeatures

__all__ = [
    "create_test_project",
    "create_mock_cloud_client",
    "create_mock_http_client",
    "TestModel",
    "TestFeatures",
]

