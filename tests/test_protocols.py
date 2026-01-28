"""Tests for protocol implementations."""

import pytest
from typing import Any

from geronimo.artifacts.protocol import ArtifactBackend
from geronimo.deploy.protocol import (
    DeploymentTarget,
    DeploymentInfo,
    DeploymentStatus,
)


# =============================================================================
# ArtifactBackend Protocol Tests
# =============================================================================

class MockArtifactBackend:
    """Mock implementation of ArtifactBackend for testing."""
    
    def __init__(self):
        self._storage: dict[str, tuple[Any, dict]] = {}
    
    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        uri = f"mock://{name}"
        self._storage[uri] = (artifact, metadata)
        return uri
    
    def load(self, uri: str) -> Any:
        if uri not in self._storage:
            raise KeyError(f"Artifact not found: {uri}")
        return self._storage[uri][0]
    
    def list(self, prefix: str) -> list[str]:
        return [uri for uri in self._storage.keys() if uri.startswith(f"mock://{prefix}")]
    
    def delete(self, uri: str) -> None:
        if uri in self._storage:
            del self._storage[uri]


class TestArtifactBackendProtocol:
    """Tests for ArtifactBackend protocol."""
    
    def test_mock_implements_protocol(self):
        """Test that mock class implements the protocol."""
        backend = MockArtifactBackend()
        assert isinstance(backend, ArtifactBackend)
    
    def test_save_returns_uri(self):
        """Test that save returns a URI."""
        backend = MockArtifactBackend()
        uri = backend.save("model", {"weights": [1, 2, 3]}, {"version": "1.0"})
        
        assert uri == "mock://model"
        assert isinstance(uri, str)
    
    def test_load_retrieves_artifact(self):
        """Test that load retrieves saved artifact."""
        backend = MockArtifactBackend()
        original = {"weights": [1, 2, 3]}
        
        uri = backend.save("model", original, {})
        loaded = backend.load(uri)
        
        assert loaded == original
    
    def test_load_raises_for_missing(self):
        """Test that load raises KeyError for missing artifact."""
        backend = MockArtifactBackend()
        
        with pytest.raises(KeyError):
            backend.load("mock://nonexistent")
    
    def test_list_returns_matching_uris(self):
        """Test that list returns URIs matching prefix."""
        backend = MockArtifactBackend()
        backend.save("model_v1", {}, {})
        backend.save("model_v2", {}, {})
        backend.save("encoder", {}, {})
        
        models = backend.list("model")
        encoders = backend.list("encoder")
        
        assert len(models) == 2
        assert len(encoders) == 1
        assert "mock://model_v1" in models
        assert "mock://model_v2" in models
    
    def test_delete_removes_artifact(self):
        """Test that delete removes artifact."""
        backend = MockArtifactBackend()
        uri = backend.save("temp", {}, {})
        
        backend.delete(uri)
        
        with pytest.raises(KeyError):
            backend.load(uri)
    
    def test_delete_idempotent(self):
        """Test that deleting non-existent artifact doesn't raise."""
        backend = MockArtifactBackend()
        # Should not raise
        backend.delete("mock://nonexistent")


# =============================================================================
# DeploymentTarget Protocol Tests
# =============================================================================

class MockDeploymentTarget:
    """Mock implementation of DeploymentTarget for testing."""
    
    def __init__(self):
        self._deployments: dict[str, DeploymentInfo] = {}
        self._logs: dict[str, list[str]] = {}
    
    def deploy(self, config: Any, artifacts: dict[str, str]) -> DeploymentInfo:
        deployment_id = f"deploy-{config.project}"
        info = DeploymentInfo(
            deployment_id=deployment_id,
            status=DeploymentStatus.RUNNING,
            endpoint_url=f"http://{config.project}.example.com",
            message="Deployment successful",
            resources={"artifacts": list(artifacts.keys())},
        )
        self._deployments[deployment_id] = info
        self._logs[deployment_id] = ["Starting deployment...", "Deployment complete."]
        return info
    
    def status(self, deployment_id: str) -> DeploymentInfo:
        if deployment_id not in self._deployments:
            return DeploymentInfo(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                message="Deployment not found",
            )
        return self._deployments[deployment_id]
    
    def teardown(self, deployment_id: str) -> None:
        if deployment_id in self._deployments:
            self._deployments[deployment_id] = DeploymentInfo(
                deployment_id=deployment_id,
                status=DeploymentStatus.STOPPED,
                message="Deployment stopped",
            )
    
    def logs(self, deployment_id: str, lines: int = 100) -> list[str]:
        return self._logs.get(deployment_id, [])[-lines:]


class MockConfig:
    """Mock config for testing."""
    def __init__(self, project: str):
        self.project = project


class TestDeploymentTargetProtocol:
    """Tests for DeploymentTarget protocol."""
    
    def test_mock_implements_protocol(self):
        """Test that mock class implements the protocol."""
        target = MockDeploymentTarget()
        assert isinstance(target, DeploymentTarget)
    
    def test_deploy_returns_deployment_info(self):
        """Test that deploy returns DeploymentInfo."""
        target = MockDeploymentTarget()
        config = MockConfig("iris-model")
        
        info = target.deploy(config, {"model": "s3://bucket/model.pkl"})
        
        assert isinstance(info, DeploymentInfo)
        assert info.deployment_id == "deploy-iris-model"
        assert info.status == DeploymentStatus.RUNNING
        assert info.endpoint_url == "http://iris-model.example.com"
    
    def test_status_returns_current_state(self):
        """Test that status returns current deployment state."""
        target = MockDeploymentTarget()
        config = MockConfig("test-model")
        
        deploy_info = target.deploy(config, {})
        status_info = target.status(deploy_info.deployment_id)
        
        assert status_info.status == DeploymentStatus.RUNNING
        assert status_info.deployment_id == deploy_info.deployment_id
    
    def test_status_returns_failed_for_unknown(self):
        """Test that status returns FAILED for unknown deployment."""
        target = MockDeploymentTarget()
        
        info = target.status("unknown-deployment")
        
        assert info.status == DeploymentStatus.FAILED
        assert "not found" in info.message.lower()
    
    def test_teardown_stops_deployment(self):
        """Test that teardown stops deployment."""
        target = MockDeploymentTarget()
        config = MockConfig("to-stop")
        
        deploy_info = target.deploy(config, {})
        target.teardown(deploy_info.deployment_id)
        
        status_info = target.status(deploy_info.deployment_id)
        assert status_info.status == DeploymentStatus.STOPPED
    
    def test_logs_returns_log_lines(self):
        """Test that logs returns deployment logs."""
        target = MockDeploymentTarget()
        config = MockConfig("logged-model")
        
        deploy_info = target.deploy(config, {})
        logs = target.logs(deploy_info.deployment_id)
        
        assert isinstance(logs, list)
        assert len(logs) == 2
        assert "Starting deployment" in logs[0]
    
    def test_logs_respects_lines_limit(self):
        """Test that logs respects lines parameter."""
        target = MockDeploymentTarget()
        config = MockConfig("many-logs")
        
        deploy_info = target.deploy(config, {})
        logs = target.logs(deploy_info.deployment_id, lines=1)
        
        assert len(logs) == 1


# =============================================================================
# DeploymentStatus Enum Tests
# =============================================================================

class TestDeploymentStatus:
    """Tests for DeploymentStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test that all expected statuses exist."""
        assert DeploymentStatus.PENDING
        assert DeploymentStatus.DEPLOYING
        assert DeploymentStatus.RUNNING
        assert DeploymentStatus.FAILED
        assert DeploymentStatus.STOPPING
        assert DeploymentStatus.STOPPED
    
    def test_status_values_are_strings(self):
        """Test that status values are strings."""
        assert DeploymentStatus.RUNNING.value == "running"
        assert DeploymentStatus.FAILED.value == "failed"
        assert DeploymentStatus.STOPPED.value == "stopped"


# =============================================================================
# DeploymentInfo Model Tests
# =============================================================================

class TestDeploymentInfo:
    """Tests for DeploymentInfo model."""
    
    def test_minimal_creation(self):
        """Test creating DeploymentInfo with minimal fields."""
        info = DeploymentInfo(
            deployment_id="test-123",
            status=DeploymentStatus.RUNNING,
        )
        
        assert info.deployment_id == "test-123"
        assert info.status == DeploymentStatus.RUNNING
        assert info.endpoint_url is None
        assert info.message is None
        assert info.resources == {}
    
    def test_full_creation(self):
        """Test creating DeploymentInfo with all fields."""
        info = DeploymentInfo(
            deployment_id="full-123",
            status=DeploymentStatus.RUNNING,
            endpoint_url="https://api.example.com",
            message="Deployed successfully",
            resources={"replicas": 3, "cpu": "2 vCPU"},
        )
        
        assert info.endpoint_url == "https://api.example.com"
        assert info.message == "Deployed successfully"
        assert info.resources["replicas"] == 3
    
    def test_serialization(self):
        """Test that DeploymentInfo serializes to dict."""
        info = DeploymentInfo(
            deployment_id="serial-123",
            status=DeploymentStatus.FAILED,
            message="Out of memory",
        )
        
        data = info.model_dump()
        
        assert data["deployment_id"] == "serial-123"
        assert data["status"] == "failed"
        assert data["message"] == "Out of memory"


# =============================================================================
# Module Export Tests
# =============================================================================

class TestModuleExports:
    """Tests for module exports."""
    
    def test_artifacts_exports_protocol(self):
        """Test that artifacts module exports ArtifactBackend."""
        from geronimo.artifacts import ArtifactBackend as AB
        assert AB is not None
    
    def test_deploy_exports_protocol(self):
        """Test that deploy module exports protocol classes."""
        from geronimo.deploy import (
            DeploymentTarget as DT,
            DeploymentInfo as DI,
            DeploymentStatus as DS,
        )
        assert DT is not None
        assert DI is not None
        assert DS is not None
