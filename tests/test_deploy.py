"""Tests for geronimo.deploy module."""

import pytest
from unittest.mock import patch, MagicMock

from geronimo.deploy.config import (
    DeploymentConfig,
    ArtifactStorageConfig,
    ServingConfig,
    BatchConfig,
)
from geronimo.deploy.targets import (
    get_available_targets,
    deploy,
    destroy,
    PulumiNotInstalledError,
    _check_pulumi_available,
)


# =============================================================================
# DeploymentConfig Tests
# =============================================================================

class TestDeploymentConfig:
    """Tests for DeploymentConfig model."""
    
    def test_minimal_config(self):
        """Test creating config with minimal required fields."""
        config = DeploymentConfig(project="test-project")
        
        assert config.project == "test-project"
        assert config.target == "aws"
        assert config.region == "us-east-1"
        assert config.stack_name == "dev"
        assert config.version == "1.0.0"
    
    def test_full_config(self):
        """Test creating config with all fields."""
        config = DeploymentConfig(
            project="iris-classifier",
            version="2.0.0",
            target="gcp",
            region="us-central1",
            stack_name="prod",
            artifacts=ArtifactStorageConfig(
                bucket_prefix="my-artifacts",
                retention_days=30,
                versioning=False,
            ),
            serving=ServingConfig(
                cpu=2,
                memory=4096,
                min_replicas=2,
                max_replicas=20,
                port=9000,
            ),
            batch=BatchConfig(
                schedule="0 */4 * * *",
                timeout_minutes=120,
            ),
        )
        
        assert config.project == "iris-classifier"
        assert config.target == "gcp"
        assert config.region == "us-central1"
        assert config.artifacts.bucket_prefix == "my-artifacts"
        assert config.artifacts.retention_days == 30
        assert config.artifacts.versioning is False
        assert config.serving.cpu == 2
        assert config.serving.memory == 4096
        assert config.batch.schedule == "0 */4 * * *"
    
    def test_target_validation(self):
        """Test that only valid targets are accepted."""
        # Valid targets
        for target in ["aws", "gcp", "azure"]:
            config = DeploymentConfig(project="test", target=target)
            assert config.target == target
        
        # Invalid target
        with pytest.raises(ValueError):
            DeploymentConfig(project="test", target="invalid")
    
    def test_default_artifact_config(self):
        """Test default artifact storage configuration."""
        config = DeploymentConfig(project="test")
        
        assert config.artifacts.bucket_prefix == "geronimo-artifacts"
        assert config.artifacts.retention_days == 90
        assert config.artifacts.versioning is True
    
    def test_serving_config_defaults(self):
        """Test default serving configuration values."""
        serving = ServingConfig()
        
        assert serving.cpu == 1
        assert serving.memory == 2048
        assert serving.min_replicas == 1
        assert serving.max_replicas == 10
        assert serving.port == 8000
    
    def test_batch_config_defaults(self):
        """Test default batch configuration values."""
        batch = BatchConfig()
        
        assert batch.schedule == "0 6 * * *"
        assert batch.timeout_minutes == 60


# =============================================================================
# Targets Module Tests
# =============================================================================

class TestGetAvailableTargets:
    """Tests for get_available_targets function."""
    
    def test_returns_all_cloud_providers(self):
        """Test that all cloud providers are returned."""
        targets = get_available_targets()
        
        assert "aws" in targets
        assert "gcp" in targets
        assert "azure" in targets
        assert len(targets) == 3


class TestPulumiDetection:
    """Tests for Pulumi runtime detection."""
    
    def test_pulumi_not_installed_error_message(self):
        """Test error message includes helpful instructions."""
        error = PulumiNotInstalledError()
        
        assert "pip install geronimo[pulumi]" in str(error)
        assert "geronimo generate" in str(error)
    
    def test_check_pulumi_detection(self):
        """Test that Pulumi detection returns boolean."""
        result = _check_pulumi_available()
        # Result depends on whether pulumi is installed
        assert isinstance(result, bool)


class TestDeployFunction:
    """Tests for deploy function."""
    
    def test_deploy_raises_when_pulumi_not_installed(self):
        """Test that deploy raises PulumiNotInstalledError without Pulumi."""
        config = DeploymentConfig(project="test")
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=False):
            with pytest.raises(PulumiNotInstalledError):
                deploy(config)
    
    def test_deploy_calls_correct_aws_provider(self):
        """Test that AWS provider is called for aws target."""
        config = DeploymentConfig(project="test", target="aws")
        mock_result = {"outputs": {"bucket": "test-bucket"}}
        mock_deploy_aws = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            # Patch the import inside the function
            with patch.dict("sys.modules", {"geronimo.deploy.providers.aws": MagicMock(deploy_aws=mock_deploy_aws)}):
                # Re-import to get patched version
                import geronimo.deploy.targets as targets_module
                result = targets_module.deploy(config)
                
                mock_deploy_aws.assert_called_once_with(config, None)
                assert result == mock_result
    
    def test_deploy_calls_correct_gcp_provider(self):
        """Test that GCP provider is called for gcp target."""
        config = DeploymentConfig(project="test", target="gcp")
        mock_result = {"outputs": {"bucket": "test-bucket"}}
        mock_deploy_gcp = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.gcp": MagicMock(deploy_gcp=mock_deploy_gcp)}):
                import geronimo.deploy.targets as targets_module
                result = targets_module.deploy(config)
                
                mock_deploy_gcp.assert_called_once_with(config, None)
                assert result == mock_result
    
    def test_deploy_calls_correct_azure_provider(self):
        """Test that Azure provider is called for azure target."""
        config = DeploymentConfig(project="test", target="azure")
        mock_result = {"outputs": {"storage": "test-storage"}}
        mock_deploy_azure = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.azure": MagicMock(deploy_azure=mock_deploy_azure)}):
                import geronimo.deploy.targets as targets_module
                result = targets_module.deploy(config)
                
                mock_deploy_azure.assert_called_once_with(config, None)
                assert result == mock_result
    
    def test_deploy_with_component_parameter(self):
        """Test deploying specific component."""
        config = DeploymentConfig(project="test", target="aws")
        mock_deploy_aws = MagicMock(return_value={})
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.aws": MagicMock(deploy_aws=mock_deploy_aws)}):
                import geronimo.deploy.targets as targets_module
                targets_module.deploy(config, component="artifacts")
                
                mock_deploy_aws.assert_called_once_with(config, "artifacts")
    
    def test_deploy_raises_for_unknown_target(self):
        """Test that deploy raises ValueError for unknown target."""
        config = DeploymentConfig(project="test", target="aws")
        # Manually override target to bypass validation
        object.__setattr__(config, "target", "invalid")
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with pytest.raises(ValueError, match="Unknown target"):
                deploy(config)


class TestDestroyFunction:
    """Tests for destroy function."""
    
    def test_destroy_raises_when_pulumi_not_installed(self):
        """Test that destroy raises PulumiNotInstalledError without Pulumi."""
        config = DeploymentConfig(project="test")
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=False):
            with pytest.raises(PulumiNotInstalledError):
                destroy(config)
    
    def test_destroy_calls_correct_aws_provider(self):
        """Test that AWS destroy is called for aws target."""
        config = DeploymentConfig(project="test", target="aws")
        mock_result = {"summary": "destroyed"}
        mock_destroy_aws = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.aws": MagicMock(destroy_aws=mock_destroy_aws)}):
                import geronimo.deploy.targets as targets_module
                result = targets_module.destroy(config)
                
                mock_destroy_aws.assert_called_once_with(config)
                assert result == mock_result
    
    def test_destroy_calls_correct_gcp_provider(self):
        """Test that GCP destroy is called for gcp target."""
        config = DeploymentConfig(project="test", target="gcp")
        mock_result = {"summary": "destroyed"}
        mock_destroy_gcp = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.gcp": MagicMock(destroy_gcp=mock_destroy_gcp)}):
                import geronimo.deploy.targets as targets_module
                result = targets_module.destroy(config)
                
                mock_destroy_gcp.assert_called_once_with(config)
                assert result == mock_result
    
    def test_destroy_calls_correct_azure_provider(self):
        """Test that Azure destroy is called for azure target."""
        config = DeploymentConfig(project="test", target="azure")
        mock_result = {"summary": "destroyed"}
        mock_destroy_azure = MagicMock(return_value=mock_result)
        
        with patch("geronimo.deploy.targets._check_pulumi_available", return_value=True):
            with patch.dict("sys.modules", {"geronimo.deploy.providers.azure": MagicMock(destroy_azure=mock_destroy_azure)}):
                import geronimo.deploy.targets as targets_module
                result = targets_module.destroy(config)
                
                mock_destroy_azure.assert_called_once_with(config)
                assert result == mock_result


# =============================================================================
# Module Import Tests
# =============================================================================

class TestModuleExports:
    """Tests for module public API."""
    
    def test_deploy_module_exports(self):
        """Test that deploy module exports expected symbols."""
        from geronimo import deploy
        
        assert hasattr(deploy, "DeploymentConfig")
        assert hasattr(deploy, "deploy")
        assert hasattr(deploy, "destroy")
        assert hasattr(deploy, "get_available_targets")
    
    def test_config_module_exports(self):
        """Test that config classes are importable."""
        from geronimo.deploy.config import (
            DeploymentConfig,
            ArtifactStorageConfig,
            ServingConfig,
            BatchConfig,
        )
        
        assert DeploymentConfig is not None
        assert ArtifactStorageConfig is not None
        assert ServingConfig is not None
        assert BatchConfig is not None
