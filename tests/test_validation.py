"""Tests for geronimo.validation module."""

import pytest

from geronimo.validation.rules import (
    ValidationRule,
    RuleResult,
    ProjectNameRule,
    ResourceSizingRule,
    ScalingConfigRule,
    EnvironmentNamesRule,
    DEFAULT_RULES,
)
from geronimo.validation.engine import ValidationEngine, ValidationResult
from geronimo.config.schema import (
    GeronimoConfig,
    ProjectConfig,
    ModelConfig,
    ModelType,
    MLFramework,
    InfrastructureConfig,
    ScalingConfig,
    DeploymentConfig,
    EnvironmentConfig,
)


@pytest.fixture
def valid_config():
    """Create a valid configuration."""
    return GeronimoConfig(
        project=ProjectConfig(name="valid-project", version="1.0.0"),
        model=ModelConfig(
            type=ModelType.REALTIME,
            framework=MLFramework.SKLEARN,
            artifact_path="model.joblib",
        ),
    )


@pytest.fixture
def config_with_bad_name():
    """Create config with invalid project name."""
    return GeronimoConfig(
        project=ProjectConfig(name="Invalid_Name", version="1.0.0"),
        model=ModelConfig(
            type=ModelType.REALTIME,
            framework=MLFramework.SKLEARN,
            artifact_path="model.joblib",
        ),
    )


class TestProjectNameRule:
    """Tests for ProjectNameRule."""

    def test_valid_project_name(self, valid_config):
        """Test valid project names pass."""
        rule = ProjectNameRule()
        result = rule.validate(valid_config)
        assert result.passed is True

    def test_uppercase_name_fails(self, config_with_bad_name):
        """Test uppercase names fail."""
        rule = ProjectNameRule()
        result = rule.validate(config_with_bad_name)
        assert result.passed is False
        assert "lowercase" in result.message.lower()

    def test_name_too_long(self):
        """Test names over 63 chars fail."""
        config = GeronimoConfig(
            project=ProjectConfig(name="a" * 64, version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
        )
        rule = ProjectNameRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "63" in result.message


class TestResourceSizingRule:
    """Tests for ResourceSizingRule."""

    def test_valid_combination(self, valid_config):
        """Test valid CPU/memory combinations pass."""
        rule = ResourceSizingRule()
        result = rule.validate(valid_config)
        assert result.passed is True

    def test_invalid_cpu(self):
        """Test invalid CPU values fail."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
            infrastructure=InfrastructureConfig(cpu=123, memory=512),
        )
        rule = ResourceSizingRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "CPU" in result.message

    def test_invalid_memory_for_cpu(self):
        """Test invalid memory for given CPU fails."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
            infrastructure=InfrastructureConfig(cpu=256, memory=8192),
        )
        rule = ResourceSizingRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "Memory" in result.message


class TestScalingConfigRule:
    """Tests for ScalingConfigRule."""

    def test_valid_scaling(self, valid_config):
        """Test valid scaling config passes."""
        rule = ScalingConfigRule()
        result = rule.validate(valid_config)
        assert result.passed is True

    def test_min_exceeds_max(self):
        """Test min > max fails."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
            infrastructure=InfrastructureConfig(
                scaling=ScalingConfig(min_instances=5, max_instances=2)
            ),
        )
        rule = ScalingConfigRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "exceed" in result.message.lower()

    def test_max_over_limit(self):
        """Test max > 100 fails."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
            infrastructure=InfrastructureConfig(
                scaling=ScalingConfig(min_instances=1, max_instances=150)
            ),
        )
        rule = ScalingConfigRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "100" in result.message


class TestEnvironmentNamesRule:
    """Tests for EnvironmentNamesRule."""

    def test_valid_environments(self, valid_config):
        """Test valid environment names pass."""
        rule = EnvironmentNamesRule()
        result = rule.validate(valid_config)
        assert result.passed is True

    def test_duplicate_names(self):
        """Test duplicate environment names fail."""
        config = GeronimoConfig(
            project=ProjectConfig(name="test", version="1.0.0"),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=MLFramework.SKLEARN,
                artifact_path="model.joblib",
            ),
            deployment=DeploymentConfig(
                environments=[
                    EnvironmentConfig(name="prod"),
                    EnvironmentConfig(name="prod"),  # Duplicate
                ]
            ),
        )
        rule = EnvironmentNamesRule()
        result = rule.validate(config)
        assert result.passed is False
        assert "Duplicate" in result.message


class TestValidationEngine:
    """Tests for ValidationEngine."""

    def test_engine_with_valid_config(self, valid_config):
        """Test engine passes valid config."""
        engine = ValidationEngine()
        result = engine.validate(valid_config)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.rules_checked == len(DEFAULT_RULES)

    def test_engine_with_invalid_config(self, config_with_bad_name):
        """Test engine catches invalid config."""
        engine = ValidationEngine()
        result = engine.validate(config_with_bad_name)
        
        assert result.is_valid is False
        assert len(result.errors) > 0

    def test_register_custom_rule(self, valid_config):
        """Test registering custom rules."""
        class CustomRule(ValidationRule):
            name = "Custom Rule"
            
            def validate(self, config):
                return RuleResult(
                    passed=True,
                    message="Custom check passed",
                    rule_name=self.name,
                )
        
        engine = ValidationEngine()
        engine.register_rule(CustomRule())
        
        result = engine.validate(valid_config)
        assert result.rules_checked == len(DEFAULT_RULES) + 1
