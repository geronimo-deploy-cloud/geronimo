"""Validation rules for Geronimo configurations.

Extensible rule system for enforcing deployment policies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geronimo.config.schema import GeronimoConfig


@dataclass
class RuleResult:
    """Result of a single rule validation."""

    passed: bool
    message: str
    rule_name: str


class ValidationRule(ABC):
    """Abstract base class for validation rules."""

    # Human-readable name for the rule
    name: str = "Base Rule"

    # Description of what the rule checks
    description: str = ""

    @abstractmethod
    def validate(self, config: "GeronimoConfig") -> RuleResult:
        """Validate the configuration against this rule.

        Args:
            config: The configuration to validate.

        Returns:
            RuleResult indicating pass/fail and any messages.
        """
        pass


class ProjectNameRule(ValidationRule):
    """Ensures project name follows naming conventions."""

    name = "Project Name Format"
    description = "Project name must be lowercase with hyphens only"

    def validate(self, config: "GeronimoConfig") -> RuleResult:
        name = config.project.name

        # Check for valid characters
        import re

        if not re.match(r"^[a-z][a-z0-9-]*[a-z0-9]$", name):
            return RuleResult(
                passed=False,
                message=f"Project name '{name}' must be lowercase, start with letter, "
                "and contain only letters, numbers, and hyphens",
                rule_name=self.name,
            )

        # Check length
        if len(name) > 63:
            return RuleResult(
                passed=False,
                message=f"Project name '{name}' exceeds 63 characters",
                rule_name=self.name,
            )

        return RuleResult(passed=True, message="Project name is valid", rule_name=self.name)


class ResourceSizingRule(ValidationRule):
    """Validates CPU and memory combinations are valid for Fargate."""

    name = "Resource Sizing"
    description = "CPU and memory must be valid Fargate combinations"

    # Valid Fargate CPU/memory combinations
    VALID_COMBINATIONS = {
        256: [512, 1024, 2048],
        512: [1024, 2048, 3072, 4096],
        1024: [2048, 3072, 4096, 5120, 6144, 7168, 8192],
        2048: list(range(4096, 16385, 1024)),
        4096: list(range(8192, 30721, 1024)),
        8192: list(range(16384, 61441, 4096)),
        16384: list(range(32768, 122881, 8192)),
    }

    def validate(self, config: "GeronimoConfig") -> RuleResult:
        cpu = config.infrastructure.cpu
        memory = config.infrastructure.memory

        if cpu not in self.VALID_COMBINATIONS:
            valid_cpus = ", ".join(str(c) for c in sorted(self.VALID_COMBINATIONS.keys()))
            return RuleResult(
                passed=False,
                message=f"CPU value {cpu} is not valid. Must be one of: {valid_cpus}",
                rule_name=self.name,
            )

        valid_memory = self.VALID_COMBINATIONS[cpu]
        if memory not in valid_memory:
            return RuleResult(
                passed=False,
                message=f"Memory {memory}MB is not valid for CPU {cpu}. "
                f"Valid values: {min(valid_memory)}-{max(valid_memory)}MB",
                rule_name=self.name,
            )

        return RuleResult(passed=True, message="Resource sizing is valid", rule_name=self.name)


class ScalingConfigRule(ValidationRule):
    """Validates scaling configuration is reasonable."""

    name = "Scaling Configuration"
    description = "Min/max instances must be valid"

    def validate(self, config: "GeronimoConfig") -> RuleResult:
        scaling = config.infrastructure.scaling

        if scaling.min_instances > scaling.max_instances:
            return RuleResult(
                passed=False,
                message=f"min_instances ({scaling.min_instances}) cannot exceed "
                f"max_instances ({scaling.max_instances})",
                rule_name=self.name,
            )

        if scaling.max_instances > 100:
            return RuleResult(
                passed=False,
                message=f"max_instances ({scaling.max_instances}) exceeds limit of 100",
                rule_name=self.name,
            )

        return RuleResult(passed=True, message="Scaling configuration is valid", rule_name=self.name)


class EnvironmentNamesRule(ValidationRule):
    """Validates environment names are unique and valid."""

    name = "Environment Names"
    description = "Environment names must be unique and valid"

    def validate(self, config: "GeronimoConfig") -> RuleResult:
        env_names = [env.name for env in config.deployment.environments]

        # Check for duplicates
        if len(env_names) != len(set(env_names)):
            return RuleResult(
                passed=False,
                message="Duplicate environment names found",
                rule_name=self.name,
            )

        # Check names are valid
        import re

        for name in env_names:
            if not re.match(r"^[a-z][a-z0-9-]*$", name):
                return RuleResult(
                    passed=False,
                    message=f"Environment name '{name}' is invalid",
                    rule_name=self.name,
                )

        return RuleResult(passed=True, message="Environment names are valid", rule_name=self.name)


# Registry of all default rules
DEFAULT_RULES: list[type[ValidationRule]] = [
    ProjectNameRule,
    ResourceSizingRule,
    ScalingConfigRule,
    EnvironmentNamesRule,
]
