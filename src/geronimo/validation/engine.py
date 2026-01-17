"""Validation engine for Geronimo configurations.

Runs all registered validation rules against a configuration.
"""

from dataclasses import dataclass, field

from geronimo.config.schema import GeronimoConfig
from geronimo.validation.rules import DEFAULT_RULES, RuleResult, ValidationRule


@dataclass
class ValidationResult:
    """Result of validating a configuration."""

    is_valid: bool
    rules_checked: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    results: list[RuleResult] = field(default_factory=list)


class ValidationEngine:
    """Runs validation rules against configurations.

    The engine maintains a registry of rules that are checked
    when validate() is called.
    """

    def __init__(self) -> None:
        """Initialize the validation engine with default rules."""
        self._rules: list[ValidationRule] = []

        # Register default rules
        for rule_class in DEFAULT_RULES:
            self._rules.append(rule_class())

    def register_rule(self, rule: ValidationRule) -> None:
        """Register a custom validation rule.

        Args:
            rule: The rule instance to register.
        """
        self._rules.append(rule)

    def validate(self, config: GeronimoConfig) -> ValidationResult:
        """Validate a configuration against all registered rules.

        Args:
            config: The configuration to validate.

        Returns:
            ValidationResult with pass/fail status and any errors.
        """
        results: list[RuleResult] = []
        errors: list[str] = []
        warnings: list[str] = []

        for rule in self._rules:
            result = rule.validate(config)
            results.append(result)

            if not result.passed:
                errors.append(f"[{result.rule_name}] {result.message}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            rules_checked=len(self._rules),
            errors=errors,
            warnings=warnings,
            results=results,
        )

    @property
    def rules(self) -> list[ValidationRule]:
        """Get list of registered rules."""
        return self._rules.copy()
