"""Geronimo Configuration Validation Module.

The validation module ensures that the project configuration (`geronimo.yaml`) is valid
and compliant with deployment requirements before any infrastructure is provisioned.

It checks for:
- Resource sizing (valid Fargate CPU/Memory combinations)
- Scaling policies (min/max instances)
- Naming conventions (project and environment names)
- Structural integrity of the deployment config

Key components:
- ValidationEngine: Runs the suite of registered validation rules.
- ValidationRule: Base class for implementing specific checks.
"""

from geronimo.validation.engine import ValidationEngine, ValidationResult
from geronimo.validation.rules import ValidationRule

__all__ = [
    "ValidationEngine",
    "ValidationResult",
    "ValidationRule",
]

__docformat__ = "google"
