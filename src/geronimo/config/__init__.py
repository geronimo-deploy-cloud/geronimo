"""Geronimo Configuration Module.

This module handles loading and validating the platform's configuration, primarily
from `geronimo.yaml` and environment variables.

It defines:
- The configuration schema using Pydantic models.
- Methods to locate and load the global `~/.geronimo/config.yaml`.
- Support for project-specific overrides.

This centralized configuration ensures consistency across different commands and
environments (local dev, CI/CD, production).
"""

__docformat__ = "google"
