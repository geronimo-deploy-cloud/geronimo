"""Geronimo Code Generators.

This module powers the `geronimo init` scaffolding and other code generation tasks.
It uses Jinja2 templates to produce production-ready code for:
- New Geronimo projects (folder structure, config files)
- Dockerfiles for training and serving containers
- Terraform configurations for AWS infrastructure
- CI/CD pipeline definitions (GitHub Actions)

The goal is to provide a "batteries-included" starting point that follows MLOps best practices.
"""

from geronimo.generators.base import BaseGenerator
from geronimo.generators.docker import DockerGenerator
from geronimo.generators.pipeline import PipelineGenerator
from geronimo.generators.project import ProjectGenerator
from geronimo.generators.terraform import TerraformGenerator

__all__ = [
    "BaseGenerator",
    "ProjectGenerator",
    "TerraformGenerator",
    "DockerGenerator",
    "PipelineGenerator",
]
