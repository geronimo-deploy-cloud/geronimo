"""Terraform generator for Geronimo.

Generates modular Terraform configuration for AWS ECS deployments.
"""

from pathlib import Path

from geronimo.config.schema import GeronimoConfig
from geronimo.generators.base import BaseGenerator
from geronimo.generators.template_engine import TemplateEngine


class TerraformGenerator(BaseGenerator):
    """Generates Terraform infrastructure files for AWS ECS deployments."""

    TEMPLATE_DIR = "terraform"

    def __init__(
        self,
        config: GeronimoConfig,
        output_dir: str = "infrastructure",
    ) -> None:
        """Initialize the Terraform generator.

        Args:
            config: Geronimo configuration.
            output_dir: Directory to write Terraform files.
        """
        super().__init__()
        self.config = config
        self.output_dir = Path(output_dir)
        self.engine = TemplateEngine()
        
        # Build context from config
        self.context = self._build_context()

    def _build_context(self) -> dict:
        """Build template context from configuration."""
        return {
            "project_name": self.config.project.name,
            "cpu": self.config.infrastructure.cpu,
            "memory": self.config.infrastructure.memory,
            "min_instances": self.config.infrastructure.scaling.min_instances,
            "max_instances": self.config.infrastructure.scaling.max_instances,
            "target_cpu": self.config.infrastructure.scaling.target_cpu_percent,
        }

    def generate(self) -> list[str]:
        """Generate all Terraform files.

        Returns:
            List of generated file paths.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        files = []

        # Generate each Terraform file from templates
        terraform_files = [
            ("main.tf.jinja2", "main.tf"),
            ("variables.tf.jinja2", "variables.tf"),
            ("ecr.tf.jinja2", "ecr.tf"),
            ("ecs.tf.jinja2", "ecs.tf"),
            ("alb.tf.jinja2", "alb.tf"),
            ("cloudwatch.tf.jinja2", "cloudwatch.tf"),
            ("iam.tf.jinja2", "iam.tf"),
            ("outputs.tf.jinja2", "outputs.tf"),
        ]

        for template_name, output_name in terraform_files:
            output_path = self.output_dir / output_name
            self.engine.render_to_file(
                f"terraform/{template_name}",
                self.context,
                output_path,
            )
            files.append(str(output_path))

        return files
