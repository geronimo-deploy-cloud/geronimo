"""Pipeline generator for Geronimo.

Generates Azure DevOps pipeline YAML for CI/CD.
"""

from pathlib import Path

from geronimo.config.schema import GeronimoConfig
from geronimo.generators.base import BaseGenerator
from geronimo.generators.template_engine import TemplateEngine


class PipelineGenerator(BaseGenerator):
    """Generates Azure DevOps pipeline configuration."""

    TEMPLATE_DIR = "pipeline"

    def __init__(
        self,
        config: GeronimoConfig,
        output_path: str = "azure-pipelines.yaml",
    ) -> None:
        """Initialize the pipeline generator.

        Args:
            config: Geronimo configuration.
            output_path: Path to write the pipeline file.
        """
        super().__init__()
        self.config = config
        self.output_path = Path(output_path)
        self.engine = TemplateEngine()

    def generate(self) -> str:
        """Generate the Azure DevOps pipeline.

        Returns:
            Path to the generated pipeline file.
        """
        # Build context
        context = {
            "project_name": self.config.project.name,
            "python_version": self.config.runtime.python_version,
            "environments": self.config.deployment.environments,
        }

        # Render pipeline template
        self.engine.render_to_file(
            "pipeline/azure-pipelines.yaml.jinja2",
            context,
            self.output_path
        )
        
        return str(self.output_path)
