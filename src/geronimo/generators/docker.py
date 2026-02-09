"""Docker generator for Geronimo.

Generates optimized Dockerfiles for ML model serving.
"""

from pathlib import Path

from geronimo.config.schema import GeronimoConfig, MLFramework
from geronimo.generators.base import BaseGenerator
from geronimo.generators.template_engine import TemplateEngine


class DockerGenerator(BaseGenerator):
    """Generates optimized Dockerfiles for ML deployments."""

    TEMPLATE_DIR = "docker"

    def __init__(
        self,
        config: GeronimoConfig,
        output_path: str = "Dockerfile",
    ) -> None:
        """Initialize the Docker generator.

        Args:
            config: Geronimo configuration.
            output_path: Path to write the Dockerfile.
        """
        super().__init__()
        self.config = config
        self.output_path = Path(output_path)
        self.engine = TemplateEngine()

    def _get_base_image(self) -> str:
        """Get the appropriate base image for the framework."""
        python_version = self.config.runtime.python_version

        # Use slim images for smaller size
        base_images = {
            MLFramework.PYTORCH: f"python:{python_version}-slim",
            MLFramework.TENSORFLOW: f"python:{python_version}-slim",
            MLFramework.SKLEARN: f"python:{python_version}-slim",
            MLFramework.XGBOOST: f"python:{python_version}-slim",
            MLFramework.CUSTOM: f"python:{python_version}-slim",
        }

        return base_images.get(self.config.model.framework, f"python:{python_version}-slim")

    def generate(self) -> str:
        """Generate the Dockerfile.

        Returns:
            Path to the generated Dockerfile.
        """
        # Render Dockerfile
        context = {
            "project_name": self.config.project.name,
            "project_name_snake": self.config.project.name.replace("-", "_"),
            "base_image": self._get_base_image(),
        }
        
        self.engine.render_to_file(
            "docker/Dockerfile.jinja2",
            context,
            self.output_path
        )

        # Render .dockerignore
        self.engine.render_to_file(
            "docker/dockerignore.jinja2",
            {},
            self.output_path.parent / ".dockerignore"
        )

        return str(self.output_path)
