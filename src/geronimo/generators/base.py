"""Base generator class for Geronimo.

Provides common template rendering functionality for all generators.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape


class BaseGenerator(ABC):
    """Abstract base class for all Geronimo generators.

    Provides Jinja2 template rendering infrastructure.
    """

    # Subclasses should override this to specify their template subdirectory
    TEMPLATE_DIR: str = ""

    def __init__(self) -> None:
        """Initialize the generator with Jinja2 environment."""
        self._env = Environment(
            loader=PackageLoader("geronimo", "templates"),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with the given context.

        Args:
            template_name: Name of the template file (relative to templates/).
            context: Dictionary of variables to pass to the template.

        Returns:
            Rendered template as a string.
        """
        template_path = (
            f"{self.TEMPLATE_DIR}/{template_name}"
            if self.TEMPLATE_DIR
            else template_name
        )
        template = self._env.get_template(template_path)
        return template.render(**context)

    def write_file(self, path: Path | str, content: str) -> None:
        """Write content to a file, creating directories as needed.

        Args:
            path: Target file path.
            content: Content to write.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @abstractmethod
    def generate(self) -> Any:
        """Generate the output artifacts.

        Subclasses must implement this method.
        """
        pass
