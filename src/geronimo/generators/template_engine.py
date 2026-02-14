"""Jinja2 template engine for code generation.

Provides a centralized template rendering system for project generation.
Templates are stored in the templates/ directory and rendered with context.
"""

import os
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape


class TemplateEngine:
    """Jinja2-based template engine for code generation.
    
    Loads templates from package directory and renders with context.
    
    Example:
        engine = TemplateEngine()
        content = engine.render("sdk/model.py.jinja2", {
            "project_name": "my-project",
            "framework": "sklearn",
        })
    """

    templates_dir: Path
    """Directory containing Jinja2 templates."""

    env: Environment
    """Jinja2 environment."""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize template engine.
        
        Args:
            templates_dir: Path to templates directory.
                           Defaults to generators/templates/.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"
        
        self.templates_dir = templates_dir
        
        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=select_autoescape(disabled_extensions=("py", "toml", "md", "yaml")),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )
        
        # Register custom filters
        self.env.filters["pascal_case"] = self._to_pascal_case
        self.env.filters["snake_case"] = self._to_snake_case
    
    @staticmethod
    def _to_pascal_case(name: str) -> str:
        """Convert kebab-case or snake_case to PascalCase."""
        return ''.join(
            word.title() for word in name.replace("-", "_").split("_")
        )
    
    @staticmethod
    def _to_snake_case(name: str) -> str:
        """Convert kebab-case to snake_case."""
        return name.replace("-", "_")
    
    def render(self, template_name: str, context: dict[str, Any]) -> str:
        """Render a template with context.
        
        Args:
            template_name: Template file name (relative to templates/).
            context: Dictionary of variables for template.
            
        Returns:
            Rendered template string.
        """
        template = self.env.get_template(template_name)
        return template.render(**context)
    
    def render_to_file(
        self,
        template_name: str,
        context: dict[str, Any],
        output_path: Path,
    ) -> None:
        """Render a template and write to file.
        
        Args:
            template_name: Template file name.
            context: Dictionary of variables for template.
            output_path: Path to write rendered content.
        """
        content = self.render(template_name, context)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
    
    def list_templates(self, prefix: str = "") -> list[str]:
        """List available templates.
        
        Args:
            prefix: Optional prefix to filter (e.g., "sdk/").
            
        Returns:
            List of template names.
        """
        templates = []
        for root, dirs, files in os.walk(self.templates_dir):
            for file in files:
                if file.endswith(".jinja2"):
                    rel_path = os.path.relpath(
                        os.path.join(root, file),
                        self.templates_dir
                    )
                    if rel_path.startswith(prefix):
                        templates.append(rel_path)
        return sorted(templates)
