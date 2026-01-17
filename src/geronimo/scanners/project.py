"""Project scanner for Geronimo.

Scans existing UV-managed FastAPI projects to build a structured representation
for analysis and import.
"""

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProjectScan:
    """Structured representation of a scanned project."""

    root: Path
    """Root directory of the project."""

    pyproject: dict
    """Parsed pyproject.toml contents."""

    project_name: str
    """Name of the project from pyproject.toml."""

    python_version: str
    """Python version requirement."""

    dependencies: list[str]
    """List of project dependencies."""

    source_files: list[Path] = field(default_factory=list)
    """All Python source files found."""

    model_artifacts: list[Path] = field(default_factory=list)
    """Model artifact files (.joblib, .pkl, .pt, .h5, etc.)."""

    services_dir: Path | None = None
    """Path to services directory if found."""

    api_dir: Path | None = None
    """Path to API directory if found."""

    routes_dir: Path | None = None
    """Path to routes directory if found."""

    main_app_file: Path | None = None
    """Path to the main FastAPI application file."""

    src_package: Path | None = None
    """Path to the main source package."""


class ProjectScanner:
    """Scans UV-managed FastAPI projects.

    Identifies project structure, dependencies, model artifacts,
    and key directories following ArjanCodes patterns.
    """

    # Common model artifact extensions
    MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle", ".pt", ".pth", ".h5", ".keras", ".onnx"}

    # Directories to skip when scanning
    SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache", "node_modules", ".tox"}

    def __init__(self, project_path: str | Path) -> None:
        """Initialize the scanner.

        Args:
            project_path: Path to the project root directory.

        Raises:
            FileNotFoundError: If the project path doesn't exist.
            ValueError: If pyproject.toml is not found.
        """
        self.root = Path(project_path).resolve()

        if not self.root.exists():
            raise FileNotFoundError(f"Project path not found: {self.root}")

        self.pyproject_path = self.root / "pyproject.toml"
        if not self.pyproject_path.exists():
            raise ValueError(f"No pyproject.toml found in {self.root}")

    def scan(self) -> ProjectScan:
        """Scan the project and return structured representation.

        Returns:
            ProjectScan with all discovered project information.
        """
        # Parse pyproject.toml
        pyproject = self._parse_pyproject()

        # Extract project metadata
        project_name = self._extract_project_name(pyproject)
        python_version = self._extract_python_version(pyproject)
        dependencies = self._extract_dependencies(pyproject)

        # Create initial scan
        scan = ProjectScan(
            root=self.root,
            pyproject=pyproject,
            project_name=project_name,
            python_version=python_version,
            dependencies=dependencies,
        )

        # Find source package
        scan.src_package = self._find_src_package(project_name)

        # Scan for files and directories
        scan.source_files = self._find_source_files()
        scan.model_artifacts = self._find_model_artifacts()

        # Find key directories
        if scan.src_package:
            scan.services_dir = self._find_directory(scan.src_package, "services")
            scan.api_dir = self._find_directory(scan.src_package, "api")
            scan.routes_dir = self._find_directory(
                scan.api_dir or scan.src_package, "routes"
            )
            scan.main_app_file = self._find_main_app(scan.api_dir or scan.src_package)

        return scan

    def _parse_pyproject(self) -> dict:
        """Parse pyproject.toml file."""
        with open(self.pyproject_path, "rb") as f:
            return tomllib.load(f)

    def _extract_project_name(self, pyproject: dict) -> str:
        """Extract project name from pyproject.toml."""
        # Try [project] table first (PEP 621)
        if "project" in pyproject and "name" in pyproject["project"]:
            return pyproject["project"]["name"]

        # Fall back to [tool.poetry] for Poetry projects
        if "tool" in pyproject and "poetry" in pyproject["tool"]:
            return pyproject["tool"]["poetry"].get("name", "unknown")

        return "unknown"

    def _extract_python_version(self, pyproject: dict) -> str:
        """Extract Python version requirement."""
        # Try [project] table
        if "project" in pyproject:
            requires = pyproject["project"].get("requires-python", "")
            # Parse ">=3.11" to "3.11"
            if requires:
                import re
                match = re.search(r"(\d+\.\d+)", requires)
                if match:
                    return match.group(1)

        # Try [tool.poetry]
        if "tool" in pyproject and "poetry" in pyproject["tool"]:
            deps = pyproject["tool"]["poetry"].get("dependencies", {})
            python_req = deps.get("python", "")
            if python_req:
                import re
                match = re.search(r"(\d+\.\d+)", python_req)
                if match:
                    return match.group(1)

        return "3.11"  # Default

    def _extract_dependencies(self, pyproject: dict) -> list[str]:
        """Extract project dependencies."""
        deps = []

        # PEP 621 style
        if "project" in pyproject:
            deps.extend(pyproject["project"].get("dependencies", []))

        # Poetry style
        if "tool" in pyproject and "poetry" in pyproject["tool"]:
            poetry_deps = pyproject["tool"]["poetry"].get("dependencies", {})
            for name, version in poetry_deps.items():
                if name.lower() != "python":
                    if isinstance(version, str):
                        deps.append(f"{name}{version}" if version[0] in "<>=^~" else name)
                    else:
                        deps.append(name)

        return deps

    def _find_src_package(self, project_name: str) -> Path | None:
        """Find the main source package directory."""
        # Convert project name to package name (hyphens to underscores)
        package_name = project_name.replace("-", "_")

        # Check common locations
        candidates = [
            self.root / "src" / package_name,
            self.root / package_name,
            self.root / "src",
        ]

        for candidate in candidates:
            if candidate.exists() and candidate.is_dir():
                # Verify it's a Python package
                if (candidate / "__init__.py").exists():
                    return candidate
                # Or has Python files
                if list(candidate.glob("*.py")):
                    return candidate

        return None

    def _find_source_files(self) -> list[Path]:
        """Find all Python source files."""
        source_files = []

        for path in self.root.rglob("*.py"):
            # Skip excluded directories
            if any(skip in path.parts for skip in self.SKIP_DIRS):
                continue
            source_files.append(path)

        return sorted(source_files)

    def _find_model_artifacts(self) -> list[Path]:
        """Find model artifact files."""
        artifacts = []

        # Check models/ directory first
        models_dir = self.root / "models"
        if models_dir.exists():
            for ext in self.MODEL_EXTENSIONS:
                artifacts.extend(models_dir.glob(f"*{ext}"))

        # Also check root and other common locations
        for ext in self.MODEL_EXTENSIONS:
            for path in self.root.rglob(f"*{ext}"):
                if any(skip in path.parts for skip in self.SKIP_DIRS):
                    continue
                if path not in artifacts:
                    artifacts.append(path)

        return sorted(artifacts)

    def _find_directory(self, parent: Path, name: str) -> Path | None:
        """Find a named directory within a parent."""
        candidate = parent / name
        if candidate.exists() and candidate.is_dir():
            return candidate
        return None

    def _find_main_app(self, search_dir: Path) -> Path | None:
        """Find the main FastAPI application file."""
        # Common names for FastAPI main files
        candidates = ["main.py", "app.py", "application.py", "__main__.py"]

        for name in candidates:
            path = search_dir / name
            if path.exists():
                # Verify it contains FastAPI
                content = path.read_text()
                if "FastAPI" in content or "fastapi" in content:
                    return path

        return None
