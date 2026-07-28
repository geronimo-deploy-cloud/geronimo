"""Project generator for Geronimo.

Generates complete FastAPI ML project structure with model serving scaffolding.
"""

from pathlib import Path

import geronimo
from geronimo.config.loader import save_config
from geronimo.config.schema import (
    DeploymentConfig,
    EnvironmentConfig,
    GeronimoConfig,
    InfrastructureConfig,
    MLFramework,
    ModelConfig,
    ModelType,
    MonitoringConfig,
    ProjectConfig,
    RuntimeConfig,
    ScalingConfig,
)

from geronimo.generators.base import BaseGenerator
from geronimo.generators.template_engine import TemplateEngine

CORE_DEPS = [
    "geronimo",
    "pydantic>=2.5.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "boto3>=1.34.0",
]

REALTIME_DEPS = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "fastmcp>=2.0.0",
]

BATCH_DEPS = [
    "metaflow>=2.10.0",
    "evidently>=0.7.20",
]

class ProjectGenerator(BaseGenerator):
    """Generates a complete FastAPI ML project structure."""

    TEMPLATE_DIR = "project"

    project_name: str
    """The project name (kebab-case)."""

    framework: MLFramework
    """The selected ML framework."""

    output_dir: Path
    """The output directory path."""

    project_dir: Path
    """The full path to the project directory."""

    template: str
    """The selected project template (realtime/batch/both)."""

    engine: TemplateEngine
    """The template rendering engine."""

    def __init__(
        self,
        project_name: str,
        framework: str = "sklearn",
        output_dir: str = ".",
        template: str = "realtime",
    ) -> None:
        """Initialize the project generator.

        Args:
            project_name: Name of the project.
            framework: ML framework to use.
            output_dir: Directory to create the project in.
            template: Project template (realtime, batch, or both).
        """
        super().__init__()
        self.project_name = project_name.lower().replace(" ", "-")
        self.framework = MLFramework(framework.lower())
        self.output_dir = Path(output_dir)
        self.project_dir = self.output_dir / self.project_name
        self.template = template
        self.engine = TemplateEngine()

    def _get_framework_dependencies(self) -> list[str]:
        """Get framework-specific dependencies."""
        deps = {
            MLFramework.SKLEARN: ["scikit-learn>=1.3.0", "joblib>=1.3.0"],
            MLFramework.PYTORCH: ["torch>=2.0.0"],
            MLFramework.TENSORFLOW: ["tensorflow>=2.13.0"],
            MLFramework.XGBOOST: ["xgboost>=2.0.0"],
            MLFramework.CUSTOM: [],
        }
        return deps.get(self.framework, [])

    def _get_template_dependencies(self) -> list[str]:
        """Get template-specific dependencies."""
        # Core deps for all templates
        core = CORE_DEPS

        # Template-specific deps
        if self.template == "realtime":
            template_deps = REALTIME_DEPS
        elif self.template == "batch":
            template_deps = BATCH_DEPS
        else:  # both
            template_deps = REALTIME_DEPS + BATCH_DEPS

        # Framework-specific deps
        framework_deps = self._get_framework_dependencies()

        return core + template_deps + framework_deps

    def _to_pascal_case(self, name: str) -> str:
        """Convert kebab-case or snake_case name to PascalCase.
        
        Args:
            name: Name in kebab-case (my-project) or snake_case (my_project)
            
        Returns:
            PascalCase version (MyProject)
            
        Examples:
            >>> self._to_pascal_case("my-project")
            'MyProject'
            >>> self._to_pascal_case("test_batch")
            'TestBatch'
        """
        return ''.join(
            word.title() for word in name.replace("-", "_").split("_")
        )

    def _create_config(self) -> GeronimoConfig:
        """Create the default configuration for this project."""
        # Determine model type based on template
        model_type = ModelType.BATCH if self.template == "batch" else ModelType.REALTIME
        
        # Base dependencies
        base_deps = [
            "pydantic>=2.5.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            *self._get_framework_dependencies(),
        ]
        
        # Template-specific dependencies
        if self.template == "batch":
            runtime_deps = base_deps + ["metaflow>=2.10.0", "pyarrow>=14.0.0"]
        else:
            runtime_deps = base_deps + [
                "fastapi>=0.109.0",
                "uvicorn[standard]>=0.27.0",
            ]
        
        # Template-specific monitoring metrics
        if self.template == "batch":
            metrics = [
                "records_processed",
                "pipeline_duration",
                "error_rate",
                "drift_score",
            ]
        else:
            metrics = [
                "latency_p50",
                "latency_p99",
                "error_rate",
                "request_count",
            ]

        # Template-specific infrastructure
        if self.template == "batch":
            infrastructure = InfrastructureConfig(
                cpu=1024,
                memory=4096,
            )
        else:
            infrastructure = InfrastructureConfig(
                cpu=512,
                memory=1024,
                scaling=ScalingConfig(
                    min_instances=1,
                    max_instances=4,
                ),
            )

        config = GeronimoConfig(
            project=ProjectConfig(
                name=self.project_name,
                version="1.0.0",
                description=f"ML batch pipeline for {self.project_name}" if self.template == "batch" else f"ML model serving API for {self.project_name}",
            ),
            model=ModelConfig(
                type=model_type,
                framework=self.framework,
                artifact_path="models/model.joblib",
                mcp_enabled=self.template != "batch",
            ),
            runtime=RuntimeConfig(
                python_version="3.11",
                dependencies=runtime_deps,
            ),
            infrastructure=infrastructure,
            monitoring=MonitoringConfig(
                metrics=metrics,
                dashboard_enabled=True,
            ),
            deployment=DeploymentConfig(
                environments=[
                    EnvironmentConfig(name="dev", auto_deploy=True),
                    EnvironmentConfig(name="prod", approval_required=True),
                ],
            ),
        )

        # Enable batch config with a starter job for batch projects
        if self.template == "batch":
            from geronimo.config.schema import BatchConfig, BatchJobConfig

            config.batch = BatchConfig(
                enabled=True,
                jobs=[
                    BatchJobConfig(
                        name=f"{self.project_name}-scoring",
                        flow_file=f"src/{self.project_name.replace('-', '_')}/flow.py",
                    ),
                ],
            )

        return config

    def generate(self) -> Path:
        """Generate the complete project structure.

        Returns:
            Path to the created project directory.
        """
        # Create project directory
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Generate configuration
        config = self._create_config()
        save_config(config, self.project_dir / "geronimo.yaml")

        # Generate source code
        self._generate_source_code()

        # Generate monitoring code
        # Monitoring is needed for both realtime (metrics) and batch (drift detection)
        if self.template in ("realtime", "batch", "both"):
            self._generate_monitoring()

        # Generate project files
        self._generate_project_files()

        return self.project_dir

    def _generate_source_code(self) -> None:
        """Generate SDK-first application structure.
        
        SDK files (user edits):
            - sdk/model.py - Model train/predict
            - sdk/features.py - FeatureSet definition
            - sdk/data_sources.py - DataSource config
            - sdk/endpoint.py - [realtime] preprocess/postprocess
            - sdk/pipeline.py - [batch] run() logic
        
        Wrappers (thin, rarely edited):
            - app.py - [realtime] FastAPI imports SDK
            - flow.py - [batch] Metaflow imports SDK
        """
        src = self.project_dir / "src"
        context = {
            "project_name": self.project_name,
            "project_name_snake": self.project_name.replace("-", "_"),
            "framework": self.framework.value,
        }

        # Main package
        pkg_dir = src / context["project_name_snake"]
        pkg_dir.mkdir(parents=True, exist_ok=True)
        self.write_file(pkg_dir / "__init__.py", f'"""ML package for {self.project_name}."""\n')

        # ==============================
        # SDK Core (always generated) - use templates
        # ==============================
        sdk_dir = pkg_dir / "sdk"
        sdk_dir.mkdir(exist_ok=True)
        self.write_file(sdk_dir / "__init__.py", '"""Geronimo SDK - define your model lifecycle here."""\n')

        # ==============================
        # Experiments package (always generated)
        # ==============================
        experiments_dir = pkg_dir / "experiments"
        experiments_dir.mkdir(exist_ok=True)
        self.write_file(
            experiments_dir / "__init__.py",
            '"""Experiments package — ad-hoc model testing and iteration.\n'
            '\n'
            '    This directory is intentionally excluded from production code paths.\n'
            '    It is the designated space for ad-hoc testing and experimentation\n'
            '    of the model during development.\n'
            '\n'
            '    Scripts here are not expected to be production-quality; they\n'
            '    exist to help developers explore, iterate, and validate\n'
            '    behavior quickly.\n'
            '"""',
        )

        # Use template engine for SDK files
        self.engine.render_to_file("sdk/model.py.jinja2", context, sdk_dir / "model.py")
        self.engine.render_to_file("sdk/features.py.jinja2", context, sdk_dir / "features.py")
        self.engine.render_to_file("sdk/data_sources.py.jinja2", context, sdk_dir / "data_sources.py")

        # ==============================
        # Template-specific SDK files
        # ==============================
        if self.template in ("realtime", "both"):
            self.engine.render_to_file("sdk/endpoint.py.jinja2", context, sdk_dir / "endpoint.py")
            self.engine.render_to_file("sdk/monitoring_config.py.jinja2", context, sdk_dir / "monitoring_config.py")
            self.engine.render_to_file("project/app.py.jinja2", context, pkg_dir / "app.py")
            # Generate MCP agent package for AI integration
            self._generate_agent_package(context)
        
        if self.template in ("batch", "both"):
            self.engine.render_to_file("sdk/pipeline.py.jinja2", context, sdk_dir / "pipeline.py")
            self.engine.render_to_file("sdk/batch_monitoring_config.py.jinja2", context, sdk_dir / "monitoring_config.py")
            self.engine.render_to_file("project/flow.py.jinja2", context, pkg_dir / "flow.py")

        # ==============================
        # Tests
        # ==============================
        tests_dir = self.project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        self.write_file(tests_dir / "__init__.py", '"""Tests package."""\n')
        
        self.engine.render_to_file("project/test_sdk.py.jinja2", context, tests_dir / "test_sdk.py")

    def _generate_monitoring(self) -> None:
        """Generate monitoring package."""
        src = self.project_dir / "src"
        pkg_dir = src / self.project_name.replace("-", "_")
        monitor_dir = pkg_dir / "monitoring"
        monitor_dir.mkdir(exist_ok=True)
        
        # Template-specific __init__.py — batch projects skip FastAPI middleware
        if self.template == "batch":
            self.write_file(
                monitor_dir / "__init__.py", 
                '"""Monitoring package."""\n\n'
                'from .metrics import MetricsCollector, MetricType\n'
                'from .alerts import AlertManager, SlackAlert\n'
                'from .drift import DriftDetector\n'
                '\n'
                '__all__ = [\n'
                '    "MetricsCollector",\n'
                '    "MetricType",\n'
                '    "AlertManager",\n'
                '    "SlackAlert",\n'
                '    "DriftDetector",\n'
                ']\n'
            )
        else:
            self.write_file(
                monitor_dir / "__init__.py", 
                '"""Monitoring package."""\n\n'
                'from .metrics import MetricsCollector, MetricType\n'
                'from .alerts import AlertManager, SlackAlert\n'
                'from .middleware import MonitoringMiddleware\n'
                'from .drift import DriftDetector\n'
                '\n'
                '__all__ = [\n'
                '    "MetricsCollector",\n'
                '    "MetricType",\n'
                '    "AlertManager",\n'
                '    "SlackAlert",\n'
                '    "MonitoringMiddleware",\n'
                '    "DriftDetector",\n'
                ']\n'
            )

        # Read templates from installed package
        package_root = Path(geronimo.__file__).parent
        template_dir = package_root / "generators" / "templates" / "monitoring"
        
        # Batch projects don't need the FastAPI middleware
        if self.template == "batch":
            files = {
                "metrics.py": "metrics.py",
                "alerts.py": "alerts.py",
                "drift.py": "drift.py",
            }
        else:
            files = {
                "metrics.py": "metrics.py",
                "alerts.py": "alerts.py",
                "middleware.py": "middleware.py",
                "drift.py": "drift.py",
            }

        for dest_name, src_name in files.items():
            template_path = template_dir / src_name
            if not template_path.exists():
                continue
                
            source = template_path.read_text()
            
            # Replace absolute imports with relative imports
            source = source.replace("from geronimo.monitoring.metrics", "from .metrics")
            source = source.replace("from geronimo.monitoring.alerts", "from .alerts")
            source = source.replace("from geronimo.monitoring.middleware", "from .middleware")
            source = source.replace("from geronimo.monitoring.drift", "from .drift")
            
            self.write_file(monitor_dir / dest_name, source)

    def _generate_agent_package(self, context: dict) -> None:
        """Generate MCP agent file for AI agent integration.
        
        Creates agent.py at the package level (alongside app.py).
        """
        src = self.project_dir / "src"
        pkg_dir = src / context["project_name_snake"]

        # Generate agent.py at package level (like app.py)
        self.engine.render_to_file(
            "sdk/agent_server.py.jinja2", 
            context, 
            pkg_dir / "agent.py"
        )

    def _generate_project_files(self) -> None:
        """Generate project-level configuration files."""
        context = {
            "project_name": self.project_name,
            "project_name_snake": self.project_name.replace("-", "_"),
        }

        # Template-specific dependencies
        deps = self._get_template_dependencies()
        deps_str = ",\n    ".join(f'"{d}"' for d in deps)

        # pyproject.toml
        description = "ML batch pipeline" if self.template == "batch" else "ML model serving API"
        pyproject_context = {**context, "deps_str": deps_str, "description": description}
        self.engine.render_to_file("project/pyproject.toml.jinja2", pyproject_context, self.project_dir / "pyproject.toml")

        # Generate training script
        self._generate_training_script(context)

        # README.md - use template-appropriate README
        readme_template = "project/README_batch.md.jinja2" if self.template == "batch" else "project/README.md.jinja2"
        self.engine.render_to_file(readme_template, context, self.project_dir / "README.md")

        # .gitignore
        self.engine.render_to_file("project/gitignore.jinja2", context, self.project_dir / ".gitignore")

    def _generate_training_script(self, context: dict) -> None:
        """Generate training script template."""
        pkg_dir = self.project_dir / "src" / context["project_name_snake"]
        project_name_pascal = self._to_pascal_case(context["project_name"])

        train_context = {**context, "project_name_pascal": project_name_pascal}
        self.engine.render_to_file("project/train.py.jinja2", train_context, pkg_dir / "train.py")


