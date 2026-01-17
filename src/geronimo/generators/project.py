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


class ProjectGenerator(BaseGenerator):
    """Generates a complete FastAPI ML project structure."""

    TEMPLATE_DIR = "project"

    def __init__(
        self,
        project_name: str,
        framework: str = "sklearn",
        output_dir: str = ".",
    ) -> None:
        """Initialize the project generator.

        Args:
            project_name: Name of the project.
            framework: ML framework to use.
            output_dir: Directory to create the project in.
        """
        super().__init__()
        self.project_name = project_name.lower().replace(" ", "-")
        self.framework = MLFramework(framework.lower())
        self.output_dir = Path(output_dir)
        self.project_dir = self.output_dir / self.project_name

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

    def _create_config(self) -> GeronimoConfig:
        """Create the default configuration for this project."""
        return GeronimoConfig(
            project=ProjectConfig(
                name=self.project_name,
                version="1.0.0",
                description=f"ML model serving API for {self.project_name}",
            ),
            model=ModelConfig(
                type=ModelType.REALTIME,
                framework=self.framework,
                artifact_path="models/model.joblib",
            ),
            runtime=RuntimeConfig(
                python_version="3.11",
                dependencies=[
                    "fastapi>=0.109.0",
                    "uvicorn[standard]>=0.27.0",
                    "pydantic>=2.5.0",
                    "numpy>=1.24.0",
                    "pandas>=2.0.0",
                    *self._get_framework_dependencies(),
                ],
            ),
            infrastructure=InfrastructureConfig(
                cpu=512,
                memory=1024,
                scaling=ScalingConfig(
                    min_instances=1,
                    max_instances=4,
                ),
            ),
            monitoring=MonitoringConfig(
                metrics=[
                    "latency_p50",
                    "latency_p99",
                    "error_rate",
                    "request_count",
                ],
                dashboard_enabled=True,
            ),
            deployment=DeploymentConfig(
                environments=[
                    EnvironmentConfig(name="dev", auto_deploy=True),
                    EnvironmentConfig(name="prod", approval_required=True),
                ],
            ),
        )

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
        self._generate_monitoring()

        # Generate project files
        self._generate_project_files()

        # Create models directory
        (self.project_dir / "models").mkdir(exist_ok=True)
        (self.project_dir / "models" / ".gitkeep").touch()

        return self.project_dir

    def _generate_source_code(self) -> None:
        """Generate the FastAPI application source code."""
        src = self.project_dir / "src"

        # Create package structure
        context = {
            "project_name": self.project_name,
            "project_name_snake": self.project_name.replace("-", "_"),
            "framework": self.framework.value,
        }

        # Main package
        pkg_dir = src / context["project_name_snake"]
        pkg_dir.mkdir(parents=True, exist_ok=True)
        self.write_file(pkg_dir / "__init__.py", '"""ML serving package."""\n')

        # API module
        api_dir = pkg_dir / "api"
        api_dir.mkdir(exist_ok=True)
        self.write_file(api_dir / "__init__.py", '"""API package."""\n')

        # Generate main.py
        main_content = self._generate_main_py(context)
        self.write_file(api_dir / "main.py", main_content)
        # Generate deps.py
        deps_content = self._generate_deps(context)
        self.write_file(api_dir / "deps.py", deps_content)

        # Generate agent package
        self._generate_agent_package(context)




        # Routes
        routes_dir = api_dir / "routes"
        routes_dir.mkdir(exist_ok=True)
        self.write_file(routes_dir / "__init__.py", '"""Routes package."""\n')

        # Health route
        health_content = self._generate_health_route()
        self.write_file(routes_dir / "health.py", health_content)

        # Predict route
        predict_content = self._generate_predict_route(context)
        self.write_file(routes_dir / "predict.py", predict_content)

        # Models (schemas)
        models_dir = api_dir / "models"
        models_dir.mkdir(exist_ok=True)
        self.write_file(models_dir / "__init__.py", '"""Pydantic models."""\n')

        schemas_content = self._generate_schemas(context)
        self.write_file(models_dir / "schemas.py", schemas_content)

        # ML module
        ml_dir = pkg_dir / "ml"
        ml_dir.mkdir(exist_ok=True)
        self.write_file(ml_dir / "__init__.py", '"""ML module."""\n')

        predictor_content = self._generate_predictor(context)
        self.write_file(ml_dir / "predictor.py", predictor_content)

        # Tests
        tests_dir = self.project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        self.write_file(tests_dir / "__init__.py", '"""Tests package."""\n')

        test_api_content = self._generate_test_api(context)
        self.write_file(tests_dir / "test_api.py", test_api_content)



    def _generate_monitoring(self) -> None:
        """Generate monitoring package."""
        src = self.project_dir / "src"
        pkg_dir = src / self.project_name.replace("-", "_")
        monitor_dir = pkg_dir / "monitoring"
        monitor_dir.mkdir(exist_ok=True)
        
        # Create __init__.py for new package
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
        template_dir = package_root / "templates" / "monitoring"
        
        files = {
            "metrics.py": "metrics.py",
            "alerts.py": "alerts.py",
            "middleware.py": "middleware.py",
            "drift.py": "drift.py",
        }

        for dest_name, src_name in files.items():
            template_path = template_dir / src_name
            if not template_path.exists():
                # Fallback implementation or error
                # For basic functionality in development mode where files might not be moved yet?
                # No, we assume it exists.
                continue
                
            source = template_path.read_text()
            
            # Fix imports using simple string replacement
            # The original files had "from geronimo.monitoring..."
            # We need to change that to "from ." or "from <pkg>.monitoring"
            
            # Replace absolute imports with relative imports which is cleaner for internal package
            source = source.replace("from geronimo.monitoring.metrics", "from .metrics")
            source = source.replace("from geronimo.monitoring.alerts", "from .alerts")
            source = source.replace("from geronimo.monitoring.middleware", "from .middleware")
            source = source.replace("from geronimo.monitoring.drift", "from .drift")
            
            self.write_file(monitor_dir / dest_name, source)

    def _generate_main_py(self, context: dict) -> str:
        """Generate the FastAPI main application."""
        return f'''"""FastAPI application for {context["project_name"]} ML serving."""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from {context["project_name_snake"]}.api.routes import health, predict
from {context["project_name_snake"]}.ml.predictor import ModelPredictor
from {context["project_name_snake"]}.monitoring.middleware import MonitoringMiddleware
from {context["project_name_snake"]}.monitoring.metrics import MetricsCollector
from {context["project_name_snake"]}.api import deps
from {context["project_name_snake"]}.agent.server import mcp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize metrics collector
metrics = MetricsCollector(project_name="{context["project_name"]}")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler for model loading."""
    logger.info("Loading model...")
    deps.predictor = ModelPredictor()
    deps.predictor.load()
    logger.info("Model loaded successfully")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title="{context["project_name"]}",
    description="ML model serving API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring middleware
app.add_middleware(MonitoringMiddleware, collector=metrics)

# Mount MCP Agent (Streamable HTTP)
if os.getenv("ENABLE_MCP_AGENT", "true").lower() == "true":
    app.mount("/mcp", mcp.streamable_http_app())

# Include routers
app.include_router(health.router, tags=["Health"])
app.include_router(predict.router, prefix="/v1", tags=["Predictions"])
'''

    def _generate_deps(self, context: dict) -> str:
        """Generate dependencies module."""
        return f'''"""API dependencies."""

from typing import Optional
from {context["project_name_snake"]}.ml.predictor import ModelPredictor

# Global model instance
predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """Get the loaded model predictor."""
    if predictor is None:
        raise RuntimeError("Model not loaded")
    return predictor
'''

    def _generate_health_route(self) -> str:
        """Generate the health check route."""
        return '''"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy"}


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check for load balancer."""
    # Add model readiness check here if needed
    return {"status": "ready"}
'''

    def _generate_predict_route(self, context: dict) -> str:
        """Generate the prediction route."""
        return f'''"""Prediction endpoints."""

import time
import logging

from fastapi import APIRouter, HTTPException, Depends

from {context["project_name_snake"]}.api.models.schemas import (
    PredictionRequest,
    PredictionResponse,
)
from {context["project_name_snake"]}.api.deps import get_predictor
from {context["project_name_snake"]}.ml.predictor import ModelPredictor

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    predictor: ModelPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Generate predictions for the input features.

    Args:
        request: Input features for prediction.
        predictor: The loaded model predictor (injected).

    Returns:
        Model predictions with metadata.
    """
    start_time = time.perf_counter()

    try:
        prediction = predictor.predict(request.features)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Prediction completed in {{latency_ms:.2f}}ms")

        return PredictionResponse(
            prediction=prediction,
            model_version=predictor.version,
            latency_ms=latency_ms,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {{e}}")
        raise HTTPException(status_code=500, detail=str(e))
'''

    def _generate_schemas(self, context: dict) -> str:
        """Generate Pydantic schemas for request/response."""
        return '''"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request schema for predictions."""

    features: dict[str, float | int | str | list] = Field(
        ...,
        description="Input features as key-value pairs",
        examples=[{"feature_1": 1.5, "feature_2": "category_a", "feature_3": [1, 2, 3]}],
    )


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: float | int | str | list = Field(
        ...,
        description="Model prediction result",
    )
    model_version: str = Field(
        ...,
        description="Version of the model used for prediction",
    )
    latency_ms: float = Field(
        ...,
        description="Prediction latency in milliseconds",
    )


class ErrorResponse(BaseModel):
    """Error response schema."""

    detail: str = Field(..., description="Error message")
'''

    def _generate_predictor(self, context: dict) -> str:
        """Generate the model predictor class."""
        load_code = self._get_framework_load_code(context["framework"])

        return f'''"""Model predictor for ML inference.

Handles model loading, caching, and prediction logic.
"""

import logging
from pathlib import Path
from typing import Any

{self._get_framework_imports(context["framework"])}

logger = logging.getLogger(__name__)

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = Path("models/model.joblib")


class ModelPredictor:
    """Handles model loading and predictions.

    Implements lazy loading and caching for efficient inference.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        """Initialize the predictor.

        Args:
            model_path: Path to the model artifact. Uses default if not provided.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model: Any = None
        self._version: str = "1.0.0"

    @property
    def version(self) -> str:
        """Get the model version."""
        return self._version

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        if not self.model_path.exists():
            logger.warning(
                f"Model file not found at {{self.model_path}}. "
                "Using placeholder for development."
            )
            self._model = self._create_placeholder_model()
            return

        try:
            logger.info(f"Loading model from {{self.model_path}}")
            {load_code}
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {{e}}")

    def _create_placeholder_model(self) -> Any:
        """Create a placeholder model for development/testing."""
        # Returns a simple function that echoes input
        return lambda x: 0.5

    def predict(self, features: dict[str, Any]) -> Any:
        """Generate predictions for input features.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Model prediction (type depends on model).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert features to model input format
        # This should be customized based on your model's requirements
        try:
            if callable(self._model):
                # Placeholder model
                return self._model(features)

            # For sklearn-style models with predict method
            import pandas as pd
            import numpy as np

            # Convert dict to DataFrame for sklearn compatibility
            df = pd.DataFrame([features])

            # Get prediction
            prediction = self._model.predict(df)

            # Return single value if single prediction
            if isinstance(prediction, np.ndarray) and len(prediction) == 1:
                return float(prediction[0])

            return prediction.tolist()

        except Exception as e:
            logger.error(f"Prediction failed: {{e}}")
            raise
'''

    def _get_framework_imports(self, framework: str) -> str:
        """Get framework-specific imports."""
        imports = {
            "sklearn": "import joblib",
            "pytorch": "import torch",
            "tensorflow": "import tensorflow as tf",
            "xgboost": "import xgboost as xgb\nimport joblib",
            "custom": "",
        }
        return imports.get(framework, "")

    def _get_framework_load_code(self, framework: str) -> str:
        """Get framework-specific model loading code."""
        load_code = {
            "sklearn": "self._model = joblib.load(self.model_path)",
            "pytorch": "self._model = torch.load(self.model_path)\n            self._model.eval()",
            "tensorflow": "self._model = tf.keras.models.load_model(self.model_path)",
            "xgboost": "self._model = joblib.load(self.model_path)",
            "custom": "# Implement custom model loading",
        }
        return load_code.get(framework, "# Unknown framework")

    def _generate_test_api(self, context: dict) -> str:
        """Generate API tests."""
        return f'''"""Tests for the ML serving API."""

import pytest
from fastapi.testclient import TestClient

from {context["project_name_snake"]}.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    with TestClient(app) as c:
        yield c


def test_health_check(client: TestClient):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_check(client: TestClient):
    """Test readiness endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_predict(client: TestClient):
    """Test prediction endpoint."""
    response = client.post(
        "/v1/predict",
        json={{"features": {{"feature_1": 1.0, "feature_2": 2.0}}}},
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "model_version" in data
    assert "latency_ms" in data
'''

    def _generate_agent_package(self, context: dict) -> None:
        """Generate agent package (MCP)."""
        src = self.project_dir / "src"
        pkg_dir = src / context["project_name_snake"] / "agent"
        pkg_dir.mkdir(exist_ok=True)
        self.write_file(pkg_dir / "__init__.py", '"""Agent package."""\n')

        # Read template
        template_path = Path(geronimo.__file__).parent / "templates" / "agent" / "server.py"
        if template_path.exists():
            content = template_path.read_text()
            # Fix imports
            content = content.replace(
                "from geronimo.", 
                f"from {context['project_name_snake']}."
            )
            self.write_file(pkg_dir / "server.py", content)

    def _generate_project_files(self) -> None:
        """Generate project-level configuration files."""
        context = {
            "project_name": self.project_name,
            "project_name_snake": self.project_name.replace("-", "_"),
        }

        # pyproject.toml
        pyproject = f'''[project]
name = "{context["project_name"]}"
version = "1.0.0"
description = "ML model serving API"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.0",
    "boto3>=1.34.0",
    "evidently>=0.4.0",
    "mcp>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "httpx>=0.25.0",
    "pytest-cov>=4.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
'''
        self.write_file(self.project_dir / "pyproject.toml", pyproject)

        # README.md
        readme = f'''# {context["project_name"]}

ML model serving API generated by Geronimo.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the API locally
uv run uvicorn {context["project_name_snake"]}.api.main:app --reload

# Run tests
uv run pytest
```

## Project Structure

```
{context["project_name"]}/
├── geronimo.yaml          # Deployment configuration
├── pyproject.toml         # Python project config
├── Dockerfile             # Container definition
├── azure-pipelines.yaml   # CI/CD pipeline
├── infrastructure/        # Terraform files
├── src/
│   └── {context["project_name_snake"]}/
│       ├── api/          # FastAPI application
│       │   ├── main.py
│       │   ├── routes/
│       │   └── models/
│       └── ml/           # Model loading & inference
│           └── predictor.py
├── models/               # Model artifacts
└── tests/
```

## Deployment

```bash
# Generate all deployment artifacts
geronimo generate all

# Deploy infrastructure
cd infrastructure && terraform apply
```
'''
        self.write_file(self.project_dir / "README.md", readme)

        # .gitignore
        gitignore = '''# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env

# IDE
.idea/
.vscode/
*.swp

# Testing
.coverage
htmlcov/
.pytest_cache/

# Build
dist/
build/
*.egg-info/

# Terraform
.terraform/
*.tfstate
*.tfstate.*
.terraform.lock.hcl

# Models (large files)
models/*.joblib
models/*.pkl
models/*.pt
models/*.h5
!models/.gitkeep
'''
        self.write_file(self.project_dir / ".gitignore", gitignore)
