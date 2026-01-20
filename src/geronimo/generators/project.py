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
        core = [
            "geronimo",
            "pydantic>=2.5.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "boto3>=1.34.0",
        ]

        # Template-specific deps
        if self.template == "realtime":
            template_deps = [
                "fastapi>=0.109.0",
                "uvicorn[standard]>=0.27.0",
                "mcp>=0.1.0",
            ]
        elif self.template == "batch":
            template_deps = [
                "metaflow>=2.10.0",
            ]
        else:  # both
            template_deps = [
                "fastapi>=0.109.0",
                "uvicorn[standard]>=0.27.0",
                "mcp>=0.1.0",
                "metaflow>=2.10.0",
            ]

        # Framework-specific deps
        framework_deps = self._get_framework_dependencies()

        return core + template_deps + framework_deps

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
        
        return GeronimoConfig(
            project=ProjectConfig(
                name=self.project_name,
                version="1.0.0",
                description=f"ML model serving API for {self.project_name}" if self.template != "batch" else f"ML batch pipeline for {self.project_name}",
            ),
            model=ModelConfig(
                type=model_type,
                framework=self.framework,
                artifact_path="models/model.joblib",
            ),
            runtime=RuntimeConfig(
                python_version="3.11",
                dependencies=runtime_deps,
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

        # Generate monitoring code (only for realtime/both)
        if self.template in ("realtime", "both"):
            self._generate_monitoring()

        # Generate project files
        self._generate_project_files()

        # Create models directory
        (self.project_dir / "models").mkdir(exist_ok=True)
        (self.project_dir / "models" / ".gitkeep").touch()

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
        # SDK Core (always generated)
        # ==============================
        sdk_dir = pkg_dir / "sdk"
        sdk_dir.mkdir(exist_ok=True)
        self.write_file(sdk_dir / "__init__.py", '"""Geronimo SDK - define your model lifecycle here."""\n')
        
        # model.py
        self.write_file(sdk_dir / "model.py", self._generate_sdk_model(context))
        
        # features.py
        self.write_file(sdk_dir / "features.py", self._generate_sdk_features(context))
        
        # data_sources.py
        self.write_file(sdk_dir / "data_sources.py", self._generate_sdk_data_sources(context))

        # ==============================
        # Template-specific SDK files
        # ==============================
        if self.template in ("realtime", "both"):
            self.write_file(sdk_dir / "endpoint.py", self._generate_sdk_endpoint(context))
            self.write_file(sdk_dir / "monitoring_config.py", self._generate_sdk_monitoring_config(context))
            self.write_file(pkg_dir / "app.py", self._generate_app_wrapper(context))
        
        if self.template in ("batch", "both"):
            self.write_file(sdk_dir / "pipeline.py", self._generate_sdk_pipeline(context))
            self.write_file(sdk_dir / "monitoring_config.py", self._generate_sdk_batch_monitoring_config(context))
            self.write_file(pkg_dir / "flow.py", self._generate_flow_wrapper(context))
            
            # Batch directory structure
            batch_dir = self.project_dir / "batch"
            batch_dir.mkdir(exist_ok=True)
            (batch_dir / "data").mkdir(exist_ok=True)
            (batch_dir / "output").mkdir(exist_ok=True)

        # ==============================
        # Tests
        # ==============================
        tests_dir = self.project_dir / "tests"
        tests_dir.mkdir(exist_ok=True)
        self.write_file(tests_dir / "__init__.py", '"""Tests package."""\n')
        
        self.write_file(tests_dir / "test_sdk.py", self._generate_test_sdk(context))

    def _generate_sdk_model(self, context: dict) -> str:
        """Generate SDK model.py file."""
        return f'''"""Model definition - implement your ML model here."""

from geronimo.models import Model, HyperParams
from .features import ProjectFeatures
from .data_sources import training_data  # Import your data source


class ProjectModel(Model):
    """Main model class.
    
    Define your model's train and predict methods.
    The features attribute connects to your FeatureSet.
    The data_source attribute defines where training data comes from.
    
    Example:
        from sklearn.ensemble import RandomForestClassifier
        
        def train(self, X, y, params):
            self.estimator = RandomForestClassifier(**params.to_dict())
            self.estimator.fit(X, y)
    """

    name = "{context["project_name"]}"
    version = "1.0.0"
    features = ProjectFeatures()
    data_source = training_data  # Connect to data source

    def train(self, X, y, params: HyperParams) -> None:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            params: Hyperparameters from HyperParams
        """
        # TODO: Implement training logic
        # self.estimator = YourModel(**params.to_dict())
        # self.estimator.fit(X, y)
        raise NotImplementedError("Implement train() method")

    def predict(self, X):
        """Generate predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        # TODO: Implement prediction logic
        # return self.estimator.predict(X)
        raise NotImplementedError("Implement predict() method")
'''

    def _generate_sdk_features(self, context: dict) -> str:
        """Generate SDK features.py file."""
        return '''"""Feature definitions - define your feature engineering here."""

from geronimo.features import FeatureSet, Feature
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ProjectFeatures(FeatureSet):
    """Define your features here.
    
    Each Feature describes a column with its type, transformer, and encoder.
    
    Example:
        age = Feature(dtype='numeric', transformer=StandardScaler())
        income = Feature(dtype='numeric', transformer=StandardScaler())
        category = Feature(dtype='categorical', encoder=OneHotEncoder())
        
        # Derived feature from multiple columns
        age_income_ratio = Feature(
            dtype='numeric',
            derived_feature_fn=lambda df: df['age'] / df['income']
        )
    """
    
    # TODO: Define your features
    # feature_1 = Feature(dtype='numeric')
    # feature_2 = Feature(dtype='categorical')
    pass
'''

    def _generate_sdk_data_sources(self, context: dict) -> str:
        """Generate SDK data_sources.py file."""
        return f'''"""Data source definitions - configure where your data comes from.

This module is imported by model.py and pipeline.py to load training/scoring data.
"""

from geronimo.data import DataSource, Query


# =============================================================================
# Training Data Source (used by model.py)
# =============================================================================

# TODO: Configure your training data source
# Option 1: Local CSV file
training_data = DataSource(
    name="training",
    source="file",
    path="data/train.csv",  # Update with your path
)

# Option 2: Snowflake query
# training_data = DataSource(
#     name="training",
#     source="snowflake",
#     query=Query.from_file("queries/train.sql"),
# )

# Option 3: S3 parquet
# training_data = DataSource(
#     name="training",
#     source="file",
#     path="s3://my-bucket/data/train.parquet",
# )


# =============================================================================
# Scoring Data Source (used by pipeline.py for batch scoring)
# =============================================================================

scoring_data = DataSource(
    name="scoring",
    source="file",
    path="batch/data/input.csv",  # Update with your path
)
'''

    def _generate_sdk_endpoint(self, context: dict) -> str:
        """Generate SDK endpoint.py file for realtime serving."""
        return f'''"""Endpoint definition - handle incoming prediction requests."""

from geronimo.serving import Endpoint
from .model import ProjectModel


class PredictEndpoint(Endpoint):
    """Prediction endpoint for real-time serving.
    
    This is a working demo endpoint. Replace the preprocess/postprocess
    methods with your actual implementation once you have a trained model.
    
    To train a model:
        uv run python -m {context["project_name_snake"]}.train
    """

    model_class = ProjectModel

    def preprocess(self, request: dict):
        """Transform incoming request to model input.
        
        Args:
            request: JSON request body with "features" key
            
        Returns:
            Feature matrix ready for model.predict()
        """
        # Demo mode: just return the features dict
        # TODO: Replace with actual preprocessing once model is trained
        # import pandas as pd
        # df = pd.DataFrame([request["features"]])
        # return self.model.features.transform(df)
        return request.get("features", request)

    def postprocess(self, prediction):
        """Format model output for response.
        
        Args:
            prediction: Raw model output
            
        Returns:
            JSON-serializable response
        """
        # Demo mode: echo the input back
        # TODO: Replace with actual postprocessing once model is trained
        # return {{"prediction": int(prediction[0]), "confidence": 0.95}}
        return {{"result": prediction, "status": "demo_mode"}}
    
    def initialize(self, project=None, version=None):
        """Initialize endpoint.
        
        Demo mode: Skip model loading if artifacts don't exist.
        """
        try:
            super().initialize(project=project, version=version)
        except Exception:
            # Demo mode: continue without trained model
            self.model = None
            self._is_initialized = True
    
    def handle(self, request: dict) -> dict:
        """Handle prediction request.
        
        Demo mode: If no model loaded, echo the request back.
        """
        if self.model is None:
            # Demo mode
            features = self.preprocess(request)
            return self.postprocess(features)
        
        # Normal mode with trained model
        return super().handle(request)
'''

    def _generate_sdk_monitoring_config(self, context: dict) -> str:
        """Generate SDK monitoring_config.py file for realtime - focuses on metrics + alerts."""
        return f'''"""Realtime monitoring configuration - metrics thresholds and alerts.

Realtime endpoints focus on:
- Latency monitoring (p50, p99 thresholds)
- Error rate tracking
- Alerts for threshold breaches (Slack/email)

NOTE: For drift detection in realtime, consider using batch monitoring
to periodically analyze sampled request data. Full realtime drift is
planned for a future release.

TODO: Future enhancement - add realtime drift detection by:
- Sampling incoming requests to a buffer
- Periodically comparing buffer against reference data
- See: https://docs.evidentlyai.com/user-guide/installation/monitor_with_evidently
"""

import os
from {context["project_name_snake"]}.monitoring.alerts import AlertManager, SlackAlert, AlertSeverity
from {context["project_name_snake"]}.monitoring.metrics import MetricsCollector


# =============================================================================
# Latency Thresholds - customize these values
# =============================================================================

# Latency thresholds (milliseconds)
LATENCY_P50_WARNING = 100.0  # Alert if p50 latency exceeds this
LATENCY_P99_WARNING = 500.0  # Alert if p99 latency exceeds this

# Error rate thresholds (percentage)
ERROR_RATE_WARNING = 1.0     # Alert if error rate exceeds 1%
ERROR_RATE_CRITICAL = 5.0    # Critical alert if error rate exceeds 5%

# TODO: Future - drift thresholds for realtime (requires request sampling)
# DRIFT_THRESHOLD = 0.3      # Alert if drift score exceeds 30%


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager() -> AlertManager:
    \"\"\"Create and configure the alert manager.
    
    To enable Slack alerts:
    1. Create a Slack incoming webhook
    2. Set SLACK_WEBHOOK_URL environment variable
    
    Example:
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    \"\"\"
    alerts = AlertManager(cooldown_seconds=300)  # 5 min between duplicate alerts
    
    # Add Slack if configured
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(
            webhook_url=slack_webhook,
            channel=os.getenv("SLACK_CHANNEL"),  # Optional channel override
        )
    
    return alerts


# =============================================================================
# Drift Detection Configuration (Optional - requires Evidently)
# =============================================================================

def create_drift_detector(reference_data=None) -> DriftDetector:
    \"\"\"Create and configure the drift detector.
    
    Args:
        reference_data: Training data to compare production data against.
                       Load this from your data warehouse or artifact store.
    
    Example:
        import pandas as pd
        training_df = pd.read_parquet("data/training_sample.parquet")
        detector = create_drift_detector(reference_data=training_df)
    \"\"\"
    return DriftDetector(
        reference_data=reference_data,
        # TODO: Specify your feature types
        # categorical_features=["category", "region"],
        # numerical_features=["age", "income", "score"],
        # target_column="prediction",
    )


# =============================================================================
# Threshold Monitoring (call periodically or after each request)
# =============================================================================

def check_thresholds(metrics: MetricsCollector, alerts: AlertManager) -> None:
    \"\"\"Check metrics against thresholds and send alerts if breached.
    
    Call this periodically (e.g., every minute) or after batch processing.
    
    Example:
        # In app.py or a background task
        from {context["project_name_snake"]}.sdk.monitoring_config import (
            create_alert_manager, check_thresholds
        )
        from {context["project_name_snake"]}.monitoring.metrics import metrics
        
        alerts = create_alert_manager()
        check_thresholds(metrics, alerts)
    \"\"\"
    # Check latency
    p50 = metrics.get_latency_p50()
    if p50 > LATENCY_P50_WARNING:
        alerts.alert(
            title="High P50 Latency",
            message=f"P50 latency is {{p50:.1f}}ms (threshold: {{LATENCY_P50_WARNING}}ms)",
            severity=AlertSeverity.WARNING,
            metadata={{"current": p50, "threshold": LATENCY_P50_WARNING}},
        )
    
    p99 = metrics.get_latency_p99()
    if p99 > LATENCY_P99_WARNING:
        alerts.alert(
            title="High P99 Latency",
            message=f"P99 latency is {{p99:.1f}}ms (threshold: {{LATENCY_P99_WARNING}}ms)",
            severity=AlertSeverity.WARNING,
            metadata={{"current": p99, "threshold": LATENCY_P99_WARNING}},
        )
    
    # Check error rate
    error_count = metrics.get_error_count()
    request_count = metrics.get_request_count()
    if request_count > 0:
        error_rate = (error_count / request_count) * 100
        if error_rate > ERROR_RATE_CRITICAL:
            alerts.alert(
                title="Critical Error Rate",
                message=f"Error rate is {{error_rate:.1f}}% (threshold: {{ERROR_RATE_CRITICAL}}%)",
                severity=AlertSeverity.CRITICAL,
                metadata={{"error_rate": error_rate, "errors": error_count, "requests": request_count}},
            )
        elif error_rate > ERROR_RATE_WARNING:
            alerts.alert(
                title="Elevated Error Rate",
                message=f"Error rate is {{error_rate:.1f}}% (threshold: {{ERROR_RATE_WARNING}}%)",
                severity=AlertSeverity.WARNING,
                metadata={{"error_rate": error_rate}},
            )
'''

    def _generate_sdk_batch_monitoring_config(self, context: dict) -> str:
        """Generate SDK monitoring_config.py for batch jobs - focuses on drift + alerts."""
        return f'''"""Batch monitoring configuration - drift detection and alerts.

Batch jobs focus on:
- Drift detection (data drift between training and scoring data)
- Alerts for drift/failures (Slack/email notifications)

Unlike realtime endpoints, batch jobs don't need latency tracking.
"""

import os
import pandas as pd


# =============================================================================
# Drift Thresholds - customize these values
# =============================================================================

# Feature drift threshold (percentage of features drifted)
FEATURE_DRIFT_THRESHOLD = 0.3     # Alert if >30% of features show drift

# Dataset drift threshold (PSI/KS statistic)
DATASET_DRIFT_THRESHOLD = 0.1     # Alert if dataset drift score > 0.1

# Prediction drift threshold
PREDICTION_DRIFT_THRESHOLD = 0.2  # Alert if prediction distribution shifts


# =============================================================================
# Alert Configuration
# =============================================================================

def create_alert_manager():
    \"\"\"Create alert manager for batch job notifications.
    
    To enable Slack alerts:
        export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
    \"\"\"
    from {context["project_name_snake"]}.monitoring.alerts import AlertManager
    
    alerts = AlertManager(cooldown_seconds=0)  # No cooldown for batch (runs periodically)
    
    slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
    if slack_webhook:
        alerts.add_slack(
            webhook_url=slack_webhook,
            channel=os.getenv("SLACK_CHANNEL"),
        )
    
    return alerts


# =============================================================================
# Drift Detection - integrate into your pipeline.run() method
# =============================================================================

def create_drift_detector(reference_data: pd.DataFrame = None):
    \"\"\"Create drift detector for batch scoring.
    
    Args:
        reference_data: Training data sample to compare against.
                       Load from your artifact store or data warehouse.
    
    Usage in pipeline.py:
        from .monitoring_config import create_drift_detector, check_drift
        
        def run(self):
            # Load reference data (training sample)
            reference = pd.read_parquet("data/training_sample.parquet")
            detector = create_drift_detector(reference_data=reference)
            
            # Load scoring data
            scoring_data = self.data_source.load()
            
            # Check for drift before scoring
            drift_result = check_drift(detector, scoring_data)
            if drift_result["has_drift"]:
                # Log warning or alert
                pass
            
            # Continue with scoring...
    \"\"\"
    from {context["project_name_snake"]}.monitoring.drift import DriftDetector
    
    return DriftDetector(
        reference_data=reference_data,
        # TODO: Configure feature types for your model
        # categorical_features=["category", "region"],
        # numerical_features=["feature_1", "feature_2"],
        # target_column="prediction",
    )


def check_drift(detector, current_data: pd.DataFrame, alert_manager=None) -> dict:
    \"\"\"Check for drift and optionally send alerts.
    
    Args:
        detector: DriftDetector instance with reference data
        current_data: Current batch data to check for drift
        alert_manager: Optional AlertManager for notifications
    
    Returns:
        Dict with drift results and alert status
    \"\"\"
    from {context["project_name_snake"]}.monitoring.alerts import AlertSeverity
    
    result = detector.calculate_drift(current_data)
    
    has_drift = False
    if "drift_share" in result:
        has_drift = result["drift_share"] > FEATURE_DRIFT_THRESHOLD
        
        if has_drift and alert_manager:
            alert_manager.alert(
                title="Data Drift Detected",
                message=f"{{result['drift_share']*100:.1f}}% of features show drift (threshold: {{FEATURE_DRIFT_THRESHOLD*100}}%)",
                severity=AlertSeverity.WARNING,
                metadata={{
                    "drift_share": result["drift_share"],
                    "threshold": FEATURE_DRIFT_THRESHOLD,
                }},
            )
    
    return {{
        "has_drift": has_drift,
        "drift_result": result,
    }}


def send_pipeline_completion_alert(alert_manager, result: dict, success: bool = True):
    \"\"\"Send alert when pipeline completes.
    
    Usage in pipeline.py:
        from .monitoring_config import create_alert_manager, send_pipeline_completion_alert
        
        def run(self):
            alerts = create_alert_manager()
            result = {{"samples_scored": 1000}}
            send_pipeline_completion_alert(alerts, result)
    \"\"\"
    from {context["project_name_snake"]}.monitoring.alerts import AlertSeverity
    
    if success:
        alert_manager.alert(
            title="Batch Pipeline Complete",
            message=f"Successfully processed {{result.get('samples_scored', 'N/A')}} samples",
            severity=AlertSeverity.INFO,
            metadata=result,
        )
    else:
        alert_manager.alert(
            title="Batch Pipeline Failed",
            message=f"Pipeline failed: {{result.get('error', 'Unknown error')}}",
            severity=AlertSeverity.CRITICAL,
            metadata=result,
        )
'''

    def _generate_sdk_pipeline(self, context: dict) -> str:
        """Generate SDK pipeline.py file for batch processing."""
        return f'''"""Pipeline definition - implement your batch processing logic."""

from geronimo.batch import BatchPipeline, Schedule
from .model import ProjectModel
from .data_sources import scoring_data


class ScoringPipeline(BatchPipeline):
    """Batch scoring pipeline.
    
    This is a working demo pipeline. Replace the run() method with your
    actual implementation once you have a trained model.
    
    To train a model:
        uv run python -m {context["project_name_snake"]}.train
    """

    name = "{context["project_name"]}-scoring"
    model_class = ProjectModel
    schedule = Schedule.daily(hour=6, minute=0)
    data_source = scoring_data

    def initialize(self):
        """Initialize pipeline.
        
        Demo mode: Skip model loading if no artifacts exist.
        """
        try:
            super().initialize()
        except Exception:
            # Demo mode: continue without trained model
            self.model = None
            self._is_initialized = True
            print("Running in DEMO MODE (no trained model)")

    def execute(self):
        """Execute the pipeline.
        
        Demo mode: Return sample results if no model loaded.
        """
        if self.model is None:
            # Demo mode
            return self.run()
        return super().execute()

    def run(self):
        """Execute batch processing.
        
        Demo mode implementation - replace with your actual logic.
        
        Returns:
            Dict with execution results
        """
        # Demo mode: return sample results
        # TODO: Replace with actual batch logic once model is trained
        #
        # Example implementation:
        # data = self.data_source.load()
        # X = self.model.features.transform(data)
        # predictions = self.model.predict(X)
        # results = data.assign(prediction=predictions)
        # output_path = self.save_results(results)
        # return {{"samples_scored": len(results), "output_path": output_path}}
        
        return {{
            "status": "demo_mode",
            "message": "Pipeline executed successfully in demo mode",
            "samples_scored": 0,
        }}
'''


    def _generate_app_wrapper(self, context: dict) -> str:
        """Generate thin FastAPI wrapper that imports SDK endpoint."""
        return f'''"""FastAPI application - thin wrapper around SDK endpoint.

This app integrates:
- SDK endpoint for predictions
- Monitoring middleware for latency/error tracking
- Metrics collector for CloudWatch/custom backends
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from {context["project_name_snake"]}.sdk.endpoint import PredictEndpoint
from {context["project_name_snake"]}.monitoring.middleware import MonitoringMiddleware
from {context["project_name_snake"]}.monitoring.metrics import MetricsCollector


# =============================================================================
# Configuration - customize these values
# =============================================================================

PROJECT_NAME = "{context["project_name"]}"

# Metrics backend: "cloudwatch", "local", or custom
METRICS_BACKEND = "local"  # TODO: Change to "cloudwatch" for production


# =============================================================================
# Initialize components
# =============================================================================

# Initialize metrics collector
# For CloudWatch: MetricsCollector(project_name=PROJECT_NAME, namespace="MLModels")
metrics = MetricsCollector(project_name=PROJECT_NAME)

# Lazy-load endpoint
_endpoint = None


def get_endpoint():
    global _endpoint
    if _endpoint is None:
        _endpoint = PredictEndpoint()
        _endpoint.initialize()
    return _endpoint


@asynccontextmanager
async def lifespan(app: FastAPI):
    \"\"\"Application lifecycle - load model on startup.\"\"\"
    # Startup: pre-load model for faster first request
    get_endpoint()
    yield
    # Shutdown: cleanup if needed


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title=PROJECT_NAME,
    description="ML model serving API with monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - customize origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monitoring middleware - tracks latency, errors, request counts
app.add_middleware(MonitoringMiddleware, collector=metrics)


# =============================================================================
# Request/Response Models
# =============================================================================

class PredictRequest(BaseModel):
    \"\"\"Prediction request schema.\"\"\"
    features: dict[str, Any]


class PredictResponse(BaseModel):
    \"\"\"Prediction response schema.\"\"\"
    prediction: Any


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
def health():
    \"\"\"Health check endpoint.\"\"\"
    return {{"status": "ok"}}


@app.get("/metrics")
def get_metrics():
    \"\"\"Get current metrics summary.
    
    Returns latency percentiles, request counts, and error rates.
    \"\"\"
    return {{
        "latency_p50_ms": metrics.get_latency_p50(),
        "latency_p99_ms": metrics.get_latency_p99(),
        "request_count": metrics.get_request_count(),
        "error_count": metrics.get_error_count(),
    }}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    \"\"\"Generate prediction from model.
    
    The endpoint handles:
    1. preprocess() - transform request to features
    2. model.predict() - generate prediction
    3. postprocess() - format response
    
    Latency and errors are automatically tracked by MonitoringMiddleware.
    \"\"\"
    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request.model_dump())
        return PredictResponse(prediction=result)
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

    def _generate_flow_wrapper(self, context: dict) -> str:
        """Generate thin Metaflow wrapper that imports SDK pipeline."""
        return f'''"""Metaflow flow - thin wrapper around SDK pipeline.

Run locally:
    python -m {context["project_name_snake"]}.flow run

Deploy to Step Functions:
    python -m {context["project_name_snake"]}.flow step-functions create
"""

from metaflow import FlowSpec, step, schedule
from {context["project_name_snake"]}.sdk.pipeline import ScoringPipeline


@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Batch scoring flow - wraps SDK pipeline."""

    @step
    def start(self):
        """Initialize pipeline and load model."""
        self.pipeline = ScoringPipeline()
        self.pipeline.initialize()
        print(f"Initialized: {{self.pipeline}}")
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        """Execute the SDK pipeline."""
        self.result = self.pipeline.execute()
        print(f"Result: {{self.result}}")
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Pipeline complete: {{self.result}}")


if __name__ == "__main__":
    ScoringFlow()
'''

    def _generate_test_sdk(self, context: dict) -> str:
        """Generate SDK tests."""
        return f'''"""Tests for SDK components."""

import pytest


class TestProjectModel:
    """Tests for ProjectModel."""

    def test_model_import(self):
        """Test model can be imported."""
        from {context["project_name_snake"]}.sdk.model import ProjectModel
        
        model = ProjectModel()
        assert model.name == "{context["project_name"]}"


class TestProjectFeatures:
    """Tests for ProjectFeatures."""

    def test_features_import(self):
        """Test features can be imported."""
        from {context["project_name_snake"]}.sdk.features import ProjectFeatures
        
        features = ProjectFeatures()
        assert features is not None
'''

    def _generate_api_code(self, context: dict, pkg_dir: Path) -> None:
        """Generate FastAPI structure for realtime serving."""
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

    def _generate_batch_code(self, context: dict, pkg_dir: Path) -> None:
        """Generate Metaflow batch pipeline structure."""
        # Flows directory at project root
        flows_dir = self.project_dir / "batch" / "flows"
        flows_dir.mkdir(parents=True, exist_ok=True)
        
        # Data and output directories
        (self.project_dir / "batch" / "data").mkdir(exist_ok=True)
        (self.project_dir / "batch" / "output").mkdir(exist_ok=True)
        
        # Generate Metaflow flow
        flow_content = self._generate_metaflow_flow(context)
        self.write_file(flows_dir / "scoring_flow.py", flow_content)
        
        # Generate pipeline class in package
        pipeline_content = self._generate_pipeline_class(context)
        self.write_file(pkg_dir / "pipeline.py", pipeline_content)

    def _generate_metaflow_flow(self, context: dict) -> str:
        """Generate Metaflow flow file."""
        return f'''"""Metaflow flow for {context["project_name"]} batch scoring.

Run locally:
    python batch/flows/scoring_flow.py run

Deploy to Step Functions:
    python batch/flows/scoring_flow.py step-functions create
"""

from metaflow import FlowSpec, step, Parameter, schedule


@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Daily batch scoring flow."""

    input_path = Parameter(
        "input_path",
        help="Path to input data",
        default="batch/data/input.csv",
    )
    output_path = Parameter(
        "output_path",
        help="Path for output predictions",
        default="batch/output/predictions.parquet",
    )

    @step
    def start(self):
        """Initialize the flow."""
        print(f"Starting batch scoring for {context["project_name"]}")
        self.next(self.load_data)

    @step
    def load_data(self):
        """Load data to score."""
        import pandas as pd
        from pathlib import Path

        path = Path(self.input_path)
        if path.exists():
            self.data = pd.read_csv(path)
        else:
            # Generate sample data
            import numpy as np
            self.data = pd.DataFrame({{
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }})
        print(f"Loaded {{len(self.data)}} samples")
        self.next(self.predict)

    @step
    def predict(self):
        """Generate predictions."""
        import pandas as pd
        from {context["project_name_snake"]}.ml.predictor import ModelPredictor

        predictor = ModelPredictor()
        predictor.load()
        
        predictions = predictor.predict(self.data)
        
        self.results = self.data.copy()
        self.results["prediction"] = predictions
        self.results["scored_at"] = pd.Timestamp.now().isoformat()
        
        print(f"Generated {{len(self.results)}} predictions")
        self.next(self.save_results)

    @step
    def save_results(self):
        """Save predictions to storage."""
        from pathlib import Path

        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.results.to_parquet(path, index=False)
        print(f"Saved results to {{path}}")
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Scored {{len(self.results)}} samples")


if __name__ == "__main__":
    ScoringFlow()
'''

    def _generate_pipeline_class(self, context: dict) -> str:
        """Generate BatchPipeline class."""
        return f'''"""Batch pipeline using Geronimo BatchPipeline."""

from geronimo.batch import BatchPipeline, Schedule


class ScoringPipeline(BatchPipeline):
    """Daily batch scoring pipeline.
    
    Example:
        pipeline = ScoringPipeline()
        pipeline.initialize()
        result = pipeline.execute()
    """
    
    name = "{context["project_name"]}-scoring"
    schedule = Schedule.daily(hour=6, minute=0)
    
    def run(self):
        """Main pipeline logic."""
        import pandas as pd
        from pathlib import Path
        from .ml.predictor import ModelPredictor
        
        # Load predictor
        predictor = ModelPredictor()
        predictor.load()
        
        # Load data
        data_path = Path("batch/data/input.csv")
        if data_path.exists():
            data = pd.read_csv(data_path)
        else:
            # Sample data
            import numpy as np
            data = pd.DataFrame({{
                "feature_1": np.random.randn(100),
                "feature_2": np.random.randn(100),
            }})
        
        # Predict
        predictions = predictor.predict(data)
        
        # Build results
        results = data.copy()
        results["prediction"] = predictions
        results["scored_at"] = pd.Timestamp.now().isoformat()
        
        # Save
        output_path = self.save_results(results)
        
        return {{
            "samples_scored": len(results),
            "output_path": output_path,
        }}


if __name__ == "__main__":
    pipeline = ScoringPipeline()
    pipeline.initialize()
    print(pipeline.execute())
'''

    def _generate_test_batch(self, context: dict) -> str:
        """Generate batch pipeline tests."""
        return f'''"""Tests for batch pipeline."""

import pytest


class TestScoringPipeline:
    """Tests for ScoringPipeline."""

    def test_pipeline_exists(self):
        """Test pipeline can be imported."""
        from {context["project_name_snake"]}.pipeline import ScoringPipeline
        
        pipeline = ScoringPipeline()
        assert pipeline.name == "{context["project_name"]}-scoring"

    def test_pipeline_schedule(self):
        """Test pipeline has schedule."""
        from {context["project_name_snake"]}.pipeline import ScoringPipeline
        
        pipeline = ScoringPipeline()
        assert pipeline.schedule is not None
        assert "6" in pipeline.schedule.cron_expression
'''



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
        """Generate the prediction route using SDK Endpoint."""
        return f'''"""Prediction endpoints using Geronimo SDK Endpoint."""

import time
import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from {context["project_name_snake"]}.sdk.endpoint import PredictEndpoint

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize SDK endpoint
_endpoint = None


def get_endpoint() -> PredictEndpoint:
    """Get or initialize the SDK endpoint."""
    global _endpoint
    if _endpoint is None:
        _endpoint = PredictEndpoint()
        _endpoint.initialize()
        logger.info("SDK Endpoint initialized")
    return _endpoint


@router.post("/predict")
async def predict(request: dict[str, Any]) -> dict[str, Any]:
    """Generate predictions using the SDK Endpoint.

    Args:
        request: Input features for prediction.

    Returns:
        Model predictions with metadata.
    """
    start_time = time.perf_counter()

    try:
        endpoint = get_endpoint()
        result = endpoint.handle(request)

        latency_ms = (time.perf_counter() - start_time) * 1000
        logger.info(f"Prediction completed in {{latency_ms:.2f}}ms")

        return {{
            **result,
            "latency_ms": latency_ms,
        }}

    except NotImplementedError as e:
        raise HTTPException(
            status_code=501, 
            detail=f"Endpoint not implemented: {{e}}"
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

        # Template-specific dependencies
        deps = self._get_template_dependencies()
        deps_str = ",\n    ".join(f'"{d}"' for d in deps)

        # pyproject.toml
        pyproject = f'''[project]
name = "{context["project_name"]}"
version = "1.0.0"
description = "ML model serving API"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    {deps_str},
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

        # Generate training script
        self._generate_training_script(context)

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

    def _generate_training_script(self, context: dict) -> None:
        """Generate training script template."""
        pkg_dir = self.project_dir / "src" / context["project_name_snake"]

        train_script = f'''"""Training script for {context["project_name"]}.

This script demonstrates the full training workflow:
1. Load data from SDK data_sources
2. Initialize model with SDK features
3. Fit transformers and train model
4. Save artifacts to ArtifactStore

Usage:
    uv run python -m {context["project_name_snake"]}.train
"""

from pathlib import Path
import pandas as pd

from geronimo.artifacts import ArtifactStore
from geronimo.models import HyperParams

# Import from your SDK
from {context["project_name_snake"]}.sdk.model import ProjectModel
from {context["project_name_snake"]}.sdk.data_sources import training_data


def main():
    """Train and save the model."""
    print("=" * 50)
    print("Model Training")
    print("=" * 50)

    # =========================================================================
    # 1. Load data from configured data source
    # =========================================================================
    print("\\n1. Loading data...")
    
    # Option A: Load from SDK data_sources (recommended)
    # This uses the DataSource defined in sdk/data_sources.py
    # df = training_data.load()
    
    # Option B: Direct file load (for development/testing)
    # df = pd.read_csv("data/train.csv")
    
    # TODO: Uncomment one of the options above
    raise NotImplementedError(
        "Configure your data source in sdk/data_sources.py, then uncomment:\\n"
        "  df = training_data.load()"
    )

    # =========================================================================
    # 2. Prepare features and target
    # =========================================================================
    print("\\n2. Preparing data...")
    
    # TODO: Update with your target column name
    # y = df.pop("target")
    raise NotImplementedError("Set your target column: y = df.pop('your_target_column')")

    # =========================================================================
    # 3. Initialize and train model
    # =========================================================================
    print("\\n3. Training model...")
    model = ProjectModel()
    
    # Fit feature transformers (from sdk/features.py)
    print("   Fitting feature transformers...")
    model.features.fit(df)
    X = model.features.transform(df)
    
    # Train with hyperparameters
    # TODO: Customize your hyperparameters
    params = HyperParams(
        n_estimators=100,
        max_depth=5,
    )
    model.train(X, y, params)

    # =========================================================================
    # 4. Save model artifacts
    # =========================================================================
    print("\\n4. Saving artifacts...")
    models_dir = Path(__file__).parent.parent.parent / "models"
    
    store = ArtifactStore(
        project="{context["project_name"]}",
        version="1.0.0",
        backend="local",
        base_path=str(models_dir),
    )
    model.save(store)
    print(f"   Saved to {{models_dir}}")

    print("\\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
'''
        self.write_file(pkg_dir / "train.py", train_script)

