"""Configuration schema definitions for Geronimo.

This module defines Pydantic models for the geronimo.yaml configuration file.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelType(str, Enum):
    """Type of ML model deployment."""

    REALTIME = "realtime"
    BATCH = "batch"


class MLFramework(str, Enum):
    """Supported ML frameworks."""

    SKLEARN = "sklearn"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    XGBOOST = "xgboost"
    CUSTOM = "custom"


class AlertType(str, Enum):
    """Types of alert destinations."""

    SLACK = "slack"
    EMAIL = "email"
    ADO_WORKITEM = "ado_workitem"


# ============================================================================
# Sub-configurations
# ============================================================================


class ProjectConfig(BaseModel):
    """Project metadata configuration."""

    name: str = Field(..., description="Project name (lowercase, hyphens allowed)")
    version: str = Field(default="1.0.0", description="Semantic version")
    description: Optional[str] = Field(
        default=None, description="Brief project description"
    )


class ModelConfig(BaseModel):
    """ML model configuration."""

    type: ModelType = Field(
        default=ModelType.REALTIME, description="Deployment type (realtime or batch)"
    )
    framework: MLFramework = Field(
        default=MLFramework.SKLEARN, description="ML framework used"
    )
    artifact_path: Optional[str] = Field(
        default="models/model.joblib", description="Path to model artifact"
    )


class RuntimeConfig(BaseModel):
    """Python runtime configuration."""

    python_version: str = Field(default="3.11", description="Python version")
    dependencies: list[str] = Field(
        default_factory=list, description="Additional pip dependencies"
    )


class ScalingConfig(BaseModel):
    """Auto-scaling configuration."""

    min_instances: int = Field(
        default=1, ge=0, description="Minimum number of instances"
    )
    max_instances: int = Field(
        default=4, ge=1, description="Maximum number of instances"
    )
    target_cpu_percent: int = Field(
        default=70, ge=1, le=100, description="Target CPU utilization for scaling"
    )


class InfrastructureConfig(BaseModel):
    """Infrastructure resource configuration."""

    cpu: int = Field(
        default=512,
        description="CPU units (256, 512, 1024, 2048, 4096)",
    )
    memory: int = Field(
        default=1024,
        description="Memory in MB",
    )
    scaling: ScalingConfig = Field(default_factory=ScalingConfig)
    vpc_id: Optional[str] = Field(
        default=None, description="VPC ID (supports ${ENV_VAR} interpolation)"
    )
    subnets: Optional[list[str]] = Field(
        default=None, description="Subnet IDs for deployment"
    )
    security_groups: Optional[list[str]] = Field(
        default=None, description="Security group IDs"
    )


class AlertConditions(BaseModel):
    """Conditions that trigger alerts."""

    error_rate_threshold: float = Field(
        default=0.01, ge=0, le=1, description="Error rate threshold (0-1)"
    )
    latency_p99_threshold_ms: int = Field(
        default=500, ge=1, description="P99 latency threshold in milliseconds"
    )
    data_drift_threshold: float = Field(
        default=0.1, ge=0, le=1, description="Data drift detection threshold"
    )


class AlertConfig(BaseModel):
    """Alert destination configuration."""

    type: AlertType = Field(..., description="Alert destination type")
    channel: Optional[str] = Field(
        default=None, description="Slack channel or email address"
    )
    webhook_url: Optional[str] = Field(
        default=None, description="Webhook URL for Slack"
    )
    conditions: AlertConditions = Field(default_factory=AlertConditions)


class SourceSystem(str, Enum):
    """Database source systems for query-based data capture."""

    SNOWFLAKE = "snowflake"
    POSTGRES = "postgres"
    SQLSERVER = "sqlserver"


class DriftDetectionConfig(BaseModel):
    """Drift detection configuration."""

    enabled: bool = Field(default=False, description="Enable drift detection")
    s3_bucket: str = Field(
        default="model-monitoring",
        description="S3 bucket for storing snapshots and reports",
    )
    sampling_rate: float = Field(
        default=0.05,
        ge=0.001,
        le=1.0,
        description="Fraction of requests to sample (0.001-1.0)",
    )
    window_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Rolling window size in days for recent data",
    )
    drift_threshold: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Threshold for feature drift detection",
    )
    dataset_drift_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Fraction of drifted features to trigger dataset drift",
    )
    retention_days: int = Field(
        default=90,
        ge=1,
        description="Days to retain historical snapshots",
    )
    auto_capture_on_deploy: bool = Field(
        default=True,
        description="Automatically capture reference on deployment",
    )


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "latency_p50",
            "latency_p99",
            "error_rate",
            "request_count",
            "prediction_distribution",
        ],
        description="Metrics to collect",
    )
    alerts: list[AlertConfig] = Field(
        default_factory=list, description="Alert configurations"
    )
    dashboard_enabled: bool = Field(
        default=True, description="Generate CloudWatch dashboard"
    )
    drift_detection: DriftDetectionConfig = Field(
        default_factory=DriftDetectionConfig, description="Drift detection settings"
    )


class EnvironmentConfig(BaseModel):
    """Deployment environment configuration."""

    name: str = Field(..., description="Environment name (dev, staging, prod)")
    auto_deploy: bool = Field(
        default=False, description="Auto-deploy on pipeline success"
    )
    approval_required: bool = Field(
        default=False, description="Require manual approval for deployment"
    )
    variables: dict[str, str] = Field(
        default_factory=dict, description="Environment-specific variables"
    )


class DeploymentConfig(BaseModel):
    """Deployment pipeline configuration."""

    environments: list[EnvironmentConfig] = Field(
        default_factory=lambda: [
            EnvironmentConfig(name="dev", auto_deploy=True),
            EnvironmentConfig(name="prod", approval_required=True),
        ],
        description="Deployment environments",
    )
    container_registry: Optional[str] = Field(
        default=None, description="ECR registry URL (auto-generated if not provided)"
    )


class BatchBackend(str, Enum):
    """Batch job deployment backends."""

    STEP_FUNCTIONS = "step-functions"
    AIRFLOW = "airflow"


class BatchJobConfig(BaseModel):
    """Individual batch job configuration."""

    name: str = Field(..., description="Job name")
    flow_file: str = Field(..., description="Path to Metaflow flow file")
    schedule: Optional[str] = Field(
        default=None, description="Cron schedule expression"
    )
    cpu: int = Field(default=4, description="CPU cores for batch compute")
    memory: int = Field(default=8192, description="Memory in MB")


class StepFunctionsConfig(BaseModel):
    """AWS Step Functions backend configuration."""

    s3_root: str = Field(
        default="s3://metaflow-data",
        description="S3 root for Metaflow artifacts",
    )
    batch_queue: Optional[str] = Field(
        default=None, description="AWS Batch queue name"
    )


class AirflowConfig(BaseModel):
    """Astronomer Airflow backend configuration."""

    connection_id: str = Field(
        default="astronomer_default",
        description="Airflow connection ID",
    )
    namespace: str = Field(
        default="default",
        description="Kubernetes namespace for pod operators",
    )


class BatchConfig(BaseModel):
    """Batch job configuration."""

    enabled: bool = Field(default=False, description="Enable batch job generation")
    backend: BatchBackend = Field(
        default=BatchBackend.STEP_FUNCTIONS,
        description="Deployment backend (step-functions or airflow)",
    )
    step_functions: StepFunctionsConfig = Field(default_factory=StepFunctionsConfig)
    airflow: AirflowConfig = Field(default_factory=AirflowConfig)
    jobs: list[BatchJobConfig] = Field(
        default_factory=list, description="Batch job definitions"
    )


# ============================================================================
# Main Configuration
# ============================================================================


class GeronimoConfig(BaseModel):
    """Root configuration for a Geronimo project.

    This is the Pydantic model for geronimo.yaml files.
    """

    model_config = ConfigDict(extra="forbid")

    project: ProjectConfig = Field(..., description="Project metadata")
    model: ModelConfig = Field(default_factory=ModelConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    infrastructure: InfrastructureConfig = Field(
        default_factory=InfrastructureConfig
    )
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)

