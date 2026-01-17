"""Drift detection data models.

Defines schemas for reference snapshots, recent windows, and drift reports.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class FeatureType(str, Enum):
    """Type of feature."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"


class FeatureStats(BaseModel):
    """Per-feature statistics."""

    name: str
    dtype: FeatureType

    # Numeric features
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    quantiles: Optional[dict[str, float]] = None  # p25, p50, p75

    # Categorical features
    cardinality: Optional[int] = None
    value_counts: Optional[dict[str, int]] = None

    # Missing values
    null_count: int = 0
    null_rate: float = 0.0


class ReferenceSnapshot(BaseModel):
    """Baseline data captured at model deploy time."""

    id: str = Field(..., description="Unique snapshot ID")
    project_name: str
    model_version: str
    deployment_type: Literal["realtime", "batch"]

    # When captured
    created_at: datetime

    # Data
    feature_statistics: dict[str, FeatureStats]
    sample_size: int

    # Storage
    s3_path: str = Field(..., description="S3 path to full sample data")


class RecentWindow(BaseModel):
    """Rolling window of recent model inputs."""

    id: str = Field(..., description="Unique window ID")
    project_name: str
    deployment_type: Literal["realtime", "batch"]

    window_start: datetime
    window_end: datetime

    feature_statistics: dict[str, FeatureStats]
    sample_size: int

    s3_path: str


class FeatureDrift(BaseModel):
    """Drift detection result for a single feature."""

    feature_name: str
    drift_detected: bool
    drift_score: float = Field(..., ge=0, le=1)
    stattest_name: str  # "ks", "chi2", "wasserstein", "psi"
    stattest_threshold: float

    # Distribution shift details
    reference_mean: Optional[float] = None
    current_mean: Optional[float] = None
    psi: Optional[float] = None  # Population Stability Index


class DriftReport(BaseModel):
    """Drift detection results comparing reference to recent window."""

    id: str
    project_name: str
    reference_id: str
    recent_window_id: str

    created_at: datetime

    # Overall drift
    dataset_drift: bool
    drift_score: float = Field(..., ge=0, le=1)

    # Per-feature drift
    feature_drift: dict[str, FeatureDrift]

    # Alerting
    alert_triggered: bool = False
    alert_destination: Optional[str] = None
