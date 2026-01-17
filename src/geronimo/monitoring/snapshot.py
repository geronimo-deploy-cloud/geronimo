"""Snapshot service for capturing reference data.

Provides utilities for capturing baseline statistics from model inputs.
"""

import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from geronimo.monitoring.drift_models import (
    FeatureStats,
    FeatureType,
    ReferenceSnapshot,
    RecentWindow,
)


class SnapshotService:
    """Service for capturing and storing data snapshots."""

    def __init__(self, s3_bucket: str = "model-monitoring"):
        """Initialize snapshot service.

        Args:
            s3_bucket: S3 bucket for storing snapshot data.
        """
        self.s3_bucket = s3_bucket

    def capture_reference(
        self,
        data: pd.DataFrame,
        project_name: str,
        model_version: str,
        deployment_type: Literal["realtime", "batch"],
        sample_size: int = 1000,
    ) -> ReferenceSnapshot:
        """Capture a reference snapshot from input data.

        Args:
            data: DataFrame of model inputs.
            project_name: Name of the project.
            model_version: Version of the model.
            deployment_type: Type of deployment.
            sample_size: Number of rows to sample for storage.

        Returns:
            ReferenceSnapshot with computed statistics.
        """
        snapshot_id = str(uuid.uuid4())

        # Compute feature statistics
        feature_stats = self._compute_feature_stats(data)

        # Sample data for storage
        if len(data) > sample_size:
            sample = data.sample(n=sample_size, random_state=42)
        else:
            sample = data

        # Generate S3 path
        s3_path = (
            f"s3://{self.s3_bucket}/{project_name}/references/{snapshot_id}.parquet"
        )

        return ReferenceSnapshot(
            id=snapshot_id,
            project_name=project_name,
            model_version=model_version,
            deployment_type=deployment_type,
            created_at=datetime.utcnow(),
            feature_statistics=feature_stats,
            sample_size=len(sample),
            s3_path=s3_path,
        )

    def capture_window(
        self,
        data: pd.DataFrame,
        project_name: str,
        deployment_type: Literal["realtime", "batch"],
        window_start: datetime,
        window_end: datetime,
    ) -> RecentWindow:
        """Capture a recent window of inputs.

        Args:
            data: DataFrame of recent model inputs.
            project_name: Name of the project.
            deployment_type: Type of deployment.
            window_start: Start of the window.
            window_end: End of the window.

        Returns:
            RecentWindow with computed statistics.
        """
        window_id = str(uuid.uuid4())

        feature_stats = self._compute_feature_stats(data)
        s3_path = f"s3://{self.s3_bucket}/{project_name}/windows/{window_id}.parquet"

        return RecentWindow(
            id=window_id,
            project_name=project_name,
            deployment_type=deployment_type,
            window_start=window_start,
            window_end=window_end,
            feature_statistics=feature_stats,
            sample_size=len(data),
            s3_path=s3_path,
        )

    def _compute_feature_stats(self, data: pd.DataFrame) -> dict[str, FeatureStats]:
        """Compute statistics for each feature.

        Args:
            data: Input DataFrame.

        Returns:
            Dictionary of feature name to FeatureStats.
        """
        stats: dict[str, FeatureStats] = {}

        for col in data.columns:
            dtype = self._infer_feature_type(data[col])
            null_count = data[col].isnull().sum()
            null_rate = null_count / len(data) if len(data) > 0 else 0.0

            if dtype == FeatureType.NUMERIC:
                stats[col] = FeatureStats(
                    name=col,
                    dtype=dtype,
                    mean=float(data[col].mean()),
                    std=float(data[col].std()),
                    min=float(data[col].min()),
                    max=float(data[col].max()),
                    quantiles={
                        "p25": float(data[col].quantile(0.25)),
                        "p50": float(data[col].quantile(0.50)),
                        "p75": float(data[col].quantile(0.75)),
                    },
                    null_count=int(null_count),
                    null_rate=float(null_rate),
                )
            elif dtype == FeatureType.CATEGORICAL:
                value_counts = data[col].value_counts().head(100).to_dict()
                stats[col] = FeatureStats(
                    name=col,
                    dtype=dtype,
                    cardinality=data[col].nunique(),
                    value_counts={str(k): int(v) for k, v in value_counts.items()},
                    null_count=int(null_count),
                    null_rate=float(null_rate),
                )
            else:  # TEXT
                stats[col] = FeatureStats(
                    name=col,
                    dtype=dtype,
                    cardinality=data[col].nunique(),
                    null_count=int(null_count),
                    null_rate=float(null_rate),
                )

        return stats

    def _infer_feature_type(self, series: pd.Series) -> FeatureType:
        """Infer the feature type from a pandas Series.

        Args:
            series: Pandas Series.

        Returns:
            FeatureType enum.
        """
        if pd.api.types.is_numeric_dtype(series):
            return FeatureType.NUMERIC
        elif pd.api.types.is_categorical_dtype(series) or series.nunique() < 50:
            return FeatureType.CATEGORICAL
        else:
            return FeatureType.TEXT

    def save_to_s3(self, data: pd.DataFrame, s3_path: str) -> None:
        """Save DataFrame to S3 as Parquet.

        Args:
            data: DataFrame to save.
            s3_path: S3 path (s3://bucket/key).
        """
        import boto3

        # Parse S3 path
        path_parts = s3_path.replace("s3://", "").split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1]

        # Save to local temp then upload
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".parquet") as f:
            data.to_parquet(f.name, index=False)
            s3 = boto3.client("s3")
            s3.upload_file(f.name, bucket, key)
