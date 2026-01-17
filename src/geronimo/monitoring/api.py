"""Snapshot API for programmatic access.

Provides API functions for use in Metaflow flows and other Python code.
"""

from datetime import datetime
from typing import Any, Literal, Optional

import pandas as pd

from geronimo.monitoring.drift_models import ReferenceSnapshot
from geronimo.monitoring.snapshot import SnapshotService


def capture_reference_from_data(
    data: pd.DataFrame,
    project_name: str,
    model_version: str = "1.0.0",
    deployment_type: Literal["realtime", "batch"] = "batch",
    s3_bucket: str = "model-monitoring",
    sampling_rate: float = 0.05,
    upload_to_s3: bool = True,
) -> ReferenceSnapshot:
    """Capture reference snapshot from DataFrame.

    Use this in Metaflow flows to auto-capture baselines.

    Args:
        data: Input DataFrame.
        project_name: Project name.
        model_version: Model version string.
        deployment_type: Type of deployment.
        s3_bucket: S3 bucket for storage.
        sampling_rate: Fraction of data to sample.
        upload_to_s3: Whether to upload sample to S3.

    Returns:
        ReferenceSnapshot with computed statistics.

    Example:
        ```python
        from geronimo.monitoring.api import capture_reference_from_data

        @step
        def capture_baseline(self):
            snapshot = capture_reference_from_data(
                data=self.training_data,
                project_name="my-model",
                model_version="1.2.0",
            )
            self.reference_snapshot = snapshot
        ```
    """
    sample_size = max(1, int(len(data) * sampling_rate))

    service = SnapshotService(s3_bucket=s3_bucket)
    snapshot = service.capture_reference(
        data=data,
        project_name=project_name,
        model_version=model_version,
        deployment_type=deployment_type,
        sample_size=sample_size,
    )

    if upload_to_s3 and len(data) > 0:
        # Sample and upload
        sample = data.sample(n=min(sample_size, len(data)), random_state=42)
        service.save_to_s3(sample, snapshot.s3_path)

    return snapshot


def capture_reference_from_query(
    sql: str,
    source_system: Literal["snowflake", "postgres", "sqlserver"],
    project_name: str,
    model_version: str = "1.0.0",
    deployment_type: Literal["realtime", "batch"] = "batch",
    s3_bucket: str = "model-monitoring",
    sampling_rate: float = 0.05,
    connection_params: Optional[dict[str, Any]] = None,
) -> ReferenceSnapshot:
    """Capture reference snapshot from SQL query.

    Args:
        sql: SQL query string.
        source_system: Database type.
        project_name: Project name.
        model_version: Model version string.
        deployment_type: Type of deployment.
        s3_bucket: S3 bucket for storage.
        sampling_rate: Fraction of data to sample.
        connection_params: Optional connection parameters (overrides env vars).

    Returns:
        ReferenceSnapshot with computed statistics.
    """
    import os

    if source_system == "snowflake":
        import snowflake.connector

        conn_args = connection_params or {
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
            "database": os.getenv("SNOWFLAKE_DATABASE"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA"),
        }
        conn = snowflake.connector.connect(**conn_args)
        data = pd.read_sql(sql, conn)

    elif source_system == "postgres":
        import psycopg2

        conn_str = (
            connection_params.get("connection_string")
            if connection_params
            else os.getenv("POSTGRES_CONNECTION_STRING")
        )
        conn = psycopg2.connect(conn_str)
        data = pd.read_sql(sql, conn)

    elif source_system == "sqlserver":
        import pyodbc

        conn_str = (
            connection_params.get("connection_string")
            if connection_params
            else os.getenv("SQLSERVER_CONNECTION_STRING")
        )
        conn = pyodbc.connect(conn_str)
        data = pd.read_sql(sql, conn)

    else:
        raise ValueError(f"Unsupported source system: {source_system}")

    return capture_reference_from_data(
        data=data,
        project_name=project_name,
        model_version=model_version,
        deployment_type=deployment_type,
        s3_bucket=s3_bucket,
        sampling_rate=sampling_rate,
    )
