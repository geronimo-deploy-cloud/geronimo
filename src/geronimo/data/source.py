"""DataSource abstraction for connecting to data backends."""

import os
from enum import Enum
from typing import Any, Literal, Optional

import pandas as pd

from geronimo.data.query import Query


class SourceType(str, Enum):
    """Supported data source types."""

    SNOWFLAKE = "snowflake"
    POSTGRES = "postgres"
    SQLSERVER = "sqlserver"
    FILE = "file"


class DataSource:
    """Abstraction for loading data from various backends.

    Provides a unified interface for querying data from databases
    or loading from files with automatic lineage tracking.

    Example:
        ```python
        from geronimo.data import DataSource, Query

        # Define source
        training_data = DataSource(
            name="customer_features",
            source="snowflake",
            query=Query.from_file("queries/training_data.sql"),
        )

        # Load data
        df = training_data.load(start_date="2024-01-01")
        ```
    """

    def __init__(
        self,
        name: str,
        source: SourceType | str,
        query: Optional[Query] = None,
        path: Optional[str] = None,
        connection_params: Optional[dict[str, Any]] = None,
    ):
        """Initialize data source.

        Args:
            name: Descriptive name for the data source.
            source: Source type (snowflake, postgres, sqlserver, file).
            query: Query object for database sources.
            path: File path for file-based sources.
            connection_params: Optional connection parameters (overrides env vars).
        """
        self.name = name
        self.source = SourceType(source) if isinstance(source, str) else source
        self.query = query
        self.path = path
        self.connection_params = connection_params or {}

        if self.source != SourceType.FILE and not query:
            raise ValueError("Database sources require a query")
        if self.source == SourceType.FILE and not path:
            raise ValueError("File sources require a path")

    def load(self, **params) -> pd.DataFrame:
        """Load data from source.

        Args:
            **params: Query parameters for substitution.

        Returns:
            DataFrame with loaded data.
        """
        if self.source == SourceType.FILE:
            return self._load_file()
        else:
            return self._load_database(**params)

    def _load_file(self) -> pd.DataFrame:
        """Load data from file."""
        from pathlib import Path

        path = Path(self.path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        elif path.suffix in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        elif path.suffix == ".json":
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def _load_database(self, **params) -> pd.DataFrame:
        """Load data from database."""
        sql = self.query.render(**params)

        if self.source == SourceType.SNOWFLAKE:
            return self._query_snowflake(sql)
        elif self.source == SourceType.POSTGRES:
            return self._query_postgres(sql)
        elif self.source == SourceType.SQLSERVER:
            return self._query_sqlserver(sql)
        else:
            raise ValueError(f"Unsupported source: {self.source}")

    def _query_snowflake(self, sql: str) -> pd.DataFrame:
        """Execute query against Snowflake."""
        import snowflake.connector

        conn_args = {
            "user": self.connection_params.get("user", os.getenv("SNOWFLAKE_USER")),
            "password": self.connection_params.get(
                "password", os.getenv("SNOWFLAKE_PASSWORD")
            ),
            "account": self.connection_params.get(
                "account", os.getenv("SNOWFLAKE_ACCOUNT")
            ),
            "warehouse": self.connection_params.get(
                "warehouse", os.getenv("SNOWFLAKE_WAREHOUSE")
            ),
            "database": self.connection_params.get(
                "database", os.getenv("SNOWFLAKE_DATABASE")
            ),
            "schema": self.connection_params.get("schema", os.getenv("SNOWFLAKE_SCHEMA")),
        }
        conn = snowflake.connector.connect(**conn_args)
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    def _query_postgres(self, sql: str) -> pd.DataFrame:
        """Execute query against PostgreSQL."""
        import psycopg2

        conn_str = self.connection_params.get(
            "connection_string", os.getenv("POSTGRES_CONNECTION_STRING")
        )
        conn = psycopg2.connect(conn_str)
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    def _query_sqlserver(self, sql: str) -> pd.DataFrame:
        """Execute query against SQL Server."""
        import pyodbc

        conn_str = self.connection_params.get(
            "connection_string", os.getenv("SQLSERVER_CONNECTION_STRING")
        )
        conn = pyodbc.connect(conn_str)
        try:
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    def __repr__(self) -> str:
        return f"DataSource({self.name}, source={self.source.value})"
