"""Geronimo Data Layer.

Provides abstractions for data sources, queries, and database connections.
"""

from geronimo.data_sources.source import DataSource, JoinSpec, collect_data_sources
from geronimo.data_sources.query import Query
from geronimo.data_sources.connection import (
    DatabaseConnection,
    BaseDatabaseConnection,
    SnowflakeConnection,
    PostgresConnection,
    SQLServerConnection,
    get_connection,
)

__all__ = [
    "DataSource",
    "JoinSpec",
    "collect_data_sources",
    "Query",
    "DatabaseConnection",
    "BaseDatabaseConnection",
    "SnowflakeConnection",
    "PostgresConnection",
    "SQLServerConnection",
    "get_connection",
]
