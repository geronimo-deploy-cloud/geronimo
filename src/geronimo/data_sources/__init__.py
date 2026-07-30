"""Geronimo Data Layer.

The data_sources module provides a unified abstraction for connecting to, querying,
and ingesting data from various upstream sources. It decouples the ML modeling logic
from the underlying data infrastructure.

Key components:
- DataSource: Represents a table or view in a database.
- Query: A composable query object for retrieving data.
- DatabaseConnection: Protocol for database adapters.

Supported connections:
- Snowflake
- PostgreSQL
- SQL Server
- Google BigQuery (via generic interface)

This layer handles connection pooling, query generation, and data type mapping.
"""

from geronimo.data_sources.source import DataSource, JoinSpec, ConcatSpec, SplitSpec, collect_data_sources
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
    "SplitSpec",
    "ConcatSpec",
    "collect_data_sources",
    "Query",
    "DatabaseConnection",
    "BaseDatabaseConnection",
    "SnowflakeConnection",
    "PostgresConnection",
    "SQLServerConnection",
    "get_connection",
]

__docformat__ = "google"
