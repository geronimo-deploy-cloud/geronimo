"""Database connection protocol and implementations.

Provides a standard interface for database connections used by DataSource.
"""

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator, Optional, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class DatabaseConnection(Protocol):
    """Protocol for database connections.
    
    Implement this protocol to add support for new database types.
    
    Example:
        ```python
        class MyDatabaseConnection:
            def __init__(self, connection_string: str):
                self.connection_string = connection_string
                self._conn = None
            
            def connect(self) -> None:
                self._conn = my_db_driver.connect(self.connection_string)
            
            def execute(self, sql: str) -> pd.DataFrame:
                return pd.read_sql(sql, self._conn)
            
            def close(self) -> None:
                if self._conn:
                    self._conn.close()
        ```
    """
    
    def connect(self) -> None:
        """Establish connection to the database."""
        ...
    
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame.
        
        Args:
            sql: SQL query to execute.
            
        Returns:
            DataFrame containing query results.
        """
        ...
    
    def close(self) -> None:
        """Close the database connection."""
        ...


class BaseDatabaseConnection(ABC):
    """Abstract base class for database connections.
    
    Provides common functionality like context manager support.
    """
    
    def __init__(self, connection_params: Optional[dict[str, Any]] = None):
        """Initialize connection with parameters.
        
        Args:
            connection_params: Optional connection parameters (overrides env vars).
        """
        self.connection_params = connection_params or {}
        self._connection: Any = None
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the database."""
        pass
    
    @abstractmethod
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute a SQL query and return results as DataFrame."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass
    
    def __enter__(self) -> "BaseDatabaseConnection":
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
    
    @contextmanager
    def session(self) -> Generator["BaseDatabaseConnection", None, None]:
        """Convenience context manager for connection lifecycle.
        
        Example:
            ```python
            conn = SnowflakeConnection(params)
            with conn.session():
                df = conn.execute("SELECT * FROM table")
            ```
        """
        try:
            self.connect()
            yield self
        finally:
            self.close()


class SnowflakeConnection(BaseDatabaseConnection):
    """Snowflake database connection."""
    
    def __init__(self, connection_params: Optional[dict[str, Any]] = None):
        """Initialize Snowflake connection.
        
        Args:
            connection_params: Optional dict with keys:
                - user: Snowflake username (or SNOWFLAKE_USER env var)
                - password: Snowflake password (or SNOWFLAKE_PASSWORD env var)
                - account: Snowflake account (or SNOWFLAKE_ACCOUNT env var)
                - warehouse: Snowflake warehouse (or SNOWFLAKE_WAREHOUSE env var)
                - database: Snowflake database (or SNOWFLAKE_DATABASE env var)
                - schema: Snowflake schema (or SNOWFLAKE_SCHEMA env var)
        """
        super().__init__(connection_params)
    
    def connect(self) -> None:
        """Establish connection to Snowflake."""
        import snowflake.connector
        
        conn_args = {
            "user": self.connection_params.get("user", os.getenv("SNOWFLAKE_USER")),
            "password": self.connection_params.get("password", os.getenv("SNOWFLAKE_PASSWORD")),
            "account": self.connection_params.get("account", os.getenv("SNOWFLAKE_ACCOUNT")),
            "warehouse": self.connection_params.get("warehouse", os.getenv("SNOWFLAKE_WAREHOUSE")),
            "database": self.connection_params.get("database", os.getenv("SNOWFLAKE_DATABASE")),
            "schema": self.connection_params.get("schema", os.getenv("SNOWFLAKE_SCHEMA")),
        }
        self._connection = snowflake.connector.connect(**conn_args)
    
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute query against Snowflake."""
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return pd.read_sql(sql, self._connection)
    
    def close(self) -> None:
        """Close Snowflake connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


class PostgresConnection(BaseDatabaseConnection):
    """PostgreSQL database connection."""
    
    def __init__(self, connection_params: Optional[dict[str, Any]] = None):
        """Initialize PostgreSQL connection.
        
        Args:
            connection_params: Optional dict with keys:
                - connection_string: Full connection string 
                  (or POSTGRES_CONNECTION_STRING env var)
        """
        super().__init__(connection_params)
    
    def connect(self) -> None:
        """Establish connection to PostgreSQL."""
        import psycopg2
        
        conn_str = self.connection_params.get(
            "connection_string", os.getenv("POSTGRES_CONNECTION_STRING")
        )
        self._connection = psycopg2.connect(conn_str)
    
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute query against PostgreSQL."""
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return pd.read_sql(sql, self._connection)
    
    def close(self) -> None:
        """Close PostgreSQL connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


class SQLServerConnection(BaseDatabaseConnection):
    """SQL Server database connection."""
    
    def __init__(self, connection_params: Optional[dict[str, Any]] = None):
        """Initialize SQL Server connection.
        
        Args:
            connection_params: Optional dict with keys:
                - connection_string: ODBC connection string
                  (or SQLSERVER_CONNECTION_STRING env var)
        """
        super().__init__(connection_params)
    
    def connect(self) -> None:
        """Establish connection to SQL Server."""
        import pyodbc
        
        conn_str = self.connection_params.get(
            "connection_string", os.getenv("SQLSERVER_CONNECTION_STRING")
        )
        self._connection = pyodbc.connect(conn_str)
    
    def execute(self, sql: str) -> pd.DataFrame:
        """Execute query against SQL Server."""
        if self._connection is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return pd.read_sql(sql, self._connection)
    
    def close(self) -> None:
        """Close SQL Server connection."""
        if self._connection:
            self._connection.close()
            self._connection = None


def get_connection(
    source_type: str, 
    connection_params: Optional[dict[str, Any]] = None
) -> BaseDatabaseConnection:
    """Factory function to get appropriate connection for source type.
    
    Args:
        source_type: Database type (snowflake, postgres, sqlserver).
        connection_params: Optional connection parameters.
        
    Returns:
        Appropriate connection instance.
        
    Raises:
        ValueError: If source type is not supported.
    """
    connections = {
        "snowflake": SnowflakeConnection,
        "postgres": PostgresConnection,
        "sqlserver": SQLServerConnection,
    }
    
    if source_type not in connections:
        raise ValueError(
            f"Unsupported database type: {source_type}. "
            f"Supported types: {list(connections.keys())}"
        )
    
    return connections[source_type](connection_params)
