"""DataSource abstraction for connecting to data backends."""

import os
from enum import Enum
from typing import Any, Callable, Literal, Optional

import pandas as pd

from geronimo.data.query import Query
from geronimo.data.connection import get_connection, DatabaseConnection


class SourceType(str, Enum):
    """Supported data source types."""

    SNOWFLAKE = "snowflake"
    POSTGRES = "postgres"
    SQLSERVER = "sqlserver"
    FILE = "file"
    FUNC = "function"


class DataSourceError(Exception):
    """Exception raised when a DataSource operation fails."""
    pass


class DataSource:
    """Abstraction for loading data from various backends.

    Provides a unified interface for querying data from databases,
    loading from files, or calling custom functions.

    Example (database):
        ```python
        from geronimo.data import DataSource, Query

        training_data = DataSource(
            name="customer_features",
            source="snowflake",
            query=Query.from_file("queries/training_data.sql"),
        )
        df = training_data.load(start_date="2024-01-01")
        ```

    Example (function):
        ```python
        from geronimo.data import DataSource
        from sklearn.datasets import load_iris
        import pandas as pd
        
        def load_iris_data() -> pd.DataFrame:
            iris = load_iris()
            return pd.DataFrame(iris.data, columns=iris.feature_names)
        
        training_data = DataSource(
            name="iris",
            source="function",
            handle=load_iris_data,
        )
        df = training_data.load()  # Validates return type at runtime
        ```
    
    Note:
        When using `source="function"`, the provided handle function MUST:
        1. Return a pandas DataFrame
        2. Be callable with optional keyword arguments
        
        A DataSourceError is raised at runtime if the function does not
        return a DataFrame.
    """

    def __init__(
        self,
        name: str,
        source: SourceType | str,
        query: Optional[Query] = None,
        path: Optional[str] = None,
        handle: Optional[Callable[..., pd.DataFrame]] = None,
        connection_params: Optional[dict[str, Any]] = None,
        connection: Optional[DatabaseConnection] = None,
    ):
        """Initialize data source.

        Args:
            name: Descriptive name for the data source.
            source: Source type (snowflake, postgres, sqlserver, file, function).
            query: Query object for database sources.
            path: File path for file-based sources.
            handle: Callable that returns a DataFrame (for function sources).
                    Must return pd.DataFrame - validated at runtime.
            connection_params: Optional connection parameters (overrides env vars).
            connection: Optional custom DatabaseConnection implementation.
        
        Raises:
            ValueError: If required arguments are missing for the source type.
        """
        self.name = name
        self.source = SourceType(source) if isinstance(source, str) else source
        self.query = query
        self.path = path
        self.handle = handle
        self.connection_params = connection_params or {}
        self._custom_connection = connection

        # Validate required arguments based on source type
        if self.source == SourceType.FUNC:
            if not handle:
                raise ValueError("Function sources require a handle")
            if not callable(handle):
                raise ValueError("handle must be callable")
        elif self.source == SourceType.FILE:
            if not path:
                raise ValueError("File sources require a path")
        else:
            # Database sources
            if not query:
                raise ValueError("Database sources require a query")

    def load(self, **params) -> pd.DataFrame:
        """Load data from source.

        Args:
            **params: Parameters passed to the data loading function.
                      For database sources, these are query parameters.
                      For function sources, these are passed to the handle.

        Returns:
            DataFrame with loaded data.
        
        Raises:
            DataSourceError: If function source doesn't return a DataFrame.
        """
        if self.source == SourceType.FILE:
            return self._load_file()
        elif self.source == SourceType.FUNC:
            return self._load_function(**params)
        else:
            return self._load_database(**params)
    
    def _load_function(self, **params) -> pd.DataFrame:
        """Load data by calling the handle function.
        
        Validates at runtime that the function returns a DataFrame.
        
        Args:
            **params: Keyword arguments passed to the handle function.
        
        Returns:
            DataFrame returned by the handle function.
        
        Raises:
            DataSourceError: If handle doesn't return a DataFrame or raises an exception.
        """
        try:
            result = self.handle(**params)
        except Exception as e:
            raise DataSourceError(
                f"DataSource '{self.name}' handle function raised an exception: {e}"
            ) from e
        
        # Runtime validation: ensure result is a DataFrame
        if not isinstance(result, pd.DataFrame):
            actual_type = type(result).__name__
            raise DataSourceError(
                f"DataSource '{self.name}' handle function must return a pandas DataFrame, "
                f"but returned {actual_type}. "
                f"Ensure your function returns pd.DataFrame."
            )
        
        return result

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
        """Load data from database using connection interface."""
        sql = self.query.render(**params)
        
        # Use custom connection if provided, otherwise create from factory
        if self._custom_connection is not None:
            connection = self._custom_connection
        else:
            connection = get_connection(self.source.value, self.connection_params)
        
        # Use context manager for automatic connection cleanup
        with connection:
            return connection.execute(sql)

    def __repr__(self) -> str:
        return f"DataSource({self.name}, source={self.source.value})"

