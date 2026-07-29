"""DataSource abstraction for connecting to data backends."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # JoinSpec forward reference handled via string annotation

import pandas as pd

from geronimo.data_sources.query import Query
from geronimo.data_sources.connection import get_connection, DatabaseConnection

logger = logging.getLogger(__name__)


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
        from geronimo.data_sources import DataSource, Query

        training_data = DataSource(
            name="customer_features",
            source="snowflake",
            query=Query.from_file("queries/training_data.sql"),
        )
        df = training_data.load(start_date="2024-01-01")
        ```

    Example (function):
        ```python
        from geronimo.data_sources import DataSource
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

    name: str
    """The name of the data source."""

    source: SourceType
    """The type of the data source."""

    query: Optional[Query]
    """The query object (for database sources)."""

    path: Optional[str]
    """The file path (for file sources)."""

    handle: Optional[Callable[..., pd.DataFrame]]
    """The function handle (for function sources)."""

    connection_params: dict[str, Any]
    """Connection parameters."""

    _custom_connection: Optional[DatabaseConnection]
    """Internal custom connection instance."""

    join_spec: Optional["JoinSpec"]
    """Specification for joining to this source."""

    concat_spec: Optional["ConcatSpec"]
    """Specification for concatenating multiple DataSources row-wise."""

    def __init__(
        self,
        name: str,
        source: SourceType | str,
        query: Optional[Query] = None,
        path: Optional[str] = None,
        handle: Optional[Callable[..., pd.DataFrame]] = None,
        connection_params: Optional[dict[str, Any]] = None,
        connection: Optional[DatabaseConnection] = None,
        join_spec: Optional["JoinSpec"] = None,
        concat_spec: Optional["ConcatSpec"] = None,
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
            ValueError: If required arguments are missing for the source type,
                or if both join_spec and concat_spec are provided.
        """
        self.name = name
        self.source = SourceType(source) if isinstance(source, str) else source
        self.query = query
        self.path = path
        self.handle = handle
        self.connection_params = connection_params or {}
        self._custom_connection = connection
        self.join_spec = join_spec
        self.concat_spec = concat_spec

        # Mutual-exclusion guard: cannot have both join_spec and concat_spec
        if join_spec is not None and concat_spec is not None:
            raise ValueError(
                f"DataSource '{self.name}' cannot have both join_spec and concat_spec. "
                "Use one or the other, not both."
            )

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
            DataSourceError: If function source doesn't return a DataFrame,
                or if concat_spec column validation fails.
        """
        if self.concat_spec is not None:
            return self._load_concat(**params)
        elif self.source == SourceType.FILE:
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

    def _load_concat(self, **params) -> pd.DataFrame:
        """Load data by concatenating multiple constituent DataSources row-wise.

        Loads each constituent DataSource in the order declared in concat_spec,
        validates column consistency, and concatenates with a reset index.

        Args:
            **params: Keyword arguments passed to each constituent source's load.

        Returns:
            Single concatenated DataFrame with reset index.

        Raises:
            DataSourceError: If constituent sources have mismatched column names.
        """
        sources = self.concat_spec.sources
        dataframes = []

        for source in sources:
            df = source.load(**params)
            dataframes.append(df)

        # Validate column names across all sources
        base_columns = list(dataframes[0].columns)
        base_columns_set = set(base_columns)

        for i, df in enumerate(dataframes[1:], start=1):
            source_name = sources[i].name
            df_columns_set = set(df.columns)

            if df_columns_set != base_columns_set:
                missing = base_columns_set - df_columns_set
                extra = df_columns_set - base_columns_set
                delta = ""
                if missing:
                    delta += f"Missing columns: {sorted(missing)}. "
                if extra:
                    delta += f"Extra columns: {sorted(extra)}."
                raise DataSourceError(
                    f"DataSource '{source_name}' has different columns than the "
                    f"first source. {delta.strip()}"
                )

        # Warn about dtype mismatches for shared columns
        all_columns = base_columns
        for i, df in enumerate(dataframes[1:], start=1):
            source_name = sources[i].name
            for col in all_columns:
                dtype_first = dataframes[0][col].dtype
                dtype_current = df[col].dtype
                if dtype_first != dtype_current:
                    logger.warning(
                        f"DataSource '{source_name}' has dtype '{dtype_current}' for "
                        f"column '{col}' which differs from the first source's "
                        f"dtype '{dtype_first}'. Proceeding with concatenation."
                    )

        # Concatenate row-wise with reset index
        result = pd.concat(dataframes, axis=0, ignore_index=True)
        return result

    def __repr__(self) -> str:
        return f"DataSource({self.name}, source={self.source.value})"


@dataclass
class JoinSpec:
    """Specification for joining a DataSource to the primary source.
    
    Used when combining multiple DataSources that share a common key.
    The first DataSource in a list is treated as the primary; subsequent
    sources are joined to it using their JoinSpec.
    
    Example:
        ```python
        from geronimo.data_sources import DataSource, JoinSpec
        
        # Primary training source
        training_customers = DataSource(
            name="customers",
            source="file",
            path="data/customers.csv",
        )
        
        # Secondary source to join
        training_transactions = DataSource(
            name="transactions",
            source="file",
            path="data/transactions.csv",
            join_spec=JoinSpec(
                left_on="customer_id",
                right_on="customer_id",
                how="left",
            ),
        )
        ```
    
    Attributes:
        left_on: Column name in the primary (left) source.
        right_on: Column name in this (right) source.
        how: Join type - 'left', 'right', 'inner', or 'outer'.
    """
    left_on: str
    right_on: str
    how: str = "left"


@dataclass
class ConcatSpec:
    """Specification for concatenating multiple DataSources row-wise.
    
    Used when stacking datasets that share the same schema — for example,
    training data from multiple time windows, regions, or upstream sources.
    All constituent sources must have identical column names; dtypes may
    differ (a warning is emitted in that case).
    
    Example:
        ```python
        from geronimo.data_sources import DataSource, ConcatSpec
        
        # Two sources with the same schema
        q1_data = DataSource(
            name="sales_q1",
            source="file",
            path="data/sales_q1.csv",
        )
        q2_data = DataSource(
            name="sales_q2",
            source="file",
            path="data/sales_q2.csv",
        )
        
        # Concatenate row-wise
        sales = DataSource(
            name="sales_all",
            source="concat",
            concat_spec=ConcatSpec(sources=[q1_data, q2_data]),
        )
        df = sales.load()
        ```
    
    Attributes:
        sources: List of two or more DataSource instances to concatenate
            row-wise. Sources are loaded in the order declared.
    """
    sources: list[DataSource]

    def __post_init__(self):
        """Validate that at least two sources are provided."""
        if len(self.sources) < 2:
            raise ValueError(
                f"ConcatSpec requires at least two DataSource instances, "
                f"got {len(self.sources)}."
            )


def collect_data_sources(module, prefix: str) -> list[DataSource]:
    """Collect all DataSource objects whose variable names start with prefix.
    
    Useful for dynamically collecting training_* or production_* DataSources.
    
    Example:
        ```python
        # In data_sources.py
        from geronimo.data_sources import DataSource, collect_data_sources
        import sys
        
        training_customers = DataSource(...)
        training_transactions = DataSource(...)
        production_customers = DataSource(...)
        
        # Auto-collect by prefix
        training_sources = collect_data_sources(sys.modules[__name__], "training_")
        production_sources = collect_data_sources(sys.modules[__name__], "production_")
        ```
    
    Args:
        module: The module to search (typically sys.modules[__name__]).
        prefix: Variable name prefix to match (e.g., "training_").
    
    Returns:
        List of DataSource objects whose variable names start with prefix.
    """
    sources = []
    for name in dir(module):
        if name.startswith(prefix):
            obj = getattr(module, name)
            if isinstance(obj, DataSource):
                sources.append(obj)
    return sources

