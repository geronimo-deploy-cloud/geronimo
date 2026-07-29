"""DataSource abstraction for connecting to data backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass  # JoinSpec forward reference handled via string annotation

import numpy as np
import pandas as pd

from geronimo.data_sources.query import Query
from geronimo.data_sources.connection import get_connection, DatabaseConnection


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

    split_spec: Optional["SplitSpec"]
    """Specification for splitting this source into train/eval sets."""

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
        split_spec: Optional["SplitSpec"] = None,
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
            join_spec: Optional specification for joining to this source.
            split_spec: Optional specification for splitting into train/eval sets.
        
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
        self.join_spec = join_spec
        self.split_spec = split_spec

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

    def split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into training and evaluation sets.

        Convenience method that loads the DataSource and delegates to
        ``self.split_spec.split(df)``.  Raises ``ValueError`` if no
        ``split_spec`` was provided.

        Returns:
            Tuple of (train_df, eval_df) DataFrames.

        Raises:
            ValueError: If this DataSource has no ``split_spec``.
        """
        if self.split_spec is None:
            raise ValueError(
                f"DataSource '{self.name}' has no split_spec. "
                f"Provide a SplitSpec to enable splitting."
            )
        return self.split_spec.split(self.load())
    
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


@dataclass
class SplitSpec:
    """Specification for splitting a DataSource into training and evaluation sets.

    Declared on a DataSource alongside ``join_spec`` so that how a dataset
    is split into training and evaluation portions is inspectable at
    definition time, alongside how it is joined to other sources.

    This design attaches ``SplitSpec`` to ``DataSource`` (not ``Model``)
    because splitting is a property of the *data*, not the model.
    ``DataSource.load()`` returns a DataFrame; the split is the natural
    next operation. ``Model.train()`` simply receives (X, y) post-split.

    Example::

        ```python
        from geronimo.data_sources import DataSource, SplitSpec

        training_data = DataSource(
            name="sales",
            source="file",
            path="data/sales.csv",
            split_spec=SplitSpec(
                strategy="random",
                ratio=0.8,
                random_seed=42,
            ),
        )

        train_df, eval_df = training_data.split()
        ```

    Attributes:
        strategy: One of ``"random"`` or ``"time"``.
        ratio: Fraction of rows assigned to the training set (0 < ratio < 1).
               Defaults to ``0.8`` (80/20 split).
        random_seed: Seed for reproducibility on random splits. Defaults to ``42``.
        datetime_column: Column name used for time-based splits (required when
                         ``strategy="time"``).
        cutoff: Timestamp string or datetime object defining the time-based
                split boundary. Rows with ``datetime_column <= cutoff`` go
                to training; rows with ``datetime_column > cutoff`` go to
                evaluation (required when ``strategy="time"``).
    """

    strategy: str = "random"
    ratio: float = 0.8
    random_seed: int = 42
    datetime_column: Optional[str] = None
    cutoff: Optional[str] = None

    def __post_init__(self):
        """Validate split_spec parameters after initialization."""
        if self.ratio <= 0 or self.ratio >= 1:
            raise ValueError(
                f"Split ratio must be between 0 and 1 (exclusive). "
                f"Got: {self.ratio}."
            )

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split a DataFrame into training and evaluation sets.

        Args:
            df: Input DataFrame to split.

        Returns:
            Tuple of (train_df, eval_df) DataFrames.

        Raises:
            ValueError: If the strategy is unsupported, ratio is invalid,
                        required time-based fields are missing, or the
                        datetime column does not exist in *df*.
        """
        if self.strategy == "random":
            return self._random_split(df)
        elif self.strategy == "time":
            return self._time_split(df)
        else:
            raise ValueError(
                f"Unsupported split strategy: '{self.strategy}'. "
                f"Supported strategies: 'random', 'time'."
            )

    def _random_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Random split by row index, deterministic given the seed."""
        n = len(df)
        if n == 0:
            raise ValueError("Cannot split an empty DataFrame.")

        rng = np.random.RandomState(self.random_seed)
        indices = rng.permutation(n)

        split_idx = int(n * self.ratio)
        train_idx = indices[:split_idx]
        eval_idx = indices[split_idx:]

        return df.iloc[train_idx].reset_index(drop=True), df.iloc[eval_idx].reset_index(
            drop=True
        )

    def _time_split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Time-based split: rows <= cutoff go to training, > cutoff to eval."""
        if not self.datetime_column:
            raise ValueError(
                f"Time-based split requires 'datetime_column' to be set. "
                f"Got: {self.datetime_column!r}."
            )
        if not self.cutoff:
            raise ValueError(
                f"Time-based split requires 'cutoff' to be set. "
                f"Got: {self.cutoff!r}."
            )
        if self.datetime_column not in df.columns:
            raise ValueError(
                f"Datetime column '{self.datetime_column}' not found in the data. "
                f"Available columns: {list(df.columns)}."
            )

        # Parse cutoff to datetime
        cutoff_dt = pd.to_datetime(self.cutoff)
        df_copy = df.copy()

        # Ensure the datetime column is parsed
        df_copy[self.datetime_column] = pd.to_datetime(df_copy[self.datetime_column])

        train_mask = df_copy[self.datetime_column] <= cutoff_dt
        eval_mask = df_copy[self.datetime_column] > cutoff_dt

        train_df = df_copy[train_mask].reset_index(drop=True)
        eval_df = df_copy[eval_mask].reset_index(drop=True)

        return train_df, eval_df


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

