"""Geronimo Data Layer.

Provides abstractions for data sources and queries.
"""

from geronimo.data.source import DataSource
from geronimo.data.query import Query

__all__ = ["DataSource", "Query"]
