"""Query abstraction for SQL-based data sources."""

from pathlib import Path
from typing import Optional


class Query:
    """SQL query wrapper with parameter substitution.

    Supports loading from files and inline SQL definitions.

    Example:
        ```python
        # From file
        query = Query.from_file("queries/training_data.sql")

        # Inline
        query = Query("SELECT * FROM features WHERE date >= :start_date")

        # With parameters
        sql = query.render(start_date="2024-01-01")
        ```
    """

    def __init__(self, sql: str, name: Optional[str] = None):
        """Initialize query.

        Args:
            sql: SQL query string with optional :param placeholders.
            name: Optional query name for tracking.
        """
        self.sql = sql
        self.name = name

    @classmethod
    def from_file(cls, path: str | Path) -> "Query":
        """Load query from SQL file.

        Args:
            path: Path to .sql file.

        Returns:
            Query instance.
        """
        path = Path(path)
        sql = path.read_text()
        return cls(sql=sql, name=path.stem)

    def render(self, **params) -> str:
        """Render query with parameter substitution.

        Args:
            **params: Named parameters to substitute.

        Returns:
            Rendered SQL string.
        """
        sql = self.sql
        for key, value in params.items():
            placeholder = f":{key}"
            if isinstance(value, str):
                sql = sql.replace(placeholder, f"'{value}'")
            else:
                sql = sql.replace(placeholder, str(value))
        return sql

    def __repr__(self) -> str:
        name = self.name or "unnamed"
        preview = self.sql[:50] + "..." if len(self.sql) > 50 else self.sql
        return f"Query({name}: {preview})"
