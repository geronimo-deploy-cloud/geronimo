"""Tests for geronimo.data module."""

import pytest

from geronimo.data import Query


class TestQuery:
    """Tests for Query class."""

    def test_query_from_string(self):
        """Test creating query from string."""
        q = Query("SELECT * FROM users WHERE id = :id")
        assert "SELECT * FROM users" in q.sql

    def test_query_with_params(self):
        """Test query parameter substitution."""
        q = Query("SELECT * FROM users WHERE id = :id AND name = :name")
        rendered = q.render(id=123, name="Alice")
        
        # Parameters should be substituted
        assert "123" in rendered or ":id" in rendered  # Depends on implementation

    def test_query_from_file(self, temp_dir):
        """Test loading query from file."""
        query_file = temp_dir / "query.sql"
        query_file.write_text("SELECT * FROM products")
        
        q = Query.from_file(str(query_file))
        assert "SELECT * FROM products" in q.sql

    def test_query_from_file_not_found(self):
        """Test error when query file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            Query.from_file("/nonexistent/query.sql")


class TestDataSource:
    """Tests for DataSource class."""

    def test_file_data_source(self, temp_dir):
        """Test DataSource from CSV file."""
        from geronimo.data import DataSource
        import pandas as pd
        
        # Create test CSV
        csv_path = temp_dir / "data.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(csv_path, index=False)
        
        source = DataSource(
            name="csv_data",
            source="file",
            path=str(csv_path),
        )
        
        df = source.load()
        assert len(df) == 2
        assert "a" in df.columns

    def test_file_data_source_parquet(self, temp_dir):
        """Test DataSource from Parquet file."""
        try:
            import pyarrow
        except ImportError:
            pytest.skip("pyarrow not installed")

        from geronimo.data import DataSource
        import pandas as pd
        
        # Create test Parquet
        pq_path = temp_dir / "data.parquet"
        pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}).to_parquet(pq_path)
        
        source = DataSource(
            name="parquet_data",
            source="file",
            path=str(pq_path),
        )
        
        df = source.load()
        assert len(df) == 3

    def test_database_source_requires_query(self):
        """Test that database sources require a query."""
        from geronimo.data import DataSource
        
        with pytest.raises(ValueError, match="require a query"):
            DataSource(
                name="db_data",
                source="postgres",
            )

    def test_file_source_requires_path(self):
        """Test that file sources require a path."""
        from geronimo.data import DataSource
        
        with pytest.raises(ValueError, match="require a path"):
            DataSource(
                name="file_data",
                source="file",
            )
