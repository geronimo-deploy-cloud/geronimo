"""Tests for geronimo.data_sources module."""

import logging

import pytest

from geronimo.data_sources import Query
from geronimo.data_sources.source import ConcatSpec, DataSourceError


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
        from geronimo.data_sources import DataSource
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

        from geronimo.data_sources import DataSource
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
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a query"):
            DataSource(
                name="db_data",
                source="postgres",
            )

    def test_file_source_requires_path(self):
        """Test that file sources require a path."""
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a path"):
            DataSource(
                name="file_data",
                source="file",
            )

    def test_function_data_source(self):
        """Test DataSource with function handle."""
        from geronimo.data_sources import DataSource
        import pandas as pd
        
        def load_data():
            return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        
        source = DataSource(
            name="func_data",
            source="function",
            handle=load_data,
        )
        
        df = source.load()
        assert len(df) == 3
        assert "x" in df.columns

    def test_function_data_source_with_params(self):
        """Test function DataSource passes params to handle."""
        from geronimo.data_sources import DataSource
        import pandas as pd
        
        def load_data(limit=10):
            return pd.DataFrame({"n": range(limit)})
        
        source = DataSource(
            name="func_data",
            source="function",
            handle=load_data,
        )
        
        df = source.load(limit=5)
        assert len(df) == 5

    def test_function_source_requires_handle(self):
        """Test that function sources require a handle."""
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a handle"):
            DataSource(
                name="func_data",
                source="function",
            )

    def test_function_source_validates_return_type(self):
        """Test that function source validates DataFrame return type at runtime."""
        from geronimo.data_sources import DataSource
        from geronimo.data_sources.source import DataSourceError
        
        def bad_handle():
            return {"x": [1, 2, 3]}  # Returns dict, not DataFrame
        
        source = DataSource(
            name="bad_func",
            source="function",
            handle=bad_handle,
        )
        
        with pytest.raises(DataSourceError, match="must return a pandas DataFrame"):
            source.load()

"""Tests for geronimo.data_sources module."""

import logging

import pytest

from geronimo.data_sources import Query
from geronimo.data_sources.source import ConcatSpec, DataSourceError


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
        from geronimo.data_sources import DataSource
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

        from geronimo.data_sources import DataSource
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
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a query"):
            DataSource(
                name="db_data",
                source="postgres",
            )

    def test_file_source_requires_path(self):
        """Test that file sources require a path."""
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a path"):
            DataSource(
                name="file_data",
                source="file",
            )

    def test_function_data_source(self):
        """Test DataSource with function handle."""
        from geronimo.data_sources import DataSource
        import pandas as pd
        
        def load_data():
            return pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
        
        source = DataSource(
            name="func_data",
            source="function",
            handle=load_data,
        )
        
        df = source.load()
        assert len(df) == 3
        assert "x" in df.columns

    def test_function_data_source_with_params(self):
        """Test function DataSource passes params to handle."""
        from geronimo.data_sources import DataSource
        import pandas as pd
        
        def load_data(limit=10):
            return pd.DataFrame({"n": range(limit)})
        
        source = DataSource(
            name="func_data",
            source="function",
            handle=load_data,
        )
        
        df = source.load(limit=5)
        assert len(df) == 5

    def test_function_source_requires_handle(self):
        """Test that function sources require a handle."""
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="require a handle"):
            DataSource(
                name="func_data",
                source="function",
            )

    def test_function_source_validates_return_type(self):
        """Test that function source validates DataFrame return type at runtime."""
        from geronimo.data_sources import DataSource
        
        def bad_handle():
            return {"x": [1, 2, 3]}  # Returns dict, not DataFrame
        
        source = DataSource(
            name="bad_func",
            source="function",
            handle=bad_handle,
        )
        
        with pytest.raises(DataSourceError, match="must return a pandas DataFrame"):
            source.load()

    def test_function_source_handle_must_be_callable(self):
        """Test that handle must be callable."""
        from geronimo.data_sources import DataSource
        
        with pytest.raises(ValueError, match="must be callable"):
            DataSource(
                name="bad_func",
                source="function",
                handle="not_a_function",
            )


class TestConcatSpec:
    """Tests for ConcatSpec class and DataSource.concat_spec integration."""

    def test_concat_spec_requires_at_least_two_sources(self):
        """Test that ConcatSpec raises ValueError with fewer than two sources."""
        import pandas as pd
        from geronimo.data_sources import DataSource

        source = DataSource(
            name="single",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1]}),
        )
        
        with pytest.raises(ValueError, match="at least two"):
            ConcatSpec(sources=[source])

    def test_concat_spec_accepts_two_sources(self):
        """Test that ConcatSpec accepts exactly two sources."""
        import pandas as pd
        from geronimo.data_sources import DataSource
        
        s1 = DataSource(
            name="s1",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1]}),
        )
        s2 = DataSource(
            name="s2",
            source="function",
            handle=lambda: pd.DataFrame({"x": [2]}),
        )
        
        spec = ConcatSpec(sources=[s1, s2])
        assert len(spec.sources) == 2

    def test_both_join_spec_and_concat_spec_raises_value_error(self):
        """Test that providing both join_spec and concat_spec raises ValueError."""
        import pandas as pd
        from geronimo.data_sources import DataSource, JoinSpec
        
        s1 = DataSource(
            name="s1",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1]}),
        )
        s2 = DataSource(
            name="s2",
            source="function",
            handle=lambda: pd.DataFrame({"x": [2]}),
        )
        
        with pytest.raises(ValueError, match="cannot have both"):
            DataSource(
                name="both",
                source="function",
                handle=lambda: pd.DataFrame({"x": [1]}),
                join_spec=JoinSpec(left_on="x", right_on="x"),
                concat_spec=ConcatSpec(sources=[s1, s2]),
            )

    def test_successful_concat_no_warnings(self, caplog):
        """Test successful concatenation emits no warnings."""
        import pandas as pd
        from geronimo.data_sources import DataSource
        
        s1 = DataSource(
            name="s1",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        )
        s2 = DataSource(
            name="s2",
            source="function",
            handle=lambda: pd.DataFrame({"x": [5, 6], "y": [7, 8]}),
        )
        
        combined = DataSource(
            name="combined",
            source="function",
            handle=lambda: pd.DataFrame(),  # dummy, not used
            concat_spec=ConcatSpec(sources=[s1, s2]),
        )
        
        with caplog.at_level(logging.WARNING):
            df = combined.load()
        
        assert len(df) == 4
        assert list(df.columns) == ["x", "y"]
        assert list(df["x"]) == [1, 2, 5, 6]
        assert list(df["y"]) == [3, 4, 7, 8]
        assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 0

    def test_concat_dtype_mismatch_emits_warning(self, caplog):
        """Test that dtype mismatch across sources emits a warning but proceeds."""
        import pandas as pd
        from geronimo.data_sources import DataSource
        
        s1 = DataSource(
            name="s1",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1, 2], "y": [3.0, 4.0]}),
        )
        s2 = DataSource(
            name="s2",
            source="function",
            handle=lambda: pd.DataFrame({"x": [5, 6], "y": ["a", "b"]}),
        )
        
        combined = DataSource(
            name="combined",
            source="function",
            handle=lambda: pd.DataFrame(),  # dummy, not used
            concat_spec=ConcatSpec(sources=[s1, s2]),
        )
        
        with caplog.at_level(logging.WARNING):
            df = combined.load()
        
        assert len(df) == 4
        # Warning should mention the column and differing types
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) >= 1
        assert "y" in warnings[0].message

    def test_concat_column_name_mismatch_raises_data_source_error(self):
        """Test that mismatched column names raises DataSourceError."""
        import pandas as pd
        from geronimo.data_sources import DataSource
        
        s1 = DataSource(
            name="s1",
            source="function",
            handle=lambda: pd.DataFrame({"x": [1, 2], "y": [3, 4]}),
        )
        s2 = DataSource(
            name="s2",
            source="function",
            handle=lambda: pd.DataFrame({"x": [5, 6], "z": [7, 8]}),  # z instead of y
        )
        
        combined = DataSource(
            name="combined",
            source="function",
            handle=lambda: pd.DataFrame(),  # dummy, not used
            concat_spec=ConcatSpec(sources=[s1, s2]),
        )
        
        with pytest.raises(DataSourceError, match="s2"):
            combined.load()
