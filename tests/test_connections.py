"""Tests for geronimo.data_sources.connection module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from geronimo.data_sources.connection import (
    DatabaseConnection,
    BaseDatabaseConnection,
    SnowflakeConnection,
    PostgresConnection,
    SQLServerConnection,
    get_connection,
)


class TestDatabaseConnectionProtocol:
    """Tests for DatabaseConnection protocol."""

    def test_protocol_runtime_checkable(self):
        """Test that protocol is runtime checkable."""
        class MockConnection:
            def connect(self) -> None:
                pass
            
            def execute(self, sql: str) -> pd.DataFrame:
                return pd.DataFrame()
            
            def close(self) -> None:
                pass
        
        conn = MockConnection()
        assert isinstance(conn, DatabaseConnection)

    def test_non_conforming_class_not_connection(self):
        """Test that non-conforming classes don't match protocol."""
        class NotAConnection:
            def do_something(self):
                pass
        
        obj = NotAConnection()
        assert not isinstance(obj, DatabaseConnection)


class TestBaseDatabaseConnection:
    """Tests for BaseDatabaseConnection."""

    def test_context_manager(self):
        """Test context manager interface."""
        class TestConnection(BaseDatabaseConnection):
            def __init__(self):
                super().__init__()
                self.connected = False
                self.closed = False
            
            def connect(self):
                self.connected = True
            
            def execute(self, sql):
                return pd.DataFrame({"data": [1, 2, 3]})
            
            def close(self):
                self.closed = True
        
        conn = TestConnection()
        with conn:
            assert conn.connected
        assert conn.closed


class TestGetConnection:
    """Tests for get_connection factory."""

    def test_get_snowflake_connection(self):
        """Test factory returns SnowflakeConnection."""
        conn = get_connection("snowflake", {})
        assert isinstance(conn, SnowflakeConnection)

    def test_get_postgres_connection(self):
        """Test factory returns PostgresConnection."""
        conn = get_connection("postgres", {})
        assert isinstance(conn, PostgresConnection)

    def test_get_sqlserver_connection(self):
        """Test factory returns SQLServerConnection."""
        conn = get_connection("sqlserver", {})
        assert isinstance(conn, SQLServerConnection)

    def test_get_unknown_connection_raises(self):
        """Test factory raises for unknown type."""
        with pytest.raises(ValueError, match="Unsupported database type"):
            get_connection("unknown_db", {})


class TestSnowflakeConnection:
    """Tests for SnowflakeConnection."""

    def test_not_connected_execute_raises(self):
        """Test execute raises if not connected."""
        conn = SnowflakeConnection({})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.execute("SELECT 1")

    def test_connection_params_stored(self):
        """Test connection params are stored correctly."""
        params = {"user": "test", "password": "secret", "account": "acct"}
        conn = SnowflakeConnection(params)
        assert conn.connection_params == params
        assert conn._connection is None


class TestPostgresConnection:
    """Tests for PostgresConnection."""

    def test_not_connected_execute_raises(self):
        """Test execute raises if not connected."""
        conn = PostgresConnection({})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.execute("SELECT 1")


class TestSQLServerConnection:
    """Tests for SQLServerConnection."""

    def test_not_connected_execute_raises(self):
        """Test execute raises if not connected."""
        conn = SQLServerConnection({})
        with pytest.raises(RuntimeError, match="Not connected"):
            conn.execute("SELECT 1")
