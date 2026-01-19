"""Tests for geronimo.serving.auth module."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from geronimo.serving.auth.keys import APIKey, APIKeyManager
from geronimo.serving.auth.config import AuthConfig


class TestAPIKey:
    """Tests for APIKey dataclass."""

    def test_api_key_creation(self):
        """Test creating an API key."""
        key = APIKey(
            key_id="abc123",
            name="test-key",
            key_hash="hash123",
            scopes=["predict"],
        )
        assert key.key_id == "abc123"
        assert key.name == "test-key"
        assert key.enabled is True

    def test_api_key_is_valid(self):
        """Test key validity checks."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            scopes=["predict"],
        )
        assert key.is_valid() is True

    def test_api_key_disabled(self):
        """Test disabled key is not valid."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            enabled=False,
        )
        assert key.is_valid() is False

    def test_api_key_expired(self):
        """Test expired key is not valid."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            expires_at=datetime.now(timezone.utc) - timedelta(days=1),
        )
        assert key.is_valid() is False

    def test_api_key_has_scope(self):
        """Test scope checking."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            scopes=["predict", "batch"],
        )
        assert key.has_scope("predict") is True
        assert key.has_scope("batch") is True
        assert key.has_scope("admin") is False

    def test_api_key_wildcard_scope(self):
        """Test wildcard scope."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            scopes=["*"],
        )
        assert key.has_scope("anything") is True

    def test_api_key_to_dict(self):
        """Test serialization to dict."""
        key = APIKey(
            key_id="abc",
            name="test",
            key_hash="hash",
            scopes=["predict"],
        )
        d = key.to_dict()
        assert d["key_id"] == "abc"
        assert d["name"] == "test"
        assert d["scopes"] == ["predict"]

    def test_api_key_from_dict(self):
        """Test deserialization from dict."""
        d = {
            "key_id": "xyz",
            "name": "loaded",
            "key_hash": "hash",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": None,
            "scopes": ["predict"],
            "enabled": True,
        }
        key = APIKey.from_dict(d)
        assert key.key_id == "xyz"
        assert key.name == "loaded"


class TestAPIKeyManager:
    """Tests for APIKeyManager."""

    def test_create_key(self, temp_dir: Path):
        """Test creating a new API key."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        raw_key, api_key = manager.create_key("test-key", scopes=["predict"])
        
        assert raw_key.startswith("grn_")
        assert api_key.name == "test-key"
        assert api_key.scopes == ["predict"]
        assert keys_file.exists()

    def test_validate_key(self, temp_dir: Path):
        """Test validating a key."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        raw_key, _ = manager.create_key("test-key")
        validated = manager.validate(raw_key)
        
        assert validated is not None
        assert validated.name == "test-key"

    def test_validate_invalid_key(self, temp_dir: Path):
        """Test that invalid keys are rejected."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        assert manager.validate("invalid") is None
        assert manager.validate("grn_invalid") is None
        assert manager.validate("") is None
        assert manager.validate(None) is None

    def test_revoke_key(self, temp_dir: Path):
        """Test revoking a key."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        raw_key, api_key = manager.create_key("test-key")
        manager.revoke(api_key.key_id)
        
        # Key should no longer validate
        assert manager.validate(raw_key) is None

    def test_delete_key(self, temp_dir: Path):
        """Test deleting a key."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        _, api_key = manager.create_key("test-key")
        assert len(manager.list_keys()) == 1
        
        manager.delete(api_key.key_id)
        assert len(manager.list_keys()) == 0

    def test_list_keys(self, temp_dir: Path):
        """Test listing all keys."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        manager.create_key("key1")
        manager.create_key("key2")
        manager.create_key("key3")
        
        keys = manager.list_keys()
        assert len(keys) == 3

    def test_persistence(self, temp_dir: Path):
        """Test keys persist across manager instances."""
        keys_file = temp_dir / "keys.json"
        
        # Create keys with first manager
        manager1 = APIKeyManager(str(keys_file))
        raw_key, _ = manager1.create_key("persistent-key")
        
        # Load with new manager
        manager2 = APIKeyManager(str(keys_file))
        validated = manager2.validate(raw_key)
        
        assert validated is not None
        assert validated.name == "persistent-key"


class TestAuthConfig:
    """Tests for AuthConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AuthConfig()
        assert config.enabled is False
        assert config.method == "api_key"
        assert config.header_name == "X-API-Key"

    def test_enabled_config(self):
        """Test enabled configuration."""
        config = AuthConfig(enabled=True, method="api_key")
        assert config.enabled is True
        assert config.method == "api_key"

    def test_jwt_config(self):
        """Test JWT configuration."""
        config = AuthConfig(
            enabled=True,
            method="jwt",
            jwt_secret="secret123",
        )
        assert config.method == "jwt"
        assert config.jwt_secret == "secret123"
