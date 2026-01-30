"""Security-focused tests for geronimo.serving.auth module.

Tests for SOC2 compliance requirements including:
- Constant-time hash comparison
- Rate limiting
- Lockout mechanisms
"""

import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from geronimo.serving.auth.keys import APIKey, APIKeyManager
from geronimo.serving.auth.config import AuthConfig


class TestSecurityHashComparison:
    """Tests for secure hash comparison."""

    def test_validate_uses_constant_time_comparison(self, temp_dir: Path):
        """Test that validation uses hmac.compare_digest."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        # Create a key
        raw_key, api_key = manager.create_key("test-key")
        
        # Patch hmac.compare_digest to verify it's called
        with patch("geronimo.serving.auth.keys.hmac.compare_digest", return_value=True) as mock_compare:
            validated = manager.validate(raw_key)
            # compare_digest should have been called
            assert mock_compare.called

    def test_wrong_key_rejected(self, temp_dir: Path):
        """Test that wrong keys are properly rejected."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        # Create a key
        raw_key, _ = manager.create_key("test-key")
        
        # Try to validate wrong key
        wrong_key = "grn_wrongkeywrongkeywrongkey"
        assert manager.validate(wrong_key) is None

    def test_empty_key_rejected(self, temp_dir: Path):
        """Test that empty keys are rejected."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        assert manager.validate("") is None
        assert manager.validate(None) is None

    def test_key_without_prefix_rejected(self, temp_dir: Path):
        """Test that keys without proper prefix are rejected."""
        keys_file = temp_dir / "keys.json"
        manager = APIKeyManager(str(keys_file))
        
        # Create a real key
        raw_key, _ = manager.create_key("test-key")
        
        # Try without prefix
        no_prefix_key = raw_key.replace("grn_", "")
        assert manager.validate(no_prefix_key) is None


class TestRateLimiting:
    """Tests for authentication rate limiting."""

    @pytest.fixture
    def mock_middleware(self, temp_dir: Path):
        """Create middleware instance for testing."""
        # We need to test the middleware class directly
        from geronimo.serving.auth.middleware import AuthMiddleware
        
        config = AuthConfig(
            enabled=True,
            method="api_key",
            keys_file=str(temp_dir / "keys.json"),
        )
        
        mock_app = MagicMock()
        middleware = AuthMiddleware(mock_app, config)
        return middleware

    def test_initial_request_not_rate_limited(self, mock_middleware):
        """Test first request is not rate limited."""
        assert mock_middleware._is_rate_limited("192.168.1.1") is False

    def test_lockout_after_max_attempts(self, mock_middleware):
        """Test lockout after maximum failed attempts."""
        client_ip = "192.168.1.100"
        
        # Record max failed attempts
        for _ in range(mock_middleware.MAX_FAILED_ATTEMPTS):
            mock_middleware._record_failed_attempt(client_ip)
        
        # Should now be rate limited
        assert mock_middleware._is_rate_limited(client_ip) is True

    def test_lockout_expires(self, mock_middleware):
        """Test lockout expires after duration."""
        client_ip = "192.168.1.101"
        
        # Set an expired lockout
        mock_middleware._lockouts[client_ip] = datetime.utcnow() - timedelta(minutes=1)
        
        # Should not be rate limited (expired)
        assert mock_middleware._is_rate_limited(client_ip) is False
        # Lockout should be cleared
        assert client_ip not in mock_middleware._lockouts

    def test_clear_attempts_on_success(self, mock_middleware):
        """Test failed attempts cleared on successful auth."""
        client_ip = "192.168.1.102"
        
        # Record some failed attempts
        mock_middleware._record_failed_attempt(client_ip)
        mock_middleware._record_failed_attempt(client_ip)
        assert len(mock_middleware._failed_attempts[client_ip]) == 2
        
        # Clear on success
        mock_middleware._clear_failed_attempts(client_ip)
        assert client_ip not in mock_middleware._failed_attempts

    def test_old_attempts_cleaned(self, mock_middleware):
        """Test old attempts are cleaned from tracking."""
        client_ip = "192.168.1.103"
        
        # Add old timestamp directly
        old_time = datetime.utcnow() - timedelta(minutes=10)
        mock_middleware._failed_attempts[client_ip] = [old_time]
        
        # Record new attempt - should clean old one
        mock_middleware._record_failed_attempt(client_ip)
        
        # Should only have one attempt (the new one)
        assert len(mock_middleware._failed_attempts[client_ip]) == 1


class TestKeyExpiration:
    """Tests for API key expiration."""

    def test_expired_key_not_valid(self):
        """Test that expired keys are rejected."""
        key = APIKey(
            key_id="test",
            name="expired",
            key_hash="hash",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert key.is_valid() is False

    def test_future_expiry_valid(self):
        """Test that keys with future expiry are valid."""
        key = APIKey(
            key_id="test",
            name="future",
            key_hash="hash",
            expires_at=datetime.now(timezone.utc) + timedelta(days=30),
        )
        assert key.is_valid() is True

    def test_no_expiry_valid(self):
        """Test that keys without expiry are valid."""
        key = APIKey(
            key_id="test",
            name="no-expiry",
            key_hash="hash",
            expires_at=None,
        )
        assert key.is_valid() is True
