"""API Key management for endpoint authentication."""

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class APIKey:
    """Represents an API key with metadata."""

    key_id: str
    name: str
    key_hash: str  # SHA-256 hash of the actual key
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    scopes: list[str] = field(default_factory=lambda: ["predict"])
    enabled: bool = True

    def is_valid(self) -> bool:
        """Check if key is valid (enabled and not expired)."""
        if not self.enabled:
            return False
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        return True

    def has_scope(self, scope: str) -> bool:
        """Check if key has the specified scope."""
        return scope in self.scopes or "*" in self.scopes

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            "key_id": self.key_id,
            "name": self.name,
            "key_hash": self.key_hash,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "scopes": self.scopes,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "APIKey":
        """Create from dictionary."""
        return cls(
            key_id=data["key_id"],
            name=data["name"],
            key_hash=data["key_hash"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            scopes=data.get("scopes", ["predict"]),
            enabled=data.get("enabled", True),
        )


class APIKeyManager:
    """Manage API keys for endpoint authentication.

    Example:
        ```python
        manager = APIKeyManager(".geronimo/keys.json")

        # Generate a new key
        key, api_key = manager.create_key("production")
        print(f"Save this key: {key}")  # Only shown once!

        # Validate a key
        api_key = manager.validate("grn_abc123...")
        if api_key and api_key.has_scope("predict"):
            # Allow request
        ```
    """

    KEY_PREFIX = "grn_"  # Geronimo key prefix

    def __init__(self, keys_file: str = ".geronimo/keys.json"):
        """Initialize key manager.

        Args:
            keys_file: Path to JSON file storing key metadata.
        """
        self.keys_file = Path(keys_file)
        self._keys: dict[str, APIKey] = {}
        self._load_keys()

    def _load_keys(self) -> None:
        """Load keys from file."""
        if self.keys_file.exists():
            try:
                data = json.loads(self.keys_file.read_text())
                self._keys = {
                    k: APIKey.from_dict(v) for k, v in data.get("keys", {}).items()
                }
            except (json.JSONDecodeError, KeyError):
                self._keys = {}

    def _save_keys(self) -> None:
        """Save keys to file."""
        self.keys_file.parent.mkdir(parents=True, exist_ok=True)
        data = {"keys": {k: v.to_dict() for k, v in self._keys.items()}}
        self.keys_file.write_text(json.dumps(data, indent=2))

    def _hash_key(self, key: str) -> str:
        """Generate SHA-256 hash of a key."""
        return hashlib.sha256(key.encode()).hexdigest()

    def create_key(
        self,
        name: str,
        scopes: Optional[list[str]] = None,
        expires_at: Optional[datetime] = None,
    ) -> tuple[str, APIKey]:
        """Create a new API key.

        Args:
            name: Human-readable name for the key.
            scopes: List of allowed scopes (default: ["predict"]).
            expires_at: Optional expiration datetime.

        Returns:
            Tuple of (raw_key, APIKey object).
            The raw_key is only returned once - store it securely!
        """
        # Generate cryptographically secure key
        raw_key = self.KEY_PREFIX + secrets.token_urlsafe(32)
        key_hash = self._hash_key(raw_key)
        key_id = secrets.token_hex(8)

        api_key = APIKey(
            key_id=key_id,
            name=name,
            key_hash=key_hash,
            scopes=scopes or ["predict"],
            expires_at=expires_at,
        )

        self._keys[key_id] = api_key
        self._save_keys()

        return raw_key, api_key

    def validate(self, key: str) -> Optional[APIKey]:
        """Validate an API key.

        Uses constant-time comparison to prevent timing attacks.

        Args:
            key: Raw API key to validate.

        Returns:
            APIKey object if valid, None otherwise.
        """
        if not key or not key.startswith(self.KEY_PREFIX):
            return None

        key_hash = self._hash_key(key)

        for api_key in self._keys.values():
            # Use constant-time comparison to prevent timing attacks (SOC2 requirement)
            if hmac.compare_digest(api_key.key_hash, key_hash) and api_key.is_valid():
                return api_key

        return None

    def revoke(self, key_id: str) -> bool:
        """Revoke an API key by ID.

        Args:
            key_id: The key ID to revoke.

        Returns:
            True if key was found and revoked.
        """
        if key_id in self._keys:
            self._keys[key_id].enabled = False
            self._save_keys()
            return True
        return False

    def delete(self, key_id: str) -> bool:
        """Permanently delete an API key.

        Args:
            key_id: The key ID to delete.

        Returns:
            True if key was found and deleted.
        """
        if key_id in self._keys:
            del self._keys[key_id]
            self._save_keys()
            return True
        return False

    def list_keys(self) -> list[APIKey]:
        """List all API keys (without the actual key values)."""
        return list(self._keys.values())

    def get_key(self, key_id: str) -> Optional[APIKey]:
        """Get a specific key by ID."""
        return self._keys.get(key_id)
