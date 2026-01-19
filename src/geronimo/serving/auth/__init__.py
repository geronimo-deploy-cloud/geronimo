"""Authentication module for Geronimo serving endpoints.

Components:
- APIKeyManager: Create, validate, revoke API keys
- AuthMiddleware: FastAPI middleware for authentication
- AuthConfig: Pydantic configuration model
- require_auth: Route decorator for scope checking

Note: FastAPI middleware components are imported lazily to avoid
requiring FastAPI for CLI-only usage.
"""

from geronimo.serving.auth.keys import APIKey, APIKeyManager
from geronimo.serving.auth.config import AuthConfig

# Lazy imports for FastAPI components
def get_auth_middleware():
    """Get AuthMiddleware class (requires FastAPI)."""
    from geronimo.serving.auth.middleware import AuthMiddleware
    return AuthMiddleware

def get_require_auth():
    """Get require_auth decorator (requires FastAPI)."""
    from geronimo.serving.auth.middleware import require_auth
    return require_auth

def get_current_api_key():
    """Get current_api_key function (requires FastAPI context)."""
    from geronimo.serving.auth.middleware import get_current_api_key as _get
    return _get()

__all__ = [
    "APIKey",
    "APIKeyManager",
    "AuthConfig",
    "get_auth_middleware",
    "get_require_auth",
    "get_current_api_key",
]
