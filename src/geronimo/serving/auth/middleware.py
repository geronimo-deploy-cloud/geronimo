"""Authentication middleware for FastAPI endpoints."""

import logging
from contextvars import ContextVar
from functools import wraps
from typing import Callable, Optional

from fastapi import HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware

from geronimo.serving.auth.config import AuthConfig
from geronimo.serving.auth.keys import APIKey, APIKeyManager

logger = logging.getLogger(__name__)

# Context variable to store current API key
_current_api_key: ContextVar[Optional[APIKey]] = ContextVar(
    "current_api_key", default=None
)


def get_current_api_key() -> Optional[APIKey]:
    """Get the current request's API key.

    Returns:
        APIKey if authenticated, None otherwise.

    Example:
        ```python
        from geronimo.serving.auth import get_current_api_key

        @router.post("/predict")
        async def predict(request: dict):
            api_key = get_current_api_key()
            if api_key:
                logger.info(f"Request from key: {api_key.name}")
        ```
    """
    return _current_api_key.get()


class AuthMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for API key authentication.

    Example:
        ```python
        from fastapi import FastAPI
        from geronimo.serving.auth import AuthMiddleware, AuthConfig

        app = FastAPI()

        config = AuthConfig(enabled=True, method="api_key")
        app.add_middleware(AuthMiddleware, config=config)
        ```
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {"/health", "/healthz", "/ready", "/docs", "/openapi.json"}

    def __init__(self, app, config: AuthConfig):
        """Initialize middleware.

        Args:
            app: FastAPI application.
            config: Authentication configuration.
        """
        super().__init__(app)
        self.config = config
        self.key_manager = (
            APIKeyManager(config.keys_file)
            if config.method == "api_key" and config.keys_file
            else None
        )

    async def dispatch(self, request: Request, call_next):
        """Process request and validate authentication."""
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip if auth disabled
        if not self.config.enabled:
            return await call_next(request)

        # Validate based on method
        if self.config.method == "api_key":
            api_key = await self._validate_api_key(request)
        elif self.config.method == "jwt":
            api_key = await self._validate_jwt(request)
        else:
            api_key = None

        # Store API key in context for route handlers
        token = _current_api_key.set(api_key)
        try:
            response = await call_next(request)
            return response
        finally:
            _current_api_key.reset(token)

    async def _validate_api_key(self, request: Request) -> Optional[APIKey]:
        """Validate API key from request header."""
        if not self.key_manager:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API key authentication not configured",
            )

        key = request.headers.get(self.config.header_name)
        if not key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Missing {self.config.header_name} header",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        api_key = self.key_manager.validate(key)
        if not api_key:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )

        return api_key

    async def _validate_jwt(self, request: Request) -> Optional[APIKey]:
        """Validate JWT token from Authorization header."""
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid Authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )

        token = auth_header.split(" ", 1)[1]

        try:
            import jwt

            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm],
            )
            # Create APIKey-like object from JWT claims
            return APIKey(
                key_id=payload.get("sub", "jwt"),
                name=payload.get("name", "JWT User"),
                key_hash="",  # Not applicable for JWT
                scopes=payload.get("scopes", ["predict"]),
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )


def require_auth(scope: str = "predict") -> Callable:
    """Decorator to require authentication for a route.

    Use this for fine-grained scope checking when middleware
    provides authentication.

    Args:
        scope: Required scope for this endpoint.

    Example:
        ```python
        from geronimo.serving.auth import require_auth

        @router.post("/predict")
        @require_auth(scope="predict")
        async def predict(request: dict):
            ...

        @router.post("/admin/retrain")
        @require_auth(scope="admin")
        async def retrain():
            ...
        ```
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            api_key = get_current_api_key()
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            if not api_key.has_scope(scope):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Scope '{scope}' required",
                )
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_scopes(*scopes: str) -> Callable:
    """Decorator to require multiple scopes.

    Args:
        scopes: Required scopes (all must be present).

    Example:
        ```python
        @router.post("/admin/deploy")
        @require_scopes("admin", "deploy")
        async def deploy():
            ...
        ```
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            api_key = get_current_api_key()
            if not api_key:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required",
                )
            for scope in scopes:
                if not api_key.has_scope(scope):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Scope '{scope}' required",
                    )
            return await func(*args, **kwargs)

        return wrapper

    return decorator
