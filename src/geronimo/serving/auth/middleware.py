"""Authentication middleware for FastAPI endpoints."""

import logging
from collections import defaultdict
from contextvars import ContextVar
from datetime import datetime, timedelta
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

    Includes rate limiting for failed authentication attempts
    to prevent brute-force attacks (SOC2 compliance).

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

    # Rate limiting configuration
    MAX_FAILED_ATTEMPTS = 5  # Maximum failed attempts before lockout
    LOCKOUT_DURATION = timedelta(minutes=15)  # Duration of lockout
    ATTEMPT_WINDOW = timedelta(minutes=5)  # Window for counting failed attempts

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
        # Track failed attempts per IP: {ip: [(timestamp, ...], ...}
        self._failed_attempts: dict[str, list[datetime]] = defaultdict(list)
        # Track locked out IPs: {ip: lockout_until}
        self._lockouts: dict[str, datetime] = {}

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client IP is rate limited.
        
        Args:
            client_ip: Client IP address.
            
        Returns:
            True if rate limited, False otherwise.
        """
        now = datetime.utcnow()
        
        # Check if currently locked out
        if client_ip in self._lockouts:
            if now < self._lockouts[client_ip]:
                return True
            else:
                # Lockout expired, remove it
                del self._lockouts[client_ip]
        
        return False

    def _record_failed_attempt(self, client_ip: str) -> None:
        """Record a failed authentication attempt.
        
        May trigger a lockout if too many failures.
        
        Args:
            client_ip: Client IP address.
        """
        now = datetime.utcnow()
        cutoff = now - self.ATTEMPT_WINDOW
        
        # Clean old attempts and add new one
        self._failed_attempts[client_ip] = [
            ts for ts in self._failed_attempts[client_ip] if ts > cutoff
        ]
        self._failed_attempts[client_ip].append(now)
        
        # Check if we need to lockout
        if len(self._failed_attempts[client_ip]) >= self.MAX_FAILED_ATTEMPTS:
            self._lockouts[client_ip] = now + self.LOCKOUT_DURATION
            self._failed_attempts[client_ip] = []  # Clear attempts on lockout
            logger.warning(
                f"Rate limit triggered for {client_ip}. "
                f"Locked out until {self._lockouts[client_ip].isoformat()}"
            )

    def _clear_failed_attempts(self, client_ip: str) -> None:
        """Clear failed attempts after successful auth.
        
        Args:
            client_ip: Client IP address.
        """
        if client_ip in self._failed_attempts:
            del self._failed_attempts[client_ip]

    async def dispatch(self, request: Request, call_next):
        """Process request and validate authentication."""
        # Skip auth for public paths
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Skip if auth disabled
        if not self.config.enabled:
            return await call_next(request)

        # Get client IP for rate limiting
        client_ip = request.client.host if request.client else "unknown"
        
        # Check rate limiting
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limited request from {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed authentication attempts. Please try again later.",
            )

        # Validate based on method
        try:
            if self.config.method == "api_key":
                api_key = await self._validate_api_key(request)
            elif self.config.method == "jwt":
                api_key = await self._validate_jwt(request)
            else:
                api_key = None
            
            # Successful auth - clear failed attempts
            if api_key:
                self._clear_failed_attempts(client_ip)
        except HTTPException as e:
            # Record failed attempt if it was an auth failure
            if e.status_code == status.HTTP_401_UNAUTHORIZED:
                self._record_failed_attempt(client_ip)
            raise

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
