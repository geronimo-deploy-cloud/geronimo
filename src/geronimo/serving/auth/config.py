"""Authentication configuration."""

from typing import Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class AuthConfig(BaseModel):
    """Authentication configuration for endpoints.

    Example in geronimo.yaml:
        serving:
          auth:
            enabled: true
            method: api_key
            header_name: X-API-Key
            keys_file: .geronimo/keys.json
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable authentication for endpoints",
    )
    method: Literal["api_key", "jwt", "none"] = Field(
        default="api_key",
        description="Authentication method",
    )
    header_name: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication",
    )
    keys_file: Optional[str] = Field(
        default=".geronimo/keys.json",
        description="Path to API keys file (for api_key method)",
    )
    jwt_secret: Optional[str] = Field(
        default=None,
        description="JWT secret for jwt method (use env var in production)",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )

