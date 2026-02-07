"""Centralized HTTP client utilities for Geronimo Cloud API.

This module provides a unified HTTP client context manager to reduce
code duplication and ensure consistent configuration across all API calls.
"""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import httpx

logger = logging.getLogger(__name__)

# Default timeouts
API_TIMEOUT = 30.0  # Standard API calls
TRANSFER_TIMEOUT = 60.0  # Upload/download operations


@contextmanager
def api_client(
    base_url: str,
    headers: dict,
    timeout: float = API_TIMEOUT,
    operation: Optional[str] = None,
) -> Generator[httpx.Client, None, None]:
    """Context manager for API HTTP client with consistent configuration.
    
    Args:
        base_url: Base URL for the API (e.g., "https://api.geronimo.dev/v1")
        headers: Request headers including authorization
        timeout: Request timeout in seconds (default: 30.0)
        operation: Optional operation name for logging
        
    Yields:
        Configured httpx.Client instance
        
    Example:
        ```python
        with api_client(self.api_url, self.headers, operation="save_artifact") as http:
            resp = http.post("/artifacts", json=data)
            resp.raise_for_status()
        ```
    """
    if operation:
        logger.debug(f"Starting API operation: {operation}")
    
    with httpx.Client(base_url=base_url, headers=headers, timeout=timeout) as client:
        try:
            yield client
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP {e.response.status_code} in {operation or 'API call'}: {e}")
            raise
        except httpx.NetworkError as e:
            logger.error(f"Network error in {operation or 'API call'}: {e}")
            raise


@contextmanager
def transfer_client(
    timeout: float = TRANSFER_TIMEOUT,
    operation: Optional[str] = None,
) -> Generator[httpx.Client, None, None]:
    """Context manager for upload/download operations (no base URL).
    
    Args:
        timeout: Request timeout in seconds (default: 60.0)
        operation: Optional operation name for logging
        
    Yields:
        Configured httpx.Client instance for direct URL access
        
    Example:
        ```python
        with transfer_client(operation="upload_artifact") as http:
            resp = http.put(upload_url, content=data)
            resp.raise_for_status()
        ```
    """
    if operation:
        logger.debug(f"Starting transfer operation: {operation}")
    
    with httpx.Client(timeout=timeout) as client:
        try:
            yield client
        except httpx.HTTPError as e:
            logger.error(f"Transfer error in {operation or 'transfer'}: {e}")
            raise
