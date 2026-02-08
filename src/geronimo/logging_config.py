"""Centralized logging configuration for Geronimo.

Provides consistent logging setup across all modules.

Usage:
    from geronimo.logging_config import setup_logging, get_logger
    
    # In application entry point:
    setup_logging(level="INFO")
    
    # In any module:
    logger = get_logger(__name__)
    logger.info("Message")
"""

import logging
import sys
from typing import Optional


# Default format styles
FORMATS = {
    "simple": "%(levelname)s: %(message)s",
    "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "debug": "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s",
}


def setup_logging(
    level: str = "INFO",
    format_style: str = "simple",
    log_file: Optional[str] = None,
) -> None:
    """Configure logging for Geronimo.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_style: Format style ("simple", "detailed", or "debug").
        log_file: Optional file path to write logs to.
        
    Example:
        # For development:
        setup_logging(level="DEBUG", format_style="debug")
        
        # For production:
        setup_logging(level="INFO", format_style="detailed", log_file="/var/log/geronimo.log")
    """
    log_format = FORMATS.get(format_style, FORMATS["simple"])
    
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(FORMATS["detailed"]))
        handlers.append(file_handler)
    
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given module.
    
    Args:
        name: Module name (typically __name__).
        
    Returns:
        Configured logger instance.
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing started")
        logger.debug("Details: %s", data)
        logger.error("Failed: %s", error)
    """
    return logging.getLogger(name)


# Create package-level logger
logger = get_logger("geronimo")
