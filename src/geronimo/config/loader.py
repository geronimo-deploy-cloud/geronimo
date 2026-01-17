"""Configuration loader for Geronimo.

Handles loading, parsing, and environment variable interpolation
for geronimo.yaml configuration files.
"""

import os
import re
from pathlib import Path

import yaml
from pydantic import ValidationError

from geronimo.config.schema import GeronimoConfig


class ConfigurationError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""

    pass


def interpolate_env_vars(value: str) -> str:
    """Interpolate environment variables in a string.

    Supports ${VAR_NAME} and ${VAR_NAME:-default} syntax.

    Args:
        value: String potentially containing environment variable references.

    Returns:
        String with environment variables replaced with their values.
    """
    # Pattern matches ${VAR_NAME} or ${VAR_NAME:-default}
    pattern = r"\$\{([A-Z_][A-Z0-9_]*)(?::-([^}]*))?\}"

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(2)
        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        if default is not None:
            return default
        # Return original if no value and no default
        return match.group(0)

    return re.sub(pattern, replace, value)


def interpolate_dict(data: dict) -> dict:
    """Recursively interpolate environment variables in a dictionary.

    Args:
        data: Dictionary potentially containing string values with env vars.

    Returns:
        Dictionary with all string values interpolated.
    """
    result = {}
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = interpolate_env_vars(value)
        elif isinstance(value, dict):
            result[key] = interpolate_dict(value)
        elif isinstance(value, list):
            result[key] = [
                interpolate_env_vars(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path) -> GeronimoConfig:
    """Load and validate a geronimo.yaml configuration file.

    Args:
        config_path: Path to the geronimo.yaml file.

    Returns:
        Validated GeronimoConfig instance.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        ConfigurationError: If the config is invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ConfigurationError(f"Path is not a file: {path}")

    try:
        with open(path) as f:
            raw_config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML syntax: {e}")

    if raw_config is None:
        raise ConfigurationError("Configuration file is empty")

    # Interpolate environment variables
    config_data = interpolate_dict(raw_config)

    # Validate and parse with Pydantic
    try:
        return GeronimoConfig(**config_data)
    except ValidationError as e:
        errors = []
        for error in e.errors():
            location = ".".join(str(loc) for loc in error["loc"])
            errors.append(f"  {location}: {error['msg']}")
        raise ConfigurationError(
            f"Configuration validation failed:\n" + "\n".join(errors)
        )


def save_config(config: GeronimoConfig, config_path: str | Path) -> None:
    """Save a GeronimoConfig to a YAML file.

    Args:
        config: The configuration to save.
        config_path: Path to write the YAML file.
    """
    path = Path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict, excluding None values for cleaner output
    config_dict = config.model_dump(exclude_none=True, mode="json")

    with open(path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
