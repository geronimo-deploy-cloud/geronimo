"""User configuration management for Geronimo.

Provides persistent global settings stored in ~/.geronimo/config.yaml.
Settings apply as defaults for all projects.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


# Default config location
USER_CONFIG_DIR = Path.home() / ".geronimo"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"


@dataclass
class ArtifactConfig:
    """Configuration for ArtifactStore defaults."""
    
    backend: Literal["local", "s3", "cloud"] = "local"
    s3_bucket: Optional[str] = None
    base_path: str = "~/.geronimo/artifacts"


@dataclass
class DefaultsConfig:
    """Default project initialization settings."""
    
    framework: str = "sklearn"
    template: str = "realtime"


@dataclass
class UserConfig:
    """Global user configuration.
    
    Stored in ~/.geronimo/config.yaml.
    """
    
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    defaults: DefaultsConfig = field(default_factory=DefaultsConfig)


def load_user_config() -> UserConfig:
    """Load global config from ~/.geronimo/config.yaml.
    
    Returns default config if file doesn't exist.
    
    Returns:
        UserConfig with loaded or default settings.
    """
    if not USER_CONFIG_FILE.exists():
        return UserConfig()
    
    try:
        with open(USER_CONFIG_FILE) as f:
            data = yaml.safe_load(f) or {}
        
        # Parse artifacts section
        artifacts_data = data.get("artifacts", {})
        artifacts = ArtifactConfig(
            backend=artifacts_data.get("backend", "local"),
            s3_bucket=artifacts_data.get("s3_bucket"),
            base_path=artifacts_data.get("base_path", "~/.geronimo/artifacts"),
        )
        
        # Parse defaults section
        defaults_data = data.get("defaults", {})
        defaults = DefaultsConfig(
            framework=defaults_data.get("framework", "sklearn"),
            template=defaults_data.get("template", "realtime"),
        )
        
        return UserConfig(artifacts=artifacts, defaults=defaults)
    except Exception:
        # Return defaults on any parsing error
        return UserConfig()


def save_user_config(config: UserConfig) -> None:
    """Save global config to ~/.geronimo/config.yaml.
    
    Creates ~/.geronimo directory if it doesn't exist.
    
    Args:
        config: UserConfig to save.
    """
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    data = {
        "artifacts": {
            "backend": config.artifacts.backend,
            "s3_bucket": config.artifacts.s3_bucket,
            "base_path": config.artifacts.base_path,
        },
        "defaults": {
            "framework": config.defaults.framework,
            "template": config.defaults.template,
        },
    }
    
    # Remove None values for cleaner YAML
    data["artifacts"] = {k: v for k, v in data["artifacts"].items() if v is not None}
    
    with open(USER_CONFIG_FILE, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def get_config_value(key: str) -> Optional[str]:
    """Get a specific config value by dot-notation key.
    
    Args:
        key: Dot-notation key like "artifacts.backend"
        
    Returns:
        Value as string, or None if not found.
    """
    config = load_user_config()
    
    parts = key.split(".")
    if len(parts) != 2:
        return None
    
    section, field = parts
    
    if section == "artifacts":
        return getattr(config.artifacts, field, None)
    elif section == "defaults":
        return getattr(config.defaults, field, None)
    
    return None


def set_config_value(key: str, value: str) -> bool:
    """Set a specific config value by dot-notation key.
    
    Args:
        key: Dot-notation key like "artifacts.backend"
        value: Value to set
        
    Returns:
        True if successful, False if invalid key.
    """
    config = load_user_config()
    
    parts = key.split(".")
    if len(parts) != 2:
        return False
    
    section, field_name = parts
    
    if section == "artifacts":
        if field_name == "backend":
            if value not in ("local", "s3", "cloud"):
                return False
            config.artifacts.backend = value
        elif field_name == "s3_bucket":
            config.artifacts.s3_bucket = value
        elif field_name == "base_path":
            config.artifacts.base_path = value
        else:
            return False
    elif section == "defaults":
        if field_name == "framework":
            config.defaults.framework = value
        elif field_name == "template":
            config.defaults.template = value
        else:
            return False
    else:
        return False
    
    save_user_config(config)
    return True


def reset_user_config() -> None:
    """Reset config to defaults by deleting the config file."""
    if USER_CONFIG_FILE.exists():
        USER_CONFIG_FILE.unlink()
