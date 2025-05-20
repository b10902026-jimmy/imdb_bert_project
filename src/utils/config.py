import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Must be before any HF imports

"""
Configuration utilities.

This module contains functions for loading and managing configuration files.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dictionary containing the configuration.
    """
    logger.info(f"Loading configuration from {config_path}...")
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def get_project_root() -> Path:
    """Get the project root directory.

    Returns:
        Path to the project root directory.
    """
    # Assuming this file is in src/utils/config.py
    return Path(__file__).parent.parent.parent


def get_config_path(config_name: str) -> Path:
    """Get the path to a configuration file.

    Args:
        config_name: Name of the configuration file.

    Returns:
        Path to the configuration file.
    """
    project_root = get_project_root()
    return project_root / "configs" / config_name


def load_env_vars() -> None:
    """Load environment variables from .env file."""
    project_root = get_project_root()
    env_path = project_root / ".env"
    
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}...")
        load_dotenv(env_path)
        logger.info("Environment variables loaded")
    else:
        logger.warning(f".env file not found at {env_path}")


def get_env_var(name: str, default: Optional[str] = None) -> Optional[str]:
    """Get an environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value to return if the environment variable is not set.

    Returns:
        Value of the environment variable, or the default value if not set.
    """
    value = os.environ.get(name, default)
    
    if value is None:
        logger.warning(f"Environment variable {name} not set")
    
    return value


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple configuration dictionaries.

    Args:
        *configs: Configuration dictionaries to merge.

    Returns:
        Merged configuration dictionary.
    """
    merged_config = {}
    
    for config in configs:
        for key, value in config.items():
            if (
                key in merged_config
                and isinstance(merged_config[key], dict)
                and isinstance(value, dict)
            ):
                merged_config[key] = merge_configs(merged_config[key], value)
            else:
                merged_config[key] = value
    
    return merged_config


def load_all_configs() -> Dict[str, Any]:
    """Load all configuration files and merge them.

    Returns:
        Merged configuration dictionary.
    """
    project_root = get_project_root()
    configs_dir = project_root / "configs"
    
    configs = []
    
    for config_file in configs_dir.glob("*.yaml"):
        config = load_config(str(config_file))
        configs.append(config)
    
    return merge_configs(*configs)
