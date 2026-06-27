"""
Configuration loader for the Admission Prediction project.

Reads ``configs/config.yaml`` and exposes all settings as a nested dict.
Every other module imports paths, feature lists, and hyperparameters from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.logger import get_logger

logger = get_logger(__name__)

# Project root — two levels up from src/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str or Path, optional
        Override path to config.yaml. Defaults to ``configs/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    """
    path = Path(config_path) if config_path else CONFIG_PATH

    if not path.exists():
        msg = f"Configuration file not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {path}")
        return config
    except yaml.YAMLError as e:
        msg = f"Failed to parse config file: {e}"
        logger.error(msg)
        raise ValueError(msg) from e


def get_data_config(config: dict | None = None) -> dict[str, Any]:
    """Return the ``data`` section of the config."""
    cfg = config or load_config()
    return cfg.get("data", {})


def get_feature_config(config: dict | None = None) -> dict[str, Any]:
    """Return the ``features`` section of the config."""
    cfg = config or load_config()
    return cfg.get("features", {})


def get_model_config(config: dict | None = None) -> dict[str, Any]:
    """Return the ``models`` section (hyperparameter grids)."""
    cfg = config or load_config()
    return cfg.get("models", {})


def get_report_paths(config: dict | None = None) -> dict[str, str]:
    """Return the ``reports`` section of the config."""
    cfg = config or load_config()
    return cfg.get("reports", {})


def get_model_paths(config: dict | None = None) -> dict[str, str]:
    """Return the ``model_paths`` section of the config."""
    cfg = config or load_config()
    return cfg.get("model_paths", {})


def resolve_path(relative_path: str) -> Path:
    """Resolve a project-relative path to an absolute path."""
    return PROJECT_ROOT / relative_path


# Pre-load config on import for convenience
try:
    CONFIG = load_config()
except FileNotFoundError:
    logger.warning("config.yaml not found — using empty config")
    CONFIG = {}

# Convenience constants
RANDOM_STATE = CONFIG.get("project", {}).get("random_state", 42)
