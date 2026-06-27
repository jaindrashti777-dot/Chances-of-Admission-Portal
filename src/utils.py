"""
Utility helpers for the Admission Prediction project.

Provides seed setting, model save/load, and figure saving helpers.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


def save_model(model: Any, path: str | Path) -> None:
    """
    Save a model (or any Python object) to disk using joblib.

    Parameters
    ----------
    model : Any
        The object to persist (model, pipeline, scaler, etc.).
    path : str or Path
        Destination file path.
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save model to {path}: {e}")
        raise


def load_pickle(path: str | Path) -> Any:
    """
    Load a joblib-serialized object from disk.

    Parameters
    ----------
    path : str or Path
        Path to the .pkl file.

    Returns
    -------
    Any
        The deserialized object.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        msg = f"File not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        obj = joblib.load(path)
        logger.info(f"Loaded object from {path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        raise


def save_figure(fig, path: str | Path, dpi: int = 150) -> None:
    """
    Save a matplotlib figure to disk.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Destination file path (e.g., ``reports/figures/plot.png``).
    dpi : int
        Resolution in dots per inch (default: 150).
    """
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        logger.info(f"Figure saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save figure to {path}: {e}")
        raise


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
