"""
Data loading and saving utilities.

Handles reading raw/processed CSV files with proper exception handling
and logging. Never silently fails — all errors are logged.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import resolve_path
from src.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the raw admission dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Override path. Defaults to ``data/raw/admission_data.csv``.

    Returns
    -------
    pd.DataFrame
        The raw dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    csv_path = resolve_path(path or "data/raw/admission_data.csv")

    if not csv_path.exists():
        msg = (
            f"Raw dataset not found at {csv_path}. "
            "Run 'make generate' or 'python data/raw/generate_admission_data.py' "
            "to create it."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(csv_path)
        logger.info(
            f"Loaded raw data: {df.shape[0]} rows × {df.shape[1]} columns "
            f"from {csv_path}"
        )
        return df
    except pd.errors.ParserError as e:
        msg = f"Failed to parse CSV at {csv_path}: {e}"
        logger.error(msg)
        raise ValueError(msg) from e
    except Exception as e:
        logger.error(f"Unexpected error loading {csv_path}: {e}")
        raise


def load_processed_data(path: str | None = None) -> pd.DataFrame:
    """
    Load the processed (cleaned) dataset from CSV.

    Parameters
    ----------
    path : str, optional
        Override path. Defaults to ``data/processed/admission_cleaned.csv``.

    Returns
    -------
    pd.DataFrame
        The processed dataset.
    """
    csv_path = resolve_path(path or "data/processed/admission_cleaned.csv")

    if not csv_path.exists():
        msg = (
            f"Processed dataset not found at {csv_path}. "
            "Run the preprocessing pipeline first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(csv_path)
        logger.info(
            f"Loaded processed data: {df.shape[0]} rows × {df.shape[1]} columns"
        )
        return df
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        raise


def save_processed_data(
    df: pd.DataFrame,
    path: str | None = None,
) -> None:
    """
    Save a processed DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    path : str, optional
        Override path. Defaults to ``data/processed/admission_cleaned.csv``.
    """
    csv_path = resolve_path(path or "data/processed/admission_cleaned.csv")

    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(
            f"Saved processed data: {df.shape[0]} rows × {df.shape[1]} columns "
            f"to {csv_path}"
        )
    except Exception as e:
        logger.error(f"Failed to save processed data to {csv_path}: {e}")
        raise
