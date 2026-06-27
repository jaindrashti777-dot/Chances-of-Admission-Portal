"""
Feature engineering module.

Handles encoding, train/val/test splitting, and feature transformation.
Follows the ML best practice: split BEFORE fitting any transformers.
"""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE
from src.logger import get_logger

logger = get_logger(__name__, log_file="logs/training.log")


def split_data(
    df: pd.DataFrame,
    target_col: str = "Admission_Chance",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.

    IMPORTANT: This must be called BEFORE any encoding or scaling
    to prevent data leakage.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    target_col : str
        Name of the target column.
    test_size : float
        Fraction for test set.
    val_size : float
        Fraction for validation set (from remaining after test split).
    random_state : int, optional
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    seed = random_state or RANDOM_STATE
    logger.info(
        f"Splitting data: test_size={test_size}, val_size={val_size}, "
        f"random_state={seed}"
    )

    try:
        if target_col not in df.columns:
            raise KeyError(f"Target column '{target_col}' not found in DataFrame")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # First split: train+val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=seed,
        )

        # Second split: train vs val (from train+val)
        val_fraction = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val,
            y_train_val,
            test_size=val_fraction,
            random_state=seed,
        )

        logger.info(
            f"Split complete — "
            f"Train: {len(X_train)} ({len(X_train)/len(df)*100:.1f}%), "
            f"Val: {len(X_val)} ({len(X_val)/len(df)*100:.1f}%), "
            f"Test: {len(X_test)} ({len(X_test)/len(df)*100:.1f}%)"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    except Exception as e:
        logger.error(f"Data splitting failed: {e}")
        raise


def get_feature_lists() -> dict[str, list[str]]:
    """
    Return categorized feature lists for use with ColumnTransformer.

    Returns
    -------
    dict
        Keys: 'numerical', 'categorical', 'ordinal', 'binary'.
    """
    return {
        "numerical": [
            "Tenth_Percentage",
            "Twelfth_Percentage",
            "JEE_Percentile",
            "CUET_Score",
            "Family_Income",
            "CGPA",
            "Gap_Year",
            "Backlogs",
            "Research_Paper",
            "Internship",
        ],
        "categorical": [
            "Category",
            "State",
            "Gender",
            "Desired_Branch",
        ],
        "ordinal": [
            "College_Tier",
        ],
        "binary": [
            "Extracurricular",
        ],
    }
