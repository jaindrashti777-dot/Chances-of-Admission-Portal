"""
Data preprocessing module.

Handles data quality checks, missing values, outliers, and dtype fixes.
All operations are logged and wrapped in exception handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__, log_file="logs/training.log")


def check_data_quality(df: pd.DataFrame) -> dict:
    """
    Perform comprehensive data quality assessment.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Quality report with shape, dtypes, nulls, duplicates, and stats.
    """
    logger.info("=" * 60)
    logger.info("DATA QUALITY CHECK")
    logger.info("=" * 60)

    report = {}

    try:
        # Shape
        report["shape"] = df.shape
        logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Data types
        report["dtypes"] = df.dtypes.to_dict()
        logger.info(f"Data types:\n{df.dtypes}")

        # Missing values
        null_counts = df.isnull().sum()
        null_pct = (null_counts / len(df) * 100).round(2)
        report["null_counts"] = null_counts[null_counts > 0].to_dict()
        report["null_percentage"] = null_pct[null_pct > 0].to_dict()

        if null_counts.sum() > 0:
            logger.warning(f"Missing values found:\n" f"{null_counts[null_counts > 0]}")
        else:
            logger.info("No missing values found")

        # Duplicates
        n_duplicates = df.duplicated().sum()
        report["n_duplicates"] = n_duplicates
        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate row(s)")
        else:
            logger.info("No duplicate rows found")

        # Basic statistics
        report["describe"] = df.describe()
        logger.info(f"Statistical summary:\n{df.describe()}")

    except Exception as e:
        logger.error(f"Data quality check failed: {e}")
        raise

    return report


def handle_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "mode",
) -> pd.DataFrame:
    """
    Handle missing values with appropriate imputation strategies.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    numerical_strategy : str
        Strategy for numerical columns: 'median', 'mean', or 'zero'.
    categorical_strategy : str
        Strategy for categorical columns: 'mode' or 'unknown'.

    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled.
    """
    df = df.copy()
    initial_nulls = df.isnull().sum().sum()

    if initial_nulls == 0:
        logger.info("No missing values to handle")
        return df

    logger.info(
        f"Handling {initial_nulls} missing values "
        f"(numerical={numerical_strategy}, categorical={categorical_strategy})"
    )

    try:
        # Numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                if numerical_strategy == "median":
                    fill_val = df[col].median()
                elif numerical_strategy == "mean":
                    fill_val = df[col].mean()
                else:
                    fill_val = 0
                df[col].fillna(fill_val, inplace=True)
                logger.info(
                    f"  {col}: filled {n_missing} nulls with "
                    f"{numerical_strategy}={fill_val:.2f}"
                )

        # Categorical columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            n_missing = df[col].isnull().sum()
            if n_missing > 0:
                if categorical_strategy == "mode":
                    fill_val = df[col].mode()[0]
                else:
                    fill_val = "Unknown"
                df[col].fillna(fill_val, inplace=True)
                logger.info(
                    f"  {col}: filled {n_missing} nulls with "
                    f"{categorical_strategy}='{fill_val}'"
                )

        remaining = df.isnull().sum().sum()
        logger.info(
            f"Missing value handling complete. "
            f"Before: {initial_nulls}, After: {remaining}"
        )

    except Exception as e:
        logger.error(f"Missing value handling failed: {e}")
        raise

    return df


def handle_outliers(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Handle outliers using IQR-based capping.

    Values below Q1 - factor*IQR are capped to that lower bound.
    Values above Q3 + factor*IQR are capped to that upper bound.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    columns : list[str], optional
        Columns to check. Defaults to all numerical columns.
    method : str
        Outlier detection method ('iqr').
    factor : float
        IQR multiplier (default: 1.5).

    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    logger.info(f"Handling outliers using {method} method (factor={factor})")
    total_capped = 0

    try:
        for col in columns:
            if col not in df.columns:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr

            n_below = (df[col] < lower).sum()
            n_above = (df[col] > upper).sum()
            n_capped = n_below + n_above

            if n_capped > 0:
                df[col] = df[col].clip(lower=lower, upper=upper)
                logger.info(
                    f"  {col}: capped {n_capped} outlier(s) "
                    f"(range: [{lower:.2f}, {upper:.2f}])"
                )
                total_capped += n_capped

        logger.info(f"Outlier handling complete. Total values capped: {total_capped}")

    except Exception as e:
        logger.error(f"Outlier handling failed: {e}")
        raise

    return df


def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types for known columns.

    Ensures integer columns are ints and categorical columns are strings.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected data types.
    """
    df = df.copy()
    logger.info("Fixing data types")

    try:
        # Integer columns that might have been read as float
        int_columns = [
            "Gap_Year",
            "Backlogs",
            "Research_Paper",
            "Internship",
            "Extracurricular",
        ]
        for col in int_columns:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Ensure categorical columns are strings
        cat_columns = ["Category", "State", "Gender", "Desired_Branch", "College_Tier"]
        for col in cat_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)

        logger.info("Data type fixing complete")

    except Exception as e:
        logger.error(f"Dtype fixing failed: {e}")
        raise

    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline.

    Steps: quality check → missing values → dtypes → outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    logger.info("Starting full preprocessing pipeline")

    try:
        check_data_quality(df)
        df = handle_missing_values(df)
        df = fix_dtypes(df)

        # Only apply outlier capping to academic score columns, not binary/count cols
        outlier_cols = [
            "Tenth_Percentage",
            "Twelfth_Percentage",
            "JEE_Percentile",
            "CUET_Score",
            "CGPA",
            "Family_Income",
        ]
        df = handle_outliers(df, columns=outlier_cols)

        # Remove duplicates
        n_before = len(df)
        df = df.drop_duplicates()
        n_removed = n_before - len(df)
        if n_removed > 0:
            logger.info(f"Removed {n_removed} duplicate row(s)")

        logger.info(
            f"Preprocessing complete. Final shape: {df.shape[0]} × {df.shape[1]}"
        )

    except Exception as e:
        logger.error(f"Preprocessing pipeline failed: {e}")
        raise

    return df
