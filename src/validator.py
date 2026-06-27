"""
Data validation layer for the Admission Prediction project.

Validates both DataFrames (batch) and single input dicts (API/Streamlit)
before any data enters the preprocessing pipeline. All violations are
logged and collected into a structured error list.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Validation rules
# ==============================================================================

NUMERICAL_RULES: dict[str, dict[str, Any]] = {
    "Tenth_Percentage": {"min": 0, "max": 100, "type": (int, float)},
    "Twelfth_Percentage": {"min": 0, "max": 100, "type": (int, float)},
    "JEE_Percentile": {"min": 0, "max": 100, "type": (int, float)},
    "CUET_Score": {"min": 0, "max": 800, "type": (int, float)},
    "Family_Income": {"min": 0, "max": 200, "type": (int, float)},
    "CGPA": {"min": 0, "max": 10, "type": (int, float)},
    "Gap_Year": {"min": 0, "max": 5, "type": (int, float, np.integer)},
    "Backlogs": {"min": 0, "max": 50, "type": (int, float, np.integer)},
    "Research_Paper": {"min": 0, "max": 20, "type": (int, float, np.integer)},
    "Internship": {"min": 0, "max": 10, "type": (int, float, np.integer)},
    "Extracurricular": {"min": 0, "max": 1, "type": (int, float, np.integer)},
}

VALID_CATEGORIES = {"General", "OBC", "SC", "ST", "EWS"}
VALID_GENDERS = {"Male", "Female"}
VALID_TIERS = {"Tier_1", "Tier_2", "Tier_3"}
VALID_BRANCHES = {"CSE", "ECE", "ME", "CE", "EE", "IT", "Chemical", "Biotech"}
VALID_STATES = {
    "Maharashtra",
    "Delhi",
    "Karnataka",
    "Tamil_Nadu",
    "Uttar_Pradesh",
    "West_Bengal",
    "Rajasthan",
    "Gujarat",
    "Telangana",
    "Kerala",
}

CATEGORICAL_RULES: dict[str, set[str]] = {
    "Category": VALID_CATEGORIES,
    "Gender": VALID_GENDERS,
    "College_Tier": VALID_TIERS,
    "Desired_Branch": VALID_BRANCHES,
    "State": VALID_STATES,
}

REQUIRED_COLUMNS = list(NUMERICAL_RULES.keys()) + list(CATEGORICAL_RULES.keys())


# ==============================================================================
# Custom exception
# ==============================================================================


class ValidationError(Exception):
    """Raised when data fails validation checks."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        msg = f"{len(errors)} validation error(s): " + "; ".join(errors[:5])
        if len(errors) > 5:
            msg += f" ... and {len(errors) - 5} more"
        super().__init__(msg)


# ==============================================================================
# DataFrame validation (batch)
# ==============================================================================


def validate_required_columns(
    df: pd.DataFrame,
    expected_columns: list[str] | None = None,
) -> list[str]:
    """Check that all required columns are present in the DataFrame."""
    errors: list[str] = []
    expected = expected_columns or REQUIRED_COLUMNS
    missing = set(expected) - set(df.columns)
    if missing:
        msg = f"Missing required columns: {sorted(missing)}"
        logger.error(msg)
        errors.append(msg)
    return errors


def validate_numerical_ranges(df: pd.DataFrame) -> list[str]:
    """Validate that numerical columns fall within allowed ranges."""
    errors: list[str] = []
    for col, rules in NUMERICAL_RULES.items():
        if col not in df.columns:
            continue
        below = df[col] < rules["min"]
        above = df[col] > rules["max"]
        n_below = below.sum()
        n_above = above.sum()
        if n_below > 0:
            msg = f"{col}: {n_below} values below minimum ({rules['min']})"
            logger.warning(msg)
            errors.append(msg)
        if n_above > 0:
            msg = f"{col}: {n_above} values above maximum ({rules['max']})"
            logger.warning(msg)
            errors.append(msg)
    return errors


def validate_categorical_values(df: pd.DataFrame) -> list[str]:
    """Validate that categorical columns contain only allowed values."""
    errors: list[str] = []
    for col, valid_values in CATEGORICAL_RULES.items():
        if col not in df.columns:
            continue
        invalid = set(df[col].dropna().unique()) - valid_values
        if invalid:
            msg = f"{col}: invalid values found: {sorted(invalid)}"
            logger.warning(msg)
            errors.append(msg)
    return errors


def validate_dataframe(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Run all validation checks on a DataFrame.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    logger.info(f"Validating DataFrame with shape {df.shape}")
    errors: list[str] = []

    try:
        errors.extend(validate_required_columns(df))
        errors.extend(validate_numerical_ranges(df))
        errors.extend(validate_categorical_values(df))

        # Check for fully empty rows
        empty_rows = df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            msg = f"Found {empty_rows} completely empty row(s)"
            logger.warning(msg)
            errors.append(msg)

    except Exception as e:
        msg = f"Validation failed with unexpected error: {e}"
        logger.error(msg)
        errors.append(msg)

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("DataFrame validation passed — no issues found")
    else:
        logger.warning(f"DataFrame validation found {len(errors)} issue(s)")

    return is_valid, errors


# ==============================================================================
# Single-input validation (API / Streamlit)
# ==============================================================================


def validate_single_input(input_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate a single prediction input (dict of feature values).

    Used by the FastAPI endpoint and Streamlit app before prediction.

    Returns
    -------
    tuple[bool, list[str]]
        (is_valid, list_of_error_messages)
    """
    errors: list[str] = []

    # Check required fields
    missing = set(REQUIRED_COLUMNS) - set(input_dict.keys())
    if missing:
        errors.append(f"Missing required fields: {sorted(missing)}")

    # Validate numerical ranges
    for col, rules in NUMERICAL_RULES.items():
        if col not in input_dict:
            continue
        val = input_dict[col]
        if not isinstance(val, rules["type"]):
            errors.append(f"{col}: expected number, got {type(val).__name__}")
            continue
        if val < rules["min"] or val > rules["max"]:
            errors.append(
                f"{col}: value {val} out of range " f"[{rules['min']}, {rules['max']}]"
            )

    # Validate categorical values
    for col, valid_values in CATEGORICAL_RULES.items():
        if col not in input_dict:
            continue
        val = input_dict[col]
        if val not in valid_values:
            errors.append(
                f"{col}: invalid value '{val}'. "
                f"Must be one of {sorted(valid_values)}"
            )

    is_valid = len(errors) == 0
    if is_valid:
        logger.info("Single input validation passed")
    else:
        logger.warning(f"Single input validation failed: {errors}")

    return is_valid, errors
