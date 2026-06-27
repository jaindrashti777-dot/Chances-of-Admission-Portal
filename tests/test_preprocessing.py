"""
Unit tests for data preprocessing module.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    fix_dtypes,
    handle_missing_values,
    handle_outliers,
    preprocess_data,
)


@pytest.fixture
def raw_data():
    return pd.DataFrame({
        "Tenth_Percentage": [80, 90, 85, np.nan, 200],  # One NaN, one outlier
        "Category": ["General", "OBC", "General", np.nan, "SC"],
        "Gap_Year": [0, 1.0, 0, 0, 0],
    })


def test_handle_missing_values(raw_data):
    df_clean = handle_missing_values(raw_data, numerical_strategy="median", categorical_strategy="mode")
    assert df_clean["Tenth_Percentage"].isnull().sum() == 0
    assert df_clean["Category"].isnull().sum() == 0
    # Median of [80, 90, 85, 200] is 87.5
    assert df_clean.loc[3, "Tenth_Percentage"] == 87.5
    # Mode of ["General", "OBC", "General", "SC"] is "General"
    assert df_clean.loc[3, "Category"] == "General"


def test_handle_outliers():
    df = pd.DataFrame({"A": [10, 12, 11, 13, 10, 12, 11, 100, -50]})
    df_clean = handle_outliers(df, columns=["A"], factor=1.5)
    assert df_clean["A"].max() < 100
    assert df_clean["A"].min() > -50


def test_fix_dtypes(raw_data):
    df_clean = fix_dtypes(raw_data)
    assert df_clean["Gap_Year"].dtype == int
    assert df_clean["Category"].dtype == object


def test_preprocess_data_pipeline(raw_data):
    # Just checking it runs end-to-end without failing
    # In real world, we'd mock the check_data_quality or test full DataFrame
    df_clean = preprocess_data(raw_data)
    assert df_clean.isnull().sum().sum() == 0
    assert df_clean["Gap_Year"].dtype == int
