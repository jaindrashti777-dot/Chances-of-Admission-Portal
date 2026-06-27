"""
Unit tests for data validation layer.
"""

import pandas as pd
import pytest

from src.validator import validate_dataframe, validate_single_input


@pytest.fixture
def valid_input():
    return {
        "Tenth_Percentage": 85.0,
        "Twelfth_Percentage": 88.0,
        "JEE_Percentile": 90.0,
        "CUET_Score": 600.0,
        "Category": "General",
        "State": "Maharashtra",
        "Family_Income": 8.0,
        "Gender": "Male",
        "Gap_Year": 0,
        "CGPA": 8.5,
        "Backlogs": 0,
        "Extracurricular": 1,
        "Research_Paper": 1,
        "Internship": 2,
        "Desired_Branch": "CSE",
        "College_Tier": "Tier_1",
    }


def test_validate_single_input_valid(valid_input):
    is_valid, errors = validate_single_input(valid_input)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_single_input_missing_fields(valid_input):
    del valid_input["CGPA"]
    is_valid, errors = validate_single_input(valid_input)
    assert is_valid is False
    assert any("Missing required fields" in e for e in errors)


def test_validate_single_input_out_of_range(valid_input):
    valid_input["CGPA"] = 11.0
    is_valid, errors = validate_single_input(valid_input)
    assert is_valid is False
    assert any("CGPA: value 11.0 out of range" in e for e in errors)


def test_validate_single_input_invalid_category(valid_input):
    valid_input["Category"] = "InvalidCategory"
    is_valid, errors = validate_single_input(valid_input)
    assert is_valid is False
    assert any("Category: invalid value" in e for e in errors)


def test_validate_dataframe_valid(valid_input):
    df = pd.DataFrame([valid_input] * 5)
    is_valid, errors = validate_dataframe(df)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_dataframe_missing_column(valid_input):
    df = pd.DataFrame([valid_input] * 5)
    df = df.drop(columns=["CGPA"])
    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("Missing required columns" in e for e in errors)


def test_validate_dataframe_out_of_range(valid_input):
    df = pd.DataFrame([valid_input] * 5)
    df.loc[0, "CGPA"] = 15.0
    is_valid, errors = validate_dataframe(df)
    assert is_valid is False
    assert any("CGPA:" in e and "above maximum" in e for e in errors)
