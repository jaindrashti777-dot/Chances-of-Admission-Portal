"""
Unit tests for feature engineering module.
"""

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.feature_engineering import split_data
from src.pipeline_builder import build_preprocessor


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        "Tenth_Percentage": [80] * 100,
        "Twelfth_Percentage": [85] * 100,
        "JEE_Percentile": [90] * 100,
        "CUET_Score": [600] * 100,
        "Category": ["General"] * 100,
        "State": ["Maharashtra"] * 100,
        "Family_Income": [8] * 100,
        "Gender": ["Male"] * 100,
        "Gap_Year": [0] * 100,
        "CGPA": [8.5] * 100,
        "Backlogs": [0] * 100,
        "Extracurricular": [1] * 100,
        "Research_Paper": [1] * 100,
        "Internship": [1] * 100,
        "Desired_Branch": ["CSE"] * 100,
        "College_Tier": ["Tier_1"] * 100,
        "Admission_Chance": [0.85] * 100,
    })


def test_split_data(sample_data):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        sample_data, target_col="Admission_Chance", test_size=0.15, val_size=0.15
    )

    # Total 100 rows
    assert len(X_test) == 15
    assert len(X_val) in (15, 16)
    assert len(X_train) in (69, 70)

    assert "Admission_Chance" not in X_train.columns
    assert len(y_train) in (69, 70)


def test_build_preprocessor():
    preprocessor = build_preprocessor()
    assert preprocessor is not None
    # Check if transformers are set up
    transformers = [t[0] for t in preprocessor.transformers]
    assert "num" in transformers
    assert "cat" in transformers
    assert "ord" in transformers
    assert "bin" in transformers
