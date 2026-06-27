"""
Unit tests for model training module.
"""

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from src.pipeline_builder import build_full_pipeline, build_preprocessor
from src.train import get_models


@pytest.fixture
def mock_data():
    X_train = pd.DataFrame(
        {
            "Tenth_Percentage": [80, 90, 85, 75],
            "Twelfth_Percentage": [85, 92, 88, 70],
            "JEE_Percentile": [90, 95, 80, 60],
            "CUET_Score": [600, 700, 500, 400],
            "Category": ["General", "OBC", "SC", "ST"],
            "State": ["Maharashtra", "Delhi", "Karnataka", "Tamil_Nadu"],
            "Family_Income": [8, 12, 5, 2],
            "Gender": ["Male", "Female", "Male", "Female"],
            "Gap_Year": [0, 0, 1, 2],
            "CGPA": [8.5, 9.2, 7.5, 6.0],
            "Backlogs": [0, 0, 1, 3],
            "Extracurricular": [1, 1, 0, 0],
            "Research_Paper": [1, 2, 0, 0],
            "Internship": [1, 2, 0, 0],
            "Desired_Branch": ["CSE", "ECE", "ME", "CE"],
            "College_Tier": ["Tier_1", "Tier_1", "Tier_2", "Tier_3"],
        }
    )
    y_train = pd.Series([0.85, 0.95, 0.60, 0.30])
    return X_train, y_train


def test_get_models():
    models = get_models()
    assert "Linear_Regression" in models
    assert "Random_Forest" in models
    assert "XGBoost" in models


def test_pipeline_fit_predict(mock_data):
    X_train, y_train = mock_data
    preprocessor = build_preprocessor()
    pipeline = build_full_pipeline(LinearRegression(), preprocessor)

    # Should fit without error
    pipeline.fit(X_train, y_train)

    # Should predict without error
    preds = pipeline.predict(X_train)
    assert len(preds) == len(y_train)
