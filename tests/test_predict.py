"""
Unit tests for the prediction pipeline.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.predict import batch_predict, predict_admission


@pytest.fixture
def mock_pipeline():
    pipeline = MagicMock()
    pipeline.predict.return_value = [0.84]
    return pipeline


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


def test_predict_admission(mock_pipeline, valid_input):
    with patch("src.predict._log_prediction") as mock_log:
        result = predict_admission(valid_input, pipeline=mock_pipeline)

        assert result["admission_chance"] == 0.84
        assert result["confidence_level"] == "High"
        assert "timestamp" in result
        assert result["input"] == valid_input

        # Check that it tried to log the prediction
        mock_log.assert_called_once()


def test_predict_admission_invalid_input(mock_pipeline, valid_input):
    valid_input["CGPA"] = 15.0  # Invalid CGPA
    with pytest.raises(ValueError, match="Invalid input"):
        predict_admission(valid_input, pipeline=mock_pipeline)


def test_batch_predict(mock_pipeline, valid_input):
    inputs = [valid_input, valid_input]
    with patch("src.predict._log_prediction"):
        results = batch_predict(inputs, pipeline=mock_pipeline)

        assert len(results) == 2
        assert results[0]["admission_chance"] == 0.84
        assert results[1]["admission_chance"] == 0.84
