"""
Unit tests for the FastAPI application.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert "model_available" in data


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "predict" in data
    assert "docs" in data
    assert "scope" in data


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_predictions_served" in data
    assert "status" in data


@pytest.fixture
def valid_payload():
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


def test_predict_endpoint_success(valid_payload):
    # We mock the get_pipeline function to return a dummy pipeline
    dummy_pipeline = MagicMock()
    with patch("api.main.get_pipeline", return_value=dummy_pipeline):
        with patch("src.predict.predict_admission") as mock_predict:
            mock_predict.return_value = {
                "admission_chance": 0.85,
                "confidence_level": "High",
                "input": valid_payload,
                "timestamp": "2023-01-01T12:00:00",
            }
            # We mock the explain module since it's optional
            with patch(
                "src.explain.explain_prediction", side_effect=Exception("No SHAP")
            ):
                response = client.post("/predict", json=valid_payload)

            assert response.status_code == 200
            data = response.json()
            assert data["admission_chance"] == 0.85
            assert data["confidence_level"] == "High"
            assert data["top_factors"] == []  # Since we mocked SHAP exception


def test_predict_endpoint_validation_error_invalid_value(valid_payload):
    # Pass invalid CGPA
    valid_payload["CGPA"] = 15.0
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"
    assert any(err["field"] == "CGPA" for err in data["details"])


def test_predict_endpoint_validation_error_missing_field(valid_payload):
    # Remove a required field
    del valid_payload["CGPA"]
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422
    data = response.json()
    assert data["error"] == "validation_error"
    assert any(err["field"] == "CGPA" for err in data["details"])


def test_predict_endpoint_validation_error_invalid_category(valid_payload):
    # Pass an invalid category
    valid_payload["Category"] = "Unknown"
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 422


def test_model_artifact_is_available():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_available"] is True


def test_chat_endpoint_success():
    response = client.post(
        "/api/chat", json={"message": "How do backlogs affect this?"}
    )
    assert response.status_code == 200
    assert "response" in response.json()
