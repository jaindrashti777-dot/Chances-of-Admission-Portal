"""
End-to-end smoke test for the admission prediction workflow.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression

from api.main import app
from data.raw.generate_admission_data import generate_dataset
from src.feature_engineering import split_data
from src.pipeline_builder import build_full_pipeline, build_preprocessor
from src.predict import load_pipeline, predict_admission
from src.preprocessing import preprocess_data
from src.utils import save_model


def test_end_to_end_prediction_smoke(tmp_path):
    df = generate_dataset(n_samples=50, seed=42)
    df = preprocess_data(df)
    X_train, _X_val, _X_test, y_train, _y_val, _y_test = split_data(df)

    pipeline = build_full_pipeline(LinearRegression(), build_preprocessor())
    pipeline.fit(X_train, y_train)

    model_path = tmp_path / "pipeline.pkl"
    save_model(pipeline, model_path)
    loaded_pipeline = load_pipeline(str(model_path))

    sample = X_train.iloc[0].to_dict()
    result = predict_admission(sample, pipeline=loaded_pipeline)

    assert isinstance(result["admission_chance"], float)
    assert 0.0 <= result["admission_chance"] <= 1.0
    assert result["confidence_level"] in {"High", "Medium", "Low"}

    client = TestClient(app)
    with patch("api.main.get_pipeline", return_value=loaded_pipeline):
        response = client.post("/predict", json=sample)

    assert response.status_code == 200
    payload = response.json()
    assert 0.0 <= payload["admission_chance"] <= 1.0
    assert payload["confidence_level"] in {"High", "Medium", "Low"}
    assert "timestamp" in payload
