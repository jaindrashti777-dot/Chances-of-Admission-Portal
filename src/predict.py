"""
Prediction pipeline.

Loads the saved pipeline, validates inputs, makes predictions, and
logs every prediction to a CSV history file.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import resolve_path, load_config
from src.logger import get_prediction_logger
from src.utils import load_pickle
from src.validator import validate_single_input

logger = get_prediction_logger()


def load_pipeline(path: str | None = None):
    """
    Load the saved ML pipeline.

    Parameters
    ----------
    path : str, optional
        Override path. Defaults to ``models/pipeline.pkl``.

    Returns
    -------
    Pipeline
        Fitted sklearn Pipeline.
    """
    config = load_config()
    pipeline_path = resolve_path(
        path or config.get("model_paths", {}).get("pipeline", "models/pipeline.pkl")
    )

    logger.info(f"Loading pipeline from {pipeline_path}")
    return load_pickle(pipeline_path)


def predict_admission(
    input_data: dict[str, Any],
    pipeline=None,
) -> dict[str, Any]:
    """
    Predict admission chance for a single student.

    Steps: validate → create DataFrame → predict → log → return

    Parameters
    ----------
    input_data : dict
        Feature values (all 16 features).
    pipeline : Pipeline, optional
        Pre-loaded pipeline. Loads from disk if not provided.

    Returns
    -------
    dict
        {
            "admission_chance": float,
            "confidence_level": str,  # "High" / "Medium" / "Low"
            "input": dict,
            "timestamp": str,
        }
    """
    logger.info("New prediction request received")

    try:
        # 1. Validate input
        is_valid, errors = validate_single_input(input_data)
        if not is_valid:
            logger.error(f"Input validation failed: {errors}")
            raise ValueError(f"Invalid input: {errors}")

        # 2. Load pipeline if needed
        if pipeline is None:
            pipeline = load_pipeline()

        # 3. Create DataFrame (pipeline expects DataFrame input)
        input_df = pd.DataFrame([input_data])

        # 4. Predict
        prediction = pipeline.predict(input_df)[0]
        prediction = round(float(prediction), 4)

        # Clip to valid range
        prediction = max(0.0, min(1.0, prediction))

        # 5. Determine confidence level
        if prediction >= 0.70:
            confidence = "High"
        elif prediction >= 0.40:
            confidence = "Medium"
        else:
            confidence = "Low"

        result = {
            "admission_chance": prediction,
            "confidence_level": confidence,
            "input": input_data,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        logger.info(
            f"Prediction: {prediction:.4f} ({confidence}) "
            f"for CGPA={input_data.get('CGPA')}, "
            f"JEE={input_data.get('JEE_Percentile')}"
        )

        # 6. Log to prediction history
        _log_prediction(input_data, prediction, confidence)

        return result

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def _log_prediction(
    input_data: dict[str, Any],
    prediction: float,
    confidence: str,
) -> None:
    """Append the prediction to the CSV history file."""
    try:
        config = load_config()
        log_path = resolve_path(
            config.get("data", {}).get("predictions_log", "reports/predictions.csv")
        )

        # Build row
        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            **input_data,
            "prediction": prediction,
            "confidence_level": confidence,
        }

        row_df = pd.DataFrame([row])

        # Append (create if doesn't exist)
        if log_path.exists():
            row_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            row_df.to_csv(log_path, index=False)

        logger.info(f"Prediction logged to {log_path}")

    except Exception as e:
        # Don't fail the prediction if logging fails
        logger.warning(f"Failed to log prediction: {e}")


def batch_predict(
    inputs: list[dict[str, Any]],
    pipeline=None,
) -> list[dict[str, Any]]:
    """
    Make predictions for multiple inputs.

    Parameters
    ----------
    inputs : list[dict]
        List of feature dicts.
    pipeline : Pipeline, optional
        Pre-loaded pipeline.

    Returns
    -------
    list[dict]
        List of prediction results.
    """
    if pipeline is None:
        pipeline = load_pipeline()

    results = []
    for i, input_data in enumerate(inputs):
        try:
            result = predict_admission(input_data, pipeline=pipeline)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch prediction failed for input {i}: {e}")
            results.append({"error": str(e), "input_index": i})

    logger.info(f"Batch prediction complete: {len(results)} results")
    return results
