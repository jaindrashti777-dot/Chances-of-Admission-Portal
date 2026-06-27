"""
FastAPI Application — Admission Prediction API

Endpoints:
    GET  /health  — Health check
    POST /predict — Predict admission chance

Usage:
    uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Docs:
    http://localhost:8000/docs  (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Literal

# Resolve project root for imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Admission Prediction API",
    description=(
        "Predict the probability of a student getting college admission in India "
        "based on 16 academic and demographic features."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ==============================================================================
# Pydantic models (input validation built-in)
# ==============================================================================


class AdmissionInput(BaseModel):
    """Input schema for admission prediction."""

    Tenth_Percentage: float = Field(
        ..., ge=0, le=100, description="10th class percentage"
    )
    Twelfth_Percentage: float = Field(
        ..., ge=0, le=100, description="12th class percentage"
    )
    JEE_Percentile: float = Field(..., ge=0, le=100, description="JEE exam percentile")
    CUET_Score: float = Field(..., ge=0, le=800, description="CUET exam score")
    Category: Literal["General", "OBC", "SC", "ST", "EWS"] = Field(
        ..., description="Category: General, OBC, SC, ST, EWS"
    )
    State: Literal[
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
    ] = Field(..., description="Home state")
    Family_Income: float = Field(
        ..., ge=0, le=200, description="Family income in Lakhs/year"
    )
    Gender: Literal["Male", "Female"] = Field(..., description="Gender: Male or Female")
    Gap_Year: int = Field(..., ge=0, le=5, description="Number of gap years")
    CGPA: float = Field(..., ge=0, le=10, description="Current CGPA")
    Backlogs: int = Field(..., ge=0, le=50, description="Number of backlogs")
    Extracurricular: int = Field(
        ..., ge=0, le=1, description="Extracurricular activities (0 or 1)"
    )
    Research_Paper: int = Field(
        ..., ge=0, le=20, description="Number of research papers"
    )
    Internship: int = Field(..., ge=0, le=10, description="Number of internships")
    Desired_Branch: Literal[
        "CSE", "ECE", "ME", "CE", "EE", "IT", "Chemical", "Biotech"
    ] = Field(..., description="Branch: CSE, ECE, ME, CE, EE, IT, Chemical, Biotech")
    College_Tier: Literal["Tier_1", "Tier_2", "Tier_3"] = Field(
        ..., description="College tier: Tier_1, Tier_2, Tier_3"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "Tenth_Percentage": 88.5,
                    "Twelfth_Percentage": 85.0,
                    "JEE_Percentile": 92.3,
                    "CUET_Score": 650.0,
                    "Category": "General",
                    "State": "Maharashtra",
                    "Family_Income": 12.0,
                    "Gender": "Male",
                    "Gap_Year": 0,
                    "CGPA": 8.9,
                    "Backlogs": 0,
                    "Extracurricular": 1,
                    "Research_Paper": 1,
                    "Internship": 2,
                    "Desired_Branch": "CSE",
                    "College_Tier": "Tier_1",
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for admission prediction."""

    admission_chance: float = Field(
        ..., description="Predicted admission probability (0-1)"
    )
    admission_percentage: str = Field(
        ..., description="Admission chance as percentage string"
    )
    confidence_level: str = Field(
        ..., description="Confidence level: High, Medium, or Low"
    )
    top_factors: list[dict] = Field(
        default_factory=list,
        description="Top contributing factors (SHAP-based)",
    )
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    version: str


# ==============================================================================
# Model loading (lazy, cached)
# ==============================================================================

_pipeline = None


def get_pipeline():
    """Load or return cached pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.predict import load_pipeline

            _pipeline = load_pipeline()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not available. Run training first. Error: {e}",
            )
    return _pipeline


# ==============================================================================
# Endpoints
# ==============================================================================


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Check API health and model availability."""
    model_loaded = False
    try:
        get_pipeline()
        model_loaded = True
    except HTTPException:
        pass

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: AdmissionInput):
    """
    Predict admission chance for a student.

    Accepts 16 features and returns the predicted admission probability
    along with confidence level and top contributing factors.
    """
    pipeline = get_pipeline()

    try:
        from src.predict import predict_admission

        input_dict = input_data.model_dump()
        result = predict_admission(input_dict, pipeline=pipeline)

        # Try to get SHAP factors
        top_factors = []
        try:
            import pandas as pd

            from src.explain import explain_prediction
            from src.pipeline_builder import get_feature_names_from_pipeline

            input_df = pd.DataFrame([input_dict])
            preprocessor = pipeline.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(input_df)
            feature_names = get_feature_names_from_pipeline(pipeline)
            model = pipeline.named_steps["model"]

            contributions = explain_prediction(model, X_transformed, feature_names)
            top_factors = [
                {"feature": k, "contribution": v}
                for k, v in list(contributions.items())[:5]
            ]
        except Exception as e:
            logger.info("SHAP explanation unavailable: %s", e)

        return PredictionOutput(
            admission_chance=result["admission_chance"],
            admission_percentage=f"{result['admission_chance'] * 100:.1f}%",
            confidence_level=result["confidence_level"],
            top_factors=top_factors,
            timestamp=result["timestamp"],
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.get("/metrics", tags=["System"])
def metrics():
    """
    Retrieve basic system and model usage metrics.
    """
    import pandas as pd

    from src.config import load_config, resolve_path

    config = load_config()
    log_path = resolve_path(
        config.get("data", {}).get("predictions_log", "reports/predictions.csv")
    )

    total_predictions = 0
    average_prediction = 0.0

    if log_path.exists():
        try:
            df = pd.read_csv(log_path)
            total_predictions = len(df)
            if total_predictions > 0 and "prediction" in df.columns:
                average_prediction = round(float(df["prediction"].mean()), 4)
        except Exception:
            pass

    return {
        "total_predictions_served": total_predictions,
        "average_admission_chance_predicted": average_prediction,
        "model_version": "1.0.0",
        "status": "active",
    }


@app.get("/", tags=["System"])
def root():
    """API root — redirects to documentation."""
    return {
        "message": "Admission Prediction API",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }
