"""FastAPI application for the admission prediction demo."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

# Resolve project root for imports when the app is started from Docker/Render.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette import status

from src.config import load_config, resolve_path
from src.logger import get_logger

logger = get_logger(__name__)
API_VERSION = "1.0.0"

app = FastAPI(
    title="Admission Prediction API",
    description=(
        "Demonstration API for a synthetic college admission prediction model. "
        "The service showcases ML inference, validation, explainability-ready output, "
        "and deployment patterns."
    ),
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

origins_env = os.environ.get("CORS_ORIGINS")
if origins_env:
    allowed_origins = [
        origin.strip() for origin in origins_env.split(",") if origin.strip()
    ]
else:
    allowed_origins = [
        "http://localhost:3000",
        "https://chances-of-admission-portal.vercel.app",
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Return compact validation errors that frontend clients can display."""
    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"] if part != "body"),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": "Request payload failed validation.",
            "details": errors,
        },
    )


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
        ..., description="Reservation category"
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
        ..., ge=0, le=200, description="Family income in lakhs per year"
    )
    Gender: Literal["Male", "Female"] = Field(..., description="Gender")
    Gap_Year: int = Field(..., ge=0, le=5, description="Number of gap years")
    CGPA: float = Field(..., ge=0, le=10, description="Current CGPA")
    Backlogs: int = Field(
        ..., ge=0, le=50, description="Number of active or historical backlogs"
    )
    Extracurricular: int = Field(
        ..., ge=0, le=1, description="Extracurricular activity flag"
    )
    Research_Paper: int = Field(
        ..., ge=0, le=20, description="Number of research papers"
    )
    Internship: int = Field(..., ge=0, le=10, description="Number of internships")
    Desired_Branch: Literal[
        "CSE", "ECE", "ME", "CE", "EE", "IT", "Chemical", "Biotech"
    ] = Field(..., description="Desired engineering branch")
    College_Tier: Literal["Tier_1", "Tier_2", "Tier_3"] = Field(
        ..., description="Target college tier"
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
        ..., ge=0, le=1, description="Predicted probability from 0 to 1"
    )
    admission_percentage: str = Field(
        ..., description="Admission chance as a percentage string"
    )
    confidence_level: Literal["High", "Medium", "Low"] = Field(
        ..., description="Band derived from prediction"
    )
    resume_score: int = Field(
        ..., ge=0, le=100, description="Heuristic profile readiness score"
    )
    top_factors: list[dict[str, float | str]] = Field(
        default_factory=list, description="Top SHAP factors"
    )
    timestamp: str = Field(..., description="Prediction timestamp")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_available: bool
    version: str


_pipeline = None


def get_pipeline_path() -> Path:
    """Resolve the configured pipeline artifact path."""
    config = load_config()
    return resolve_path(
        config.get("model_paths", {}).get("pipeline", "models/pipeline.pkl")
    )


def get_pipeline():
    """Load or return the cached pipeline."""
    global _pipeline
    if _pipeline is None:
        try:
            from src.predict import load_pipeline

            _pipeline = load_pipeline()
        except Exception as exc:
            logger.exception("Model pipeline failed to load")
            raise HTTPException(
                status_code=503,
                detail={
                    "error": "model_unavailable",
                    "message": "Model artifact could not be loaded.",
                    "detail": str(exc),
                },
            ) from exc
    return _pipeline


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Return service health without forcing model deserialization."""
    return HealthResponse(
        status="healthy",
        model_loaded=_pipeline is not None,
        model_available=get_pipeline_path().exists(),
        version=API_VERSION,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(input_data: AdmissionInput):
    """Predict admission chance for one synthetic profile."""
    pipeline = get_pipeline()

    try:
        import pandas as pd

        from src.explain import explain_prediction
        from src.pipeline_builder import get_feature_names_from_pipeline
        from src.predict import predict_admission

        input_dict = input_data.model_dump()
        result = predict_admission(input_dict, pipeline=pipeline)

        top_factors: list[dict[str, float | str]] = []
        try:
            input_df = pd.DataFrame([input_dict])
            preprocessor = pipeline.named_steps["preprocessor"]
            X_transformed = preprocessor.transform(input_df)
            feature_names = get_feature_names_from_pipeline(pipeline)
            model = pipeline.named_steps["model"]
            contributions = explain_prediction(model, X_transformed, feature_names)
            top_factors = [
                {"feature": feature, "contribution": contribution}
                for feature, contribution in list(contributions.items())[:5]
            ]
        except Exception as exc:
            logger.info("SHAP explanation unavailable: %s", exc)

        academics_score = (
            input_dict["Tenth_Percentage"]
            + input_dict["Twelfth_Percentage"]
            + (input_dict["CGPA"] * 10)
        ) / 3
        extracurricular_score = min(
            100,
            (input_dict["Extracurricular"] * 20)
            + (input_dict["Research_Paper"] * 10)
            + (input_dict["Internship"] * 10),
        )
        resume_score = (academics_score * 0.7) + (extracurricular_score * 0.3)
        resume_score -= input_dict["Backlogs"] * 5
        resume_score = max(0, min(100, resume_score))

        return PredictionOutput(
            admission_chance=result["admission_chance"],
            admission_percentage=f"{result['admission_chance'] * 100:.1f}%",
            confidence_level=result["confidence_level"],
            resume_score=int(resume_score),
            top_factors=top_factors,
            timestamp=result["timestamp"],
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_error",
                "message": "Prediction input failed domain validation.",
                "detail": str(exc),
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_failed",
                "message": "Prediction failed while processing the request.",
                "detail": str(exc),
            },
        ) from exc


@app.get("/metrics", tags=["System"])
def metrics():
    """Retrieve basic API usage metrics from the local prediction log."""
    import pandas as pd

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
        except Exception as exc:
            logger.warning("Failed to read prediction metrics: %s", exc)

    return {
        "total_predictions_served": total_predictions,
        "average_admission_chance_predicted": average_prediction,
        "model_version": API_VERSION,
        "status": "active",
    }


@app.get("/", tags=["System"])
def root():
    """Return API metadata."""
    return {
        "message": "Admission Prediction API",
        "scope": "Synthetic-data portfolio demonstration; not for real admissions decisions.",
        "docs": "/docs",
        "health": "/health",
        "predict": "POST /predict",
    }


@app.get("/api/colleges", tags=["Reference Data"])
def get_colleges():
    """Return a small static college list used by the demo UI."""
    return [
        {
            "name": "IIT Bombay",
            "branch": "CSE",
            "state": "Maharashtra",
            "tier": "Tier_1",
            "package": "25 LPA",
            "nirf": 3,
        },
        {
            "name": "NIT Trichy",
            "branch": "CSE",
            "state": "Tamil Nadu",
            "tier": "Tier_1",
            "package": "15 LPA",
            "nirf": 9,
        },
        {
            "name": "BITS Pilani",
            "branch": "CSE",
            "state": "Rajasthan",
            "tier": "Tier_1",
            "package": "20 LPA",
            "nirf": 20,
        },
        {
            "name": "COEP Pune",
            "branch": "Computer",
            "state": "Maharashtra",
            "tier": "Tier_2",
            "package": "10 LPA",
            "nirf": 73,
        },
        {
            "name": "VJTI Mumbai",
            "branch": "IT",
            "state": "Maharashtra",
            "tier": "Tier_2",
            "package": "9 LPA",
            "nirf": 84,
        },
        {
            "name": "VIT Vellore",
            "branch": "CSE",
            "state": "Tamil Nadu",
            "tier": "Tier_2",
            "package": "8 LPA",
            "nirf": 11,
        },
        {
            "name": "SRM Chennai",
            "branch": "CSE",
            "state": "Tamil Nadu",
            "tier": "Tier_3",
            "package": "5 LPA",
            "nirf": 24,
        },
        {
            "name": "Amity University",
            "branch": "CSE",
            "state": "Delhi",
            "tier": "Tier_3",
            "package": "4 LPA",
            "nirf": 31,
        },
    ]


class ChatMessage(BaseModel):
    """Request schema for the rule-based advisor."""

    message: str = Field(..., min_length=1, max_length=500)


@app.post("/api/chat", tags=["Advisor"])
def chat(payload: ChatMessage):
    """Return deterministic profile guidance for the demo UI."""
    user_msg = payload.message.lower()
    if "improve" in user_msg or "cgpa" in user_msg:
        return {
            "response": "For this synthetic model, CGPA is usually an influential input. Try adjusting CGPA in the form to see how the predicted score changes."
        }
    if "backlog" in user_msg:
        return {
            "response": "Backlogs reduce the synthetic readiness score and can lower the prediction. The app treats fewer active backlogs as a stronger profile signal."
        }
    return {
        "response": "This is a rule-based demo helper. It can explain how fields like CGPA, backlogs, internships, and research papers affect this synthetic model."
    }
