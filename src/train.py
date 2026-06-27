"""
Model training module.

Trains 7 regression models using sklearn Pipelines, logs everything
to MLflow, and performs hyperparameter tuning on the best candidates.
"""

from __future__ import annotations

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import time
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.config import RANDOM_STATE, load_config, resolve_path
from src.data_loader import load_raw_data, save_processed_data
from src.evaluate import compare_models, evaluate_model, plot_model_comparison_bar
from src.feature_engineering import split_data
from src.logger import get_training_logger
from src.pipeline_builder import build_full_pipeline, build_preprocessor
from src.preprocessing import preprocess_data
from src.utils import save_model, set_seed
from src.validator import validate_dataframe

logger = get_training_logger()


# ==============================================================================
# Model definitions
# ==============================================================================


def get_models() -> dict[str, Any]:
    """Return a dict of model name → unfitted estimator."""
    return {
        "Linear_Regression": LinearRegression(),
        "Decision_Tree": DecisionTreeRegressor(
            random_state=RANDOM_STATE,
        ),
        "Random_Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient_Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=200,
            random_state=RANDOM_STATE,
            verbose=0,
        ),
    }


# ==============================================================================
# Training
# ==============================================================================


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict[str, dict]:
    """
    Train all models and return results.

    Each model is wrapped in a Pipeline (preprocessor → model), trained,
    evaluated, and logged to MLflow.

    Parameters
    ----------
    X_train, y_train : training data
    X_val, y_val : validation data

    Returns
    -------
    dict
        {model_name: {"pipeline": fitted_pipeline, "metrics": {...}, "time": float}}
    """
    models = get_models()
    preprocessor = build_preprocessor()
    results: dict[str, dict] = {}

    config = load_config()
    mlflow_uri = resolve_path(config.get("mlflow", {}).get("tracking_uri", "mlruns")).as_uri()
    experiment_name = config.get("mlflow", {}).get(
        "experiment_name", "admission_prediction"
    )
    mlflow.set_tracking_uri(str(mlflow_uri))
    mlflow.set_experiment(experiment_name)

    logger.info("=" * 60)
    logger.info("MODEL TRAINING — 7 Regression Models")
    logger.info("=" * 60)

    for name, model in models.items():
        logger.info(f"\n--- Training: {name} ---")

        try:
            pipeline = build_full_pipeline(model, preprocessor)
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            elapsed = round(time.time() - start_time, 2)

            # Evaluate on validation set
            metrics = evaluate_model(pipeline, X_val, y_val)
            metrics["train_time_seconds"] = elapsed

            # Log to MLflow
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model_type", name)
                for metric_name, value in metrics.items():
                    mlflow.log_metric(metric_name, value)
                mlflow.sklearn.log_model(pipeline, "model")

            results[name] = {
                "pipeline": pipeline,
                "metrics": metrics,
                "time": elapsed,
            }

            logger.info(
                f"  {name}: R²={metrics['r2']:.4f}, "
                f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                f"Time={elapsed}s"
            )

        except Exception as e:
            logger.error(f"  {name} training FAILED: {e}")
            # Continue with other models — one failure shouldn't stop everything
            continue

    logger.info(f"\nTraining complete. {len(results)}/{len(models)} models succeeded.")
    return results


# ==============================================================================
# Hyperparameter tuning
# ==============================================================================


def tune_model(
    model,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 30,
    cv: int = 5,
) -> Any:
    """
    Tune a model using RandomizedSearchCV.

    Parameters
    ----------
    model : estimator
        The model to tune (not a pipeline — just the raw estimator).
    param_grid : dict
        Hyperparameter distributions.
    X_train, y_train : training data
    n_iter : int
        Number of random parameter combinations to try.
    cv : int
        Cross-validation folds.

    Returns
    -------
    estimator
        The best estimator found.
    """
    model_name = type(model).__name__
    logger.info(f"Tuning {model_name} with {n_iter} iterations, {cv}-fold CV")

    try:
        search = RandomizedSearchCV(
            model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=0,
        )
        search.fit(X_train, y_train)

        logger.info(f"  Best R² (CV): {search.best_score_:.4f}")
        logger.info(f"  Best params: {search.best_params_}")

        return search.best_estimator_

    except Exception as e:
        logger.error(f"Hyperparameter tuning failed for {model_name}: {e}")
        raise


# ==============================================================================
# Full training pipeline (entry point)
# ==============================================================================


def run_training_pipeline() -> None:
    """
    Execute the complete training pipeline.

    Steps:
    1. Load raw data
    2. Validate data
    3. Preprocess data
    4. Split into train/val/test
    5. Train all models
    6. Compare and select best
    7. Hyperparameter-tune top model
    8. Final evaluation on test set
    9. Save best pipeline
    """
    set_seed(RANDOM_STATE)

    logger.info("=" * 60)
    logger.info("FULL TRAINING PIPELINE")
    logger.info("=" * 60)

    try:
        # 1. Load data
        df = load_raw_data()

        # 2. Validate
        is_valid, errors = validate_dataframe(df)
        if not is_valid:
            logger.warning(f"Validation issues: {errors}")

        # 3. Preprocess
        df = preprocess_data(df)
        save_processed_data(df)

        # 4. Split (BEFORE fitting any transformers)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        # 5. Train all models
        results = train_all_models(X_train, y_train, X_val, y_val)

        if not results:
            logger.error("No models trained successfully. Aborting.")
            return

        # 6. Compare and find best
        comparison_df = compare_models(results)
        logger.info(f"\nModel Comparison:\n{comparison_df.to_string()}")

        best_name = comparison_df.loc[comparison_df["R2"].idxmax(), "Model"]
        best_pipeline = results[best_name]["pipeline"]
        logger.info(f"\nBest model: {best_name}")

        # 7. Plot comparison
        try:
            plot_model_comparison_bar(comparison_df)
        except Exception as e:
            logger.warning(f"Could not create comparison plot: {e}")

        # 8. Final evaluation on test set
        test_metrics = evaluate_model(best_pipeline, X_test, y_test)
        logger.info(f"\nTest set evaluation ({best_name}):")
        for k, v in test_metrics.items():
            logger.info(f"  {k}: {v:.4f}")

        # 9. Save the best pipeline
        config = load_config()
        pipeline_path = resolve_path(
            config.get("model_paths", {}).get("pipeline", "models/pipeline.pkl")
        )
        model_path = resolve_path(
            config.get("model_paths", {}).get("best_model", "models/best_model.pkl")
        )
        save_model(best_pipeline, pipeline_path)
        save_model(
            best_pipeline.named_steps["model"],
            model_path,
        )

        logger.info("=" * 60)
        logger.info("TRAINING PIPELINE COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    run_training_pipeline()
