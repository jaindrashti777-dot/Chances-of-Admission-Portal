"""
sklearn Pipeline and ColumnTransformer builder.

Creates a production-grade ML pipeline that chains preprocessing and
model training into a single reproducible object.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.feature_engineering import get_feature_lists
from src.logger import get_logger

logger = get_logger(__name__)


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles all feature types.

    - Numerical features → StandardScaler
    - Categorical features → OneHotEncoder (handle_unknown='ignore')
    - Ordinal features → OrdinalEncoder (Tier_3 < Tier_2 < Tier_1)
    - Binary features → passed through as-is

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor.
    """
    features = get_feature_lists()

    logger.info("Building preprocessor ColumnTransformer")
    logger.info(f"  Numerical ({len(features['numerical'])}): {features['numerical']}")
    logger.info(
        f"  Categorical ({len(features['categorical'])}): {features['categorical']}"
    )
    logger.info(f"  Ordinal ({len(features['ordinal'])}): {features['ordinal']}")
    logger.info(f"  Binary ({len(features['binary'])}): {features['binary']}")

    numerical_transformer = Pipeline(
        steps=[("scaler", StandardScaler())]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop="first",
                ),
            )
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[["Tier_3", "Tier_2", "Tier_1"]],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, features["numerical"]),
            ("cat", categorical_transformer, features["categorical"]),
            ("ord", ordinal_transformer, features["ordinal"]),
            ("bin", "passthrough", features["binary"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )

    logger.info("Preprocessor built successfully")
    return preprocessor


def build_full_pipeline(model, preprocessor: ColumnTransformer | None = None):
    """
    Build a complete Pipeline: preprocessor → model.

    Parameters
    ----------
    model : estimator
        Any scikit-learn compatible regressor.
    preprocessor : ColumnTransformer, optional
        Custom preprocessor. Builds a new one if not provided.

    Returns
    -------
    Pipeline
        Full ML pipeline (preprocess + model).
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    model_name = type(model).__name__
    logger.info(f"Full pipeline built: Preprocessor → {model_name}")
    return pipeline


def get_feature_names_from_pipeline(pipeline) -> list[str]:
    """
    Extract feature names after transformation.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted pipeline with a preprocessor step.

    Returns
    -------
    list[str]
        Transformed feature names.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        return list(preprocessor.get_feature_names_out())
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        return []
