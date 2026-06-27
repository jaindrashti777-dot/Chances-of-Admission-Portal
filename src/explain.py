"""
Model explainability module.

Uses SHAP (TreeExplainer for tree-based models, KernelExplainer as fallback)
and permutation importance to explain predictions.
"""

from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend for saving figures

from src.config import resolve_path
from src.logger import get_logger
from src.utils import save_figure

logger = get_logger(__name__)


def explain_prediction(
    model,
    X_sample: pd.DataFrame | np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """
    Generate SHAP-based explanation for a single prediction.

    Returns a dict of feature → SHAP contribution.
    Example: {"CGPA": +0.18, "Research_Paper": +0.08, ...}

    Parameters
    ----------
    model : estimator
        A fitted tree-based model (not a pipeline).
    X_sample : array-like
        Single sample (1 row) or small batch.
    feature_names : list[str]
        Feature names matching the columns.

    Returns
    -------
    dict[str, float]
        Feature name → SHAP value (sorted by absolute magnitude).
    """
    try:
        import shap

        # Try TreeExplainer first (fast, exact for tree models)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            logger.info("TreeExplainer not available, falling back to KernelExplainer")
            explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X_sample, min(100, len(X_sample))),
            )

        if isinstance(X_sample, pd.DataFrame):
            X_sample = X_sample.values

        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = explainer.shap_values(X_sample)

        if shap_values.ndim > 1:
            shap_values = shap_values[0]

        # Build sorted dict
        contributions = {}
        for name, val in zip(feature_names, shap_values):
            contributions[name] = round(float(val), 4)

        # Sort by absolute value (most impactful first)
        contributions = dict(
            sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        logger.info(f"SHAP explanation generated for {len(feature_names)} features")
        return contributions

    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return {}
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}")
        return {}


def generate_shap_summary(
    model,
    X: pd.DataFrame | np.ndarray,
    feature_names: list[str],
    save_path: str | None = None,
    max_display: int = 20,
) -> None:
    """
    Generate and save a SHAP summary (beeswarm) plot.

    Parameters
    ----------
    model : estimator
        Fitted tree-based model.
    X : array-like
        Feature matrix (validation or test set).
    feature_names : list[str]
        Feature names.
    save_path : str, optional
        Where to save the plot.
    max_display : int
        Maximum features to display.
    """
    try:
        import shap

        logger.info("Generating SHAP summary plot")

        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            logger.info("Using KernelExplainer for SHAP summary")
            explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X, min(100, len(X))),
            )

        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X

        shap_values = explainer.shap_values(X_array)

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_array,
            feature_names=feature_names,
            max_display=max_display,
            show=False,
        )
        plt.title("SHAP Feature Importance", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = save_path or str(resolve_path("reports/figures/shap_summary.png"))
        plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"SHAP summary plot saved to {path}")

    except ImportError:
        logger.warning("SHAP not installed. Skipping summary plot.")
    except Exception as e:
        logger.error(f"SHAP summary plot failed: {e}")


def generate_feature_importance_report(
    model,
    feature_names: list[str],
    save_path: str | None = None,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Generate a feature importance report from the model.

    Works with any model that has ``feature_importances_`` attribute
    (tree-based models). Falls back to coefficients for linear models.

    Parameters
    ----------
    model : estimator
        Fitted model.
    feature_names : list[str]
        Feature names.
    save_path : str, optional
        Where to save the bar chart.
    top_n : int
        Number of top features to display.

    Returns
    -------
    pd.DataFrame
        Feature importance table.
    """
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_)
        else:
            logger.warning("Model does not support feature importance extraction")
            return pd.DataFrame()

        # Build DataFrame
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances,
        }).sort_values("Importance", ascending=False).reset_index(drop=True)

        # Plot top N
        top = importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        ax.barh(
            range(len(top)),
            top["Importance"].values[::-1],
            color="#2196F3",
            edgecolor="white",
        )
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["Feature"].values[::-1], fontsize=10)
        ax.set_xlabel("Importance", fontsize=12)
        ax.set_title(
            f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold"
        )
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()

        path = save_path or str(resolve_path("reports/figures/feature_importance.png"))
        save_figure(fig, path)
        plt.close(fig)

        logger.info(f"Feature importance report:\n{importance_df.head(10).to_string()}")
        return importance_df

    except Exception as e:
        logger.error(f"Feature importance report failed: {e}")
        return pd.DataFrame()
