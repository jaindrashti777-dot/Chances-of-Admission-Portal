"""
Model evaluation module.

Provides metrics computation, model comparison tables,
and visualization functions (actual vs predicted, residuals, etc.).
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import resolve_path
from src.logger import get_logger
from src.utils import save_figure

logger = get_logger(__name__)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """
    Evaluate a model on the given data.

    Parameters
    ----------
    model : estimator or Pipeline
        Fitted model.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True target values.

    Returns
    -------
    dict[str, float]
        Dictionary with R², MAE, RMSE, MSE.
    """
    try:
        y_pred = model.predict(X)
        metrics = {
            "r2": r2_score(y, y_pred),
            "mae": mean_absolute_error(y, y_pred),
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mse": mean_squared_error(y, y_pred),
        }
        return metrics
    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """
    Create a comparison DataFrame from training results.

    Parameters
    ----------
    results : dict
        {model_name: {"metrics": {...}, "time": float, ...}}

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by R² descending.
    """
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "R2": round(m["r2"], 4),
            "MAE": round(m["mae"], 4),
            "RMSE": round(m["rmse"], 4),
            "MSE": round(m["mse"], 6),
            "Train_Time_s": res.get("time", m.get("train_time_seconds", 0)),
        })

    df = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(drop=True)
    logger.info(f"Model comparison table:\n{df.to_string(index=False)}")
    return df


def plot_actual_vs_predicted(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Scatter plot of actual vs predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(y_true, y_pred, alpha=0.4, s=20, color="#2196F3", edgecolors="none")

    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "r--", linewidth=2, label="Perfect prediction",
    )

    ax.set_xlabel("Actual Admission Chance", fontsize=12)
    ax.set_ylabel("Predicted Admission Chance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    title: str = "Residual Plot",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot residuals (errors) against predicted values.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residuals vs Predicted
    axes[0].scatter(
        y_pred, residuals, alpha=0.4, s=20, color="#FF5722", edgecolors="none"
    )
    axes[0].axhline(y=0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("Predicted Values", fontsize=12)
    axes[0].set_ylabel("Residuals", fontsize=12)
    axes[0].set_title("Residuals vs Predicted", fontsize=13, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Residual distribution
    axes[1].hist(
        residuals, bins=40, color="#4CAF50", edgecolor="white", alpha=0.8
    )
    axes[1].axvline(x=0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual Value", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("Residual Distribution", fontsize=13, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)

    return fig


def plot_model_comparison_bar(
    comparison_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Bar chart comparing model metrics.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output from compare_models().
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = [
        "#2196F3", "#4CAF50", "#FF9800", "#F44336",
        "#9C27B0", "#00BCD4", "#FF5722",
    ]

    models = comparison_df["Model"]

    for ax, metric, label in zip(
        axes,
        ["R2", "MAE", "RMSE"],
        ["R² Score (higher is better)", "MAE (lower is better)", "RMSE (lower is better)"],
    ):
        bars = ax.bar(
            range(len(models)),
            comparison_df[metric],
            color=colors[: len(models)],
            edgecolor="white",
            linewidth=0.5,
        )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, comparison_df[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    fig.suptitle(
        "Model Comparison", fontsize=16, fontweight="bold", y=1.02
    )
    plt.tight_layout()

    path = save_path or str(resolve_path("reports/figures/model_comparison.png"))
    save_figure(fig, path)

    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance (Top 20)",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of top feature importances.

    Parameters
    ----------
    importances : array-like
        Feature importance values.
    feature_names : list[str]
        Feature names.
    top_n : int
        Number of top features to show.
    title : str
        Plot title.
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    indices = np.argsort(importances)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    bars = ax.barh(
        range(len(top_features)),
        top_importances,
        color="#2196F3",
        edgecolor="white",
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features, fontsize=10)
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = save_path or str(resolve_path("reports/figures/feature_importance.png"))
    save_figure(fig, path)

    return fig
