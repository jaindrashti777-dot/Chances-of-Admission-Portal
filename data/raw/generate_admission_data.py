"""
Synthetic dataset generator for the India Admission Prediction project.

Generates ~5,000 realistic student records with 16 India-specific features
and a computed Admission_Chance target (0–1).

Usage:
    python data/raw/generate_admission_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Resolve project root so we can import src modules when run as a script
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def generate_dataset(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic synthetic admission dataset.

    The target ``Admission_Chance`` is computed via a weighted formula
    that reflects realistic Indian admission dynamics:
    - Academic scores (10th, 12th, JEE, CUET, CGPA) are the strongest predictors
    - Research, internships, and extracurriculars provide a boost
    - Backlogs and gap years reduce chances
    - College tier and branch affect difficulty
    - Category provides reservation-based adjustments

    Parameters
    ----------
    n_samples : int
        Number of records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Generated dataset with 16 features + 1 target.
    """
    rng = np.random.default_rng(seed)

    # ---- Academic Scores (correlated) ----
    # Students who score well in 10th tend to score well in 12th, JEE, etc.
    base_ability = rng.beta(5, 2, n_samples)  # skew toward higher ability

    tenth = np.clip(base_ability * 60 + rng.normal(35, 8, n_samples), 40, 99)
    twelfth = np.clip(
        tenth * 0.7 + rng.normal(20, 8, n_samples), 40, 99
    )
    jee = np.clip(
        base_ability * 70 + rng.normal(20, 15, n_samples), 0, 99.99
    )
    cuet = np.clip(
        base_ability * 500 + rng.normal(200, 80, n_samples), 100, 800
    )
    cgpa = np.clip(
        base_ability * 4 + rng.normal(4.5, 1.0, n_samples), 4.0, 10.0
    )

    # ---- Categorical Features ----
    categories = rng.choice(
        ["General", "OBC", "SC", "ST", "EWS"],
        n_samples,
        p=[0.35, 0.30, 0.15, 0.10, 0.10],
    )

    states = rng.choice(
        [
            "Maharashtra", "Delhi", "Karnataka", "Tamil_Nadu",
            "Uttar_Pradesh", "West_Bengal", "Rajasthan", "Gujarat",
            "Telangana", "Kerala",
        ],
        n_samples,
        p=[0.15, 0.12, 0.12, 0.10, 0.12, 0.08, 0.08, 0.08, 0.08, 0.07],
    )

    genders = rng.choice(
        ["Male", "Female"], n_samples, p=[0.58, 0.42]
    )

    branches = rng.choice(
        ["CSE", "ECE", "ME", "CE", "EE", "IT", "Chemical", "Biotech"],
        n_samples,
        p=[0.25, 0.15, 0.12, 0.10, 0.10, 0.12, 0.08, 0.08],
    )

    # Tier correlates with academic ability
    tier_probs = np.column_stack([
        np.clip(0.1 + (1 - base_ability) * 0.6, 0, 1),  # Tier 3
        np.clip(0.3 + base_ability * 0.1, 0, 1),          # Tier 2
        np.clip(base_ability * 0.5, 0, 1),                 # Tier 1
    ])
    tier_probs = tier_probs / tier_probs.sum(axis=1, keepdims=True)
    tiers = np.array([
        rng.choice(["Tier_3", "Tier_2", "Tier_1"], p=p)
        for p in tier_probs
    ])

    # ---- Numerical Features ----
    income = np.clip(rng.lognormal(2.0, 0.8, n_samples), 1, 50).round(2)
    gap_year = rng.choice([0, 1, 2], n_samples, p=[0.70, 0.22, 0.08])
    backlogs = np.clip(rng.poisson(0.5, n_samples), 0, 10)
    extracurricular = rng.choice([0, 1], n_samples, p=[0.40, 0.60])
    research_paper = np.clip(rng.poisson(0.4, n_samples), 0, 5)
    internship = np.clip(rng.poisson(0.8, n_samples), 0, 4)

    # ==================================================================
    # Compute target: Admission_Chance (0 to 1)
    # ==================================================================
    # Weighted academic score (normalized to 0-1)
    academic_score = (
        (tenth / 100) * 0.10
        + (twelfth / 100) * 0.12
        + (jee / 100) * 0.20
        + (cuet / 800) * 0.15
        + (cgpa / 10) * 0.20
    )

    # Boost from extras
    extras_boost = (
        research_paper * 0.02
        + internship * 0.02
        + extracurricular * 0.03
    )

    # Penalties
    penalties = (
        backlogs * 0.03
        + gap_year * 0.02
    )

    # Tier difficulty modifier
    tier_modifier = np.where(tiers == "Tier_1", -0.10,
                    np.where(tiers == "Tier_2", -0.03, 0.05))

    # Branch competition (CSE is harder)
    branch_modifier = np.where(
        np.isin(branches, ["CSE", "IT"]), -0.05,
        np.where(np.isin(branches, ["ECE", "EE"]), -0.02, 0.0)
    )

    # Category reservation boost
    category_boost = np.where(
        np.isin(categories, ["SC", "ST"]), 0.08,
        np.where(categories == "OBC", 0.04,
        np.where(categories == "EWS", 0.03, 0.0))
    )

    # Final score with noise
    admission_chance = (
        academic_score
        + extras_boost
        - penalties
        + tier_modifier
        + branch_modifier
        + category_boost
        + rng.normal(0, 0.04, n_samples)  # random noise
    )

    # Clip to [0, 1] and round
    admission_chance = np.clip(admission_chance, 0.05, 0.98).round(4)

    # ==================================================================
    # Build DataFrame
    # ==================================================================
    df = pd.DataFrame({
        "Tenth_Percentage": tenth.round(2),
        "Twelfth_Percentage": twelfth.round(2),
        "JEE_Percentile": jee.round(2),
        "CUET_Score": cuet.round(2),
        "Category": categories,
        "State": states,
        "Family_Income": income,
        "Gender": genders,
        "Gap_Year": gap_year,
        "CGPA": cgpa.round(2),
        "Backlogs": backlogs,
        "Extracurricular": extracurricular,
        "Research_Paper": research_paper,
        "Internship": internship,
        "Desired_Branch": branches,
        "College_Tier": tiers,
        "Admission_Chance": admission_chance,
    })

    return df


def main() -> None:
    """Generate and save the synthetic dataset."""
    from src.logger import get_logger  # noqa: E402

    logger = get_logger(__name__)

    try:
        logger.info("=" * 60)
        logger.info("GENERATING SYNTHETIC ADMISSION DATASET")
        logger.info("=" * 60)

        df = generate_dataset(n_samples=5000, seed=42)

        # Save to data/raw/
        output_path = PROJECT_ROOT / "data" / "raw" / "admission_data.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

        logger.info(f"Dataset saved to {output_path}")
        logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.info(f"Target distribution:")
        logger.info(f"  Mean:   {df['Admission_Chance'].mean():.4f}")
        logger.info(f"  Median: {df['Admission_Chance'].median():.4f}")
        logger.info(f"  Std:    {df['Admission_Chance'].std():.4f}")
        logger.info(f"  Min:    {df['Admission_Chance'].min():.4f}")
        logger.info(f"  Max:    {df['Admission_Chance'].max():.4f}")
        logger.info("Dataset generation complete!")

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
