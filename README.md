# College Admission Predictor (India) -- Production ML System

> **Disclaimer:** This is a **synthetic demonstration project** for portfolio
> purposes. The dataset is generated from a deterministic formula, so the high
> model accuracy (R^2 > 0.90) is expected and does not reflect real-world
> admissions prediction performance. Do not use this system for actual
> admissions decisions.

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end, production-grade machine learning system that predicts a
student's probability of getting college admission in India based on 16
academic and demographic features.

---

## Overview

This project goes beyond a simple Jupyter Notebook and demonstrates a
professional MLOps workflow. It includes:

- **Production Pipelines:** Custom `sklearn.pipeline.Pipeline` with
  `ColumnTransformer`.
- **Hyperparameter Tuning:** Automated `RandomizedSearchCV` on the best
  base model using grids defined in `config.yaml`.
- **Explainable AI:** SHAP integration to show *why* a prediction was made.
- **Robust Engineering:** Validation layers, structured logging, and full
  exception handling.
- **Dual Deployment:** A Streamlit dashboard and a FastAPI backend.
- **DevOps:** Dockerized services, GitHub Actions CI/CD, and a `Makefile`
  for developer ergonomics.

---

## Architecture

The system is designed with a clear separation of concerns:

1. **Data Validation:** Strict input checking (`validator.py`).
2. **Preprocessing:** Handling missing values, outliers, and data types
   (`preprocessing.py`).
3. **Modeling:** 7 regression models tracked via **MLflow**, best model
   selected and tuned.
4. **Inference:** Fast prediction with full history logging (`predict.py`).

*(See `docs/architecture.md` for the full Mermaid diagram.)*

---

## Quickstart

### Prerequisites

- Python 3.12+
- (Optional) Docker & Docker Compose

### Local Setup

1. **Clone & Install**
   ```bash
   git clone https://github.com/username/Chance-of-Admission.git
   cd Chance-of-Admission
   make install
   ```

2. **Generate Data & Train Model**
   ```bash
   make generate   # Generates 5,000 synthetic records
   make train      # Runs full pipeline, logs to MLflow, saves best model
   ```

3. **Run the Apps**
   ```bash
   make app        # Launches the Streamlit Dashboard
   make api        # Launches the FastAPI server
   ```

### Docker Setup

> **Important:** You must generate data and train a model *before* starting
> Docker, because the containers mount `./models` from the host.

```bash
make generate
make train
make docker-up
```

- Streamlit: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs

---

## Results & Model Comparison

7 models are trained and compared. Tree-based ensembles (XGBoost, LightGBM,
RandomForest) typically perform best on this synthetic dataset.

| Model             | R2 Score | RMSE  |
|-------------------|----------|-------|
| LightGBM          | ~0.95    | ~0.02 |
| XGBoost           | ~0.94    | ~0.02 |
| Random Forest     | ~0.93    | ~0.03 |
| Linear Regression | ~0.85    | ~0.06 |

> **Note:** These metrics are on synthetic data generated from a known
> formula. High R2 is expected and does not imply the model would generalise
> to real admissions data.

---

## Limitations & Model Card

| Item                 | Detail                                                     |
|----------------------|------------------------------------------------------------|
| **Dataset**          | 5,000 synthetic records generated via a weighted formula   |
| **Target**           | `Admission_Chance` (0--1), computed deterministically      |
| **Feature count**    | 16 (10 numerical, 4 categorical, 1 ordinal, 1 binary)     |
| **Known limitation** | Model learns the generating formula, not real patterns     |
| **Intended use**     | Portfolio demonstration of MLOps skills                    |
| **Not intended for** | Real admissions decisions or policy making                 |
| **Fairness**         | Category/reservation boosts are baked into synthetic data  |

---

## Author & Resume Section

**Admission Chance Prediction System (End-to-End ML Project)**
> Built a production-grade machine learning system using Python, Scikit-learn,
> XGBoost, Streamlit, FastAPI, and Docker. Implemented preprocessing
> pipelines with ColumnTransformer, hyperparameter tuning via
> RandomizedSearchCV, SHAP explainability, unit and end-to-end testing,
> MLflow experiment tracking, and deployment-ready architecture with an
> interactive prediction dashboard. Dataset is synthetic (formula-generated).

---

*License: MIT*
