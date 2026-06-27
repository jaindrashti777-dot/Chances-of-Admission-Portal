# 🎓 College Admission Predictor (India) — Production ML System

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An end-to-end, production-ready machine learning system that predicts a student's probability of getting college admission in India based on 16 academic and demographic features.

---

## 🌟 Overview

This project goes beyond a simple Jupyter Notebook and demonstrates a professional MLOps workflow. It includes:
- **Production Pipelines:** Custom `sklearn.pipeline.Pipeline` with `ColumnTransformer`.
- **Explainable AI:** SHAP integration to show *why* a prediction was made.
- **Robust Engineering:** Validation layers, structured logging, and full exception handling.
- **Dual Deployment:** A beautiful **Streamlit** dashboard and a highly-performant **FastAPI** backend.
- **DevOps:** Dockerized services, GitHub Actions CI/CD, and a `Makefile` for developer ergonomics.

---

## 🏗️ Architecture

The system is designed with a clear separation of concerns:

1. **Data Validation:** Strict input checking (`validator.py`).
2. **Preprocessing:** Handling missing values, outliers, and data types (`preprocessing.py`).
3. **Modeling:** 7 regression models tracked via **MLflow**, best model saved.
4. **Inference:** Fast prediction with full history logging (`predict.py`).

*(See `docs/architecture.md` for the full Mermaid diagram).*

---

## 🚀 Quickstart

### Local Setup

1. **Clone & Install**
   ```bash
   git clone https://github.com/username/Chance-of-Admission.git
   cd Chance-of-Admission
   make install
   ```

2. **Generate Data & Train Model**
   ```bash
   make generate   # Generates 5,000 realistic synthetic records
   make train      # Runs full pipeline, logs to MLflow, saves best model
   ```

3. **Run the Apps**
   ```bash
   make app        # Launches the Streamlit Dashboard
   make api        # Launches the FastAPI server
   ```

### Docker Setup

Run both the API and Streamlit dashboard instantly:
```bash
make docker-up
```
- Streamlit: http://localhost:8501
- FastAPI Docs: http://localhost:8000/docs

---

## 📊 Results & Model Comparison

7 models are trained and compared. Tree-based ensembles (XGBoost, LightGBM, RandomForest) typically perform best on this dataset.

| Model | R² Score | RMSE |
|-------|----------|------|
| LightGBM | ~0.95 | ~0.02 |
| XGBoost | ~0.94 | ~0.02 |
| Random Forest | ~0.93 | ~0.03 |
| Linear Regression | ~0.85 | ~0.06 |

---

## 🎥 Demo & Screenshots

*(Add your animated demo GIF here)*
<!-- ![Demo GIF](docs/demo.gif) -->

### 1. Streamlit Dashboard
*(Add your Streamlit UI screenshot here)*
<!-- ![Streamlit Dashboard](docs/streamlit_dashboard.png) -->

### 2. SHAP Explainability
*(Add your SHAP feature importance plot here)*
<!-- ![SHAP Plot](docs/shap_plot.png) -->

### 3. FastAPI Swagger Docs
*(Add your API documentation screenshot here)*
<!-- ![FastAPI Docs](docs/fastapi_docs.png) -->

---

## 👨‍💻 Author & Resume Section

**Admission Chance Prediction System (End-to-End ML Project)**
> Built a production-ready machine learning system using Python, Scikit-learn, XGBoost, Streamlit, FastAPI, and Docker. Implemented preprocessing pipelines, hyperparameter tuning, SHAP explainability, unit testing, deployment-ready architecture, and an interactive prediction dashboard.

---
*License: MIT*
