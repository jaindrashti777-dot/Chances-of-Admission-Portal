# ==============================================================================
# Chance of Admission Prediction (India) — Makefile
# ==============================================================================
# Usage: make <target>
# ==============================================================================

.PHONY: install generate train app api test lint format docker-up docker-down clean help

# --- Default ------------------------------------------------------------------
help:
	@echo.
	@echo   Chance of Admission Prediction - Available Commands
	@echo   ==================================================
	@echo   make install       Create venv and install dependencies
	@echo   make generate      Generate synthetic dataset
	@echo   make train         Run the full training pipeline
	@echo   make app           Launch Streamlit web application
	@echo   make api           Launch FastAPI server
	@echo   make test          Run all unit tests
	@echo   make lint          Run flake8 linter
	@echo   make format        Format code with black + isort
	@echo   make docker-up     Start Docker containers
	@echo   make docker-down   Stop Docker containers
	@echo   make clean         Remove caches, logs, pycache
	@echo.

# --- Setup --------------------------------------------------------------------
install:
	python -m venv .venv
	.venv\Scripts\pip install --upgrade pip
	.venv\Scripts\pip install -r requirements.txt

# --- Data ---------------------------------------------------------------------
generate:
	.venv\Scripts\python data\raw\generate_admission_data.py

# --- Training -----------------------------------------------------------------
train:
	.venv\Scripts\python -m src.train

# --- Applications -------------------------------------------------------------
app:
	.venv\Scripts\streamlit run app\streamlit_app.py

api:
	.venv\Scripts\uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# --- Testing ------------------------------------------------------------------
test:
	.venv\Scripts\pytest tests\ -v --tb=short

# --- Code Quality -------------------------------------------------------------
lint:
	.venv\Scripts\flake8 src\ api\ app\ tests\

format:
	.venv\Scripts\black src\ api\ app\ tests\
	.venv\Scripts\isort src\ api\ app\ tests\

# --- Docker -------------------------------------------------------------------
docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

# --- Cleanup ------------------------------------------------------------------
clean:
	if exist __pycache__ rmdir /s /q __pycache__
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist logs\*.log del /q logs\*.log
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rmdir /s /q "%%d"
