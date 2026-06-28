FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (lightweight API-only set for Render free tier)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy project files
COPY . .

# Render and other cloud providers inject a PORT environment variable
ENV PORT=8000
EXPOSE $PORT

# Default command runs FastAPI.
CMD uvicorn api.main:app --host 0.0.0.0 --port $PORT
