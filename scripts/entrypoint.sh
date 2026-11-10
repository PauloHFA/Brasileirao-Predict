#!/bin/sh
set -e

echo "=== Initializing Brasileirão Predictor Database ==="
python -m src.database

echo "=== Starting FastAPI Server ==="
exec uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
