#!/bin/bash
set -e

# Run MLflow database upgrade
echo "Running MLflow database upgrade..."
mlflow db upgrade sqlite:///mlflow.db || echo "MLflow upgrade failed or not needed, continuing..."

# Start the application
echo "Starting application..."
uvicorn Backend.main:app --host 0.0.0.0 --port 8000 --workers 4
