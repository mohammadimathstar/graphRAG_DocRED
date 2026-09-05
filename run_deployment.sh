#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting database infrastructure..."
docker compose up -d

echo "Waiting for database to be ready..."
sleep 10

echo "Running offline pipeline (Extraction & Evaluation)..."
# Uncomment the lines below to run the full pipeline automatically
# uv run python scripts/00_ingestion.py
# uv run python scripts/01_extract_kg.py
# uv run python scripts/02_generate_qa.py
# uv run python scripts/04_evaluate_retrieval.py

echo "Starting FastAPI server..."
uv run uvicorn app:app --reload --port 8000

