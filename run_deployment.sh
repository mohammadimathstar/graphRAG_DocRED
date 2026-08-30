#!/bin/bash

# Download and Preprocess dataset 
# uv run python scripts/00_ingestion.py

# Initialize Postgresql DB
docker compose up -d

sleep 10

# Extract KG from documents
# uv run python scripts/01_extract_kg.py

# # Generate QA (for evaluating Retrieval)
# uv run python scripts/02_generate_qa.py

# # Evaluate Retrieval
# uv run python scripts/04_evaluate_retrieval.py

# Run FastAPI
uv run uvicorn app:app --reload --port 8000


# # Run deployment command and automatically answer 'n' to prompt
# echo "n" | prefect deploy monitoring_pipeline.py:monitor_model -n mydeployment -p mypool


# # Keep container alive to keep services running
# tail -f /dev/null
