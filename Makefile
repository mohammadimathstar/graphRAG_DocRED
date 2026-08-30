# Makefile for M5 Forecasting Project

.PHONY: install test lint format clean clean-mlflow run up-db down-db

# Create virtual environment and install dependencies
install:
	uv venv
	uv sync

# Run all unit tests
test:
	uv pip install pytest && PYTHONPATH=. pytest tests -v

# Check code formatting and linting (does NOT modify files)
lint:
	uv pip install ruff && ruff check src

# Auto-fix common issues (formatting, imports, etc.)
lint-fix:
	ruff check src scripts --fix

# Format code (like black/isort)
format:
	ruff format src scripts

# Clean pycache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Load database
up-db:
	docker compose up -d

down-db:
	docker compose down

# Run the pipeline
run:
	chmod +x run_deployment.sh
	./run_deployment.sh
