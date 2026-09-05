.PHONY: install lint format clean up-db down-db pipeline run

# Install dependencies
install:
	uv sync

# Run unit tests
# test:
# 	uv pip install pytest && PYTHONPATH=. pytest tests -v

# Check code formatting and linting (does NOT modify files)
lint:
	uv run ruff check src scripts app.py

# Auto-fix common issues (formatting, imports, etc.)
lint-fix:
	uv run ruff check src scripts app.py --fix

# Format code (like black/isort)
format:
	uv run ruff format src scripts app.py

# Clean pycache and temp files
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

# Start Docker containers (Postgres, pgAdmin, Grafana)
up-db:
	docker compose up -d

# Stop Docker containers
down-db:
	docker compose down

# Run the offline data pipeline (Extract, Evaluate, Generate QA)
pipeline:
	uv run python scripts/00_extract_data.py
	uv run python scripts/01_extract_kg.py
	uv run python scripts/02_generate_qa.py
	uv run python scripts/04_evaluate_retrieval.py

# Run the deployment script (Starts DB, runs pipeline, starts API)
run:
	chmod +x run_deployment.sh
	./run_deployment.sh
