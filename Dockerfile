# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install system dependencies for psycopg
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy dependency files and install
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Pre-download HuggingFace models to avoid downloading on every startup
RUN uv run python -c "from transformers import AutoModel; AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"
# RUN uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('nomic-ai/nomic-embed-text-v1.5').save('/app/local_model')"

# Copy application code (app.py, src/, static/)
COPY app.py ./
COPY src ./src
COPY configs ./configs
COPY static ./static

EXPOSE 8000

# Run the app
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]