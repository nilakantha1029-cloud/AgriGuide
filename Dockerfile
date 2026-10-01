# AgriGuide - single-container image (FastAPI serves both the API and the
# static frontend, so there is only one service to run).
#
# Build:  docker build -t agriguide .
# Run:    docker run -p 8000:8000 --env-file .env agriguide
# (docker-compose.yml wraps this same build+run for convenience.)

FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

# ---------------------------------------------------------------------------
# Layer 1: Python dependencies. Only rebuilt when requirements.txt changes.
# CPU-only torch wheel keeps the image several GB smaller than the default.
# ---------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ---------------------------------------------------------------------------
# Layer 2: Build the search index and pre-download the reranker model, so a
# running container never needs internet access except for Groq API calls.
# This layer only reruns when the PDFs or the ingest/config code change - it
# is deliberately copied BEFORE the rest of src/, so editing api.py,
# generator.py, pipeline.py etc. later never forces re-embedding the corpus.
# ---------------------------------------------------------------------------
COPY src/__init__.py src/config.py src/ingest.py src/
COPY data/ data/
RUN python -m src.ingest \
    && python -c "from sentence_transformers import CrossEncoder; from src import config; CrossEncoder(config.RERANK_MODEL)"

# ---------------------------------------------------------------------------
# Layer 3: application code. Changes here are cheap to rebuild.
# ---------------------------------------------------------------------------
COPY src/ src/
COPY frontend/ frontend/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request as u; exit(0 if u.urlopen('http://localhost:8000/api/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
