# Multi-stage Docker build for macOS/ARM64 Development
# For Linux with NVIDIA GPU, use Dockerfile.cuda instead
FROM python:3.10-slim-bullseye AS base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .

# Install dependencies without GPU packages
RUN pip3 install --upgrade pip && \
    # Install core dependencies first
    pip3 install \
        psycopg2-binary==2.9.9 \
        pgvector==0.2.4 \
        sqlalchemy==2.0.23 \
        alembic==1.13.0 \
        openai==1.7.2 \
        anthropic==0.8.1 \
        fastapi==0.109.0 \
        uvicorn[standard]==0.25.0 \
        pydantic==2.5.3 \
        pydantic-settings==2.1.0 \
        pyyaml==6.0.1 \
        python-dotenv==1.0.0 \
        celery==5.3.4 \
        redis==5.0.1 \
        prometheus-client==0.19.0 \
        python-json-logger==2.0.7 \
        structlog==24.1.0 \
        tqdm==4.66.1 \
        rich==13.7.0 \
        click==8.1.7 \
        tenacity==8.2.3 && \
    # Install ML dependencies (CPU-only versions)
    pip3 install \
        numpy==1.26.3 \
        scipy==1.11.4 && \
    pip3 install \
        torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install \
        transformers==4.36.2 \
        sentence-transformers==2.3.1 \
        datasets==2.16.0 \
        accelerate==0.25.0 \
        peft==0.7.1 && \
    # Install testing dependencies
    pip3 install \
        pytest==7.4.4 \
        pytest-asyncio==0.23.3 \
        pytest-cov==4.1.0 \
        pytest-mock==3.12.0 \
        httpx==0.26.0

# Copy application code
COPY src/ ./src/
COPY config.yaml .
COPY .env.example .env

# Create necessary directories
RUN mkdir -p /app/data/{conversations,models/{adapters,checkpoints},backups,embeddings_cache} \
    /app/logs \
    /app/models/cache

# Set permissions
RUN chmod -R 755 /app

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# Development stage
FROM base AS development

# Install development dependencies
RUN pip3 install \
    ipython==8.19.0 \
    black==23.12.1 \
    flake8==7.0.0 \
    mypy==1.8.0 \
    isort==5.13.2

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]


# Production stage
FROM base AS production

# Run as non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
