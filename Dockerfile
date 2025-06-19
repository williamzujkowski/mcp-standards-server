# Multi-stage build for security and efficiency
# @nist-controls: CM-2, CM-7, SC-28
# @evidence: Minimal container with security hardening

# Build stage
FROM python:3.13-slim as builder

# Security: Run as non-root user
# @nist-controls: AC-6
# @evidence: Least privilege principle
RUN useradd -m -u 1000 appuser

WORKDIR /build

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry \
    POETRY_VERSION=1.6.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies
RUN poetry install --no-dev --no-root

# Copy application code
COPY src/ ./src/
COPY templates/ ./templates/
COPY data/ ./data/

# Production stage
FROM python:3.13-slim

# Security hardening
# @nist-controls: CM-7, SC-2
# @evidence: Minimal attack surface
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 appuser

# Copy from builder
COPY --from=builder --chown=appuser:appuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=appuser:appuser /build /app

WORKDIR /app

# Security: Drop privileges
# @nist-controls: AC-6
# @evidence: Non-root container execution
USER appuser

# Health check
# @nist-controls: SI-13, AU-5
# @evidence: Container health monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/bin:$PATH"

# Run the application
CMD ["uvicorn", "src.core.mcp.server:app", "--host", "0.0.0.0", "--port", "8000"]