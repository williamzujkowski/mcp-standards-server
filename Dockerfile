# Multi-stage build for security and efficiency
# @nist-controls: CM-2, CM-7, SC-28
# @evidence: Minimal container with security hardening

# Build stage
FROM python:3.13-slim AS builder

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

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files
COPY pyproject.toml ./
COPY README.md ./

# Create virtual environment and install dependencies
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN uv pip install -e .

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

# Copy virtual environment from builder
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /build /app

# Set up environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

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

# Expose port (MCP servers typically don't expose HTTP ports, but keeping for compatibility)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run the MCP server
CMD ["mcp-standards-server"]