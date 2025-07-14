# Multi-stage build for MCP Standards Server
FROM python:3.12-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy only files needed for dependency installation first (better caching)
COPY pyproject.toml .
COPY README.md .

# Create minimal src structure for editable install
RUN mkdir -p src && touch src/__init__.py

# Install Python dependencies using pyproject.toml (this layer will be cached)
RUN pip install --no-cache-dir .

# Now copy the actual source code (this will invalidate cache less frequently)
COPY src/ ./src/

# Production stage
FROM python:3.12-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r mcpuser && useradd -r -g mcpuser mcpuser

# Create app directory and set permissions
WORKDIR /app
RUN mkdir -p /app/data /app/logs && \
    chown -R mcpuser:mcpuser /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=mcpuser:mcpuser . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')" || true

# Switch to non-root user
USER mcpuser

# Expose ports
EXPOSE 8080 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command (can be overridden in docker-compose)
CMD ["python", "-m", "src.main"]