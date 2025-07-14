#!/bin/bash

# Development startup script for MCP Standards Server
# This script provides a quick way to start the development environment

set -e

echo "🚀 Starting MCP Standards Server Development Environment"

# Create necessary directories
mkdir -p logs data/standards

# Check if Docker is available
if command -v docker &> /dev/null && command -v docker-compose &> /dev/null; then
    echo "📦 Docker detected - using Docker Compose"
    
    # Start dependencies only (Redis, ChromaDB)
    docker-compose up -d redis chromadb
    
    # Wait for services to be ready
    echo "⏳ Waiting for services to be ready..."
    sleep 5
    
    # Check if services are healthy
    echo "🔍 Checking service health..."
    docker-compose ps
    
    export REDIS_URL="redis://localhost:6379"
    export CHROMADB_URL="http://localhost:8000"
    
    echo "✅ Dependencies started successfully"
else
    echo "⚠️  Docker not found - running without containerized dependencies"
    echo "   You may need to install Redis and ChromaDB manually"
fi

# Install Python dependencies if needed
if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "📦 Installing/updating Python dependencies..."
source venv/bin/activate
pip install -r requirements.txt

# Download NLTK data if needed
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)" || true

# Set development environment variables
export LOG_LEVEL=DEBUG
export HTTP_HOST=0.0.0.0
export HTTP_PORT=8080
export DATA_DIR=data
export HTTP_ONLY=true  # Run HTTP server only for development

echo "🌟 Starting MCP Standards Server (HTTP mode for development)"
echo "   Health checks: http://localhost:8080/health"
echo "   Standards API: http://localhost:8080/api/standards"
echo "   Metrics: http://localhost:8080/metrics"
echo "   Service info: http://localhost:8080/info"
echo ""
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the server
python -m src.main