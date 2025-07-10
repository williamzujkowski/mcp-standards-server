#!/bin/bash

# MCP Standards Server Web UI Start Script

echo "🚀 Starting MCP Standards Server Web UI..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from example..."
    cp .env.example .env
    
    # Generate a secure secret key
    SECRET_KEY=$(openssl rand -hex 32)
    sed -i "s/your-secret-key-here-change-in-production/$SECRET_KEY/g" .env
    echo "✅ Generated secure SECRET_KEY"
fi

# Build and start services
echo "🔨 Building Docker images..."
docker compose build

echo "🎯 Starting services..."
docker compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 5

# Check if services are running
if docker compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo ""
    echo "📌 Access the application at:"
    echo "   - Frontend: http://localhost"
    echo "   - Backend API: http://localhost:8000"
    echo "   - API Documentation: http://localhost:8000/docs"
    echo ""
    echo "📊 View logs with: docker compose logs -f"
    echo "🛑 Stop services with: docker compose down"
else
    echo "❌ Failed to start services. Check logs with: docker compose logs"
    exit 1
fi