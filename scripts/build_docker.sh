#!/bin/bash
# Docker build script with PyTorch optimization options

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
BUILD_TYPE=${1:-cpu}
TAG_SUFFIX=""

# Print usage
usage() {
    echo "Usage: $0 [cpu|cuda|help]"
    echo ""
    echo "Options:"
    echo "  cpu   - Build with CPU-only PyTorch (default, saves ~2GB disk space)"
    echo "  cuda  - Build with CUDA-enabled PyTorch (for GPU support)"
    echo "  help  - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Build CPU version (default)"
    echo "  $0 cpu          # Build CPU version explicitly"
    echo "  $0 cuda         # Build GPU version"
    echo ""
    echo "Environment variables:"
    echo "  DOCKER_BUILDKIT=1  # Enable BuildKit for better caching (recommended)"
}

# Check command line argument
case "$BUILD_TYPE" in
    cpu)
        echo -e "${GREEN}Building CPU-only version (optimized for CI/CD)...${NC}"
        PYTORCH_TYPE="cpu"
        TAG_SUFFIX="-cpu"
        ;;
    cuda|gpu)
        echo -e "${YELLOW}Building CUDA-enabled version (for GPU support)...${NC}"
        PYTORCH_TYPE="cuda"
        TAG_SUFFIX="-cuda"
        ;;
    help|--help|-h)
        usage
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid option: $BUILD_TYPE${NC}"
        usage
        exit 1
        ;;
esac

# Check if we're in the right directory
if [ ! -f "Dockerfile" ]; then
    echo -e "${RED}Error: Dockerfile not found. Please run this script from the project root.${NC}"
    exit 1
fi

# Clean up old containers if they exist
echo "Cleaning up any existing containers..."
docker-compose down 2>/dev/null || true

# Set build timestamp
BUILD_TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Build the Docker image
echo -e "${GREEN}Starting Docker build...${NC}"
echo "PyTorch Type: $PYTORCH_TYPE"
echo "Tag: mcp-standards-server:latest${TAG_SUFFIX}"
echo ""

# Enable BuildKit for better caching
export DOCKER_BUILDKIT=1

# Build command
docker build \
    --build-arg PYTORCH_TYPE="$PYTORCH_TYPE" \
    --tag "mcp-standards-server:latest${TAG_SUFFIX}" \
    --tag "mcp-standards-server:${BUILD_TIMESTAMP}${TAG_SUFFIX}" \
    --progress=plain \
    .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Build completed successfully!${NC}"
    echo ""
    echo "Tagged as:"
    echo "  - mcp-standards-server:latest${TAG_SUFFIX}"
    echo "  - mcp-standards-server:${BUILD_TIMESTAMP}${TAG_SUFFIX}"
    echo ""
    
    # Show image size
    echo "Image size:"
    docker images "mcp-standards-server:latest${TAG_SUFFIX}" --format "table {{.Repository}}:{{.Tag}}\t{{.Size}}"
    echo ""
    
    # Test PyTorch installation
    echo "Testing PyTorch installation..."
    docker run --rm "mcp-standards-server:latest${TAG_SUFFIX}" python -c "
import torch
import sys
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
"
    echo ""
    echo -e "${GREEN}To run the container:${NC}"
    echo "  docker-compose up -d"
    echo ""
    echo -e "${GREEN}Or directly:${NC}"
    echo "  docker run -p 8080:8080 mcp-standards-server:latest${TAG_SUFFIX}"
else
    echo -e "${RED}✗ Build failed!${NC}"
    exit 1
fi