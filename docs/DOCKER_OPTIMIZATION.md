# Docker Build Optimization for CI/CD

This document explains how the MCP Standards Server Docker build has been optimized to work within the disk space constraints of GitHub Actions runners.

## Problem

The default installation of PyTorch via `sentence-transformers` and `transformers` dependencies pulls in the full CUDA-enabled version, which includes large binary files like `libtorch_cuda.so`. This causes disk space issues in CI/CD environments:

- Full PyTorch with CUDA: ~2-3GB
- CPU-only PyTorch: ~500MB

## Solution

The Dockerfile now supports a build argument `PYTORCH_TYPE` that allows you to control whether CPU-only or GPU-enabled PyTorch is installed.

### How It Works

1. **Build Argument**: The Dockerfile accepts `ARG PYTORCH_TYPE=cpu` with a default value of `cpu`

2. **Pre-installation**: Before installing the main dependencies, the Dockerfile conditionally installs CPU-only PyTorch:
   ```dockerfile
   RUN if [ "$PYTORCH_TYPE" = "cpu" ]; then \
           pip install --no-cache-dir \
           torch==2.5.1+cpu \
           torchvision==0.20.1+cpu \
           -f https://download.pytorch.org/whl/torch_stable.html; \
       fi
   ```

3. **Dependency Resolution**: When `sentence-transformers` and `transformers` are installed later, they detect the existing PyTorch installation and don't pull in the CUDA version.

## Usage

### For CI/CD (CPU-only)

The GitHub Actions workflow is configured to use CPU-only builds:

```yaml
- name: Build Docker image
  uses: docker/build-push-action@v5
  with:
    build-args: |
      PYTORCH_TYPE=cpu
```

### For Local Development

#### Option 1: Using docker-compose (Recommended)

```bash
# For CPU-only build (default)
docker-compose build

# For GPU build
PYTORCH_TYPE=cuda docker-compose build

# Or set in .env file
echo "PYTORCH_TYPE=cpu" >> .env
docker-compose build
```

#### Option 2: Using docker build directly

```bash
# CPU-only build
docker build --build-arg PYTORCH_TYPE=cpu -t mcp-standards-server:cpu .

# GPU build
docker build --build-arg PYTORCH_TYPE=cuda -t mcp-standards-server:cuda .

# Default (CPU)
docker build -t mcp-standards-server:latest .
```

## Environment Configuration

The `.env.example` file includes the configuration option:

```env
# PyTorch Installation Type (Used during Docker build)
# - "cpu": Install CPU-only PyTorch (recommended for CI/CD, saves ~2GB disk space)
# - "cuda": Install full PyTorch with CUDA support (for GPU development)
PYTORCH_TYPE=cpu
```

## Benefits

1. **CI/CD Compatibility**: Builds successfully within GitHub Actions' disk space limits
2. **Flexibility**: Developers can still use GPU acceleration locally
3. **Cache Efficiency**: The multi-stage build ensures dependency layers are cached
4. **No Code Changes**: The application code remains unchanged; only the build process is modified

## Performance Impact

For the MCP Standards Server use case:
- **Semantic Search**: CPU version 1.0.0
- **Model Loading**: Initial model loading might be slightly slower on CPU
- **Inference**: For small to medium batch sizes, CPU performance is acceptable

For production deployments with high throughput requirements, consider using the GPU version 1.0.0

## Troubleshooting

### Verifying PyTorch Installation

To check which version 1.0.0

```bash
docker run --rm mcp-standards-server:latest python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Disk Space Issues

If you still encounter disk space issues:

1. Clean Docker build cache: `docker builder prune -f`
2. Remove unused images: `docker image prune -a`
3. Use `--no-cache` flag: `docker build --no-cache --build-arg PYTORCH_TYPE=cpu .`

### Model Download Issues

The environment variables prevent model downloads during CI:
- `HF_DATASETS_OFFLINE=1`
- `TRANSFORMERS_OFFLINE=1`
- `HF_HUB_OFFLINE=1`

Ensure models are pre-cached or available in the mounted volumes.