# Development Setup Guide

This guide helps you set up your development environment for contributing to MCP Standards Server.

## Prerequisites

- Python 3.11 or higher
- Git
- Redis (for caching features)
- Node.js 18+ (for web UI development)

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in development mode with all extras
pip install -e ".[dev,test,performance,visualization,full]"

# Install pre-commit hooks
pre-commit install
```

### 4. Set Up Redis

```bash
# Ubuntu/Debian
sudo apt-get install redis-server

# macOS
brew install redis

# Start Redis
redis-server
```

### 5. Environment Variables

Create a `.env` file in the project root:

```bash
# .env
MCP_ENV=development
REDIS_URL=redis://localhost:6379
LOG_LEVEL=DEBUG
PYTHONPATH=${PWD}
```

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow the coding standards:
- Use type hints
- Write docstrings
- Add unit tests
- Update documentation

### 3. Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_your_feature.py

# Run with coverage
pytest --cov=src --cov-report=html
```

### 4. Run Linters

```bash
# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/

# Or run all checks
make lint
```

### 5. Test Locally

```bash
# Start the MCP server
python -m src

# In another terminal, test with CLI
mcp-standards validate examples/

# Test the web UI
cd web && npm start
```

## IDE Setup

### VS Code

Recommended extensions:
- Python
- Pylance
- Black Formatter
- GitLens

Settings (`.vscode/settings.json`):
```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "python.testing.pytestEnabled": true
}
```

### PyCharm

1. Set Python interpreter to virtual environment
2. Enable Django support if working on web components
3. Configure code style to use Black
4. Set up pytest as test runner

## Debugging

### Debug MCP Server

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use VS Code/PyCharm debugger with launch configurations
```

### Debug Configuration (VS Code)

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug MCP Server",
            "type": "python",
            "request": "launch",
            "module": "src",
            "env": {
                "PYTHONPATH": "${workspaceFolder}",
                "MCP_ENV": "development"
            }
        }
    ]
}
```

## Common Issues

### Import Errors

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=${PWD}

# Or install in editable mode
pip install -e .
```

### Redis Connection

```bash
# Check Redis is running
redis-cli ping

# Should return: PONG
```

### Test Failures

```bash
# Clear test cache
find . -type d -name __pycache__ -exec rm -r {} +
pytest --cache-clear
```

## Next Steps

- Read [Contributing Standards](./standards.md)
- Learn about [Writing Validators](./validators.md)
- Understand [Testing Guidelines](./testing.md)

## Getting Help

- Check existing issues on GitHub
- Join our Discord community
- Ask questions in discussions

Happy coding! 🚀