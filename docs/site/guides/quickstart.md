# Quick Start Guide

Get up and running with MCP Standards Server in minutes.

## Prerequisites

- Python 3.11 or higher
- pip package manager
- Git (for standards synchronization)

## Installation

```bash
# Install the package
pip install mcp-standards-server

# Verify installation
mcp-standards --version
```

## Initial Setup

### 1. Initialize Configuration

```bash
# Create default configuration
mcp-standards config --init

# This creates ~/.mcp-standards/config.yaml
```

### 2. Sync Standards

```bash
# Download latest standards from repository
mcp-standards sync

# Check sync status
mcp-standards status
```

### 3. Validate Your First Project

```bash
# Navigate to your project
cd /path/to/your/project

# Run validation
mcp-standards validate .

# Get applicable standards for your project
mcp-standards query applicable
```

## Start MCP Server

```bash
# Start the MCP server (default port 8080)
mcp-standards serve

# Start on specific port
mcp-standards serve --port 8081

# Start with verbose logging
mcp-standards serve --verbose
```

## Test MCP Integration

```bash
# Test MCP tools functionality
curl -X POST http://localhost:8080/tools/get_applicable_standards \
  -H "Content-Type: application/json" \
  -d '{"project_type": "web_application", "framework": "react"}'
```

## Next Steps

- [Configuration Guide](./configuration.md) - Customize your setup
- [IDE Integration](./ide-integration.md) - Integrate with your editor
- [CI/CD Integration](./cicd-integration.md) - Add to your pipeline
- [CLI Commands Reference](../reference/cli-commands.md) - Complete command reference

## Common First Steps

### For Web Applications
```bash
# Get web app standards
mcp-standards query applicable --project-type web_application

# Validate with specific framework
mcp-standards validate . --framework react
```

### For API Projects
```bash
# Get API standards
mcp-standards query applicable --project-type api

# Validate API endpoints
mcp-standards validate ./src/api --standard api-design
```

### For Python Projects
```bash
# Get Python-specific standards
mcp-standards query applicable --language python

# Validate Python code
mcp-standards validate . --language python --fix
```

## Getting Help

- Use `mcp-standards --help` for command overview
- Use `mcp-standards <command> --help` for specific command help
- Check [Troubleshooting Guide](../reference/troubleshooting.md) for common issues