# MCP Standards Server CLI Documentation

The MCP Standards Server CLI (`mcp-standards`) provides a comprehensive command-line interface for managing, syncing, and querying development standards. This documentation covers all aspects of using the CLI effectively.

## Quick Links

- [Command Reference](./commands/README.md) - Complete reference for all CLI commands
- [Getting Started](./tutorials/getting-started.md) - Quick start guide for new users
- [Configuration Guide](./configuration.md) - Detailed configuration options
- [Troubleshooting](./troubleshooting.md) - Common issues and solutions
- [Examples](./examples/README.md) - Real-world usage examples

## Overview

The MCP Standards Server CLI is designed to:

- **Sync Standards**: Download and cache standards from GitHub repositories
- **Query Standards**: Search and retrieve applicable standards based on project context
- **Manage Cache**: Control local caching of standards files
- **Validate Code**: Check code against defined standards
- **Integration**: Work seamlessly with IDEs, CI/CD pipelines, and development workflows

## Installation

### Using pip

```bash
pip install mcp-standards-server
```

### Using pipx (recommended for CLI tools)

```bash
pipx install mcp-standards-server
```

### From Source

```bash
git clone https://github.com/williamzujkowski/mcp-standards-server
cd mcp-standards-server
pip install -e .
```

## Basic Usage

```bash
# Sync standards from repository
mcp-standards sync

# Check sync status
mcp-standards status

# Get applicable standards for a project
mcp-standards query --project-type web --framework react

# Clear cache
mcp-standards cache --clear
```

## Command Structure

The CLI follows a consistent command structure:

```
mcp-standards [global-options] <command> [command-options]
```

### Global Options

- `-v, --verbose`: Enable verbose output
- `-c, --config`: Specify configuration file path
- `--no-color`: Disable colored output
- `--json`: Output in JSON format (where applicable)
- `-h, --help`: Show help message

### Available Commands

- `sync`: Synchronize standards from repository
- `status`: Show sync status and statistics
- `cache`: Manage local cache
- `config`: Show or validate configuration
- `query`: Query standards based on context
- `validate`: Validate code against standards
- `serve`: Start MCP server

## Next Steps

- Read the [Getting Started Guide](./tutorials/getting-started.md)
- Explore [Command Reference](./commands/README.md)
- Learn about [Configuration Options](./configuration.md)
- See [Integration Examples](./examples/README.md)