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

### From Source (Currently Available)

```bash
git clone https://github.com/williamzujkowski/mcp-standards-server
cd mcp-standards-server
pip install -e .
```

### Future Installation Methods

Once published to PyPI:
```bash
pip install mcp-standards-server
# or
pipx install mcp-standards-server
```

## Basic Usage

```bash
# Sync standards from repository
mcp-standards sync

# Check sync status
mcp-standards status

# Generate a new standard from template
mcp-standards generate --template technical --title "GraphQL Standards"

# Clear cache
mcp-standards cache --clear

# List available templates
mcp-standards generate list-templates
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
- `generate`: Generate standards from templates
  - `list-templates`: List available templates
  - `template-info`: Get template information
  - `customize`: Create custom template
  - `validate`: Validate existing standard

### Additional Tools

The following tools are available through separate entry points:

- **MCP Server**: `python -m src`
- **Web UI**: `python -m src.web` (if web UI is implemented)
- **Query Tool**: Use the Python API or MCP server for querying standards

## Next Steps

- Read the [Getting Started Guide](./tutorials/getting-started.md)
- Explore [Command Reference](./commands/README.md)
- Learn about [Configuration Options](./configuration.md)
- See [Integration Examples](./examples/README.md)