# CLI Commands Reference

Complete reference for all MCP Standards Server CLI commands.

## Global Options

All commands support these global options:

- `--help` - Show help message
- `--version` - Show version 1.0.0
- `--verbose`, `-v` - Enable verbose output
- `--quiet`, `-q` - Suppress output
- `--config CONFIG` - Use specific configuration file

## Commands Overview

### Core Commands

- [`serve`](#serve) - Start the MCP server
- [`validate`](#validate) - Validate code against standards
- [`query`](#query) - Query standards database
- [`sync`](#sync) - Synchronize standards from repository

### Configuration Commands

- [`config`](#config) - Manage configuration
- [`status`](#status) - Show system status
- [`cache`](#cache) - Manage cache

## Command Details

### serve

Start the MCP Standards Server.

```bash
mcp-standards serve [OPTIONS]
```

**Options:**
- `--host HOST` - Server host (default: 0.0.0.0)
- `--port PORT` - Server port (default: 8080)
- `--reload` - Enable auto-reload on code changes
- `--workers N` - Number of worker processes

**Examples:**
```bash
# Start server on default port
mcp-standards serve

# Start on specific port
mcp-standards serve --port 8081

# Development mode with auto-reload
mcp-standards serve --reload
```

### validate

Validate code against development standards.

```bash
mcp-standards validate [PATH] [OPTIONS]
```

**Arguments:**
- `PATH` - Path to validate (default: current directory)

**Options:**
- `--standard STANDARD` - Use specific standard
- `--language LANG` - Target language
- `--framework FRAMEWORK` - Target framework
- `--fix` - Auto-fix issues where possible
- `--format FORMAT` - Output format (json, junit, sarif, text)
- `--severity LEVEL` - Minimum severity (error, warning, info)
- `--output FILE` - Output file

**Examples:**
```bash
# Validate current directory
mcp-standards validate

# Validate specific path with auto-fix
mcp-standards validate ./src --fix

# Validate Python code with specific standard
mcp-standards validate --language python --standard pep8

# Output SARIF format for CI integration
mcp-standards validate --format sarif --output results.sarif
```

### query

Query the standards database.

```bash
mcp-standards query [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**
- `applicable` - Find applicable standards
- `search` - Search standards content
- `list` - List all standards
- `show` - Show specific standard

**Options:**
- `--project-type TYPE` - Project type filter
- `--language LANG` - Language filter
- `--framework FRAMEWORK` - Framework filter
- `--tags TAGS` - Tag filters
- `--format FORMAT` - Output format

**Examples:**
```bash
# Find applicable standards for web app
mcp-standards query applicable --project-type web_application

# Search for accessibility standards
mcp-standards query search "accessibility"

# Show specific standard
mcp-standards query show react-patterns

# List all Python standards
mcp-standards query list --language python
```

### sync

Synchronize standards from remote repository.

```bash
mcp-standards sync [OPTIONS]
```

**Options:**
- `--force` - Force full resync
- `--url URL` - Custom repository URL
- `--branch BRANCH` - Specific branch
- `--no-cache` - Skip cache update

**Examples:**
```bash
# Standard sync
mcp-standards sync

# Force full resync
mcp-standards sync --force

# Sync from custom repository
mcp-standards sync --url https://github.com/custom/standards
```

### config

Manage configuration settings.

```bash
mcp-standards config [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**
- `init` - Initialize default configuration
- `show` - Show current configuration
- `set` - Set configuration value
- `get` - Get configuration value
- `validate` - Validate configuration
- `test` - Test configuration

**Examples:**
```bash
# Initialize configuration
mcp-standards config init

# Show current config
mcp-standards config show

# Set configuration value
mcp-standards config set server.port 8081

# Validate configuration
mcp-standards config validate
```

### status

Show system status and health.

```bash
mcp-standards status [OPTIONS]
```

**Options:**
- `--detailed` - Show detailed status
- `--json` - JSON output format

**Examples:**
```bash
# Basic status
mcp-standards status

# Detailed status
mcp-standards status --detailed

# JSON output for scripts
mcp-standards status --json
```

### cache

Manage cache operations.

```bash
mcp-standards cache [SUBCOMMAND] [OPTIONS]
```

**Subcommands:**
- `clear` - Clear cache
- `status` - Show cache status
- `warm` - Warm cache
- `stats` - Show cache statistics

**Examples:**
```bash
# Clear all cache
mcp-standards cache clear

# Show cache status
mcp-standards cache status

# Warm cache with common standards
mcp-standards cache warm
```

## Exit Codes

- `0` - Success
- `1` - General error
- `2` - Invalid arguments
- `3` - Configuration error
- `4` - Network error
- `5` - Validation failures found

## Environment Variables

- `MCP_CONFIG_FILE` - Configuration file path
- `MCP_CACHE_DIR` - Cache directory
- `MCP_LOG_LEVEL` - Logging level
- `MCP_NO_COLOR` - Disable colored output

## Shell Completion

Enable shell completion:

```bash
# Bash
eval "$(mcp-standards completion bash)"

# Zsh
eval "$(mcp-standards completion zsh)"

# Fish
mcp-standards completion fish | source
```

## Tips and Tricks

### Batch Validation

```bash
# Validate multiple directories
find . -name "*.py" -exec dirname {} \; | sort -u | xargs -I {} mcp-standards validate {}
```

### CI Integration

```bash
# GitHub Actions example
mcp-standards validate --format sarif --output results.sarif
gh api repos/:owner/:repo/code-scanning/sarifs --method POST --input results.sarif
```

### Custom Standards Development

```bash
# Test custom standard during development
mcp-standards validate --standard ./custom-standard.yaml .
```
