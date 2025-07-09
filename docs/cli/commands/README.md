# Command Reference

This section provides detailed documentation for all MCP Standards Server CLI commands.

## Command Index

### Core Commands

- [sync](./sync.md) - Synchronize standards from repository
- [status](./status.md) - Show sync status and statistics
- [cache](./cache.md) - Manage local cache
- [config](./config.md) - Show or validate configuration
- [generate](./generate.md) - Generate standards from templates

### Planned Commands

The following commands are documented but not yet fully implemented in the CLI:

- [query](./query.md) - Query standards based on context (use Python API or MCP server)
- [validate](./validate.md) - Validate code against standards (available as generate subcommand)
- [serve](./serve.md) - Start MCP server (use `python -m src`)

## Global Options

These options are available for all commands:

### `-v, --verbose`
Enable verbose output for debugging.

```bash
mcp-standards -v sync
```

### `-c, --config <path>`
Specify a custom configuration file.

```bash
mcp-standards -c /path/to/config.yaml sync
```

### `--no-color`
Disable colored output (useful for CI/CD environments).

```bash
mcp-standards --no-color status
```

### `--json`
Output results in JSON format where applicable.

```bash
mcp-standards --json status
```

### `-h, --help`
Show help message for any command.

```bash
mcp-standards --help
mcp-standards sync --help
```

## Exit Codes

The CLI uses standard exit codes:

- `0`: Success
- `1`: General error
- `2`: Command line syntax error
- `3`: Configuration error
- `4`: Network/sync error
- `5`: Validation error

## Environment Variables

The following environment variables affect CLI behavior:

- `MCP_STANDARDS_CONFIG`: Default configuration file path
- `MCP_STANDARDS_DATA_DIR`: Data directory location
- `MCP_STANDARDS_CACHE_DIR`: Cache directory location
- `MCP_DISABLE_SEARCH`: Disable semantic search features
- `NO_COLOR`: Disable colored output (same as --no-color)
- `MCP_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)

## Configuration Precedence

Configuration is loaded in the following order (later sources override earlier ones):

1. Default configuration
2. System-wide configuration (`/etc/mcp-standards/config.yaml`)
3. User configuration (`~/.config/mcp-standards/config.yaml`)
4. Project configuration (`./.mcp-standards.yaml`)
5. Environment variables
6. Command-line options

## Common Patterns

### Automated Sync in CI/CD

```bash
# In CI/CD pipeline
mcp-standards sync --check || mcp-standards sync --force
```

### Project-Specific Configuration

```bash
# Use project-specific config
mcp-standards -c .mcp-standards.yaml sync
```

### JSON Output for Scripting

```bash
# Get status as JSON for processing
STATUS=$(mcp-standards --json status)
echo $STATUS | jq '.total_files'
```

### Verbose Debugging

```bash
# Debug sync issues
MCP_LOG_LEVEL=DEBUG mcp-standards -v sync
```