# config Command

Show, validate, and manage MCP Standards Server configuration.

## Synopsis

```bash
mcp-standards config [options]
```

## Description

The `config` command helps you view, validate, and manage configuration files for the MCP Standards Server. It supports multiple configuration formats and provides tools for migration and validation.

## Options

### `--show`
Display current configuration.

```bash
mcp-standards config --show
```

### `--validate`
Validate configuration file syntax and values.

```bash
mcp-standards config --validate
```

### `--init`
Initialize a new configuration file with defaults.

```bash
mcp-standards config --init
```

### `--edit`
Open configuration in default editor.

```bash
mcp-standards config --edit
```

### `--get <key>`
Get a specific configuration value.

```bash
mcp-standards config --get repository.owner
```

### `--set <key> <value>`
Set a specific configuration value.

```bash
mcp-standards config --set sync.cache_ttl_hours 48
```

### `--migrate`
Migrate configuration from older format.

```bash
mcp-standards config --migrate
```

### `--schema`
Show configuration schema.

```bash
mcp-standards config --schema
```

## Examples

### Show Configuration

```bash
mcp-standards config --show
```

Output:
```yaml
# Current configuration from: /home/user/.config/mcp-standards/config.yaml

repository:
  owner: williamzujkowski
  repo: standards
  branch: main
  path: standards
  auth:
    type: none  # or 'token', 'app'

sync:
  cache_ttl_hours: 24
  parallel_downloads: 5
  retry_attempts: 3
  timeout_seconds: 30
  include_patterns:
    - "*.yaml"
    - "*.md"
  exclude_patterns:
    - "*.draft.*"
    - ".git*"

cache:
  directory: ~/.cache/mcp-standards
  max_size_mb: 500
  compression: true
  auto_cleanup:
    enabled: true
    threshold_percent: 80

search:
  enabled: true
  model: sentence-transformers/all-MiniLM-L6-v2
  index_on_sync: true
  cache_embeddings: true

server:
  host: localhost
  port: 3000
  log_level: info
  token_optimization:
    enabled: true
    default_budget: 8000
    model_type: gpt-4

validation:
  enabled: true
  strict_mode: false
  custom_rules: []
```

### Validate Configuration

```bash
mcp-standards config --validate
```

Output:
```
Validating configuration file: /home/user/.config/mcp-standards/config.yaml

✓ File exists and is readable
✓ YAML syntax is valid
✓ Required fields present
✓ Repository configuration valid
✓ Sync settings within allowed ranges
✓ Cache directory writable
⚠ Authentication not configured (using anonymous access)

Configuration is valid with 1 warning

Suggestions:
- Consider configuring GitHub authentication for higher rate limits
- Run 'mcp-standards config --set repository.auth.type token' to configure
```

### Initialize Configuration

```bash
# Create default configuration
mcp-standards config --init
```

Output:
```
Creating configuration file: /home/user/.config/mcp-standards/config.yaml

? Repository owner: williamzujkowski
? Repository name: standards
? Branch: main
? Path in repository: standards
? Cache directory: (~/.cache/mcp-standards) 
? Configure GitHub authentication? No
? Enable semantic search? Yes

Configuration file created successfully!

Next steps:
- Review configuration: mcp-standards config --show
- Test sync: mcp-standards sync --check
- Start server: mcp-standards serve
```

### Get/Set Configuration Values

```bash
# Get a specific value
mcp-standards config --get sync.cache_ttl_hours
24

# Set a value
mcp-standards config --set sync.cache_ttl_hours 48
Configuration updated: sync.cache_ttl_hours = 48

# Get nested values
mcp-standards config --get repository.owner
williamzujkowski

# Set complex values
mcp-standards config --set sync.include_patterns '["*.yaml", "*.json", "*.md"]'
Configuration updated: sync.include_patterns = ["*.yaml", "*.json", "*.md"]
```

### Edit Configuration

```bash
# Open in default editor
mcp-standards config --edit

# Opens $EDITOR or falls back to nano/vi
```

### Show Configuration Schema

```bash
mcp-standards config --schema
```

Output:
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "MCP Standards Server Configuration",
  "type": "object",
  "required": ["repository", "sync", "cache"],
  "properties": {
    "repository": {
      "type": "object",
      "required": ["owner", "repo", "branch", "path"],
      "properties": {
        "owner": {
          "type": "string",
          "description": "GitHub repository owner"
        },
        "repo": {
          "type": "string",
          "description": "GitHub repository name"
        },
        "branch": {
          "type": "string",
          "default": "main",
          "description": "Branch to sync from"
        },
        "path": {
          "type": "string",
          "description": "Path within repository"
        },
        "auth": {
          "type": "object",
          "properties": {
            "type": {
              "enum": ["none", "token", "app"],
              "default": "none"
            },
            "token": {
              "type": "string",
              "description": "GitHub personal access token"
            }
          }
        }
      }
    },
    "sync": {
      "type": "object",
      "properties": {
        "cache_ttl_hours": {
          "type": "integer",
          "minimum": 1,
          "default": 24
        },
        "parallel_downloads": {
          "type": "integer",
          "minimum": 1,
          "maximum": 20,
          "default": 5
        }
      }
    }
  }
}
```

## Configuration Sources

Configuration is loaded from multiple sources in order of precedence:

1. **Default Configuration** (built-in)
2. **System Configuration**: `/etc/mcp-standards/config.yaml`
3. **User Configuration**: `~/.config/mcp-standards/config.yaml`
4. **Project Configuration**: `./.mcp-standards.yaml`
5. **Environment Variables**: `MCP_STANDARDS_*`
6. **Command Line Options**: `--config`, `--set`

### Environment Variables

All configuration options can be set via environment variables:

```bash
# Repository settings
export MCP_STANDARDS_REPOSITORY_OWNER=williamzujkowski
export MCP_STANDARDS_REPOSITORY_REPO=standards
export MCP_STANDARDS_REPOSITORY_BRANCH=main

# Sync settings
export MCP_STANDARDS_SYNC_CACHE_TTL_HOURS=48
export MCP_STANDARDS_SYNC_PARALLEL_DOWNLOADS=10

# Cache settings
export MCP_STANDARDS_CACHE_DIRECTORY=/var/cache/mcp-standards
export MCP_STANDARDS_CACHE_MAX_SIZE_MB=1000

# Authentication
export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=ghp_xxxxxxxxxxxx
```

## Authentication Configuration

### Personal Access Token

```bash
# Set GitHub token
mcp-standards config --set repository.auth.type token
mcp-standards config --set repository.auth.token "ghp_xxxxxxxxxxxx"

# Or use environment variable
export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=ghp_xxxxxxxxxxxx
```

### GitHub App

```yaml
repository:
  auth:
    type: app
    app_id: 123456
    private_key_path: ~/.config/mcp-standards/github-app.pem
    installation_id: 789012
```

## Migration from Older Versions

```bash
# Migrate from v1 configuration
mcp-standards config --migrate
```

Output:
```
Detecting configuration format...
Found v1 configuration at: ~/.mcp-standards.json

Migrating configuration:
✓ Repository settings
✓ Cache settings
✓ New sync options added with defaults
✓ Search configuration added

Backup created: ~/.mcp-standards.json.backup
New configuration saved: ~/.config/mcp-standards/config.yaml

Migration completed successfully!
Please review the new configuration: mcp-standards config --show
```

## Advanced Configuration

### Multiple Profiles

```bash
# Use different configurations for different projects
mcp-standards -c ./web-project.yaml sync
mcp-standards -c ./api-project.yaml sync

# Or use environment variable
export MCP_STANDARDS_CONFIG=./custom-config.yaml
mcp-standards sync
```

### Configuration Templates

```bash
# Generate template for CI/CD
mcp-standards config --template ci > .github/mcp-standards.yaml

# Generate template for development
mcp-standards config --template dev > .mcp-standards.dev.yaml
```

### Validation Rules

```yaml
# Custom validation rules
validation:
  enabled: true
  custom_rules:
    - name: require-auth
      description: Ensure authentication is configured
      rule: repository.auth.type != "none"
      severity: warning
    
    - name: min-cache-ttl
      description: Ensure reasonable cache TTL
      rule: sync.cache_ttl_hours >= 12
      severity: error
```

## Troubleshooting

### Common Issues

```bash
# Debug configuration loading
MCP_LOG_LEVEL=DEBUG mcp-standards config --show

# Check which config file is being used
mcp-standards config --which
/home/user/.config/mcp-standards/config.yaml

# Validate without loading
mcp-standards config --validate --file ./test-config.yaml
```

### Configuration Conflicts

When multiple configuration sources are present:

```bash
# Show effective configuration with sources
mcp-standards config --show --sources
```

Output:
```yaml
# Effective configuration (merged from multiple sources)
repository:
  owner: williamzujkowski  # from: user config
  repo: standards         # from: user config
  branch: develop        # from: environment variable
  
sync:
  cache_ttl_hours: 48    # from: command line --set
  parallel_downloads: 5   # from: default config
```

## Related Commands

- [sync](./sync.md) - Use the configuration to sync standards
- [serve](./serve.md) - Start server with configuration
- [validate](./validate.md) - Validate code using configuration