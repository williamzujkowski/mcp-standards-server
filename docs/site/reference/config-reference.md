# Configuration Reference

Complete reference for all MCP Standards Server configuration options.

## Configuration File

The main configuration file is `.mcp-standards.yml` in your project root.

```yaml
# .mcp-standards.yml
version: 1.0

project:
  name: "My Project"
  type: "web"
  description: "Project description"

standards:
  - python-best-practices
  - security-baseline
  - custom-team-standards

validation:
  severity_threshold: "warning"  # error, warning, info
  fail_on_violation: true
  parallel: true
  
  exclude:
    - "vendor/"
    - "node_modules/"
    - "*.min.js"
    - "**/__pycache__/"
  
  include_only:
    - "src/"
    - "tests/"

cache:
  enabled: true
  ttl: 3600  # seconds
  backend: "redis"
  redis_url: "redis://localhost:6379"

reporting:
  formats: ["json", "html", "sarif"]
  output_dir: "reports/"
  
mcp:
  server:
    host: "0.0.0.0"
    port: 8080
  
  auth:
    enabled: true
    api_key_header: "X-API-Key"
```

## Environment Variables

Override configuration with environment variables:

```bash
# Server configuration
MCP_HOST=0.0.0.0
MCP_PORT=8080
MCP_ENV=production  # development, production

# Redis configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=secret
REDIS_DB=0

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT=json  # json, text

# Performance
MAX_WORKERS=4
CACHE_TTL=3600
REQUEST_TIMEOUT=30

# Security
API_KEY_REQUIRED=true
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60  # seconds
```

## CLI Configuration

```bash
# Global config file
~/.mcp-standards/config.yml

# Project config file
./.mcp-standards.yml

# Override with flags
mcp-standards validate --config custom-config.yml
```

### CLI Config Example

```yaml
# ~/.mcp-standards/config.yml
defaults:
  output_format: "table"
  color: true
  verbose: false

profiles:
  strict:
    severity_threshold: "info"
    fail_on_violation: true
  
  lenient:
    severity_threshold: "error"
    fail_on_violation: false

aliases:
  check: "validate --format json"
  fix: "validate --auto-fix"
```

## Standards Configuration

### Standard Definition

```yaml
# standards/my-standard.yaml
standard:
  id: "my-standard"
  name: "My Standard"
  version: "1.0.0"
  enabled: true
  
config:
  # Standard-specific configuration
  max_line_length: 88
  indent_size: 4
  quote_style: "double"
  
rules:
  - id: "line-length"
    enabled: true
    config:
      max_length: 88
      ignore_comments: true
```

### Rule Configuration

```yaml
rules:
  # Override specific rules
  overrides:
    - rule_id: "no-console-log"
      severity: "info"  # downgrade from error
      enabled: false
    
    - rule_id: "max-function-length"
      config:
        max_lines: 100  # increase from default 50
```

## Validation Configuration

### File Patterns

```yaml
validation:
  file_patterns:
    python:
      - "*.py"
      - "*.pyi"
    javascript:
      - "*.js"
      - "*.jsx"
      - "*.mjs"
    typescript:
      - "*.ts"
      - "*.tsx"
```

### Custom Validators

```yaml
validators:
  custom:
    - module: "myproject.validators"
      class: "CustomValidator"
      config:
        strict_mode: true
```

## Cache Configuration

### Redis Options

```yaml
cache:
  redis:
    host: "localhost"
    port: 6379
    db: 0
    password: "secret"
    ssl: true
    ssl_cert_reqs: "required"
    connection_pool:
      max_connections: 50
      socket_timeout: 5
      socket_connect_timeout: 5
```

### In-Memory Cache

```yaml
cache:
  backend: "memory"
  memory:
    max_size: 1000  # entries
    ttl: 300  # seconds
    eviction: "lru"  # lru, lfu, fifo
```

## Security Configuration

### Authentication

```yaml
security:
  auth:
    providers:
      - type: "api_key"
        header: "X-API-Key"
        keys:
          - name: "production"
            key: "prod-key-hash"
            scopes: ["read", "write"]
      
      - type: "jwt"
        issuer: "https://auth.example.com"
        audience: "mcp-standards"
        algorithms: ["RS256"]
```

### Rate Limiting

```yaml
security:
  rate_limit:
    enabled: true
    storage: "redis"  # redis, memory
    rules:
      - path: "/api/*"
        limit: 100
        window: 60  # seconds
      
      - path: "/api/validate"
        limit: 10
        window: 60
```

## Performance Configuration

### Concurrency

```yaml
performance:
  workers: 4
  thread_pool_size: 10
  async_enabled: true
  
  batching:
    enabled: true
    size: 100
    timeout: 5  # seconds
```

### Resource Limits

```yaml
performance:
  limits:
    max_file_size: 10485760  # 10MB
    max_files_per_validation: 1000
    max_memory_usage: 536870912  # 512MB
    request_timeout: 30  # seconds
```

## Logging Configuration

```yaml
logging:
  level: "INFO"
  format: "json"
  
  handlers:
    - type: "console"
      level: "INFO"
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    - type: "file"
      level: "DEBUG"
      filename: "logs/mcp-standards.log"
      max_bytes: 10485760  # 10MB
      backup_count: 5
    
    - type: "syslog"
      level: "WARNING"
      address: "/dev/log"
      facility: "local0"
```

## Integration Configuration

### CI/CD Integration

```yaml
ci:
  github_actions:
    enabled: true
    annotations: true
    status_checks: true
  
  gitlab:
    enabled: true
    merge_request_comments: true
  
  jenkins:
    enabled: true
    pipeline_integration: true
```

### IDE Integration

```yaml
ide:
  vscode:
    extension_settings:
      real_time_validation: true
      auto_fix_on_save: false
  
  jetbrains:
    inspection_profile: "mcp-standards"
```

## Advanced Configuration

### Plugins

```yaml
plugins:
  - name: "custom-reporter"
    module: "myproject.plugins.reporter"
    config:
      template: "custom-template.html"
  
  - name: "slack-notifier"
    module: "mcp_slack"
    config:
      webhook_url: "${SLACK_WEBHOOK}"
      channel: "#dev-standards"
```

### Hooks

```yaml
hooks:
  pre_validation:
    - command: "git fetch origin main"
    - script: "scripts/prepare-validation.py"
  
  post_validation:
    - command: "npm run lint"
      on_failure: "continue"
```

## Configuration Precedence

1. Command line arguments
2. Environment variables
3. Project config file (`.mcp-standards.yml`)
4. User config file (`~/.mcp-standards/config.yml`)
5. System config file (`/etc/mcp-standards/config.yml`)
6. Default values

## Validation

Validate your configuration:

```bash
# Check configuration syntax
mcp-standards config validate

# Show effective configuration
mcp-standards config show

# Test specific config file
mcp-standards config test -f custom-config.yml
```

## Related Documentation

- [Environment Variables](../../ENVIRONMENT_VARIABLES.md)
- [Security Configuration](../../SECURITY_CONFIGURATION.md)
- [CLI Commands](../cli/commands/config.md)