# Configuration Guide

Configure MCP Standards Server to match your workflow and requirements.

## Configuration File Location

The configuration file is located at:
- **Linux/macOS**: `~/.mcp-standards/config.yaml`
- **Windows**: `%APPDATA%\mcp-standards\config.yaml`

## Basic Configuration

### Default Configuration

```yaml
# ~/.mcp-standards/config.yaml
server:
  host: "0.0.0.0"
  port: 8080
  enable_cors: true
  
standards:
  repository_url: "https://github.com/williamzujkowski/standards"
  sync_interval: 3600  # 1 hour
  cache_directory: "~/.mcp-standards/cache"
  
validation:
  auto_fix: false
  severity_level: "warning"
  output_format: "json"
  
logging:
  level: "INFO"
  file: "~/.mcp-standards/mcp-server.log"
```

## Server Configuration

### Network Settings

```yaml
server:
  host: "127.0.0.1"  # Localhost only
  port: 8081         # Custom port
  enable_cors: true  # Enable CORS for web clients
  cors_origins:      # Specific origins
    - "http://localhost:3000"
    - "https://your-domain.com"
```

### Security Settings

```yaml
server:
  auth:
    enabled: true
    api_key: "your-secret-api-key"
    token_expiry: 86400  # 24 hours
  
  rate_limiting:
    enabled: true
    requests_per_minute: 100
    burst_size: 20
```

## Standards Configuration

### Repository Settings

```yaml
standards:
  repository_url: "https://github.com/your-org/standards"
  branch: "main"
  sync_interval: 1800  # 30 minutes
  auto_sync: true
  
  # Local standards directory
  local_directory: "/path/to/local/standards"
  
  # Exclude specific standards
  exclude_patterns:
    - "deprecated/*"
    - "experimental/*"
```

### Cache Configuration

```yaml
standards:
  cache_directory: "~/.mcp-standards/cache"
  cache_ttl: 3600  # 1 hour
  max_cache_size: "1GB"
  
  # Redis cache (optional)
  redis:
    enabled: true
    host: "localhost"
    port: 6379
    db: 0
```

## Validation Configuration

### Default Validation Settings

```yaml
validation:
  auto_fix: true
  severity_level: "error"  # error, warning, info
  output_format: "sarif"   # json, junit, sarif, text
  
  # Language-specific settings
  languages:
    python:
      enabled: true
      standards: ["python-pep8", "python-security"]
    
    javascript:
      enabled: true
      standards: ["javascript-es6", "react-patterns"]
    
    go:
      enabled: true
      standards: ["go-effective", "go-security"]
```

### Custom Validators

```yaml
validation:
  custom_validators:
    - name: "company-style"
      path: "/path/to/custom/validator.py"
      languages: ["python", "javascript"]
    
    - name: "security-scanner"
      command: "custom-security-tool"
      args: ["--format", "json"]
```

## Integration Configuration

### IDE Integration

```yaml
ide:
  vscode:
    enabled: true
    real_time_validation: true
    auto_fix_on_save: true
  
  jetbrains:
    enabled: true
    plugin_port: 8082
```

### CI/CD Integration

```yaml
ci_cd:
  github_actions:
    enabled: true
    fail_on_error: true
    comment_pr: true
  
  jenkins:
    enabled: true
    webhook_url: "https://jenkins.company.com/webhook"
```

## Advanced Configuration

### Performance Tuning

```yaml
performance:
  max_workers: 4
  batch_size: 100
  timeout: 30
  
  # Memory limits
  max_memory_mb: 1024
  gc_threshold: 0.8
```

### Monitoring

```yaml
monitoring:
  metrics:
    enabled: true
    port: 9090
    path: "/metrics"
  
  health_check:
    enabled: true
    endpoint: "/health"
    interval: 30
```

## Environment Variables

Override configuration with environment variables:

```bash
# Server settings
export MCP_SERVER_HOST="0.0.0.0"
export MCP_SERVER_PORT="8080"

# Standards settings
export MCP_STANDARDS_REPO_URL="https://github.com/your-org/standards"
export MCP_CACHE_DIR="/custom/cache/path"

# Validation settings
export MCP_AUTO_FIX="true"
export MCP_SEVERITY_LEVEL="warning"
```

## Configuration Validation

Validate your configuration:

```bash
# Check configuration syntax
mcp-standards config --validate

# Show current configuration
mcp-standards config --show

# Test configuration
mcp-standards config --test
```

## Multiple Environments

### Development Environment

```yaml
# config.dev.yaml
server:
  host: "127.0.0.1"
  port: 8080
  
validation:
  auto_fix: true
  severity_level: "warning"
```

### Production Environment

```yaml
# config.prod.yaml
server:
  host: "0.0.0.0"
  port: 80
  
validation:
  auto_fix: false
  severity_level: "error"
  
monitoring:
  metrics:
    enabled: true
```

### Using Environment-Specific Configs

```bash
# Use specific config file
mcp-standards serve --config config.prod.yaml

# Set environment
export MCP_ENVIRONMENT="production"
mcp-standards serve
```

## Troubleshooting Configuration

### Common Issues

1. **Port already in use**
   ```bash
   # Check what's using the port
   lsof -i :8080
   
   # Use different port
   mcp-standards serve --port 8081
   ```

2. **Permission denied**
   ```bash
   # Fix cache directory permissions
   chmod 755 ~/.mcp-standards/cache
   ```

3. **Invalid YAML syntax**
   ```bash
   # Validate YAML
   python -c "import yaml; yaml.safe_load(open('config.yaml'))"
   ```

For more troubleshooting, see the [Troubleshooting Guide](../reference/troubleshooting.md).