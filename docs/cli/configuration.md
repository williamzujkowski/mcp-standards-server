# Configuration Guide

The MCP Standards Server uses a flexible configuration system that supports multiple formats, sources, and environments. This guide covers all configuration options and best practices.

## Configuration Files

### File Locations

Configuration files are loaded in the following order (later files override earlier ones):

1. **Built-in Defaults** - Hardcoded defaults
2. **System Configuration** - `/etc/mcp-standards/config.yaml`
3. **User Configuration** - `~/.config/mcp-standards/config.yaml`
4. **Project Configuration** - `./.mcp-standards.yaml`
5. **Environment Variables** - `MCP_STANDARDS_*`
6. **Command Line Options** - `--config`, `--set`

### File Formats

The server supports multiple configuration formats:

- **YAML** (recommended): `.yaml`, `.yml`
- **JSON**: `.json`
- **TOML**: `.toml`
- **Environment**: `.env`

## Complete Configuration Reference

```yaml
# Complete configuration with all options
# ~/.config/mcp-standards/config.yaml

# Repository settings
repository:
  # GitHub repository details
  owner: williamzujkowski
  repo: standards
  branch: main
  path: standards  # Path within repository
  
  # Authentication (optional)
  auth:
    type: none  # Options: none, token, app
    # For token auth:
    token: ghp_xxxxxxxxxxxx
    # For GitHub App auth:
    app_id: 123456
    private_key_path: ~/.config/mcp-standards/github-app.pem
    installation_id: 789012

# Synchronization settings
sync:
  # Cache time-to-live in hours
  cache_ttl_hours: 24
  
  # Parallel download settings
  parallel_downloads: 5
  max_retries: 3
  timeout_seconds: 30
  
  # File patterns
  include_patterns:
    - "*.yaml"
    - "*.yml"
    - "*.md"
    - "*.json"
  
  exclude_patterns:
    - "*.draft.*"
    - "*.backup.*"
    - ".git*"
    - "_*"
  
  # Sync behavior
  auto_sync: true
  sync_on_startup: true
  verify_checksums: true

# Cache configuration
cache:
  # Cache directory
  directory: ~/.cache/mcp-standards
  
  # Size limits
  max_size_mb: 500
  warning_threshold_mb: 400
  
  # Compression
  compression:
    enabled: true
    algorithm: gzip  # Options: gzip, zstd, lz4
    level: 6
  
  # Automatic cleanup
  auto_cleanup:
    enabled: true
    threshold_percent: 80
    remove_oldest: true
    keep_recent_hours: 168  # 7 days
    run_interval_hours: 24

# Search configuration
search:
  # Semantic search
  enabled: true
  model: sentence-transformers/all-MiniLM-L6-v2
  
  # Indexing
  index_on_sync: true
  index_format: faiss  # Options: faiss, annoy, simple
  
  # Performance
  cache_embeddings: true
  embedding_batch_size: 32
  max_results: 10
  
  # Advanced settings
  similarity_threshold: 0.7
  use_gpu: false
  num_workers: 4

# Server configuration
server:
  # Network settings
  host: localhost
  port: 3000
  public_url: https://standards.example.com
  
  # Performance
  workers: auto  # auto = CPU count
  max_connections: 1000
  request_timeout: 30
  keepalive_timeout: 65
  
  # Logging
  log_level: info  # debug, info, warning, error
  log_file: /var/log/mcp-standards.log
  log_format: json  # json, text
  log_rotation:
    enabled: true
    max_size_mb: 100
    max_files: 5
  
  # Security
  auth:
    type: token  # none, token, oauth
    token_file: ~/.config/mcp-standards/tokens.json
    session_timeout: 3600
    
    # OAuth settings
    oauth:
      provider: github  # github, google, okta
      client_id: ${OAUTH_CLIENT_ID}
      client_secret: ${OAUTH_CLIENT_SECRET}
      redirect_uri: https://standards.example.com/auth/callback
      scopes:
        - read:user
        - read:org
  
  # TLS/SSL
  tls:
    enabled: false
    cert: /etc/ssl/certs/server.crt
    key: /etc/ssl/private/server.key
    ca: /etc/ssl/certs/ca.crt
    min_version: "1.2"
    ciphers:
      - ECDHE-RSA-AES256-GCM-SHA384
      - ECDHE-RSA-AES128-GCM-SHA256
  
  # CORS
  cors:
    enabled: true
    origins:
      - http://localhost:3001
      - https://app.example.com
    methods:
      - GET
      - POST
      - OPTIONS
    headers:
      - Content-Type
      - Authorization
    credentials: true
  
  # Rate limiting
  rate_limit:
    enabled: true
    window_minutes: 1
    max_requests: 60
    burst: 100
    by_ip: true
    by_token: true
    whitelist:
      - 127.0.0.1
      - 10.0.0.0/8

# Token optimization
token_optimization:
  # Default settings
  enabled: true
  default_budget: 8000
  model_type: gpt-4  # gpt-4, gpt-3.5, claude, custom
  
  # Format preferences
  prefer_formats:
    - condensed
    - structured
    - full
  
  # Compression settings
  compression:
    enabled: true
    min_size_bytes: 1000
    algorithms:
      - sentencepiece
      - bpe
  
  # Caching
  cache_optimized: true
  cache_ttl_hours: 168

# Validation configuration
validation:
  # General settings
  enabled: true
  strict_mode: false
  parallel_workers: 4
  
  # File patterns
  include_patterns:
    - "**/*.js"
    - "**/*.jsx"
    - "**/*.ts"
    - "**/*.tsx"
    - "**/*.py"
    - "**/*.yaml"
  
  exclude_patterns:
    - "**/node_modules/**"
    - "**/dist/**"
    - "**/build/**"
    - "**/.git/**"
  
  # Rules configuration
  rules:
    # Global rule settings
    global:
      severity: warning
      auto_fix: true
    
    # Specific rule overrides
    overrides:
      no-console: off
      max-line-length:
        severity: warning
        options:
          limit: 100
          ignore_comments: true
      
      security-headers:
        severity: error
        auto_fix: false
  
  # Reporting
  report:
    format: text  # text, json, junit, sarif
    file: validation-report.json
    fail_on: error
    summary: true
    details: true

# Development settings
development:
  # Debug options
  debug: false
  verbose: false
  trace: false
  
  # Hot reload
  watch: true
  watch_patterns:
    - "**/*.yaml"
    - "**/*.json"
  
  # Testing
  test_mode: false
  mock_data: false
  fixtures_path: ./tests/fixtures

# Analytics (optional)
analytics:
  enabled: false
  provider: mixpanel  # mixpanel, segment, custom
  api_key: ${ANALYTICS_API_KEY}
  track_usage: true
  track_errors: true
  anonymize_ip: true

# Experimental features
experimental:
  # AI-powered features
  ai_suggestions: false
  auto_fix_complex: false
  natural_language_rules: false
  
  # Performance features
  lazy_loading: true
  incremental_sync: true
  distributed_cache: false

# Plugin system
plugins:
  enabled: true
  directory: ~/.config/mcp-standards/plugins
  auto_load: true
  allowed:
    - official/*
    - community/verified/*
```

## Environment Variables

All configuration options can be set via environment variables using the `MCP_STANDARDS_` prefix:

```bash
# Repository settings
export MCP_STANDARDS_REPOSITORY_OWNER=williamzujkowski
export MCP_STANDARDS_REPOSITORY_REPO=standards
export MCP_STANDARDS_REPOSITORY_BRANCH=main
export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=ghp_xxxxxxxxxxxx

# Sync settings
export MCP_STANDARDS_SYNC_CACHE_TTL_HOURS=48
export MCP_STANDARDS_SYNC_PARALLEL_DOWNLOADS=10

# Cache settings
export MCP_STANDARDS_CACHE_DIRECTORY=/var/cache/mcp-standards
export MCP_STANDARDS_CACHE_MAX_SIZE_MB=1000

# Server settings
export MCP_STANDARDS_SERVER_HOST=0.0.0.0
export MCP_STANDARDS_SERVER_PORT=8080
export MCP_STANDARDS_SERVER_LOG_LEVEL=debug

# Feature flags
export MCP_STANDARDS_SEARCH_ENABLED=true
export MCP_STANDARDS_VALIDATION_ENABLED=true
```

## Configuration Profiles

### Development Profile

```yaml
# .mcp-standards.dev.yaml
development:
  debug: true
  verbose: true

server:
  host: localhost
  port: 3001
  log_level: debug
  auth:
    type: none

sync:
  cache_ttl_hours: 1
  auto_sync: true

validation:
  strict_mode: true
  auto_fix: true
```

### Production Profile

```yaml
# .mcp-standards.prod.yaml
server:
  host: 0.0.0.0
  port: 443
  workers: 16
  log_level: warning
  
  tls:
    enabled: true
    cert: /etc/ssl/certs/server.crt
    key: /etc/ssl/private/server.key
  
  auth:
    type: oauth
    oauth:
      provider: github
  
  rate_limit:
    enabled: true
    max_requests: 30
    burst: 50

cache:
  compression:
    enabled: true
    algorithm: zstd
    level: 9

sync:
  verify_checksums: true
  cache_ttl_hours: 168  # 1 week
```

### CI/CD Profile

```yaml
# .mcp-standards.ci.yaml
sync:
  cache_ttl_hours: 24
  parallel_downloads: 1  # Avoid rate limits

validation:
  enabled: true
  fail_on: error
  report:
    format: junit
    file: test-results.xml

server:
  auth:
    type: token
    token_file: /secrets/mcp-token

cache:
  directory: /tmp/mcp-cache
```

## Advanced Configuration

### Dynamic Configuration

```yaml
# Use environment variable substitution
repository:
  owner: ${GITHUB_OWNER}
  repo: ${GITHUB_REPO:-standards}  # With default
  auth:
    token: ${GITHUB_TOKEN:?Error: GITHUB_TOKEN required}

# Conditional configuration
$if: ${ENV} == "production"
server:
  workers: 16
  log_level: warning
$else:
server:
  workers: 2
  log_level: debug
```

### Multi-Repository Configuration

```yaml
# Support multiple standard sources
repositories:
  - name: main
    owner: williamzujkowski
    repo: standards
    branch: main
    priority: 1
    
  - name: enterprise
    owner: company
    repo: enterprise-standards
    branch: master
    priority: 2
    auth:
      token: ${ENTERPRISE_TOKEN}
```

### Configuration Validation

```bash
# Validate configuration
mcp-standards config --validate

# Test specific values
mcp-standards config --test repository.auth

# Dry run with config
mcp-standards --dry-run -c custom-config.yaml sync
```

## Best Practices

1. **Use YAML** for human-readable configuration
2. **Environment variables** for secrets and deployment-specific values
3. **Separate profiles** for dev/staging/production
4. **Version control** your configuration (except secrets)
5. **Validate** configuration in CI/CD pipelines
6. **Document** custom configuration options
7. **Use defaults** where possible to minimize configuration

## Migration Guide

### From v1.x to v2.x

```bash
# Automatic migration
mcp-standards config --migrate

# Manual migration
mcp-standards config --export-v1 > old-config.json
mcp-standards config --import-v2 old-config.json
```

### From Other Tools

```bash
# Import from ESLint config
mcp-standards config --import-eslint .eslintrc.json

# Import from Prettier config
mcp-standards config --import-prettier .prettierrc
```

## Troubleshooting

### Debug Configuration Loading

```bash
# Show configuration sources
MCP_DEBUG_CONFIG=1 mcp-standards config --show-sources

# Trace configuration resolution
mcp-standards config --trace
```

### Common Issues

1. **Configuration not loading**
   - Check file permissions
   - Verify YAML syntax
   - Ensure correct file extension

2. **Environment variables not working**
   - Check variable naming (MCP_STANDARDS_ prefix)
   - Verify shell export
   - Check for typos in nested paths

3. **Validation failures**
   - Run with --validate flag
   - Check required fields
   - Verify value types and ranges