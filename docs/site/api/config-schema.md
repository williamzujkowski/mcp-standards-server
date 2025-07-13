# Configuration Schema Reference

Complete schema reference for MCP Standards Server configuration.

## Configuration File Format

The configuration file uses YAML format and is typically located at:
- **Linux/macOS:** `~/.mcp-standards/config.yaml`
- **Windows:** `%APPDATA%\mcp-standards\config.yaml`

## Schema Overview

```yaml
# Root configuration schema
server:          # Server configuration
standards:       # Standards management
validation:      # Validation settings
logging:         # Logging configuration
performance:     # Performance tuning
monitoring:      # Monitoring settings
security:        # Security configuration
integrations:    # External integrations
```

## Server Configuration

### `server`

HTTP server settings for the MCP server.

```yaml
server:
  host: string           # Server bind address
  port: integer          # Server port number
  enable_cors: boolean   # Enable CORS headers
  cors_origins: [string] # Allowed CORS origins
  workers: integer       # Number of worker processes
  timeout: integer       # Request timeout (seconds)
  max_request_size: string # Maximum request size (e.g., "10MB")
```

**Default values:**
```yaml
server:
  host: "0.0.0.0"
  port: 8080
  enable_cors: true
  cors_origins: ["*"]
  workers: 1
  timeout: 30
  max_request_size: "10MB"
```

**Validation rules:**
- `host`: Valid IP address or hostname
- `port`: Integer between 1-65535
- `workers`: Positive integer
- `timeout`: Positive integer

## Standards Configuration

### `standards`

Standards repository and caching settings.

```yaml
standards:
  repository_url: string     # Git repository URL
  branch: string            # Git branch to use
  local_directory: string   # Local standards path
  sync_interval: integer    # Auto-sync interval (seconds)
  auto_sync: boolean        # Enable automatic syncing
  cache_directory: string   # Cache directory path
  cache_ttl: integer        # Cache TTL (seconds)
  max_cache_size: string    # Maximum cache size
  exclude_patterns: [string] # Exclusion patterns
```

**Default values:**
```yaml
standards:
  repository_url: "https://github.com/williamzujkowski/standards"
  branch: "main"
  local_directory: null
  sync_interval: 3600
  auto_sync: true
  cache_directory: "~/.mcp-standards/cache"
  cache_ttl: 3600
  max_cache_size: "1GB"
  exclude_patterns: []
```

### `standards.redis`

Redis cache configuration (optional).

```yaml
standards:
  redis:
    enabled: boolean    # Enable Redis caching
    host: string       # Redis server host
    port: integer      # Redis server port
    db: integer        # Redis database number
    password: string   # Redis password
    ssl: boolean       # Use SSL connection
    timeout: integer   # Connection timeout
```

**Default values:**
```yaml
standards:
  redis:
    enabled: false
    host: "localhost"
    port: 6379
    db: 0
    password: null
    ssl: false
    timeout: 5
```

## Validation Configuration

### `validation`

Code validation behavior settings.

```yaml
validation:
  auto_fix: boolean          # Enable automatic fixes
  severity_level: string     # Minimum severity (error|warning|info)
  output_format: string      # Default output format
  max_file_size: string      # Maximum file size to validate
  exclude_patterns: [string] # Files to exclude
  include_patterns: [string] # Files to include
  parallel_jobs: integer     # Number of parallel validation jobs
  timeout_per_file: integer  # Timeout per file (seconds)
```

**Default values:**
```yaml
validation:
  auto_fix: false
  severity_level: "warning"
  output_format: "text"
  max_file_size: "10MB"
  exclude_patterns:
    - "**/node_modules/**"
    - "**/*.min.js"
    - "**/dist/**"
    - "**/build/**"
  include_patterns: []
  parallel_jobs: 4
  timeout_per_file: 30
```

### `validation.languages`

Language-specific validation settings.

```yaml
validation:
  languages:
    python:
      enabled: boolean
      standards: [string]  # Default standards for Python
      file_extensions: [string]
      max_line_length: integer
    javascript:
      enabled: boolean
      standards: [string]
      file_extensions: [string]
      node_modules_path: string
    typescript:
      enabled: boolean
      standards: [string]
      file_extensions: [string]
      tsconfig_path: string
    go:
      enabled: boolean
      standards: [string]
      file_extensions: [string]
      go_mod_path: string
    rust:
      enabled: boolean
      standards: [string]
      file_extensions: [string]
      cargo_toml_path: string
```

## Logging Configuration

### `logging`

Logging behavior and output settings.

```yaml
logging:
  level: string              # Log level (DEBUG|INFO|WARNING|ERROR)
  file: string               # Log file path
  max_file_size: string      # Maximum log file size
  backup_count: integer      # Number of backup files
  format: string             # Log format string
  date_format: string        # Date format string
  enable_colors: boolean     # Enable colored output
  log_requests: boolean      # Log HTTP requests
  log_validation_details: boolean # Log detailed validation info
```

**Default values:**
```yaml
logging:
  level: "INFO"
  file: "~/.mcp-standards/mcp-server.log"
  max_file_size: "10MB"
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  date_format: "%Y-%m-%d %H:%M:%S"
  enable_colors: true
  log_requests: false
  log_validation_details: false
```

## Performance Configuration

### `performance`

Performance tuning settings.

```yaml
performance:
  max_workers: integer       # Maximum worker threads
  batch_size: integer        # Batch processing size
  memory_limit: string       # Memory limit per process
  cpu_limit: float          # CPU limit (0.0-1.0)
  gc_threshold: float       # Garbage collection threshold
  enable_profiling: boolean # Enable performance profiling
  profile_output: string    # Profiling output directory
```

**Default values:**
```yaml
performance:
  max_workers: 4
  batch_size: 100
  memory_limit: "1GB"
  cpu_limit: 0.8
  gc_threshold: 0.7
  enable_profiling: false
  profile_output: "~/.mcp-standards/profiles"
```

## Monitoring Configuration

### `monitoring`

Monitoring and metrics settings.

```yaml
monitoring:
  enabled: boolean          # Enable monitoring
  metrics_port: integer     # Prometheus metrics port
  metrics_path: string      # Metrics endpoint path
  health_check_enabled: boolean # Enable health checks
  health_check_port: integer    # Health check port
  health_check_path: string     # Health check endpoint
  telemetry_enabled: boolean    # Enable telemetry collection
  telemetry_endpoint: string    # Telemetry endpoint URL
```

**Default values:**
```yaml
monitoring:
  enabled: false
  metrics_port: 9090
  metrics_path: "/metrics"
  health_check_enabled: true
  health_check_port: 8080
  health_check_path: "/health"
  telemetry_enabled: false
  telemetry_endpoint: null
```

## Security Configuration

### `security`

Security and authentication settings.

```yaml
security:
  authentication:
    enabled: boolean        # Enable authentication
    method: string         # Auth method (api_key|jwt|oauth)
    api_key: string        # API key for authentication
    jwt_secret: string     # JWT signing secret
    token_expiry: integer  # Token expiry time (seconds)
  authorization:
    enabled: boolean       # Enable authorization
    roles: object         # Role definitions
  rate_limiting:
    enabled: boolean       # Enable rate limiting
    requests_per_minute: integer # Request limit per minute
    burst_size: integer    # Burst size allowance
  ssl:
    enabled: boolean       # Enable SSL/TLS
    cert_file: string     # SSL certificate file
    key_file: string      # SSL private key file
    ca_file: string       # CA certificate file
```

**Default values:**
```yaml
security:
  authentication:
    enabled: false
    method: "api_key"
    api_key: null
    jwt_secret: null
    token_expiry: 86400
  authorization:
    enabled: false
    roles: {}
  rate_limiting:
    enabled: false
    requests_per_minute: 100
    burst_size: 20
  ssl:
    enabled: false
    cert_file: null
    key_file: null
    ca_file: null
```

## Integration Configuration

### `integrations`

External service integrations.

```yaml
integrations:
  github:
    enabled: boolean       # Enable GitHub integration
    token: string         # GitHub personal access token
    organization: string  # GitHub organization
    webhook_secret: string # Webhook secret
  slack:
    enabled: boolean      # Enable Slack integration
    webhook_url: string   # Slack webhook URL
    channel: string       # Default Slack channel
  email:
    enabled: boolean      # Enable email notifications
    smtp_host: string     # SMTP server host
    smtp_port: integer    # SMTP server port
    username: string      # SMTP username
    password: string      # SMTP password
    from_address: string  # From email address
```

## Environment Variable Overrides

Any configuration value can be overridden using environment variables with the prefix `MCP_`:

```bash
# Server configuration
MCP_SERVER_HOST=127.0.0.1
MCP_SERVER_PORT=8081

# Standards configuration
MCP_STANDARDS_REPOSITORY_URL=https://github.com/custom/standards
MCP_STANDARDS_CACHE_DIRECTORY=/custom/cache

# Validation configuration
MCP_VALIDATION_AUTO_FIX=true
MCP_VALIDATION_SEVERITY_LEVEL=error

# Nested values use double underscores
MCP_STANDARDS__REDIS__ENABLED=true
MCP_STANDARDS__REDIS__HOST=redis.example.com
```

## Validation Rules

### Data Types

- **string**: Text values, support environment variable substitution
- **integer**: Whole numbers
- **float**: Decimal numbers
- **boolean**: true/false values
- **array**: List of values
- **object**: Nested configuration

### Size Formats

Size values support human-readable formats:
- `1024` (bytes)
- `1KB`, `1MB`, `1GB`, `1TB`
- `1K`, `1M`, `1G`, `1T` (binary)

### Time Formats

Time values support various formats:
- `30` (seconds)
- `30s` (seconds)
- `5m` (minutes)
- `1h` (hours)
- `1d` (days)

### Path Expansion

Paths support tilde expansion:
- `~/` expands to user home directory
- `$HOME/` expands to home directory
- Environment variables: `$CACHE_DIR/path`

## Configuration Validation

Validate configuration file:

```bash
# Validate current configuration
mcp-standards config validate

# Validate specific file
mcp-standards config validate --config custom-config.yaml

# Show validation errors in detail
mcp-standards config validate --verbose
```

## Example Configurations

### Development Environment

```yaml
server:
  host: "127.0.0.1"
  port: 8080
  enable_cors: true

standards:
  auto_sync: true
  sync_interval: 300  # 5 minutes

validation:
  auto_fix: true
  severity_level: "info"

logging:
  level: "DEBUG"
  log_validation_details: true

performance:
  enable_profiling: true
```

### Production Environment

```yaml
server:
  host: "0.0.0.0"
  port: 80
  workers: 4
  enable_cors: false

standards:
  sync_interval: 3600  # 1 hour
  redis:
    enabled: true
    host: "redis.internal"

validation:
  auto_fix: false
  severity_level: "error"
  parallel_jobs: 8

logging:
  level: "INFO"
  log_requests: true

security:
  authentication:
    enabled: true
    method: "api_key"
  rate_limiting:
    enabled: true
    requests_per_minute: 1000

monitoring:
  enabled: true
  telemetry_enabled: true
```

### CI/CD Environment

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  timeout: 60

standards:
  auto_sync: false  # Use pre-synced standards
  cache_directory: "/tmp/mcp-cache"

validation:
  auto_fix: false
  output_format: "sarif"
  parallel_jobs: 2
  timeout_per_file: 10

logging:
  level: "WARNING"
  file: "/dev/stdout"

performance:
  memory_limit: "512MB"
  cpu_limit: 0.5
```

---

For more configuration examples and troubleshooting, see the [Configuration Guide](../guides/configuration.md).
