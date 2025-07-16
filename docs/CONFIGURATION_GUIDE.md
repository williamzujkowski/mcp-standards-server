# Configuration Guide

## Overview

This guide provides comprehensive information about configuring the MCP Standards Server for different environments and use cases.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Environment Variables](#environment-variables)
3. [Configuration Files](#configuration-files)
4. [Environment-Specific Configurations](#environment-specific-configurations)
5. [Performance Tuning](#performance-tuning)
6. [Security Configuration](#security-configuration)
7. [Monitoring and Logging](#monitoring-and-logging)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Basic Setup

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` with your specific configuration:
   ```bash
   # Basic configuration
   HTTP_HOST=127.0.0.1
   HTTP_PORT=8080
   LOG_LEVEL=INFO
   DATA_DIR=./data
   ```

3. Start the server:
   ```bash
   python -m src.main
   ```

### Docker Quick Start

```bash
# Using docker-compose
docker-compose up -d

# Using docker directly
docker run -p 8080:8080 -e HTTP_HOST=0.0.0.0 mcp-standards-server
```

---

## Environment Variables

### Core Application Settings

#### Server Configuration

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `HTTP_HOST` | `127.0.0.1` | Host address for HTTP server | `0.0.0.0` |
| `HTTP_PORT` | `8080` | Port for HTTP server | `8080` |
| `HTTP_ONLY` | `false` | Run only HTTP server without MCP | `true` |

#### File System Paths

| Variable | Default | Description | Example |
|----------|---------|-------------|---------|
| `DATA_DIR` | `./data` | Root directory for data files | `/var/lib/mcp-standards` |
| `MCP_STANDARDS_DATA_DIR` | `./data/standards` | Directory for standards data | `/var/lib/mcp-standards/standards` |
| `MCP_CONFIG_PATH` | `./config/mcp_config.yaml` | Path to MCP configuration file | `/etc/mcp-standards/config.yaml` |

#### Feature Flags

| Variable | Default | Description | Use Case |
|----------|---------|-------------|----------|
| `MCP_DISABLE_SEARCH` | `false` | Disable semantic search features | Resource-constrained environments |

### Authentication & Security

#### Authentication Settings

| Variable | Default | Description | Security Level |
|----------|---------|-------------|----------------|
| `MCP_AUTH_ENABLED` | `false` | Enable authentication for MCP tools | Production |
| `MCP_JWT_SECRET` | `""` | JWT secret key for authentication | **Required in production** |
| `MCP_MASK_ERRORS` | `false` | Mask error details in production | Production |

**⚠️ Security Note**: Always change `MCP_JWT_SECRET` in production environments.

#### Security Best Practices

```bash
# Generate a secure JWT secret
MCP_JWT_SECRET=$(openssl rand -base64 32)

# Enable error masking in production
MCP_MASK_ERRORS=true

# Enable authentication in production
MCP_AUTH_ENABLED=true
```

### External Services

#### GitHub Integration

| Variable | Default | Description | Required For |
|----------|---------|-------------|--------------|
| `GITHUB_TOKEN` | `""` | GitHub personal access token | Standards synchronization |

```bash
# Generate token at: https://github.com/settings/tokens
GITHUB_TOKEN=ghp_your_token_here
```

#### Redis Cache

| Variable | Default | Description | Format |
|----------|---------|-------------|--------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL | `redis://[user:pass@]host:port/db` |

```bash
# Local Redis
REDIS_URL=redis://localhost:6379/0

# Redis with authentication
REDIS_URL=redis://user:password@redis-server:6379/0

# Redis cluster
REDIS_URL=redis://redis-cluster:6379/0
```

#### Vector Database

| Variable | Default | Description | Purpose |
|----------|---------|-------------|---------|
| `CHROMADB_URL` | `http://localhost:8000` | ChromaDB server URL | Semantic search |

```bash
# Local ChromaDB
CHROMADB_URL=http://localhost:8000

# Remote ChromaDB
CHROMADB_URL=http://chromadb.example.com:8000
```

### Web Application Settings

#### Frontend Configuration

| Variable | Default | Description | Environment |
|----------|---------|-------------|-------------|
| `WEB_UI_ENABLED` | `true` | Enable web UI interface | Development/Production |
| `WEB_UI_PORT` | `3000` | Port for web UI development server | Development |
| `WEB_UI_API_BASE_URL` | `http://localhost:8080` | Base URL for API calls | Production |

### Logging Configuration

#### Log Settings

| Variable | Default | Description | Options |
|----------|---------|-------------|---------|
| `LOG_LEVEL` | `INFO` | Logging level | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_FORMAT` | `json` | Log format | `json`, `text` |
| `LOG_FILE` | `""` | Log file path | `/var/log/mcp-standards.log` |
| `LOG_DIR` | `logs` | Directory for log files | `/var/log/mcp-standards/` |
| `LOG_MAX_SIZE` | `10MB` | Maximum log file size | `10MB`, `100MB` |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files | `5`, `10` |

#### Logging Examples

```bash
# Development logging
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LOG_FILE=""  # Log to stdout

# Production logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/mcp-standards/app.log
LOG_DIR=/var/log/mcp-standards/
LOG_MAX_SIZE=100MB
LOG_BACKUP_COUNT=10
```

### Performance & Caching

#### Cache Configuration

| Variable | Default | Description | Impact |
|----------|---------|-------------|--------|
| `CACHE_ENABLED` | `true` | Enable caching | Performance |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds | Memory vs. freshness |
| `CACHE_MAX_SIZE` | `1000` | Maximum cache entries | Memory usage |
| `VECTOR_CACHE_SIZE` | `10000` | Vector cache size | Search performance |
| `SEARCH_CACHE_TTL` | `300` | Search cache TTL in seconds | Search responsiveness |

#### Performance Tuning

```bash
# High-performance setup
CACHE_ENABLED=true
CACHE_TTL=7200  # 2 hours
CACHE_MAX_SIZE=5000
VECTOR_CACHE_SIZE=50000
SEARCH_CACHE_TTL=600  # 10 minutes

# Memory-constrained setup
CACHE_ENABLED=true
CACHE_TTL=1800  # 30 minutes
CACHE_MAX_SIZE=500
VECTOR_CACHE_SIZE=5000
SEARCH_CACHE_TTL=60  # 1 minute
```

### Monitoring & Metrics

#### Metrics Configuration

| Variable | Default | Description | Use Case |
|----------|---------|-------------|----------|
| `METRICS_ENABLED` | `true` | Enable metrics collection | Monitoring |
| `METRICS_PORT` | `9090` | Prometheus metrics port | Prometheus scraping |
| `METRICS_PATH` | `/metrics` | Metrics endpoint path | Custom monitoring |
| `HEALTH_CHECK_INTERVAL` | `30` | Health check interval in seconds | Service monitoring |

---

## Configuration Files

### MCP Configuration File

Create `config/mcp_config.yaml`:

```yaml
# MCP Server Configuration
server:
  name: "mcp-standards-server"
  version: "1.0.0"
  description: "MCP Standards Server"

# Tool Configuration
tools:
  get_applicable_standards:
    enabled: true
    max_results: 50
    timeout: 30
  
  search_standards:
    enabled: true
    max_results: 100
    timeout: 10
    
  get_standard:
    enabled: true
    timeout: 5

# Standards Configuration
standards:
  data_dir: "./data/standards"
  sync_interval: 3600  # 1 hour
  auto_sync: true
  
# Search Configuration
search:
  enabled: true
  index_update_interval: 300  # 5 minutes
  similarity_threshold: 0.7
```

### Logging Configuration File

Create `config/logging.yaml`:

```yaml
version: 1
disable_existing_loggers: false

formatters:
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
  
  text:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: json
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: json
    filename: /var/log/mcp-standards/app.log
    maxBytes: 104857600  # 100MB
    backupCount: 5

loggers:
  src:
    level: INFO
    handlers: [console, file]
    propagate: false
    
  aiohttp:
    level: WARNING
    handlers: [console]
    propagate: false

root:
  level: INFO
  handlers: [console, file]
```

---

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
HTTP_HOST=127.0.0.1
HTTP_PORT=8080
LOG_LEVEL=DEBUG
LOG_FORMAT=text
MCP_AUTH_ENABLED=false
MCP_MASK_ERRORS=false
CACHE_ENABLED=true
METRICS_ENABLED=true
WEB_UI_ENABLED=true
WEB_UI_PORT=3000
```

### Testing Environment

```bash
# .env.test
HTTP_HOST=127.0.0.1
HTTP_PORT=8081
LOG_LEVEL=WARNING
LOG_FORMAT=json
MCP_AUTH_ENABLED=false
MCP_MASK_ERRORS=false
CACHE_ENABLED=false  # Disable cache for consistent tests
METRICS_ENABLED=false
REDIS_URL=redis://localhost:6379/1  # Use different DB
```

### Staging Environment

```bash
# .env.staging
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
LOG_LEVEL=INFO
LOG_FORMAT=json
MCP_AUTH_ENABLED=true
MCP_MASK_ERRORS=true
CACHE_ENABLED=true
METRICS_ENABLED=true
REDIS_URL=redis://staging-redis:6379/0
CHROMADB_URL=http://staging-chromadb:8000
```

### Production Environment

```bash
# .env.production
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/mcp-standards/app.log
LOG_DIR=/var/log/mcp-standards/
MCP_AUTH_ENABLED=true
MCP_MASK_ERRORS=true
MCP_JWT_SECRET=your-production-secret-here
CACHE_ENABLED=true
CACHE_TTL=7200
METRICS_ENABLED=true
REDIS_URL=redis://prod-redis:6379/0
CHROMADB_URL=http://prod-chromadb:8000
GITHUB_TOKEN=ghp_your_production_token_here
```

---

## Performance Tuning

### Memory Optimization

```bash
# For systems with limited memory
CACHE_MAX_SIZE=500
VECTOR_CACHE_SIZE=5000
SEARCH_CACHE_TTL=300
LOG_BACKUP_COUNT=3
```

### High-Performance Setup

```bash
# For high-traffic environments
CACHE_ENABLED=true
CACHE_TTL=7200
CACHE_MAX_SIZE=10000
VECTOR_CACHE_SIZE=100000
SEARCH_CACHE_TTL=1800
HEALTH_CHECK_INTERVAL=15
```

### CPU Optimization

```bash
# Reduce CPU usage
MCP_DISABLE_SEARCH=false  # Keep search enabled but tune
SEARCH_CACHE_TTL=1800  # Cache search results longer
HEALTH_CHECK_INTERVAL=60  # Check health less frequently
```

---

## Security Configuration

### Production Security Checklist

```bash
# ✅ Enable authentication
MCP_AUTH_ENABLED=true

# ✅ Set secure JWT secret
MCP_JWT_SECRET=$(openssl rand -base64 32)

# ✅ Mask errors in production
MCP_MASK_ERRORS=true

# ✅ Secure file permissions
chmod 600 .env
chown app:app .env

# ✅ Use secure Redis connection
REDIS_URL=redis://user:password@redis:6379/0

# ✅ Enable HTTPS (via reverse proxy)
HTTP_HOST=127.0.0.1  # Bind to localhost, use reverse proxy
```

### Network Security

```bash
# Bind to localhost only (use reverse proxy)
HTTP_HOST=127.0.0.1

# Or bind to specific interface
HTTP_HOST=192.168.1.100
```

### Container Security

```dockerfile
# Use non-root user
USER app:app

# Set secure environment
ENV MCP_MASK_ERRORS=true
ENV MCP_AUTH_ENABLED=true
```

---

## Monitoring and Logging

### Prometheus Configuration

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'mcp-standards-server'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Grafana Dashboard

Key metrics to monitor:
- HTTP request rate and latency
- MCP tool call success rate
- Cache hit ratio
- Error rates
- Memory and CPU usage

### Log Aggregation

```bash
# For centralized logging
LOG_FORMAT=json
LOG_FILE=/var/log/mcp-standards/app.log

# Ship logs to ELK/EFK stack
# Configure Filebeat, Fluentd, or similar
```

---

## Troubleshooting

### Common Issues

#### 1. Server Won't Start

**Issue**: Port already in use
```bash
# Solution: Change port or kill process
HTTP_PORT=8081
# or
sudo lsof -ti:8080 | xargs kill -9
```

#### 2. Authentication Failures

**Issue**: Invalid JWT secret
```bash
# Solution: Generate new secret
MCP_JWT_SECRET=$(openssl rand -base64 32)
```

#### 3. Cache Issues

**Issue**: Redis connection failed
```bash
# Solution: Check Redis connection
redis-cli ping
# Update Redis URL if needed
REDIS_URL=redis://localhost:6379/0
```

#### 4. Search Not Working

**Issue**: ChromaDB connection failed
```bash
# Solution: Check ChromaDB status
curl http://localhost:8000/api/v1/heartbeat
# Update ChromaDB URL if needed
CHROMADB_URL=http://localhost:8000
```

### Debug Configuration

```bash
# Enable debug logging
LOG_LEVEL=DEBUG

# Disable error masking
MCP_MASK_ERRORS=false

# Enable all features for testing
MCP_DISABLE_SEARCH=false
CACHE_ENABLED=true
METRICS_ENABLED=true
```

### Health Check Debugging

```bash
# Check service health
curl http://localhost:8080/health

# Check specific components
curl http://localhost:8080/health?checks=database,cache,search

# Check readiness
curl http://localhost:8080/health/ready
```

---

## Configuration Validation

### Environment Variable Validation

The server validates configuration on startup. Common validation errors:

```bash
# Invalid log level
LOG_LEVEL=INVALID  # Error: Invalid log level

# Invalid port
HTTP_PORT=99999  # Error: Port out of range

# Missing required values (when auth enabled)
MCP_AUTH_ENABLED=true
MCP_JWT_SECRET=""  # Error: JWT secret required when auth enabled
```

### Configuration Testing

```bash
# Test configuration
python -m src.main --validate-config

# Test specific environment
python -m src.main --env=production --validate-config
```

---

## Best Practices

### 1. Environment Management

- Use different `.env` files for different environments
- Never commit `.env` files to version 1.0.0
- Use secrets management in production

### 2. Security

- Always enable authentication in production
- Use strong JWT secrets
- Mask errors in production
- Secure file permissions

### 3. Performance

- Enable caching for better performance
- Tune cache sizes based on available memory
- Monitor metrics to optimize configuration

### 4. Monitoring

- Enable metrics collection
- Set up health checks
- Configure log aggregation
- Monitor key performance indicators

### 5. Backup

- Backup configuration files
- Backup data directories
- Document configuration changes

---

## Support

For configuration help:

- Check the [API Documentation](API_DOCUMENTATION.md)
- Review [troubleshooting guide](../cli/troubleshooting.md)
- Open an issue: https://github.com/williamzujkowski/mcp-standards-server/issues