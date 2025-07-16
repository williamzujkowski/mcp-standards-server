# serve Command

Start the MCP Standards Server for Model Context Protocol integration.

## Synopsis

```bash
mcp-standards serve [options]
```

## Description

The `serve` command starts the MCP Standards Server, which provides a Model Context Protocol interface for LLMs and development tools to interact with standards programmatically. This enables real-time standard queries, validation, and code generation assistance.

## Options

### `--host <address>`
Host address to bind to (default: localhost).

```bash
mcp-standards serve --host 0.0.0.0
```

### `--port <number>`
Port to listen on (default: 3000).

```bash
mcp-standards serve --port 8080
```

### `--stdio`
Run in stdio mode for direct tool integration.

```bash
mcp-standards serve --stdio
```

### `--socket <path>`
Use Unix domain socket instead of TCP.

```bash
mcp-standards serve --socket /tmp/mcp-standards.sock
```

### `--workers <n>`
Number of worker processes (default: CPU count).

```bash
mcp-standards serve --workers 4
```

### `--log-level <level>`
Set logging level (debug, info, warning, error).

```bash
mcp-standards serve --log-level debug
```

### `--auth <type>`
Enable authentication (none, token, oauth).

```bash
mcp-standards serve --auth token
```

### `--tls-cert <file>`
TLS certificate file for HTTPS.

```bash
mcp-standards serve --tls-cert cert.pem --tls-key key.pem
```

### `--daemon`
Run as background daemon.

```bash
mcp-standards serve --daemon
```

## Examples

### Basic Server Start

```bash
mcp-standards serve
```

Output:
```
Starting MCP Standards Server v1.0.0
Loading standards from cache...
✓ Loaded 45 standards files
✓ Initialized rule engine
✓ Search index ready
✓ Token optimizer configured (gpt-4, 8000 tokens)

Server listening on http://localhost:3000
Available MCP tools:
  - get_applicable_standards
  - validate_code
  - search_standards
  - get_standard_content
  - check_compliance

Press Ctrl+C to stop
```

### Production Server

```bash
mcp-standards serve \
  --host 0.0.0.0 \
  --port 443 \
  --workers 8 \
  --auth token \
  --tls-cert /etc/ssl/certs/mcp.crt \
  --tls-key /etc/ssl/private/mcp.key \
  --log-level info \
  --daemon
```

### Stdio Mode for Tool Integration

```bash
# For direct integration with LLM tools
mcp-standards serve --stdio
```

When running in stdio mode, the server communicates via standard input/output:
```json
{"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1}
{"jsonrpc": "2.0", "result": {"capabilities": {...}}, "id": 1}
```

### Unix Socket Mode

```bash
# For local IPC communication
mcp-standards serve --socket /var/run/mcp-standards.sock
```

## MCP Tools Available

### get_applicable_standards

Get standards based on project context:

```json
{
  "tool": "get_applicable_standards",
  "arguments": {
    "context": {
      "project_type": "web-application",
      "frameworks": ["react", "nextjs"],
      "languages": ["typescript"],
      "requirements": ["accessibility", "performance"]
    },
    "include_resolution_details": true
  }
}
```

### validate_code

Validate code against standards:

```json
{
  "tool": "validate_code",
  "arguments": {
    "code": "const Button = ({onClick}) => <button onClick={onClick}>Click</button>",
    "language": "javascript",
    "standards": ["react-18-patterns", "wcag-2.2-accessibility"]
  }
}
```

### search_standards

Search standards using natural language:

```json
{
  "tool": "search_standards",
  "arguments": {
    "query": "How to implement secure authentication in Node.js?",
    "limit": 5,
    "include_content": true
  }
}
```

### get_standard_content

Retrieve specific standard content:

```json
{
  "tool": "get_standard_content",
  "arguments": {
    "standard_id": "react-18-patterns",
    "format": "condensed",
    "token_budget": 2000
  }
}
```

## Configuration

Server configuration in `.mcp-standards.yaml`:

```yaml
server:
  host: localhost
  port: 3000
  workers: auto  # or specific number
  
  # Logging
  log_level: info
  log_file: /var/log/mcp-standards.log
  
  # Authentication
  auth:
    type: token  # none, token, oauth
    token_file: /etc/mcp-standards/tokens.json
    oauth:
      provider: github
      client_id: xxx
      client_secret: xxx
  
  # TLS/SSL
  tls:
    enabled: false
    cert: /path/to/cert.pem
    key: /path/to/key.pem
    ca: /path/to/ca.pem
  
  # Performance
  cache:
    enabled: true
    ttl: 3600
    max_size: 1000
  
  # Token optimization
  token_optimization:
    enabled: true
    default_budget: 8000
    model_type: gpt-4
    
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst: 100
```

## Authentication

### Token Authentication

```bash
# Generate tokens
mcp-standards serve --generate-token --name "vscode-client"
Token generated: mcp_token_a1b2c3d4e5f6

# Start server with token auth
mcp-standards serve --auth token
```

Client usage:
```bash
curl -H "Authorization: Bearer mcp_token_a1b2c3d4e5f6" \
  http://localhost:3000/tools/get_applicable_standards
```

### OAuth Authentication

```bash
# Configure OAuth
mcp-standards config --set server.auth.type oauth
mcp-standards config --set server.auth.oauth.provider github

# Start server
mcp-standards serve --auth oauth
```

## Monitoring

### Health Check Endpoint

```bash
curl http://localhost:3000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600,
  "standards_loaded": 45,
  "cache_size": 3.42,
  "workers": 4,
  "memory_usage_mb": 128.5,
  "last_sync": "2025-07-08T10:15:30Z"
}
```

### Metrics Endpoint

```bash
curl http://localhost:3000/metrics
```

Prometheus format output:
```
# HELP mcp_requests_total Total number of MCP requests
# TYPE mcp_requests_total counter
mcp_requests_total{tool="get_applicable_standards"} 1234

# HELP mcp_request_duration_seconds Request duration in seconds
# TYPE mcp_request_duration_seconds histogram
mcp_request_duration_seconds_bucket{le="0.1"} 950
```

### Logging

```bash
# View logs
tail -f /var/log/mcp-standards.log

# Log format
2025-07-08 10:30:45 INFO: Request received: get_applicable_standards
2025-07-08 10:30:45 DEBUG: Context: {project_type: "web", frameworks: ["react"]}
2025-07-08 10:30:45 INFO: Found 7 applicable standards
2025-07-08 10:30:45 INFO: Response sent in 45ms
```

## Integration Examples

### VS Code Extension

```json
// .vscode/settings.json
{
  "mcp-standards.server.url": "http://localhost:3000",
  "mcp-standards.server.token": "mcp_token_xxx",
  "mcp-standards.validation.onSave": true,
  "mcp-standards.suggestions.enabled": true
}
```

### Claude Desktop Configuration

```json
// claude_desktop_config.json
{
  "mcpServers": {
    "standards": {
      "command": "mcp-standards",
      "args": ["serve", "--stdio"],
      "env": {
        "MCP_STANDARDS_CONFIG": "/home/user/.config/mcp-standards/config.yaml"
      }
    }
  }
}
```

### Docker Deployment

```dockerfile
FROM python:3.11-slim

RUN pip install mcp-standards-server

COPY config.yaml /etc/mcp-standards/

EXPOSE 3000

CMD ["mcp-standards", "serve", \
     "--host", "0.0.0.0", \
     "--port", "3000", \
     "--config", "/etc/mcp-standards/config.yaml"]
```

### Systemd Service

```ini
# /etc/systemd/system/mcp-standards.service
[Unit]
Description=MCP Standards Server
After=network.target

[Service]
Type=simple
User=mcp-standards
Group=mcp-standards
WorkingDirectory=/var/lib/mcp-standards
ExecStart=/usr/local/bin/mcp-standards serve --daemon
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Nginx Reverse Proxy

```nginx
server {
    listen 443 ssl http2;
    server_name standards.example.com;
    
    ssl_certificate /etc/ssl/certs/example.crt;
    ssl_certificate_key /etc/ssl/private/example.key;
    
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.0.0
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        
        # MCP specific headers
        proxy_set_header X-MCP-Client $remote_addr;
        proxy_read_timeout 300s;
    }
}
```

## Performance Tuning

### Worker Configuration

```bash
# Auto-detect optimal workers
mcp-standards serve --workers auto

# Manual configuration for high load
mcp-standards serve --workers 16 --log-level warning
```

### Memory Optimization

```yaml
# config.yaml
server:
  memory:
    max_heap_mb: 2048
    gc_interval: 300  # seconds
    cache_strategy: lru
    preload_standards: true
```

### Connection Pooling

```yaml
server:
  connections:
    max_concurrent: 1000
    timeout_seconds: 30
    keepalive: true
    compression: gzip
```

## Troubleshooting

### Debug Mode

```bash
# Enable detailed debugging
MCP_DEBUG=1 mcp-standards serve --log-level debug
```

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   lsof -i :3000
   # Use different port
   mcp-standards serve --port 3001
   ```

2. **Standards Not Loading**
   ```bash
   # Ensure standards are synced
   mcp-standards sync
   # Check cache
   mcp-standards cache --verify
   ```

3. **Authentication Failures**
   ```bash
   # Regenerate tokens
   mcp-standards serve --regenerate-tokens
   # Check token permissions
   mcp-standards serve --list-tokens
   ```

## Related Commands

- [sync](./sync.md) - Sync standards before serving
- [status](./status.md) - Check server readiness
- [config](./config.md) - Configure server settings