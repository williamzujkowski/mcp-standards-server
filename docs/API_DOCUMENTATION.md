# API Documentation

## Overview

This document provides comprehensive documentation for the MCP Standards Server API endpoints and configuration options.

## Table of Contents

1. [HTTP REST API Endpoints](#http-rest-api-endpoints)
2. [MCP Tools](#mcp-tools)
3. [Configuration Options](#configuration-options)
4. [Authentication](#authentication)
5. [Error Handling](#error-handling)
6. [Rate Limiting](#rate-limiting)

---

## HTTP REST API Endpoints

The MCP Standards Server provides a REST API for health checks, metrics, and standards access.

### Health Check Endpoints

#### `GET /health`

Comprehensive health check endpoint that returns the overall health status of the server.

**Query Parameters:**
- `checks` (optional): Comma-separated list of specific health checks to perform

**Response:**
```json
{
  "status": "healthy|degraded|unhealthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time": 0.05
    },
    "cache": {
      "status": "healthy",
      "response_time": 0.02
    }
  }
}
```

**Status Codes:**
- `200`: Healthy or degraded
- `503`: Unhealthy

#### `GET /health/live`

Kubernetes liveness probe endpoint. Returns basic service availability.

**Response:**
```json
{
  "alive": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes:**
- `200`: Service is alive
- `503`: Service is not alive

#### `GET /health/ready`

Kubernetes readiness probe endpoint. Returns service readiness to handle requests.

**Response:**
```json
{
  "ready": true,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes:**
- `200`: Service is ready
- `503`: Service is not ready

### Metrics Endpoints

#### `GET /metrics`

Prometheus metrics endpoint. Returns metrics in Prometheus format.

**Response:**
```
# HELP mcp_http_requests_total Total number of HTTP requests
# TYPE mcp_http_requests_total counter
mcp_http_requests_total{method="GET",path="/health",status="200"} 42

# HELP mcp_tool_calls_total Total number of MCP tool calls
# TYPE mcp_tool_calls_total counter
mcp_tool_calls_total{tool="get_applicable_standards",success="true"} 15
```

**Content-Type:** `text/plain; version=0.0.4`

### Service Information Endpoints

#### `GET /status`

Returns detailed service status information.

**Response:**
```json
{
  "service": "mcp-standards-server",
  "version": "1.0.0",
  "status": "running",
  "timestamp": "2024-01-01T12:00:00Z",
  "environment": {
    "host": "0.0.0.0",
    "port": 8080,
    "data_dir": "data",
    "log_level": "INFO"
  }
}
```

#### `GET /info`

Returns service information and available endpoints.

**Response:**
```json
{
  "name": "MCP Standards Server",
  "description": "Model Context Protocol server for software development standards",
  "version": "1.0.0",
  "author": "MCP Standards Team",
  "endpoints": {
    "health": "/health",
    "liveness": "/health/live",
    "readiness": "/health/ready",
    "metrics": "/metrics",
    "status": "/status",
    "standards": "/api/standards"
  },
  "documentation": "https://github.com/williamzujkowski/mcp-standards-server"
}
```

### Standards API Endpoints

#### `GET /api/standards`

List all available standards.

**Response:**
```json
{
  "standards": [
    {
      "id": "react-18-patterns",
      "title": "React 18 Patterns and Best Practices",
      "category": "Frontend",
      "description": "Modern React patterns for building scalable applications..."
    }
  ],
  "total": 25,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### `GET /api/standards/{standard_id}`

Get a specific standard by ID.

**Parameters:**
- `standard_id` (required): The ID of the standard to retrieve

**Response:**
```json
{
  "standard": {
    "id": "react-18-patterns",
    "title": "React 18 Patterns and Best Practices",
    "category": "Frontend",
    "description": "Modern React patterns for building scalable applications",
    "content": "...",
    "metadata": {
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:00:00Z"
    }
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

**Status Codes:**
- `200`: Standard found
- `404`: Standard not found

#### `GET /`

Root endpoint with basic service information.

**Response:**
```json
{
  "service": "MCP Standards Server",
  "status": "running",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "info": "/info",
    "standards": "/api/standards",
    "metrics": "/metrics"
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

## MCP Tools

The MCP Standards Server provides three main tools for interacting with standards via the Model Context Protocol.

### get_applicable_standards

Retrieves standards that are applicable to a given project context.

**Parameters:**
```json
{
  "project_context": {
    "type": "object",
    "description": "Project context for standard selection",
    "properties": {
      "project_type": {
        "type": "string",
        "description": "Type of project (e.g., 'web_application', 'mobile_app', 'api')"
      },
      "frameworks": {
        "type": "array",
        "description": "List of frameworks used in the project",
        "items": {"type": "string"}
      },
      "languages": {
        "type": "array",
        "description": "Programming languages used",
        "items": {"type": "string"}
      },
      "requirements": {
        "type": "array",
        "description": "Specific requirements or needs",
        "items": {"type": "string"}
      }
    }
  }
}
```

**Example Usage:**
```json
{
  "project_context": {
    "project_type": "web_application",
    "frameworks": ["react", "express"],
    "languages": ["javascript", "typescript"],
    "requirements": ["accessibility", "security", "performance"]
  }
}
```

**Response:**
```json
{
  "result": {
    "applicable_standards": [
      {
        "id": "react-18-patterns",
        "title": "React 18 Patterns and Best Practices",
        "relevance_score": 0.95,
        "reason": "Matches React framework and frontend requirements"
      }
    ],
    "total_standards": 1,
    "selection_criteria": {
      "framework_matches": ["react"],
      "language_matches": ["javascript", "typescript"],
      "requirement_matches": ["accessibility", "security", "performance"]
    }
  }
}
```

### search_standards

Performs semantic search across all standards.

**Parameters:**
```json
{
  "query": {
    "type": "string",
    "description": "Search query for finding relevant standards",
    "required": true
  },
  "limit": {
    "type": "integer",
    "description": "Maximum number of results to return",
    "default": 10,
    "minimum": 1,
    "maximum": 100
  }
}
```

**Example Usage:**
```json
{
  "query": "React hooks performance optimization",
  "limit": 5
}
```

**Response:**
```json
{
  "result": {
    "standards": [
      {
        "id": "react-18-patterns",
        "title": "React 18 Patterns and Best Practices",
        "relevance_score": 0.92,
        "matched_content": "React hooks performance optimization techniques...",
        "category": "Frontend"
      }
    ],
    "total_results": 1,
    "query_processed": "React hooks performance optimization",
    "search_time": 0.125
  }
}
```

### get_standard

Retrieves a specific standard by its ID.

**Parameters:**
```json
{
  "standard_id": {
    "type": "string",
    "description": "Unique identifier for the standard",
    "required": true
  },
  "version": {
    "type": "string",
    "description": "Specific version 1.0.0
    "default": "latest"
  }
}
```

**Example Usage:**
```json
{
  "standard_id": "react-18-patterns",
  "version": "1.0.0"
}
```

**Response:**
```json
{
  "result": {
    "standard": {
      "id": "react-18-patterns",
      "title": "React 18 Patterns and Best Practices",
      "version": "1.0.0",
      "category": "Frontend",
      "description": "Comprehensive guide to React 18 patterns...",
      "content": "# React 18 Patterns and Best Practices\\n\\n## Overview\\n...",
      "metadata": {
        "created_at": "2024-01-01T12:00:00Z",
        "updated_at": "2024-01-01T12:00:00Z",
        "author": "React Team",
        "tags": ["react", "frontend", "hooks", "performance"]
      }
    }
  }
}
```

---

## Configuration Options

The MCP Standards Server supports extensive configuration through environment variables.

### Core Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_HOST` | `127.0.0.1` | Host address for HTTP server |
| `HTTP_PORT` | `8080` | Port for HTTP server |
| `DATA_DIR` | `./data` | Directory for storing data files |
| `MCP_STANDARDS_DATA_DIR` | `./data/standards` | Directory for standards data |
| `MCP_CONFIG_PATH` | `./config/mcp_config.yaml` | Path to MCP configuration file |
| `LOG_LEVEL` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) |

### Feature Flags

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_ONLY` | `false` | Run only HTTP server without MCP |
| `MCP_DISABLE_SEARCH` | `false` | Disable semantic search features |

### Authentication & Security

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_AUTH_ENABLED` | `false` | Enable authentication for MCP tools |
| `MCP_JWT_SECRET` | `""` | JWT secret key for authentication |
| `MCP_MASK_ERRORS` | `false` | Mask error details in production |

### External Services

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_TOKEN` | `""` | GitHub personal access token for syncing standards |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `CHROMADB_URL` | `http://localhost:8000` | ChromaDB server URL |

### Web Application Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `WEB_UI_ENABLED` | `true` | Enable web UI interface |
| `WEB_UI_PORT` | `3000` | Port for web UI development server |
| `WEB_UI_API_BASE_URL` | `http://localhost:8080` | Base URL for API calls |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_FORMAT` | `json` | Log format (`json` or `text`) |
| `LOG_FILE` | `""` | Log file path (empty = stdout only) |
| `LOG_DIR` | `logs` | Directory for log files |
| `LOG_MAX_SIZE` | `10MB` | Maximum log file size |
| `LOG_BACKUP_COUNT` | `5` | Number of backup log files |

### Performance & Caching

| Variable | Default | Description |
|----------|---------|-------------|
| `CACHE_ENABLED` | `true` | Enable caching |
| `CACHE_TTL` | `3600` | Cache time-to-live in seconds |
| `CACHE_MAX_SIZE` | `1000` | Maximum cache entries |
| `VECTOR_CACHE_SIZE` | `10000` | Vector cache size |
| `SEARCH_CACHE_TTL` | `300` | Search cache TTL in seconds |

### Monitoring & Metrics

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICS_ENABLED` | `true` | Enable metrics collection |
| `METRICS_PORT` | `9090` | Prometheus metrics port |
| `METRICS_PATH` | `/metrics` | Metrics endpoint path |
| `HEALTH_CHECK_INTERVAL` | `30` | Health check interval in seconds |

---

## Authentication

The MCP Standards Server supports JWT-based authentication for MCP tools when enabled.

### Enabling Authentication

Set the following environment variables:

```bash
MCP_AUTH_ENABLED=true
MCP_JWT_SECRET=your-secret-key-here
```

### JWT Token Format

```json
{
  "sub": "user_id",
  "iat": 1640995200,
  "exp": 1640998800,
  "scope": ["standards:read", "standards:search"]
}
```

### Required Scopes

- `standards:read`: Access to get_standard tool
- `standards:search`: Access to search_standards tool
- `standards:list`: Access to get_applicable_standards tool

---

## Error Handling

The API uses standard HTTP status codes and provides detailed error responses.

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (validation errors)
- `401`: Unauthorized (authentication required)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found (resource not found)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error
- `503`: Service Unavailable (health check failed)

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_INVALID_PARAMETERS",
    "message": "Invalid parameter provided",
    "details": {
      "field": "standard_id",
      "reason": "Standard ID cannot be empty"
    },
    "suggestion": "Provide a valid standard ID",
    "timestamp": "2024-01-01T12:00:00Z"
  }
}
```

### MCP Error Codes

- `VALIDATION_INVALID_PARAMETERS`: Invalid input parameters
- `TOOL_NOT_FOUND`: Requested tool not available
- `TOOL_EXECUTION_FAILED`: Tool execution failed
- `AUTH_INVALID_TOKEN`: Invalid authentication token
- `AUTH_INSUFFICIENT_PERMISSIONS`: Insufficient permissions
- `SYSTEM_INTERNAL_ERROR`: Internal server error

---

## Rate Limiting

The server implements rate limiting to prevent abuse and ensure fair usage.

### Rate Limits

- HTTP API: 100 requests per minute per IP
- MCP Tools: 50 calls per minute per user
- Search Tools: 20 calls per minute per user

### Rate Limit Headers

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded Response

```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": {
      "limit": 100,
      "remaining": 0,
      "reset_time": "2024-01-01T12:01:00Z"
    }
  }
}
```

---

## Examples

### Complete cURL Examples

#### Health Check

```bash
curl -X GET http://localhost:8080/health
```

#### Get All Standards

```bash
curl -X GET http://localhost:8080/api/standards
```

#### Get Specific Standard

```bash
curl -X GET http://localhost:8080/api/standards/react-18-patterns
```

#### Prometheus Metrics

```bash
curl -X GET http://localhost:8080/metrics
```

### MCP Client Examples

#### Python MCP Client

```python
import asyncio
from mcp_client import MCPClient

async def main():
    client = MCPClient("stdio://python -m mcp_standards_server")
    
    # Search for standards
    result = await client.call_tool("search_standards", {
        "query": "React performance optimization",
        "limit": 5
    })
    
    print(result)

asyncio.run(main())
```

#### Node.js MCP Client

```javascript
const { MCPClient } = require('@modelcontextprotocol/client');

async function main() {
  const client = new MCPClient({
    command: 'python',
    args: ['-m', 'mcp_standards_server']
  });
  
  await client.connect();
  
  const result = await client.callTool('get_applicable_standards', {
    project_context: {
      project_type: 'web_application',
      frameworks: ['react'],
      languages: ['javascript']
    }
  });
  
  console.log(result);
  await client.close();
}

main();
```

---

## Configuration Examples

### Production Configuration

```bash
# .env.production
HTTP_HOST=0.0.0.0
HTTP_PORT=8080
LOG_LEVEL=INFO
LOG_FORMAT=json
MCP_AUTH_ENABLED=true
MCP_MASK_ERRORS=true
CACHE_ENABLED=true
METRICS_ENABLED=true
REDIS_URL=redis://redis:6379/0
CHROMADB_URL=http://chromadb:8000
```

### Development Configuration

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
```

### Docker Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  mcp-standards-server:
    image: mcp-standards-server:latest
    environment:
      - HTTP_HOST=0.0.0.0
      - HTTP_PORT=8080
      - REDIS_URL=redis://redis:6379/0
      - CHROMADB_URL=http://chromadb:8000
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - chromadb
```

---

## Support

For additional support and documentation:

- **GitHub Repository**: https://github.com/williamzujkowski/mcp-standards-server
- **Issues**: https://github.com/williamzujkowski/mcp-standards-server/issues
- **Documentation**: https://github.com/williamzujkowski/mcp-standards-server/docs
- **Community**: https://github.com/williamzujkowski/mcp-standards-server/discussions