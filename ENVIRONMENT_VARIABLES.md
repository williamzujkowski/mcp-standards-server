# Environment Variables Documentation

This document lists all environment variables used throughout the MCP Standards Server codebase.

## Core Application Environment Variables

### HTTP Server Configuration
- **`HTTP_HOST`**
  - **File**: `src/http_server.py` (line 317), `src/main.py` (line 46)
  - **Purpose**: Host address for the HTTP server
  - **Default**: `"0.0.0.0"`
  - **Usage**: Configures the network interface the HTTP server binds to

- **`HTTP_PORT`**
  - **File**: `src/http_server.py` (line 318), `src/main.py` (line 47)
  - **Purpose**: Port number for the HTTP server
  - **Default**: `8080`
  - **Usage**: Configures the port the HTTP server listens on

- **`DATA_DIR`**
  - **File**: `src/http_server.py` (line 181)
  - **Purpose**: Directory for storing data files
  - **Default**: `"data"`
  - **Usage**: Base directory for standards data and cache

- **`LOG_LEVEL`**
  - **File**: `src/http_server.py` (line 182), `src/main.py` (line 115)
  - **Purpose**: Logging verbosity level
  - **Default**: `"INFO"`
  - **Usage**: Controls the amount of log output (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### MCP Server Configuration
- **`MCP_CONFIG_PATH`**
  - **File**: `src/main.py` (line 132), `src/mcp_server.py` (line 1364)
  - **Purpose**: Path to MCP configuration file
  - **Default**: `"config.json"`
  - **Usage**: JSON file containing server configuration

- **`HTTP_ONLY`**
  - **File**: `src/main.py` (line 78)
  - **Purpose**: Run in HTTP-only mode without MCP server
  - **Default**: `"false"`
  - **Usage**: Set to "true" to run only the HTTP health/monitoring endpoints

- **`MCP_STANDARDS_DATA_DIR`**
  - **File**: `src/mcp_server.py` (line 86)
  - **Purpose**: Directory for MCP standards data
  - **Default**: `"data"`
  - **Usage**: Root directory for standards storage and caching

- **`MCP_DISABLE_SEARCH`**
  - **File**: `src/mcp_server.py` (line 108)
  - **Purpose**: Disable semantic search functionality
  - **Default**: Not set (search enabled)
  - **Usage**: Set to "true" to disable semantic search features

### Authentication Environment Variables
- **`MCP_AUTH_ENABLED`**
  - **File**: `src/core/auth.py` (line 42)
  - **Purpose**: Enable authentication for MCP server
  - **Default**: `"false"`
  - **Usage**: Set to "true" to require authentication for tool access

- **`MCP_JWT_SECRET`**
  - **File**: `src/core/auth.py` (line 45)
  - **Purpose**: Secret key for JWT token signing
  - **Default**: Auto-generated if auth enabled but not provided
  - **Usage**: Used to sign and verify JWT authentication tokens

### Error Handling
- **`MCP_MASK_ERRORS`**
  - **File**: `src/core/errors.py` (line 338)
  - **Purpose**: Mask sensitive information in error messages
  - **Default**: `"true"`
  - **Usage**: Set to "false" to show detailed error messages (development only)

### GitHub Integration
- **`GITHUB_TOKEN`**
  - **File**: `src/core/standards/sync.py` (lines 303, 347)
  - **Purpose**: GitHub API authentication token
  - **Default**: None (unauthenticated access)
  - **Usage**: Provides authenticated access to GitHub API for higher rate limits

## Docker Environment Variables

### Main docker-compose.yml
- **`REDIS_URL`**
  - **Purpose**: Redis connection URL
  - **Default**: `"redis://redis:6379"`
  - **Usage**: Connection string for Redis cache

- **`CHROMADB_URL`**
  - **Purpose**: ChromaDB connection URL
  - **Default**: `"http://chromadb:8000"`
  - **Usage**: Connection string for vector database

- **`MCP_SERVER_HOST`**
  - **Purpose**: MCP server host address
  - **Default**: `"0.0.0.0"`
  - **Usage**: Network interface for MCP server

- **`MCP_SERVER_PORT`**
  - **Purpose**: MCP server port
  - **Default**: `8080`
  - **Usage**: Port for MCP server

- **`WEB_UI_PORT`**
  - **Purpose**: Web UI port
  - **Default**: `3000`
  - **Usage**: Port for web interface

### Web Application (web/docker-compose.yml)
- **`SECRET_KEY`**
  - **Purpose**: Web application secret key
  - **Default**: `"your-secret-key-here"`
  - **Usage**: Used for session management and CSRF protection

- **`DATABASE_URL`**
  - **Purpose**: Database connection string
  - **Default**: `"sqlite:///./app.db"`
  - **Usage**: Connection string for web app database

- **`REACT_APP_API_URL`**
  - **Purpose**: Backend API URL for React frontend
  - **Default**: `"http://localhost:8000"`
  - **Usage**: Base URL for frontend API calls

- **`NODE_ENV`**
  - **Purpose**: Node.js environment
  - **Default**: `"production"`
  - **Usage**: Controls React build optimizations

- **`GENERATE_SOURCEMAP`**
  - **Purpose**: Generate source maps for React build
  - **Default**: `"false"`
  - **Usage**: Disable source maps in production

### Monitoring (Grafana)
- **`GF_SECURITY_ADMIN_PASSWORD`**
  - **Purpose**: Grafana admin password
  - **Default**: `"admin"`
  - **Usage**: Initial admin password for Grafana

- **`GF_USERS_ALLOW_SIGN_UP`**
  - **Purpose**: Allow user registration in Grafana
  - **Default**: `"false"`
  - **Usage**: Disable public user registration

## Python Build Environment Variables (Dockerfile)
- **`PYTHONDONTWRITEBYTECODE`**
  - **Purpose**: Prevent Python from writing .pyc files
  - **Default**: `1`
  - **Usage**: Reduces container size

- **`PYTHONUNBUFFERED`**
  - **Purpose**: Force Python output to be unbuffered
  - **Default**: `1`
  - **Usage**: Ensures logs are immediately visible

- **`PYTHONPATH`**
  - **Purpose**: Python module search path
  - **Default**: `/app/src`
  - **Usage**: Allows imports from src directory

- **`PIP_NO_CACHE_DIR`**
  - **Purpose**: Disable pip cache
  - **Default**: `1`
  - **Usage**: Reduces container size

- **`PIP_DISABLE_PIP_VERSION_CHECK`**
  - **Purpose**: Disable pip version check
  - **Default**: `1`
  - **Usage**: Speeds up pip operations

## Usage Examples

### Development
```bash
# Run with custom configuration
export MCP_CONFIG_PATH=/path/to/custom/config.json
export LOG_LEVEL=DEBUG
export MCP_MASK_ERRORS=false
python -m src.main
```

### Production with Docker
```bash
# Create .env file
cat > .env << EOF
SECRET_KEY=your-production-secret-key
GITHUB_TOKEN=ghp_yourGithubToken
MCP_AUTH_ENABLED=true
MCP_JWT_SECRET=your-jwt-secret
REDIS_URL=redis://redis-prod:6379
EOF

# Run with docker-compose
docker-compose up -d
```

### Authentication Setup
```bash
# Enable authentication
export MCP_AUTH_ENABLED=true
export MCP_JWT_SECRET=your-secret-key-here

# Run server
python -m src.main
```

## Security Notes

1. **Never commit real values** for `SECRET_KEY`, `MCP_JWT_SECRET`, or `GITHUB_TOKEN`
2. Use `.env` files for local development (already in .gitignore)
3. In production, use proper secret management (e.g., AWS Secrets Manager, Kubernetes Secrets)
4. Always set `MCP_MASK_ERRORS=true` in production to prevent information leakage
5. Generate strong random values for secret keys:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```