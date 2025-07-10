# MCP Standards Server Web UI - Verified Deployment Guide

**Date Verified:** 2025-01-10  
**Status:** ⚠️ Partial Success - Backend functional, Frontend requires fixes

## Current Web UI Structure

```
web/
├── backend/              # FastAPI backend
│   ├── main.py          # Main FastAPI application
│   ├── engine_adapter.py # Adapter for standards engine
│   ├── models.py        # Data models
│   ├── auth.py          # Authentication (if needed)
│   └── requirements.txt # Backend dependencies
├── frontend/            # React frontend
│   ├── src/            # React source code
│   ├── public/         # Static assets
│   ├── package.json    # Frontend dependencies
│   └── tsconfig.json   # TypeScript config
├── deployment/         # Docker configurations
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── docker-compose.yml  # Multi-container setup
└── start.sh           # Deployment script
```

## Deployment Test Results

### ✅ Successful Components

1. **Backend Infrastructure**
   - FastAPI backend builds successfully
   - Standards YAML files are present in `/data/standards/`
   - Engine adapter correctly loads standards from YAML files
   - Redis service starts correctly
   - Docker Compose configuration is valid

2. **Standards Data**
   - 8 YAML standards files verified in data directory
   - Standards include: AI/ML Operations, Blockchain/Web3, IoT/Edge Computing, Gaming, AR/VR, API Design, Database Design, Sustainability

### ❌ Issues Found and Fixes Applied

1. **Missing OS import in backend/main.py**
   - **Issue:** `os` module used but not imported
   - **Fix Applied:** Added `import os` to imports

2. **Invalid react-scripts version**
   - **Issue:** package.json had `"react-scripts": "^0.0.0"`
   - **Fix Applied:** Changed to `"react-scripts": "^5.0.1"`

3. **Missing .env.example**
   - **Issue:** Documentation referenced missing file
   - **Status:** File actually exists with proper configuration

4. **Docker Compose version warning**
   - **Issue:** Obsolete `version` attribute in docker-compose.yml
   - **Fix Applied:** Removed `version: '3.8'` line

5. **Docker Compose command**
   - **Issue:** Script used `docker-compose` (old syntax)
   - **Fix Applied:** Updated to `docker compose` (new syntax)

6. **Frontend build failures**
   - **Issue:** Missing default exports in React components
   - **Partial Fix:** Added default export to Layout.tsx
   - **Remaining Issues:** Dashboard and other page components need default exports

## Quick Deployment Steps (Backend Only)

Since the frontend has unresolved issues, here's how to deploy just the backend API:

```bash
# 1. Navigate to web directory
cd /home/william/git/mcp-standards-server/web

# 2. Create .env from example
cp .env.example .env

# 3. Generate secure secret key
sed -i "s/your-secret-key-here-change-in-production/$(openssl rand -hex 32)/g" .env

# 4. Run only backend and redis services
docker compose up -d backend redis

# 5. Access the API
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
# - Health Check: http://localhost:8000/health
```

## Manual Backend Testing

You can test the backend API directly:

```bash
# Get all standards
curl http://localhost:8000/api/standards

# Search standards
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI", "limit": 10}'

# Get categories
curl http://localhost:8000/api/categories

# Get tags
curl http://localhost:8000/api/tags
```

## Frontend Fixes Required

To complete the deployment, these React component issues need resolution:

1. Add default exports to all page components:
   - `/pages/Dashboard.tsx`
   - `/pages/StandardsBrowser.tsx`
   - `/pages/Search.tsx`
   - `/pages/StandardDetail.tsx`
   - `/pages/RuleTesting.tsx`

2. Fix any TypeScript type errors in components

3. Ensure all imports match the export types

## Alternative Deployment Options

### Option 1: Run Backend Locally (No Docker)

```bash
cd web/backend
pip install -r requirements.txt
pip install -e ../..
python main.py
```

### Option 2: Use Pre-built Images (When Available)

```yaml
# Modified docker-compose.yml
services:
  backend:
    image: mcp-standards-backend:latest  # Use pre-built image
    # ... rest of config
```

## Integration with MCP Server

The web UI backend integrates with the MCP server through:

1. **Standards Loading:** Reads YAML files from `/data/standards/`
2. **Engine Adapter:** Simplified adapter bypasses complex MCP engine
3. **API Endpoints:** RESTful API for standards access
4. **WebSocket Support:** Real-time updates capability

## Monitoring and Logs

```bash
# View all logs
docker compose logs -f

# View backend logs only
docker compose logs -f backend

# Check service status
docker compose ps

# Health check
curl http://localhost:8000/health
```

## Security Considerations

1. **Change SECRET_KEY:** Always generate a new secret key for production
2. **CORS Settings:** Update allowed origins in main.py for production
3. **Redis Security:** Consider adding Redis password in production
4. **HTTPS:** Use reverse proxy with SSL for production deployment

## Next Steps for Full Deployment

1. **Fix Frontend Components:** Add missing default exports
2. **Test Full Stack:** Verify frontend can communicate with backend
3. **Performance Testing:** Load test with multiple concurrent users
4. **CI/CD Integration:** Automate build and deployment process
5. **Monitoring Setup:** Add Prometheus/Grafana for production

## Troubleshooting

### Backend won't start
- Check if port 8000 is already in use
- Verify Redis is running: `docker compose ps`
- Check logs: `docker compose logs backend`

### Frontend build fails
- Clear node_modules: `rm -rf frontend/node_modules`
- Regenerate package-lock.json
- Check for TypeScript errors

### Standards not loading
- Verify YAML files exist in `/data/standards/`
- Check file permissions
- Review backend logs for parsing errors

## Conclusion

The web UI backend is functional and can serve the MCP standards through a RESTful API. The frontend requires component fixes before it can be fully deployed. For immediate use, the backend API can be accessed directly or integrated with other frontends.