# MCP Standards Server Web UI Deployment Guide

## Prerequisites

- Docker and Docker Compose installed
- Domain name (optional)
- SSL certificate (optional, recommended for production)

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/your-org/mcp-standards-server.git
cd mcp-standards-server/web
```

2. Copy environment configuration:
```bash
cp .env.example .env
```

3. Edit `.env` with your configuration:
```bash
# Generate a secure secret key
SECRET_KEY=$(openssl rand -hex 32)
```

4. Build and start the services:
```bash
docker-compose up -d --build
```

5. Access the application:
- Frontend: http://localhost
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Production Deployment

### 1. SSL Configuration

For production, use a reverse proxy like Nginx with SSL:

```nginx
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /path/to/ssl/cert.pem;
    ssl_certificate_key /path/to/ssl/key.pem;

    location / {
        proxy_pass http://localhost:80;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Environment Variables

Set these environment variables for production:

```bash
# Backend
SECRET_KEY=<generate-secure-key>
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://redis:6379

# Frontend
REACT_APP_API_URL=https://your-domain.com
```

### 3. Database Setup

For production, use PostgreSQL:

```bash
# Add to docker-compose.yml
postgres:
  image: postgres:15
  environment:
    POSTGRES_DB: mcp_standards
    POSTGRES_USER: mcp_user
    POSTGRES_PASSWORD: secure_password
  volumes:
    - postgres_data:/var/lib/postgresql/data
  restart: unless-stopped
```

### 4. Scaling

To scale the backend:

```bash
docker-compose up -d --scale backend=3
```

Add a load balancer like HAProxy or use Docker Swarm/Kubernetes.

### 5. Monitoring

Add monitoring with Prometheus and Grafana:

```yaml
# Add to docker-compose.yml
prometheus:
  image: prom/prometheus
  volumes:
    - ./prometheus.yml:/etc/prometheus/prometheus.yml
    - prometheus_data:/prometheus
  ports:
    - "9090:9090"

grafana:
  image: grafana/grafana
  ports:
    - "3000:3000"
  volumes:
    - grafana_data:/var/lib/grafana
```

### 6. Backup

Backup strategy:
```bash
# Backup database
docker exec postgres pg_dump -U mcp_user mcp_standards > backup.sql

# Backup Redis
docker exec redis redis-cli BGSAVE

# Backup volumes
docker run --rm -v mcp_standards_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Maintenance

### Update Application

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose down
docker-compose up -d --build
```

### View Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Health Checks

The application includes health check endpoints:
- Backend: `GET /health`
- Frontend: Static file serving check

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yml
2. **Memory issues**: Increase Docker memory limit
3. **Connection refused**: Check firewall rules
4. **CORS errors**: Verify API_URL in frontend .env

### Debug Mode

Enable debug logging:
```bash
# Backend
LOG_LEVEL=DEBUG

# Frontend
REACT_APP_DEBUG=true
```

## Security Best Practices

1. Use strong SECRET_KEY
2. Enable HTTPS in production
3. Set up firewall rules
4. Regular security updates
5. Use environment-specific configs
6. Enable rate limiting
7. Set up intrusion detection

## Performance Optimization

1. Enable Redis caching
2. Use CDN for static assets
3. Enable gzip compression
4. Optimize database queries
5. Use connection pooling
6. Enable HTTP/2

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/mcp-standards-server/issues
- Documentation: https://docs.your-domain.com