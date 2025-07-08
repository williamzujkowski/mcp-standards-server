# MCP Tool Response Caching

The MCP Standards Server includes a sophisticated caching system designed to improve performance and reduce load on backend systems. This document covers the caching architecture, configuration, and best practices.

## Overview

The caching system provides:

- **Intelligent per-tool caching** with configurable TTLs and strategies
- **Selective caching** - not all tools should be cached
- **Automatic compression** for large responses
- **Cache invalidation** with cascade support
- **Cache warming** capabilities for frequently accessed data
- **Detailed metrics** for monitoring and optimization
- **Redis integration** with connection pooling and circuit breakers
- **Two-tier caching** (L1 in-memory + L2 Redis)

## Architecture

### Components

1. **MCPCache** (`src/core/cache/mcp_cache.py`)
   - Core caching logic
   - Tool configuration management
   - Compression and serialization
   - Metrics collection

2. **MCPCacheMiddleware** (`src/core/cache/mcp_cache_integration.py`)
   - Integration with MCP server
   - Request interception
   - Cache management tools

3. **RedisCache** (`src/core/cache/redis_client.py`)
   - Redis connection management
   - Connection pooling
   - Circuit breaker pattern
   - L1/L2 cache tiers

### Cache Flow

```
Request → MCP Server → Cache Middleware → Check Cache
                                            ↓
                                   [Cache Hit] → Return Cached Response
                                            ↓
                                   [Cache Miss] → Execute Tool
                                            ↓
                                        Cache Response
                                            ↓
                                        Return Response
```

## Configuration

### Basic Setup

1. **Environment Variables**:
   ```bash
   export REDIS_HOST=localhost
   export REDIS_PORT=6379
   export MCP_CACHE_ENABLED=true
   ```

2. **Configuration File** (`config/cache.yaml`):
   ```yaml
   redis:
     host: localhost
     port: 6379
     default_ttl: 300
   
   warm_on_startup: true
   
   tools:
     get_standard_details:
       strategy: long_ttl
       ttl_seconds: 86400
   ```

### Cache Strategies

The system supports five caching strategies:

| Strategy | Default TTL | Use Case |
|----------|------------|----------|
| `no_cache` | N/A | Write operations, sensitive data |
| `short_ttl` | 5 minutes | Frequently changing data |
| `medium_ttl` | 30 minutes | Moderately stable data |
| `long_ttl` | 24 hours | Stable reference data |
| `permanent` | No expiry | Static data (manual invalidation) |

### Tool Configuration

Each tool can be configured with:

```yaml
tool_name:
  strategy: medium_ttl          # Caching strategy
  ttl_seconds: 1800            # Override default TTL
  compress_threshold: 1024     # Compress if larger (bytes)
  include_in_key:              # Args to include in cache key
    - arg1
    - arg2
  exclude_from_key:            # Args to exclude from key
    - timestamp
  invalidate_on:               # Tools that invalidate this cache
    - update_tool
    - delete_tool
  warm_on_startup: true        # Warm cache on startup
  warm_args:                   # Arguments for warming
    - id: 1
    - id: 2
```

## Integration

### Automatic Integration

The easiest way to add caching to your MCP server:

```python
from src.core.cache.mcp_cache_integration import integrate_cache_with_mcp_server

# In your MCP server initialization
server = MCPStandardsServer(config)
cache_config = load_config("config/cache.yaml")
integrate_cache_with_mcp_server(server, cache_config)
```

### Manual Integration

For more control:

```python
from src.core.cache.mcp_cache import MCPCache
from src.core.cache.redis_client import RedisCache, CacheConfig

# Create Redis client
redis_config = CacheConfig(host="localhost", port=6379)
redis_cache = RedisCache(redis_config)

# Create MCP cache
mcp_cache = MCPCache(redis_cache=redis_cache)

# Wrap tool execution
async def cached_execute_tool(tool_name, arguments):
    # Try cache first
    cached = await mcp_cache.get(tool_name, arguments)
    if cached is not None:
        return cached
    
    # Execute tool
    result = await original_execute_tool(tool_name, arguments)
    
    # Cache result
    await mcp_cache.set(tool_name, arguments, result)
    
    return result
```

### Using the Decorator

For individual functions:

```python
from src.core.cache.mcp_cache import cache_tool_response

@cache_tool_response(cache, "my_tool", ttl_override=3600)
async def my_tool_handler(arguments):
    # Expensive operation
    return await fetch_data(arguments)
```

## Cache Management

### Cache Management Tools

The integration adds several tools to the MCP server:

1. **get_cache_stats** - Get cache metrics and health status
   ```json
   {
     "include_redis": true
   }
   ```

2. **cache_invalidate** - Invalidate specific cache entries
   ```json
   {
     "tool_name": "get_standard_details",
     "arguments": {"standard_id": "test-std"}
   }
   ```

3. **cache_warm** - Warm cache for specific tools
   ```json
   {
     "tools": ["get_standard_details", "list_templates"]
   }
   ```

4. **cache_clear_all** - Clear all caches (requires confirmation)
   ```json
   {
     "confirm": true
   }
   ```

5. **cache_configure** - Dynamically configure tool caching
   ```json
   {
     "tool_name": "search_standards",
     "strategy": "short_ttl",
     "ttl_seconds": 180
   }
   ```

### Programmatic Management

```python
# Invalidate specific cache
await cache.invalidate("tool_name", {"arg": "value"})

# Invalidate all entries for a tool
await cache.invalidate("tool_name")

# Clear all caches
count = await cache.clear_all()

# Get metrics
metrics = cache.get_metrics()
```

## Cache Warming

### Automatic Warming

Configure tools to warm on startup:

```yaml
tools:
  get_standard_details:
    warm_on_startup: true
    warm_args:
      - standard_id: "secure-api-design"
      - standard_id: "react-best-practices"
```

### Manual Warming

```python
# Warm specific tools
results = await cache.warm_cache(executor, ["tool1", "tool2"])

# Start background warming
await cache.start_background_warming(executor, interval_seconds=3600)
```

## Monitoring

### Metrics

The cache system collects detailed metrics:

```python
metrics = cache.get_metrics()
# {
#   "overall": {
#     "hits": 1000,
#     "misses": 200,
#     "hit_rate": 0.833,
#     "errors": 5,
#     "invalidations": 50
#   },
#   "performance": {
#     "avg_hit_time_ms": 2.5,
#     "avg_miss_time_ms": 150.3
#   },
#   "compression": {
#     "compressed_saves": 100,
#     "bytes_saved": 500000
#   },
#   "by_tool": {
#     "get_standard_details": {"hits": 500, "misses": 50}
#   }
# }
```

### Health Checks

```python
health = await cache.health_check()
# {
#   "status": "healthy",
#   "cache_enabled": true,
#   "redis": {
#     "status": "healthy",
#     "latency_ms": 1.2
#   }
# }
```

## Best Practices

### 1. Choose Appropriate Strategies

- Use `no_cache` for write operations and sensitive data
- Use `short_ttl` for frequently changing data
- Use `long_ttl` for reference data that rarely changes
- Consider `permanent` for static data with manual invalidation

### 2. Configure Cache Keys Carefully

```yaml
# Good: Only relevant args in cache key
search_standards:
  include_in_key:
    - query
    - limit
    - filters
  exclude_from_key:
    - request_id
    - timestamp
```

### 3. Set Up Invalidation Rules

```yaml
# When standards are synced, invalidate related caches
sync_standards:
  strategy: no_cache

get_standard_details:
  invalidate_on:
    - sync_standards
    - update_standard
```

### 4. Monitor and Tune

- Review cache hit rates regularly
- Adjust TTLs based on usage patterns
- Monitor compression effectiveness
- Track slow queries

### 5. Handle Cache Failures Gracefully

The system automatically falls back to direct execution on cache failures:
- Circuit breaker prevents cascading failures
- L1 cache provides fallback when Redis is down
- Metrics track errors for monitoring

## Troubleshooting

### Common Issues

1. **Low Hit Rate**
   - Check if TTLs are too short
   - Verify cache keys include only necessary arguments
   - Ensure warming is configured for frequently accessed data

2. **High Memory Usage**
   - Reduce L1 cache size
   - Enable compression for large responses
   - Shorten TTLs for large objects

3. **Redis Connection Issues**
   - Check Redis connectivity
   - Verify authentication credentials
   - Monitor circuit breaker status

### Debug Logging

Enable debug logging for cache operations:

```python
import logging
logging.getLogger('src.core.cache').setLevel(logging.DEBUG)
```

## Advanced Features

### Custom Serialization

The cache supports both msgpack and pickle serialization, automatically choosing the best option.

### Compression

Responses are automatically compressed when they exceed the threshold:
- Uses zlib compression
- Configurable per tool
- Metrics track compression effectiveness

### Circuit Breaker

Protects against Redis failures:
- Opens after 5 consecutive failures
- Attempts recovery after 30 seconds
- Falls back to L1 cache when open

### Two-Tier Caching

- **L1 (In-Memory)**: Fast, limited size, short TTL
- **L2 (Redis)**: Larger capacity, longer TTL, shared across instances

## Example Configurations

### High-Performance Setup

```yaml
redis:
  max_connections: 100
  l1_max_size: 5000
  l1_ttl: 60
  enable_compression: true
  compression_threshold: 512

tools:
  # Cache everything possible
  get_standard_details:
    strategy: permanent
    warm_on_startup: true
  
  search_standards:
    strategy: medium_ttl
    compress_threshold: 256
```

### Conservative Setup

```yaml
redis:
  max_connections: 20
  l1_max_size: 100
  l1_ttl: 10
  circuit_breaker_threshold: 3

tools:
  # Only cache critical tools
  get_standard_details:
    strategy: short_ttl
    ttl_seconds: 300
```

### Development Setup

```yaml
redis:
  host: localhost
  enable_metrics: true
  slow_query_threshold: 0.05

warm_on_startup: false  # Faster startup

tools:
  # Shorter TTLs for development
  DEFAULT:
    strategy: short_ttl
    ttl_seconds: 60
```