# Redis Cache Layer Documentation

## Overview

The MCP Standards Server includes a comprehensive Redis caching layer designed to improve performance and reduce computational overhead. The cache implementation features:

- **Two-tier caching**: L1 (in-memory) and L2 (Redis) caches
- **Async support**: Full async/await compatibility
- **Resilience**: Circuit breaker, retry logic, and graceful degradation
- **Easy integration**: Decorators for seamless caching
- **Monitoring**: Built-in metrics and health checks

## Quick Start

### Basic Usage

```python
from src.core.cache import RedisCache, CacheConfig

# Initialize cache with default config
cache = RedisCache()

# Or with custom config
config = CacheConfig(
    host="redis.example.com",
    port=6379,
    password="secret",
    default_ttl=300
)
cache = RedisCache(config)

# Basic operations
cache.set("key", {"data": "value"}, ttl=60)
value = cache.get("key")
cache.delete("key")

# Batch operations
cache.mset({"key1": "value1", "key2": "value2"})
values = cache.mget(["key1", "key2", "key3"])
```

### Using Decorators

```python
from src.core.cache.decorators import cache_result, invalidate_cache

# Cache function results
@cache_result("search", ttl=300)
def search_standards(query: str) -> List[Standard]:
    # Expensive search operation
    return perform_search(query)

# Async functions
@cache_result("user", ttl=600)
async def get_user_data(user_id: str) -> dict:
    # Async database query
    return await db.get_user(user_id)

# Invalidate cache after updates
@invalidate_cache(pattern="user:{user_id}:*")
def update_user(user_id: str, data: dict):
    # Update user data
    db.update_user(user_id, data)
```

## Architecture

### Cache Layers

1. **L1 Cache (In-Memory)**
   - Ultra-fast access using Python's TTLCache
   - Limited size (configurable)
   - Short TTL (5-60 seconds)
   - Process-local

2. **L2 Cache (Redis)**
   - Distributed cache
   - Larger capacity
   - Longer TTL (minutes to hours)
   - Shared across processes

### Key Naming Convention

Cache keys follow a structured format:

```
{prefix}:{version}:{namespace}:{identifier}:{hash}
```

Examples:
- `search:v1:query:security:a3f4b2c1`
- `standards:v1:data:ISO27001:latest`
- `rules:v1:evaluation:rule_123:ctx_hash`

## Configuration

### CacheConfig Options

```python
@dataclass
class CacheConfig:
    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    
    # Pool settings
    max_connections: int = 50
    socket_keepalive: bool = True
    
    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1
    retry_backoff: float = 2.0
    
    # Cache settings
    default_ttl: int = 300  # 5 minutes
    key_prefix: str = "mcp"
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes
    
    # L1 cache settings
    l1_max_size: int = 1000
    l1_ttl: int = 30  # seconds
    
    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30
```

### TTL Policies

| Data Type | L1 TTL | L2 TTL | Use Case |
|-----------|---------|---------|----------|
| Search Results | 30s | 5m | Frequently changing |
| Standards Data | 60s | 1h | Relatively static |
| Rule Results | 15s | 2m | Context-dependent |
| Sync Metadata | 5s | 30s | Real-time updates |

## Advanced Features

### Circuit Breaker

The cache includes a circuit breaker to handle Redis failures gracefully:

```python
# Circuit breaker states:
# - Closed: Normal operation
# - Open: Failing, requests blocked
# - Half-open: Testing recovery

# Configuration
config = CacheConfig(
    circuit_breaker_threshold=5,  # Open after 5 failures
    circuit_breaker_timeout=30     # Try recovery after 30s
)
```

### Compression

Large values are automatically compressed:

```python
config = CacheConfig(
    enable_compression=True,
    compression_threshold=1024  # Compress values > 1KB
)
```

### Serialization

The cache supports multiple serialization formats:
- MessagePack (default, fastest)
- Pickle (fallback for complex objects)
- Automatic format detection on deserialization

### Batch Operations

Efficient batch operations for multiple keys:

```python
# Batch set
cache.mset({
    "key1": "value1",
    "key2": "value2",
    "key3": "value3"
}, ttl=300)

# Batch get
values = cache.mget(["key1", "key2", "key3"])
# Returns: {"key1": "value1", "key2": "value2", "key3": "value3"}
```

### Pattern-Based Operations

Delete keys matching patterns:

```python
# Delete all user cache entries
deleted_count = cache.delete_pattern("user:*")

# Delete specific user's cache
deleted_count = cache.delete_pattern("user:123:*")
```

## Integration Examples

### Semantic Search Integration

```python
from src.core.cache.integration import CachedSemanticSearch

# Wrap semantic search with caching
cached_search = CachedSemanticSearch(semantic_search_engine)

# Search with automatic caching
results = await cached_search.search("security compliance", k=10)

# Find similar standards
similar = await cached_search.find_similar("ISO27001", k=5)
```

### Standards Engine Integration

```python
from src.core.cache.integration import CachedStandardsEngine

# Wrap standards engine
cached_engine = CachedStandardsEngine(standards_engine)

# Get standard with caching
standard = await cached_engine.get_standard("NIST-CSF")

# List standards
standards = await cached_engine.list_standards(
    category="security",
    limit=50
)
```

### Cache Warming

Pre-populate cache with frequently accessed data:

```python
from src.core.cache.integration import CacheWarmer

warmer = CacheWarmer(cached_engine, cached_search)

# Warm popular searches
await warmer.warm_popular_searches([
    "security",
    "compliance",
    "data protection",
    "privacy"
])

# Warm frequently accessed standards
await warmer.warm_standards([
    "ISO27001",
    "NIST-CSF",
    "GDPR",
    "SOC2"
])
```

## Monitoring and Metrics

### Health Checks

```python
# Sync health check
health = cache.health_check()
# {
#     "status": "healthy",
#     "redis_connected": true,
#     "latency_ms": 0.5,
#     "l1_cache_size": 234,
#     "circuit_breaker_state": "closed"
# }

# Async health check
health = await cache.async_health_check()
```

### Metrics Collection

```python
metrics = cache.get_metrics()
# {
#     "l1_hits": 1520,
#     "l1_misses": 480,
#     "l1_hit_rate": 0.76,
#     "l2_hits": 350,
#     "l2_misses": 130,
#     "l2_hit_rate": 0.73,
#     "errors": 2,
#     "slow_queries": 5
# }
```

### Performance Monitoring

```python
from src.core.cache.integration import CacheMetricsCollector

collector = CacheMetricsCollector()
metrics = collector.collect_metrics()

# Report to monitoring system
await collector.report_metrics(destination="prometheus")
```

## Error Handling

The cache is designed to fail gracefully:

```python
# If Redis is unavailable:
# 1. L1 cache continues to work
# 2. Circuit breaker prevents cascade failures
# 3. Operations fall back to computation
# 4. No exceptions bubble up to application

# Example with fallback
value = cache.get("key")
if value is None:
    # Cache miss or error - compute value
    value = expensive_computation()
    # Try to cache (won't throw if Redis is down)
    cache.set("key", value)
```

## Performance

Based on benchmarks with local Redis:

| Operation | L1 Cache | L2 Cache | No Cache |
|-----------|----------|----------|----------|
| Small value GET | < 1μs | < 100μs | N/A |
| Medium value GET | < 1μs | < 200μs | N/A |
| Batch GET (100 keys) | < 10μs | < 5ms | N/A |
| Decorator overhead | < 10μs | < 10μs | 0 |

### Cache Hit Rates

Typical hit rates in production:
- L1 Cache: 70-80%
- L2 Cache: 85-95%
- Overall: 95-99%

## Best Practices

1. **Choose appropriate TTLs**
   - Shorter for frequently changing data
   - Longer for static data
   - Consider data freshness requirements

2. **Use meaningful key prefixes**
   - Helps with debugging
   - Enables pattern-based operations
   - Prevents key collisions

3. **Handle cache misses gracefully**
   - Always have a fallback
   - Don't assume cache will always work
   - Log cache errors for monitoring

4. **Warm cache for critical data**
   - Pre-populate on startup
   - Refresh before expiration
   - Monitor cache hit rates

5. **Use batch operations**
   - More efficient than individual calls
   - Reduces network overhead
   - Better performance

6. **Monitor cache health**
   - Set up alerts for low hit rates
   - Monitor Redis memory usage
   - Track error rates

## Troubleshooting

### Common Issues

1. **Low hit rates**
   - Check TTL settings
   - Verify key generation
   - Look for cache invalidation issues

2. **High latency**
   - Check Redis connection
   - Monitor network latency
   - Verify compression settings

3. **Memory issues**
   - Adjust L1 cache size
   - Check Redis memory limits
   - Review data sizes

4. **Connection errors**
   - Verify Redis is running
   - Check firewall rules
   - Review connection pool settings

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger("src.core.cache").setLevel(logging.DEBUG)
```

## Migration Guide

### From No Cache

1. Install Redis
2. Add cache configuration
3. Wrap expensive operations with `@cache_result`
4. Add cache invalidation where needed
5. Monitor performance improvements

### From Other Cache Systems

1. Map existing cache keys to new format
2. Adjust TTL policies
3. Update serialization if needed
4. Test thoroughly before switching

## Future Enhancements

Planned improvements:
- Redis Cluster support
- Cache tagging for group invalidation
- Automatic cache warming based on access patterns
- GraphQL query result caching
- Edge caching integration