# Caching Reference

Comprehensive guide to the MCP Standards Server caching system.

## Overview

The caching system improves performance by storing frequently accessed data in memory or Redis. It uses a multi-tier architecture with intelligent cache warming and invalidation.

## Cache Architecture

```
┌─────────────────┐
│   L1 Cache      │  ← In-memory LRU cache (fastest)
├─────────────────┤
│   L2 Cache      │  ← Redis cache (shared)
├─────────────────┤
│   Storage       │  ← File system/Database (persistent)
└─────────────────┘
```

## Cache Layers

### L1 Cache (In-Memory)

Fast, process-local cache for hot data.

```python
from src.core.cache import L1Cache

# Configure L1 cache
l1_cache = L1Cache(
    max_size=1000,      # Maximum entries
    ttl=300,            # Time to live (seconds)
    eviction="lru"      # Eviction policy: lru, lfu, fifo
)

# Usage
l1_cache.set("key", value, ttl=60)
value = l1_cache.get("key")
```

Configuration:
```yaml
cache:
  l1:
    enabled: true
    max_size: 1000
    default_ttl: 300
    eviction_policy: "lru"
```

### L2 Cache (Redis)

Shared cache across processes and servers.

```python
from src.core.cache import L2Cache

# Configure L2 cache
l2_cache = L2Cache(
    redis_url="redis://localhost:6379",
    prefix="mcp:",
    ttl=3600,
    serializer="json"  # json, pickle, msgpack
)

# Usage
await l2_cache.set("key", value)
value = await l2_cache.get("key")
```

Configuration:
```yaml
cache:
  l2:
    backend: "redis"
    redis:
      url: "redis://localhost:6379"
      db: 0
      prefix: "mcp:"
      connection_pool:
        max_connections: 50
    default_ttl: 3600
    serializer: "json"
```

## Cache Keys

### Key Naming Convention

```python
# Format: prefix:namespace:identifier:version
"mcp:standards:python-best-practices:1.0.0"
"mcp:validation:file_hash:abc123"
"mcp:search:query_hash:def456"
```

### Key Generation

```python
from src.core.cache import generate_cache_key

# Basic key
key = generate_cache_key("standards", "python-best-practices")
# Result: "mcp:standards:python-best-practices"

# With version
key = generate_cache_key("standards", "python-best-practices", version="1.0.0")
# Result: "mcp:standards:python-best-practices:1.0.0"

# From object
key = generate_cache_key("validation", obj={"file": "app.py", "standard": "pep8"})
# Result: "mcp:validation:hash_of_obj"
```

## Cache Decorators

### Basic Caching

```python
from src.core.cache.decorators import cache_result

@cache_result(ttl=300)
def expensive_operation(param1, param2):
    """This result will be cached for 5 minutes."""
    return compute_result(param1, param2)

# Async version
@cache_result(ttl=300)
async def async_expensive_operation(param1, param2):
    return await compute_result_async(param1, param2)
```

### Conditional Caching

```python
@cache_result(
    ttl=300,
    condition=lambda result: result is not None,
    key_prefix="compute"
)
def conditional_cache(data):
    """Only cache non-None results."""
    return process_data(data)
```

### Cache Invalidation

```python
from src.core.cache.decorators import invalidate_cache

@invalidate_cache(pattern="standards:*")
def update_standard(standard_id, data):
    """Invalidates all standards cache entries."""
    return save_standard(standard_id, data)

# Invalidate specific keys
@invalidate_cache(keys=["standards:python", "standards:javascript"])
def update_multiple_standards(updates):
    return bulk_update(updates)
```

## Cache Warming

### Startup Warming

```python
from src.core.cache import CacheWarmer

warmer = CacheWarmer()

# Define warming strategies
warmer.add_strategy("standards", warm_standards)
warmer.add_strategy("validators", warm_validators)

# Run on startup
await warmer.warm_all()
```

### Background Warming

```yaml
cache:
  warming:
    enabled: true
    schedule: "0 */6 * * *"  # Every 6 hours
    strategies:
      - name: "standards"
        priority: "high"
        batch_size: 50
      - name: "search_index"
        priority: "medium"
        batch_size: 100
```

## Cache Patterns

### Read-Through Cache

```python
async def get_standard(standard_id):
    """Read-through cache pattern."""
    # Try cache first
    cached = await cache.get(f"standard:{standard_id}")
    if cached:
        return cached
    
    # Load from source
    standard = await load_standard_from_db(standard_id)
    
    # Cache for next time
    await cache.set(f"standard:{standard_id}", standard, ttl=3600)
    
    return standard
```

### Write-Through Cache

```python
async def update_standard(standard_id, data):
    """Write-through cache pattern."""
    # Update source
    await save_standard_to_db(standard_id, data)
    
    # Update cache
    await cache.set(f"standard:{standard_id}", data, ttl=3600)
    
    # Invalidate related caches
    await cache.delete_pattern(f"search:*{standard_id}*")
```

### Cache-Aside Pattern

```python
class StandardsRepository:
    def __init__(self, cache, db):
        self.cache = cache
        self.db = db
    
    async def get(self, standard_id):
        # Application manages cache
        key = f"standard:{standard_id}"
        
        # Check cache
        cached = await self.cache.get(key)
        if cached:
            return cached
        
        # Miss - load and cache
        data = await self.db.get(standard_id)
        if data:
            await self.cache.set(key, data)
        
        return data
```

## Cache Monitoring

### Metrics

```python
from src.core.cache import CacheMetrics

metrics = CacheMetrics()

# Track hit/miss ratio
@metrics.track
async def cached_operation():
    # Your operation
    pass

# Get metrics
stats = metrics.get_stats()
print(f"Hit ratio: {stats.hit_ratio:.2%}")
print(f"Miss ratio: {stats.miss_ratio:.2%}")
print(f"Avg response time: {stats.avg_response_time}ms")
```

### Health Checks

```python
async def cache_health_check():
    """Check cache health."""
    try:
        # Test write
        await cache.set("health:check", "ok", ttl=10)
        
        # Test read
        value = await cache.get("health:check")
        
        # Test delete
        await cache.delete("health:check")
        
        return {"status": "healthy", "latency": measure_latency()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

## Cache Configuration Examples

### Development Configuration

```yaml
cache:
  l1:
    enabled: true
    max_size: 100
    ttl: 60
  l2:
    enabled: false  # Use only L1 in development
```

### Production Configuration

```yaml
cache:
  l1:
    enabled: true
    max_size: 10000
    ttl: 300
    eviction_policy: "lru"
  
  l2:
    enabled: true
    backend: "redis-cluster"
    redis:
      nodes:
        - "redis1:6379"
        - "redis2:6379"
        - "redis3:6379"
      password: "${REDIS_PASSWORD}"
      ssl: true
    ttl: 3600
    
  warming:
    enabled: true
    on_startup: true
    schedule: "0 */4 * * *"
```

### High-Performance Configuration

```yaml
cache:
  l1:
    enabled: true
    max_size: 50000
    ttl: 600
    eviction_policy: "lfu"  # Least frequently used
    
  l2:
    enabled: true
    backend: "redis-cluster"
    serializer: "msgpack"  # Faster than JSON
    compression: true
    pipeline_size: 100     # Batch operations
    
  strategies:
    aggressive_caching:
      ttl_multiplier: 2.0
      preload_related: true
      background_refresh: true
```

## Troubleshooting

### Common Issues

1. **High Miss Rate**
   ```python
   # Increase TTL
   @cache_result(ttl=3600)  # 1 hour instead of 5 minutes
   
   # Pre-warm critical data
   await cache_warmer.warm("critical_data")
   ```

2. **Memory Issues**
   ```yaml
   cache:
     l1:
       max_size: 5000  # Reduce from 10000
       eviction_policy: "lfu"  # Better for limited memory
   ```

3. **Stale Data**
   ```python
   # Implement cache invalidation
   @invalidate_cache(pattern="user:{user_id}:*")
   def update_user(user_id, data):
       pass
   ```

### Debug Mode

```python
import logging

# Enable cache debugging
logging.getLogger("mcp.cache").setLevel(logging.DEBUG)

# Track cache operations
with cache.trace() as tracer:
    result = await cached_operation()
    
print(tracer.get_report())
# Output: Cache hits: 5, misses: 2, errors: 0
```

## Performance Tips

1. **Use appropriate TTLs**
   - Static data: 1-24 hours
   - Dynamic data: 5-60 minutes
   - Real-time data: No caching or <1 minute

2. **Batch operations**
   ```python
   # Good - single round trip
   values = await cache.mget(["key1", "key2", "key3"])
   
   # Bad - multiple round trips
   value1 = await cache.get("key1")
   value2 = await cache.get("key2")
   value3 = await cache.get("key3")
   ```

3. **Use compression for large values**
   ```python
   @cache_result(compress=True, compress_threshold=1024)
   def large_data_operation():
       return generate_large_dataset()
   ```

## Related Documentation

- [Performance Optimization](./performance.md)
- [Redis Configuration](./config-reference.md#cache-configuration)
- [Cache Architecture](../../src/core/cache/CACHE_ARCHITECTURE.md)