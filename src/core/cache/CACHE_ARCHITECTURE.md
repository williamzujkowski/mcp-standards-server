# Redis Cache Architecture for MCP Standards Server

## Overview

This document describes the Redis caching layer architecture for the MCP Standards Server, designed to improve performance and reduce computational overhead.

## Cache Layers

### L1 Cache (In-Memory)
- **Purpose**: Ultra-fast access for frequently used data
- **Size**: Limited (configurable, default 100MB)
- **TTL**: Short (5-60 seconds)
- **Implementation**: LRU cache using Python's functools.lru_cache or cachetools

### L2 Cache (Redis)
- **Purpose**: Distributed cache for larger datasets
- **Size**: Configurable based on Redis instance
- **TTL**: Medium to long (5 minutes to 24 hours)
- **Implementation**: Redis with connection pooling

## Key Naming Conventions

### Format
```
{prefix}:{version}:{namespace}:{identifier}:{hash}
```

### Prefixes
- `mcp` - General MCP server cache
- `search` - Semantic search results
- `standards` - Standards data
- `rules` - Rule engine results
- `sync` - Synchronization metadata

### Examples
```
# Semantic search results
search:v1:query:{query_hash}:{params_hash}

# Standards retrieval
standards:v1:data:{standard_id}:{version}

# Rule engine results
rules:v1:evaluation:{rule_id}:{context_hash}

# Sync status
sync:v1:status:{source}:{timestamp}
```

## TTL Policies

### Default TTLs
| Data Type | L1 TTL | L2 TTL | Notes |
|-----------|---------|---------|--------|
| Search Results | 30s | 5m | Frequently changing |
| Standards Data | 60s | 1h | Relatively static |
| Rule Results | 15s | 2m | Context-dependent |
| Sync Metadata | 5s | 30s | Real-time updates |
| User Sessions | - | 30m | Session data |

### TTL Configuration
```python
TTL_CONFIG = {
    'search': {
        'l1': 30,      # seconds
        'l2': 300,     # seconds
        'refresh': 240  # refresh if older than this
    },
    'standards': {
        'l1': 60,
        'l2': 3600,
        'refresh': 3000
    },
    'rules': {
        'l1': 15,
        'l2': 120,
        'refresh': 90
    },
    'sync': {
        'l1': 5,
        'l2': 30,
        'refresh': 20
    }
}
```

## Cache Invalidation Strategy

### Automatic Invalidation
1. **TTL-based**: Natural expiration
2. **Version-based**: New versions invalidate old keys
3. **Event-based**: Updates trigger invalidation

### Manual Invalidation
1. **Pattern-based**: Invalidate by key pattern
2. **Tag-based**: Group related keys by tags
3. **Cascade**: Invalidate dependent caches

## Error Handling

### Fallback Strategy
1. L1 miss → Check L2
2. L2 miss → Compute and cache
3. Redis unavailable → L1 only mode
4. All cache miss → Direct computation

### Circuit Breaker
- Open after 5 consecutive failures
- Half-open after 30 seconds
- Close after 3 successful operations

## Monitoring Metrics

### Key Metrics
1. **Hit Rate**: L1 and L2 separately
2. **Latency**: Cache operation times
3. **Memory Usage**: L1 and Redis memory
4. **Error Rate**: Connection failures
5. **Eviction Rate**: LRU evictions

### Health Checks
- Redis connectivity
- Memory pressure
- Key distribution
- TTL compliance

## Security Considerations

1. **Encryption**: TLS for Redis connections
2. **Authentication**: Redis AUTH/ACL
3. **Key Isolation**: Namespace separation
4. **Data Sensitivity**: No PII in cache keys
5. **Access Control**: Role-based cache access

## Performance Targets

| Operation | Target Latency | Notes |
|-----------|----------------|-------|
| L1 Hit | < 1ms | In-memory |
| L2 Hit | < 10ms | Redis local |
| L2 Miss + Compute | < 100ms | Depends on operation |
| Batch Operations | < 50ms | Pipeline/MGET |

## Implementation Notes

1. **Async First**: All cache operations support async/await
2. **Batch Operations**: Support for MGET/MSET
3. **Compression**: Optional compression for large values
4. **Serialization**: JSON with MessagePack fallback
5. **Connection Pooling**: Reuse Redis connections
6. **Retry Logic**: Exponential backoff for failures