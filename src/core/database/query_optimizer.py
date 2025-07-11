"""
Database query optimization and caching system.

This module provides:
- Query result caching with intelligent cache keys
- Query performance monitoring and optimization
- Connection pooling and health monitoring
- Prepared statement caching
- Query batching and pipelining
- Database performance metrics
"""

import asyncio
import hashlib
import json
import logging
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Handle aioredis Python 3.12 compatibility issue
try:
    import aioredis
except (TypeError, ImportError) as e:
    # Python 3.12 compatibility issue with aioredis TimeoutError
    if "duplicate base class TimeoutError" in str(e):
        # Mock aioredis for compatibility
        class MockRedis:
            @staticmethod
            async def from_url(*args: Any, **kwargs: Any) -> None:
                return None

        aioredis = type(
            "aioredis", (), {"Redis": MockRedis, "from_url": MockRedis.from_url}
        )()
    else:
        raise


from ..cache.redis_client import CacheConfig, RedisCache
from ..performance.metrics import record_metric, time_operation

logger = logging.getLogger(__name__)


@dataclass
class QueryCacheConfig:
    """Configuration for query caching."""

    # Cache settings
    enable_query_cache: bool = True
    default_cache_ttl: int = 300  # 5 minutes
    max_cache_size: int = 10000
    cache_key_prefix: str = "query_cache"

    # Query optimization
    enable_prepared_statements: bool = True
    enable_query_batching: bool = True
    batch_size: int = 100
    batch_timeout: float = 0.1  # seconds

    # Connection pool settings
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600  # 1 hour

    # Performance monitoring
    enable_query_metrics: bool = True
    slow_query_threshold: float = 0.5  # seconds
    log_slow_queries: bool = True

    # Cache strategies
    cache_read_queries: bool = True
    cache_write_queries: bool = False
    cache_complex_queries: bool = True
    min_cache_query_time: float = 0.1  # seconds


@dataclass
class QueryMetrics:
    """Query performance metrics."""

    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    slow_queries: int = 0
    failed_queries: int = 0

    # Timing metrics
    query_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    cache_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Query type metrics
    query_types: dict[str, int] = field(default_factory=dict)
    slow_query_types: dict[str, int] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        total_cache_ops = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0

        avg_query_time = (
            sum(self.query_times) / len(self.query_times) if self.query_times else 0
        )
        avg_cache_time = (
            sum(self.cache_times) / len(self.cache_times) if self.cache_times else 0
        )

        return {
            "total_queries": self.total_queries,
            "cache_hit_rate": cache_hit_rate,
            "slow_query_rate": (
                self.slow_queries / self.total_queries if self.total_queries > 0 else 0
            ),
            "error_rate": (
                self.failed_queries / self.total_queries
                if self.total_queries > 0
                else 0
            ),
            "average_query_time": avg_query_time,
            "average_cache_time": avg_cache_time,
            "query_types": dict(self.query_types),
            "slow_query_types": dict(self.slow_query_types),
        }


class QueryCache:
    """Query result caching system."""

    def __init__(self, config: QueryCacheConfig, redis_cache: RedisCache) -> None:
        self.config = config
        self.redis_cache = redis_cache

        # Local cache for frequently accessed queries
        self.local_cache: dict[str, Any] = {}
        self.local_cache_times: dict[str, float] = {}
        self.local_cache_lock = threading.Lock()

        # Cache statistics
        self.cache_stats = {"hits": 0, "misses": 0, "sets": 0, "invalidations": 0}

    def _generate_cache_key(
        self, query: str, params: dict[str, Any] | None = None
    ) -> str:
        """Generate cache key for query and parameters."""
        # Normalize query (remove extra whitespace, lowercase)
        normalized_query = " ".join(query.split()).lower()

        # Include parameters in key
        params_str = json.dumps(params or {}, sort_keys=True)

        # Generate hash
        key_data = f"{normalized_query}:{params_str}"
        key_hash = hashlib.sha256(key_data.encode()).hexdigest()

        return f"{self.config.cache_key_prefix}:{key_hash}"

    async def get_cached_result(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Any | None:
        """Get cached query result."""
        if not self.config.enable_query_cache:
            return None

        cache_key = self._generate_cache_key(query, params)

        # Check local cache first
        with self.local_cache_lock:
            if cache_key in self.local_cache:
                cache_time = self.local_cache_times.get(cache_key, 0)
                if time.time() - cache_time < 60:  # 1 minute local cache
                    self.cache_stats["hits"] += 1
                    return self.local_cache[cache_key]
                else:
                    # Local cache expired
                    self.local_cache.pop(cache_key, None)
                    self.local_cache_times.pop(cache_key, None)

        # Check Redis cache
        try:
            cached_result = await self.redis_cache.async_get(cache_key)
            if cached_result is not None:
                # Store in local cache
                with self.local_cache_lock:
                    if len(self.local_cache) < 1000:  # Limit local cache size
                        self.local_cache[cache_key] = cached_result
                        self.local_cache_times[cache_key] = time.time()

                self.cache_stats["hits"] += 1
                return cached_result
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")

        self.cache_stats["misses"] += 1
        return None

    async def cache_result(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        result: Any = None,
        ttl: int | None = None,
    ) -> None:
        """Cache query result."""
        if not self.config.enable_query_cache or result is None:
            return

        cache_key = self._generate_cache_key(query, params)
        cache_ttl = ttl or self.config.default_cache_ttl

        try:
            # Store in Redis
            await self.redis_cache.async_set(cache_key, result, ttl=cache_ttl)

            # Store in local cache
            with self.local_cache_lock:
                if len(self.local_cache) < 1000:
                    self.local_cache[cache_key] = result
                    self.local_cache_times[cache_key] = time.time()

            self.cache_stats["sets"] += 1
        except Exception as e:
            logger.error(f"Error caching result: {e}")

    async def invalidate_cache(self, pattern: str) -> None:
        """Invalidate cache entries matching pattern."""
        try:
            # Invalidate Redis cache
            self.redis_cache.delete_pattern(f"{self.config.cache_key_prefix}:{pattern}")

            # Clear local cache
            with self.local_cache_lock:
                keys_to_remove = [k for k in self.local_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    self.local_cache.pop(key, None)
                    self.local_cache_times.pop(key, None)

            self.cache_stats["invalidations"] += 1
        except Exception as e:
            logger.error(f"Error invalidating cache: {e}")

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_ops = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = self.cache_stats["hits"] / total_ops if total_ops > 0 else 0

        return {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache),
        }


class QueryBatcher:
    """Batches queries for more efficient database operations."""

    def __init__(self, config: QueryCacheConfig) -> None:
        self.config = config
        self.batch_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.batch_results: dict[str, Any] = {}
        self.batch_worker_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start batch processing."""
        if self.config.enable_query_batching:
            self.batch_worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        """Stop batch processing."""
        self.shutdown_event.set()
        if self.batch_worker_task:
            self.batch_worker_task.cancel()
            try:
                await self.batch_worker_task
            except asyncio.CancelledError:
                pass

    async def queue_query(
        self, query: str, params: dict[str, Any] | None = None
    ) -> Any:
        """Queue a query for batch processing."""
        if not self.config.enable_query_batching:
            return None

        # Create future for result
        result_future: asyncio.Future[Any] = asyncio.Future()

        # Add to batch queue
        await self.batch_queue.put(
            {"query": query, "params": params, "future": result_future}
        )

        # Wait for result
        return await result_future

    async def _batch_worker(self) -> None:
        """Worker task for processing query batches."""
        while not self.shutdown_event.is_set():
            try:
                batch = []

                # Collect batch items
                try:
                    # Wait for first item
                    item = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=self.config.batch_timeout
                    )
                    batch.append(item)

                    # Collect additional items
                    while len(batch) < self.config.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.batch_queue.get(),
                                timeout=0.001,  # Very short timeout
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                # Process batch
                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch worker: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of queries."""
        # Group similar queries
        query_groups = defaultdict(list)
        for item in batch:
            query_groups[item["query"]].append(item)

        # Process each group
        for query, items in query_groups.items():
            try:
                # For now, process individually
                # In a real implementation, you'd optimize for bulk operations
                for item in items:
                    # This would be replaced with actual database execution
                    result = {"mock": "result", "query": query}
                    item["future"].set_result(result)

            except Exception as e:
                # Set exception for all items in group
                for item in items:
                    if not item["future"].done():
                        item["future"].set_exception(e)


class DatabaseOptimizer:
    """Main database optimization and caching system."""

    def __init__(self, config: QueryCacheConfig | None = None) -> None:
        self.config = config or QueryCacheConfig()
        self.metrics = QueryMetrics()

        # Initialize Redis cache
        cache_config = CacheConfig(
            max_connections=self.config.pool_size, enable_compression=True
        )
        self.redis_cache = RedisCache(cache_config)

        # Initialize cache and batcher
        self.query_cache = QueryCache(self.config, self.redis_cache)
        self.query_batcher = QueryBatcher(self.config)

        # Connection pools (these would be configured per database type)
        self.connection_pools: dict[str, Any] = {}

        # Prepared statements cache
        self.prepared_statements: dict[str, Any] = {}
        self.prepared_statements_lock = threading.Lock()

        # Query analysis
        self.query_patterns: defaultdict[str, int] = defaultdict(int)
        self.slow_queries: deque[dict[str, Any]] = deque(maxlen=100)

        # Performance monitoring
        self.performance_callbacks: list[Callable[[dict[str, Any]], None]] = []

    async def initialize(self) -> None:
        """Initialize the database optimizer."""
        # Start batch processing
        await self.query_batcher.start()

        logger.info("Database optimizer initialized")

    async def close(self) -> None:
        """Close the database optimizer."""
        # Stop batch processing
        await self.query_batcher.stop()

        # Close Redis cache
        await self.redis_cache.async_close()

        # Close connection pools
        for pool in self.connection_pools.values():
            if hasattr(pool, "close"):
                await pool.close()

        logger.info("Database optimizer closed")

    def _analyze_query(self, query: str) -> dict[str, Any]:
        """Analyze query for optimization opportunities."""
        query_lower = query.lower().strip()

        # Determine query type
        if query_lower.startswith("select"):
            query_type = "SELECT"
        elif query_lower.startswith("insert"):
            query_type = "INSERT"
        elif query_lower.startswith("update"):
            query_type = "UPDATE"
        elif query_lower.startswith("delete"):
            query_type = "DELETE"
        else:
            query_type = "OTHER"

        # Check for complex operations
        is_complex = any(
            keyword in query_lower
            for keyword in [
                "join",
                "subquery",
                "union",
                "group by",
                "order by",
                "having",
            ]
        )

        # Determine if query should be cached
        should_cache = (
            self.config.cache_read_queries
            and query_type == "SELECT"
            or self.config.cache_write_queries
            and query_type in ["INSERT", "UPDATE", "DELETE"]
            or self.config.cache_complex_queries
            and is_complex
        )

        return {
            "type": query_type,
            "is_complex": is_complex,
            "should_cache": should_cache,
            "length": len(query),
        }

    async def execute_query(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        cache_ttl: int | None = None,
        use_cache: bool = True,
    ) -> Any:
        """Execute a query with caching and optimization."""
        start_time = time.time()

        # Analyze query
        analysis = self._analyze_query(query)

        # Update metrics
        self.metrics.total_queries += 1
        self.metrics.query_types[analysis["type"]] = (
            self.metrics.query_types.get(analysis["type"], 0) + 1
        )

        # Record query pattern
        query_pattern = self._extract_query_pattern(query)
        self.query_patterns[query_pattern] += 1

        try:
            # Check cache first
            cached_result = None
            if use_cache and analysis["should_cache"]:
                with time_operation("db_cache_lookup_duration_seconds"):
                    cached_result = await self.query_cache.get_cached_result(
                        query, params
                    )

                if cached_result is not None:
                    self.metrics.cache_hits += 1

                    # Record cache timing
                    cache_time = time.time() - start_time
                    self.metrics.cache_times.append(cache_time)

                    # Record metrics
                    record_metric(
                        "app_db_query_duration_seconds",
                        cache_time,
                        {"query_type": analysis["type"], "source": "cache"},
                    )
                    record_metric("app_cache_hits", 1, {"cache_type": "query"})

                    return cached_result

            # Execute query
            result = await self._execute_query_direct(query, params, analysis)

            # Cache result if appropriate
            if use_cache and analysis["should_cache"] and result is not None:
                await self.query_cache.cache_result(query, params, result, cache_ttl)

            # Record metrics
            execution_time = time.time() - start_time
            self.metrics.query_times.append(execution_time)

            # Check for slow queries
            if execution_time > self.config.slow_query_threshold:
                self.metrics.slow_queries += 1
                self.metrics.slow_query_types[analysis["type"]] = (
                    self.metrics.slow_query_types.get(analysis["type"], 0) + 1
                )

                slow_query_info = {
                    "query": query,
                    "params": params,
                    "execution_time": execution_time,
                    "timestamp": time.time(),
                    "analysis": analysis,
                }
                self.slow_queries.append(slow_query_info)

                if self.config.log_slow_queries:
                    logger.warning(
                        f"Slow query detected: {execution_time:.3f}s - {query[:100]}..."
                    )

            # Record metrics
            record_metric(
                "app_db_query_duration_seconds",
                execution_time,
                {"query_type": analysis["type"], "source": "database"},
            )
            if use_cache and analysis["should_cache"]:
                record_metric("app_cache_misses", 1, {"cache_type": "query"})

            return result

        except Exception as e:
            self.metrics.failed_queries += 1

            # Record error metrics
            record_metric("app_error_count", 1, {"error_type": "database_query"})

            logger.error(f"Database query failed: {e}")
            raise

    async def _execute_query_direct(
        self,
        query: str,
        params: dict[str, Any] | None = None,
        analysis: dict[str, Any] | None = None,
    ) -> Any:
        """Execute query directly against database."""
        # This is a mock implementation
        # In a real implementation, you would:
        # 1. Get connection from pool
        # 2. Use prepared statements if enabled
        # 3. Execute query with proper parameter binding
        # 4. Return results

        # Simulate database execution time
        await asyncio.sleep(0.001)  # 1ms

        # Return mock result
        return {
            "query": query,
            "params": params,
            "rows_affected": 1,
            "execution_time": 0.001,
            "mock_data": ["row1", "row2", "row3"],
        }

    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for analysis."""
        # Simple pattern extraction - replace parameters with placeholders
        import re

        # Remove string literals
        pattern = re.sub(r"'[^']*'", "'?'", query)

        # Remove numeric literals
        pattern = re.sub(r"\b\d+\b", "?", pattern)

        # Normalize whitespace
        pattern = " ".join(pattern.split())

        return pattern.lower()

    async def execute_batch(
        self, queries: list[tuple[str, dict[str, Any] | None]]
    ) -> list[Any]:
        """Execute multiple queries in batch."""
        if not queries:
            return []

        results = []

        if self.config.enable_query_batching:
            # Use batch processing
            tasks = []
            for query, params in queries:
                task = self.query_batcher.queue_query(query, params)
                tasks.append(task)

            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Execute individually
            for query, params in queries:
                try:
                    result = await self.execute_query(query, params)
                    results.append(result)
                except Exception as e:
                    results.append(e)

        return results

    def get_prepared_statement(self, query: str) -> str | None:
        """Get prepared statement for query."""
        if not self.config.enable_prepared_statements:
            return None

        query_hash = hashlib.sha256(query.encode()).hexdigest()

        with self.prepared_statements_lock:
            return self.prepared_statements.get(query_hash)

    def cache_prepared_statement(self, query: str, statement: str) -> None:
        """Cache prepared statement."""
        if not self.config.enable_prepared_statements:
            return

        query_hash = hashlib.sha256(query.encode()).hexdigest()

        with self.prepared_statements_lock:
            self.prepared_statements[query_hash] = statement

    async def invalidate_cache_for_table(self, table_name: str) -> None:
        """Invalidate cache for queries affecting a specific table."""
        pattern = f"*{table_name}*"
        await self.query_cache.invalidate_cache(pattern)

    def add_performance_callback(self, callback: Callable) -> None:
        """Add performance monitoring callback."""
        self.performance_callbacks.append(callback)

    def remove_performance_callback(self, callback: Callable) -> None:
        """Remove performance monitoring callback."""
        if callback in self.performance_callbacks:
            self.performance_callbacks.remove(callback)

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "metrics": self.metrics.get_summary(),
            "cache_stats": self.query_cache.get_cache_stats(),
            "query_patterns": dict(self.query_patterns),
            "slow_queries": list(self.slow_queries),
            "prepared_statements_count": len(self.prepared_statements),
        }

    def get_slow_queries(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent slow queries."""
        return list(self.slow_queries)[-limit:]

    def get_query_recommendations(self) -> list[dict[str, Any]]:
        """Get query optimization recommendations."""
        recommendations = []

        # Check for frequently executed queries
        for pattern, count in self.query_patterns.items():
            if count > 100:  # Frequently executed
                recommendations.append(
                    {
                        "type": "frequent_query",
                        "pattern": pattern,
                        "count": count,
                        "recommendation": "Consider creating an index or optimizing this query",
                    }
                )

        # Check for slow query patterns
        slow_patterns: dict[str, int] = {}
        for slow_query in self.slow_queries:
            pattern = self._extract_query_pattern(slow_query["query"])
            slow_patterns[pattern] = slow_patterns.get(pattern, 0) + 1

        for pattern, count in slow_patterns.items():
            if count > 5:  # Repeatedly slow
                recommendations.append(
                    {
                        "type": "slow_query_pattern",
                        "pattern": pattern,
                        "count": count,
                        "recommendation": "This query pattern is consistently slow - consider optimization",
                    }
                )

        return recommendations

    async def health_check(self) -> dict[str, Any]:
        """Perform health check on database connections."""
        health: dict[str, Any] = {
            "status": "healthy",
            "cache_status": "healthy",
            "connection_pools": {},
            "metrics": self.metrics.get_summary(),
        }

        # Check Redis cache
        try:
            redis_health = await self.redis_cache.async_health_check()
            health["cache_status"] = redis_health["status"]
        except Exception as e:
            health["cache_status"] = "unhealthy"
            health["cache_error"] = str(e)

        # Check connection pools
        for pool_name, _pool in self.connection_pools.items():
            try:
                # Mock health check
                health["connection_pools"][pool_name] = {
                    "status": "healthy",
                    "active_connections": 0,
                    "idle_connections": 0,
                }
            except Exception as e:
                health["connection_pools"][pool_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                health["status"] = "degraded"

        return health


# Global database optimizer instance
_global_optimizer: DatabaseOptimizer | None = None


def get_database_optimizer() -> DatabaseOptimizer:
    """Get global database optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = DatabaseOptimizer()
    return _global_optimizer


async def initialize_database_optimizer(
    config: QueryCacheConfig | None = None,
) -> DatabaseOptimizer:
    """Initialize and start global database optimizer."""
    global _global_optimizer
    _global_optimizer = DatabaseOptimizer(config)
    await _global_optimizer.initialize()
    return _global_optimizer


async def shutdown_database_optimizer() -> None:
    """Shutdown global database optimizer."""
    global _global_optimizer
    if _global_optimizer:
        await _global_optimizer.close()
        _global_optimizer = None


# Convenience functions
async def execute_query(
    query: str, params: dict[str, Any] | None = None, **kwargs: Any
) -> Any:
    """Execute query using global optimizer."""
    optimizer = get_database_optimizer()
    return await optimizer.execute_query(query, params, **kwargs)


async def execute_batch(queries: list[tuple[str, dict[str, Any] | None]]) -> list[Any]:
    """Execute batch queries using global optimizer."""
    optimizer = get_database_optimizer()
    return await optimizer.execute_batch(queries)
