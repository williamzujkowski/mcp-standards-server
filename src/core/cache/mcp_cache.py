"""
MCP Tool Response Caching Module.

Provides intelligent caching for MCP tool responses with:
- Per-tool TTL configuration
- Selective caching (not all tools should be cached)
- Cache invalidation strategies
- Compression for large responses
- Cache warming capabilities
- Detailed metrics
"""

import asyncio
import hashlib
import json
import logging
import time
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .redis_client import RedisCache

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategy for different tool types."""

    NO_CACHE = "no_cache"  # Never cache (e.g., write operations)
    SHORT_TTL = "short_ttl"  # Cache for short period (e.g., 1-5 minutes)
    MEDIUM_TTL = "medium_ttl"  # Cache for medium period (e.g., 5-30 minutes)
    LONG_TTL = "long_ttl"  # Cache for long period (e.g., 1-24 hours)
    PERMANENT = "permanent"  # Cache permanently until invalidated


@dataclass
class ToolCacheConfig:
    """Configuration for caching a specific tool."""

    tool_name: str
    strategy: CacheStrategy = CacheStrategy.MEDIUM_TTL
    ttl_seconds: int | None = None  # Override default TTL
    compress_threshold: int = 1024  # Compress responses larger than this
    include_in_key: list[str] = field(
        default_factory=list
    )  # Which args to include in cache key
    exclude_from_key: list[str] = field(default_factory=list)  # Which args to exclude
    invalidate_on: list[str] = field(
        default_factory=list
    )  # Tool names that invalidate this cache
    warm_on_startup: bool = False  # Whether to warm this cache on startup
    warm_args: list[dict[str, Any]] = field(
        default_factory=list
    )  # Arguments for cache warming


@dataclass
class CacheMetrics:
    """Detailed cache metrics."""

    hits: int = 0
    misses: int = 0
    errors: int = 0

    total_hit_time_ms: float = 0.0
    total_miss_time_ms: float = 0.0

    compressed_saves: int = 0
    compression_bytes_saved: int = 0

    invalidations: int = 0
    warm_requests: int = 0

    by_tool: dict[str, dict[str, int]] = field(default_factory=dict)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def get_avg_hit_time_ms(self) -> float:
        """Get average cache hit time."""
        return self.total_hit_time_ms / self.hits if self.hits > 0 else 0.0

    def get_avg_miss_time_ms(self) -> float:
        """Get average cache miss time."""
        return self.total_miss_time_ms / self.misses if self.misses > 0 else 0.0


class MCPCache:
    """MCP tool response caching system."""

    # Default TTLs for different strategies (in seconds)
    DEFAULT_TTLS = {
        CacheStrategy.NO_CACHE: 0,
        CacheStrategy.SHORT_TTL: 300,  # 5 minutes
        CacheStrategy.MEDIUM_TTL: 1800,  # 30 minutes
        CacheStrategy.LONG_TTL: 86400,  # 24 hours
        CacheStrategy.PERMANENT: 0,  # No expiry
    }

    # Default tool configurations
    DEFAULT_TOOL_CONFIGS = {
        # Read-only tools with stable responses
        "get_standard_details": ToolCacheConfig(
            tool_name="get_standard_details",
            strategy=CacheStrategy.LONG_TTL,
            include_in_key=["standard_id"],
            invalidate_on=["sync_standards", "generate_standard"],
            warm_on_startup=True,
        ),
        "list_available_standards": ToolCacheConfig(
            tool_name="list_available_standards",
            strategy=CacheStrategy.MEDIUM_TTL,
            include_in_key=["category", "limit"],
            invalidate_on=["sync_standards", "generate_standard"],
        ),
        "search_standards": ToolCacheConfig(
            tool_name="search_standards",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["query", "limit", "min_relevance", "filters"],
            compress_threshold=2048,
        ),
        "get_optimized_standard": ToolCacheConfig(
            tool_name="get_optimized_standard",
            strategy=CacheStrategy.MEDIUM_TTL,
            include_in_key=[
                "standard_id",
                "format_type",
                "token_budget",
                "required_sections",
            ],
            compress_threshold=512,
        ),
        "estimate_token_usage": ToolCacheConfig(
            tool_name="estimate_token_usage",
            strategy=CacheStrategy.LONG_TTL,
            include_in_key=["standard_ids", "format_types"],
        ),
        "get_cross_references": ToolCacheConfig(
            tool_name="get_cross_references",
            strategy=CacheStrategy.MEDIUM_TTL,
            include_in_key=["standard_id", "concept", "max_depth"],
            invalidate_on=["generate_cross_references"],
        ),
        "list_templates": ToolCacheConfig(
            tool_name="list_templates",
            strategy=CacheStrategy.LONG_TTL,
            include_in_key=["domain"],
            warm_on_startup=True,
        ),
        # Analytics tools (shorter cache due to changing data)
        "get_standards_analytics": ToolCacheConfig(
            tool_name="get_standards_analytics",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["metric_type", "time_range", "standard_ids"],
            ttl_seconds=180,  # 3 minutes
        ),
        "get_recommendations": ToolCacheConfig(
            tool_name="get_recommendations",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["analysis_type", "context"],
        ),
        # Tools that should never be cached
        "sync_standards": ToolCacheConfig(
            tool_name="sync_standards", strategy=CacheStrategy.NO_CACHE
        ),
        "generate_standard": ToolCacheConfig(
            tool_name="generate_standard", strategy=CacheStrategy.NO_CACHE
        ),
        "validate_standard": ToolCacheConfig(
            tool_name="validate_standard", strategy=CacheStrategy.NO_CACHE
        ),
        "track_standards_usage": ToolCacheConfig(
            tool_name="track_standards_usage", strategy=CacheStrategy.NO_CACHE
        ),
        "generate_cross_references": ToolCacheConfig(
            tool_name="generate_cross_references", strategy=CacheStrategy.NO_CACHE
        ),
        # Context-dependent tools (cache with care)
        "get_applicable_standards": ToolCacheConfig(
            tool_name="get_applicable_standards",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["context", "include_resolution_details"],
            compress_threshold=1024,
        ),
        "suggest_improvements": ToolCacheConfig(
            tool_name="suggest_improvements",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["code", "context"],
            compress_threshold=2048,
        ),
        "validate_against_standard": ToolCacheConfig(
            tool_name="validate_against_standard",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["code", "standard", "language"],
            ttl_seconds=600,  # 10 minutes
        ),
    }

    def __init__(
        self,
        redis_cache: RedisCache | None = None,
        custom_configs: dict[str, ToolCacheConfig] | None = None,
        enable_compression: bool = True,
        enable_metrics: bool = True,
    ):
        """Initialize MCP cache.

        Args:
            redis_cache: Redis cache instance (will create default if None)
            custom_configs: Custom tool configurations to override defaults
            enable_compression: Whether to enable response compression
            enable_metrics: Whether to collect detailed metrics
        """
        self.redis = redis_cache or RedisCache()
        self.enable_compression = enable_compression
        self.enable_metrics = enable_metrics

        # Initialize tool configurations
        self.tool_configs = self.DEFAULT_TOOL_CONFIGS.copy()
        if custom_configs:
            self.tool_configs.update(custom_configs)

        # Metrics
        self.metrics = CacheMetrics()

        # Invalidation tracking
        self._invalidation_map: dict[str, set[str]] = {}
        self._build_invalidation_map()

        # Cache warming tasks
        self._warm_tasks: list[asyncio.Task] = []

    def _build_invalidation_map(self) -> None:
        """Build reverse mapping for cache invalidation."""
        for tool_name, config in self.tool_configs.items():
            for invalidator in config.invalidate_on:
                if invalidator not in self._invalidation_map:
                    self._invalidation_map[invalidator] = set()
                self._invalidation_map[invalidator].add(tool_name)

    def generate_cache_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        config: ToolCacheConfig | None = None,
    ) -> str:
        """Generate cache key for tool call.

        Uses consistent hashing with selective argument inclusion.
        """
        if config is None:
            config = self.tool_configs.get(tool_name)

        # Start with tool name
        key_parts = [f"mcp:tool:{tool_name}"]

        # Determine which arguments to include
        if config and config.include_in_key:
            # Only include specified arguments
            key_args = {k: arguments.get(k) for k in config.include_in_key}
        elif config and config.exclude_from_key:
            # Include all except excluded
            key_args = {
                k: v for k, v in arguments.items() if k not in config.exclude_from_key
            }
        else:
            # Include all arguments
            key_args = arguments

        # Sort arguments for consistency
        sorted_args = sorted(key_args.items())

        # Create string representation
        arg_str = json.dumps(sorted_args, sort_keys=True, default=str)

        # Hash the arguments
        arg_hash = hashlib.sha256(arg_str.encode()).hexdigest()[:16]
        key_parts.append(arg_hash)

        return ":".join(key_parts)

    def _should_compress(self, data: Any, config: ToolCacheConfig) -> bool:
        """Determine if response should be compressed."""
        if not self.enable_compression:
            return False

        # Estimate size
        try:
            size = len(json.dumps(data, default=str))
            return size > config.compress_threshold
        except Exception:
            return False

    def _compress_response(self, data: Any) -> tuple[bytes, int]:
        """Compress response data."""
        json_str = json.dumps(data, default=str)
        original_size = len(json_str.encode())
        compressed = zlib.compress(json_str.encode())
        compressed_size = len(compressed)

        if self.enable_metrics:
            self.metrics.compressed_saves += 1
            self.metrics.compression_bytes_saved += original_size - compressed_size

        return compressed, original_size

    def _decompress_response(self, data: bytes) -> Any:
        """Decompress response data."""
        decompressed = zlib.decompress(data)
        return json.loads(decompressed)

    async def get(self, tool_name: str, arguments: dict[str, Any]) -> Any | None:
        """Get cached response for tool call.

        Returns None if not cached or cache miss.
        """
        start_time = time.time()

        # Check if tool should be cached
        config = self.tool_configs.get(tool_name)
        if not config or config.strategy == CacheStrategy.NO_CACHE:
            return None

        # Generate cache key
        cache_key = self.generate_cache_key(tool_name, arguments, config)

        try:
            # Try to get from cache
            cached_data = await self.redis.async_get(cache_key)

            if cached_data is not None:
                # Cache hit
                elapsed_ms = (time.time() - start_time) * 1000

                if self.enable_metrics:
                    self.metrics.hits += 1
                    self.metrics.total_hit_time_ms += elapsed_ms

                    if tool_name not in self.metrics.by_tool:
                        self.metrics.by_tool[tool_name] = {"hits": 0, "misses": 0}
                    self.metrics.by_tool[tool_name]["hits"] += 1

                # Handle compressed data
                if isinstance(cached_data, dict) and cached_data.get("_compressed"):
                    return self._decompress_response(cached_data["data"])

                return cached_data
            else:
                # Cache miss
                elapsed_ms = (time.time() - start_time) * 1000

                if self.enable_metrics:
                    self.metrics.misses += 1
                    self.metrics.total_miss_time_ms += elapsed_ms

                    if tool_name not in self.metrics.by_tool:
                        self.metrics.by_tool[tool_name] = {"hits": 0, "misses": 0}
                    self.metrics.by_tool[tool_name]["misses"] += 1

                return None

        except Exception as e:
            logger.error(f"Cache get error for {tool_name}: {e}")
            if self.enable_metrics:
                self.metrics.errors += 1
            return None

    async def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        response: Any,
        ttl_override: int | None = None,
    ) -> bool:
        """Cache tool response.

        Returns True if successfully cached.
        """
        # Check if tool should be cached
        config = self.tool_configs.get(tool_name)
        if not config or config.strategy == CacheStrategy.NO_CACHE:
            return False

        # Generate cache key
        cache_key = self.generate_cache_key(tool_name, arguments, config)

        # Determine TTL
        if ttl_override is not None:
            ttl = ttl_override
        elif config.ttl_seconds is not None:
            ttl = config.ttl_seconds
        else:
            ttl = self.DEFAULT_TTLS.get(config.strategy, 1800)

        # Skip if permanent cache with 0 TTL
        if config.strategy == CacheStrategy.PERMANENT:
            ttl = 0  # Redis interprets 0 as no expiry

        try:
            # Compress if needed
            if self._should_compress(response, config):
                compressed_data, original_size = self._compress_response(response)
                cache_data = {
                    "_compressed": True,
                    "data": compressed_data,
                    "original_size": original_size,
                }
            else:
                cache_data = response

            # Set in cache
            success = await self.redis.async_set(
                cache_key, cache_data, ttl=ttl if ttl > 0 else None
            )

            # Check if this tool invalidates others
            if tool_name in self._invalidation_map:
                await self.invalidate_tools(list(self._invalidation_map[tool_name]))

            return success

        except Exception as e:
            logger.error(f"Cache set error for {tool_name}: {e}")
            if self.enable_metrics:
                self.metrics.errors += 1
            return False

    async def invalidate(
        self, tool_name: str, arguments: dict[str, Any] | None = None
    ) -> bool:
        """Invalidate cache for specific tool call or all calls to a tool.

        Args:
            tool_name: Name of the tool
            arguments: Specific arguments to invalidate (None = invalidate all)

        Returns:
            True if invalidation was successful
        """
        try:
            if arguments is not None:
                # Invalidate specific call
                config = self.tool_configs.get(tool_name)
                cache_key = self.generate_cache_key(tool_name, arguments, config)
                success = await self.redis.async_delete(cache_key)
            else:
                # Invalidate all calls to this tool
                pattern = f"mcp:tool:{tool_name}:*"
                await self.redis.async_delete_pattern(pattern)
                # Consider successful even if nothing was deleted (pattern might not match anything)
                success = True

            if self.enable_metrics:
                self.metrics.invalidations += 1

            return success

        except Exception as e:
            logger.error(f"Cache invalidation error for {tool_name}: {e}")
            return False

    async def invalidate_tools(self, tool_names: list[str]) -> dict[str, bool]:
        """Invalidate multiple tools at once.

        Returns:
            Dictionary mapping tool names to invalidation success
        """
        results = {}
        for tool_name in tool_names:
            results[tool_name] = await self.invalidate(tool_name)
        return results

    async def warm_cache(
        self,
        executor: Callable[[str, dict[str, Any]], Any],
        tools: list[str] | None = None,
    ) -> dict[str, int]:
        """Warm cache by pre-executing specified tools.

        Args:
            executor: Function to execute tool calls
            tools: List of tools to warm (None = all configured for warming)

        Returns:
            Dictionary mapping tool names to number of warmed entries
        """
        warm_results = {}

        # Determine which tools to warm
        if tools is None:
            tools = [
                name
                for name, config in self.tool_configs.items()
                if config.warm_on_startup
            ]

        for tool_name in tools:
            config = self.tool_configs.get(tool_name)
            if not config:
                continue

            warmed = 0

            # Use configured warm arguments or defaults
            warm_args_list = config.warm_args if config.warm_args else [{}]

            for args in warm_args_list:
                try:
                    # Execute tool
                    response = await executor(tool_name, args)

                    # Cache response
                    if await self.set(tool_name, args, response):
                        warmed += 1

                except Exception as e:
                    logger.error(f"Error warming cache for {tool_name}: {e}")

            warm_results[tool_name] = warmed

            if self.enable_metrics:
                self.metrics.warm_requests += warmed

        return warm_results

    async def start_background_warming(
        self,
        executor: Callable[[str, dict[str, Any]], Any],
        interval_seconds: int = 3600,
    ) -> None:
        """Start background cache warming tasks.

        Args:
            executor: Function to execute tool calls
            interval_seconds: How often to warm cache
        """

        async def warm_task() -> None:
            while True:
                try:
                    await self.warm_cache(executor)
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Background warming error: {e}")
                    await asyncio.sleep(60)  # Brief pause on error

        task = asyncio.create_task(warm_task())
        self._warm_tasks.append(task)

    def stop_background_warming(self) -> None:
        """Stop all background warming tasks."""
        for task in self._warm_tasks:
            task.cancel()
        self._warm_tasks.clear()

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive cache metrics."""
        return {
            "overall": {
                "hits": self.metrics.hits,
                "misses": self.metrics.misses,
                "hit_rate": self.metrics.get_hit_rate(),
                "errors": self.metrics.errors,
                "invalidations": self.metrics.invalidations,
                "warm_requests": self.metrics.warm_requests,
            },
            "performance": {
                "avg_hit_time_ms": self.metrics.get_avg_hit_time_ms(),
                "avg_miss_time_ms": self.metrics.get_avg_miss_time_ms(),
            },
            "compression": {
                "compressed_saves": self.metrics.compressed_saves,
                "bytes_saved": self.metrics.compression_bytes_saved,
            },
            "by_tool": self.metrics.by_tool,
            "redis_metrics": (
                self.redis.get_metrics() if hasattr(self.redis, "get_metrics") else {}
            ),
        }

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.metrics = CacheMetrics()

    async def clear_all(self) -> int:
        """Clear all MCP tool caches.

        Returns:
            Number of entries cleared
        """
        pattern = "mcp:tool:*"
        count = await self.redis.async_delete_pattern(pattern)
        return count

    def configure_tool(
        self,
        tool_name: str,
        strategy: CacheStrategy | None = None,
        ttl_seconds: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Configure or reconfigure a tool's caching behavior."""
        if tool_name not in self.tool_configs:
            self.tool_configs[tool_name] = ToolCacheConfig(tool_name=tool_name)

        config = self.tool_configs[tool_name]

        if strategy is not None:
            config.strategy = strategy
        if ttl_seconds is not None:
            config.ttl_seconds = ttl_seconds

        # Update other fields
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        # Rebuild invalidation map if needed
        if "invalidate_on" in kwargs:
            self._build_invalidation_map()

    async def health_check(self) -> dict[str, Any]:
        """Perform health check including Redis connection."""
        health = {
            "status": "healthy",
            "cache_enabled": True,
            "compression_enabled": self.enable_compression,
            "metrics_enabled": self.enable_metrics,
            "configured_tools": len(self.tool_configs),
            "active_warming_tasks": len(self._warm_tasks),
        }

        # Check Redis health
        if hasattr(self.redis, "async_health_check"):
            redis_health = await self.redis.async_health_check()
            health["redis"] = redis_health
            if redis_health.get("status") != "healthy":
                health["status"] = "degraded"

        return health


# Convenience decorator for caching tool responses
def cache_tool_response(
    cache: MCPCache, tool_name: str | None = None, ttl_override: int | None = None
) -> Callable[[Callable], Callable]:
    """Decorator to automatically cache tool responses.

    Usage:
        @cache_tool_response(cache, "my_tool")
        async def my_tool_handler(args) -> None:
            return expensive_operation(args)
    """

    def decorator(func: Callable) -> Callable:
        async def wrapper(arguments: dict[str, Any], **kwargs: Any) -> Any:
            nonlocal tool_name
            if tool_name is None:
                tool_name = func.__name__

            # Try cache first
            cached = await cache.get(tool_name, arguments)
            if cached is not None:
                return cached

            # Execute function
            result = await func(arguments, **kwargs)

            # Cache result
            await cache.set(tool_name, arguments, result, ttl_override)

            return result

        return wrapper

    return decorator
