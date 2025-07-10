#!/usr/bin/env python3
"""
Example of using the MCP caching module.

This example demonstrates:
1. Basic cache setup
2. Manual caching of tool responses
3. Using the cache decorator
4. Cache management operations
5. Monitoring cache performance
"""

import asyncio
from typing import Any

from src.core.cache import (
    CacheStrategy,
    MCPCache,
    ToolCacheConfig,
    cache_tool_response,
)
from src.core.cache.redis_client import CacheConfig, RedisCache


# Example 1: Basic Cache Setup
async def basic_cache_example():
    """Demonstrate basic cache setup and usage."""
    print("=== Basic Cache Example ===")

    # Create Redis configuration
    redis_config = CacheConfig(
        host="localhost", port=6379, default_ttl=300, enable_compression=True
    )

    # Create Redis client
    redis_cache = RedisCache(redis_config)

    # Create MCP cache
    cache = MCPCache(redis_cache=redis_cache)

    # Cache a tool response
    tool_name = "get_standard_details"
    arguments = {"standard_id": "secure-api-design"}
    response = {
        "id": "secure-api-design",
        "name": "Secure API Design Standards",
        "version": "1.0.0",
        "guidelines": ["Use HTTPS", "Implement rate limiting", "Validate inputs"],
    }

    # Set in cache
    success = await cache.set(tool_name, arguments, response)
    print(f"Cached response: {success}")

    # Get from cache
    cached = await cache.get(tool_name, arguments)
    print(f"Retrieved from cache: {cached is not None}")
    print(f"Cache hit rate: {cache.get_metrics()['overall']['hit_rate']:.2%}")


# Example 2: Using Cache Decorator
async def decorator_example():
    """Demonstrate using the cache decorator."""
    print("\n=== Cache Decorator Example ===")

    # Create cache instance
    cache = MCPCache()

    # Simulate an expensive operation
    call_count = 0

    @cache_tool_response(cache, "expensive_search", ttl_override=60)
    async def expensive_search(arguments: dict[str, Any]) -> dict[str, Any]:
        nonlocal call_count
        call_count += 1

        # Simulate expensive operation
        await asyncio.sleep(0.5)

        return {
            "results": [f"Result for {arguments['query']}" for _ in range(10)],
            "call_count": call_count,
        }

    # First call - will execute the function
    result1 = await expensive_search({"query": "security standards"})
    print(f"First call - Call count: {result1['call_count']}")

    # Second call - will use cache
    result2 = await expensive_search({"query": "security standards"})
    print(f"Second call - Call count: {result2['call_count']}")

    # Different query - will execute again
    result3 = await expensive_search({"query": "api standards"})
    print(f"Different query - Call count: {result3['call_count']}")


# Example 3: Custom Tool Configuration
async def custom_config_example():
    """Demonstrate custom tool configuration."""
    print("\n=== Custom Configuration Example ===")

    # Define custom configurations
    custom_configs = {
        "frequent_tool": ToolCacheConfig(
            tool_name="frequent_tool",
            strategy=CacheStrategy.SHORT_TTL,
            ttl_seconds=60,
            compress_threshold=512,
        ),
        "stable_tool": ToolCacheConfig(
            tool_name="stable_tool",
            strategy=CacheStrategy.LONG_TTL,
            ttl_seconds=86400,  # 24 hours
            warm_on_startup=True,
            warm_args=[{"id": 1}, {"id": 2}],
        ),
        "write_tool": ToolCacheConfig(
            tool_name="write_tool", strategy=CacheStrategy.NO_CACHE
        ),
    }

    # Create cache with custom configs
    cache = MCPCache(custom_configs=custom_configs)

    # Test different strategies
    for tool_name in ["frequent_tool", "stable_tool", "write_tool"]:
        config = cache.tool_configs[tool_name]
        print(
            f"{tool_name}: strategy={config.strategy.value}, "
            f"ttl={config.ttl_seconds or cache.DEFAULT_TTLS.get(config.strategy)}s"
        )


# Example 4: Cache Management
async def cache_management_example():
    """Demonstrate cache management operations."""
    print("\n=== Cache Management Example ===")

    cache = MCPCache()

    # Add some test data
    tools_data = [
        ("tool1", {"arg": 1}, {"result": "data1"}),
        ("tool1", {"arg": 2}, {"result": "data2"}),
        ("tool2", {"arg": 1}, {"result": "data3"}),
    ]

    for tool, args, response in tools_data:
        await cache.set(tool, args, response)

    print("Initial cache state:")
    metrics = cache.get_metrics()
    print(
        f"  Total entries: {metrics['overall']['hits'] + metrics['overall']['misses']}"
    )

    # Invalidate specific entry
    await cache.invalidate("tool1", {"arg": 1})
    print("\nAfter invalidating tool1 with arg=1:")
    print(f"  tool1, arg=1: {await cache.get('tool1', {'arg': 1})}")
    print(f"  tool1, arg=2: {await cache.get('tool1', {'arg': 2}) is not None}")

    # Invalidate all entries for a tool
    await cache.invalidate("tool1")
    print("\nAfter invalidating all tool1 entries:")
    print(f"  tool1, arg=2: {await cache.get('tool1', {'arg': 2})}")
    print(f"  tool2, arg=1: {await cache.get('tool2', {'arg': 1}) is not None}")

    # Clear all caches
    count = await cache.clear_all()
    print(f"\nCleared {count} cache entries")


# Example 5: Cache Warming
async def cache_warming_example():
    """Demonstrate cache warming."""
    print("\n=== Cache Warming Example ===")

    # Configure tools for warming
    cache = MCPCache()
    cache.configure_tool(
        "api_standards",
        strategy=CacheStrategy.LONG_TTL,
        warm_on_startup=True,
        warm_args=[{"category": "rest"}, {"category": "graphql"}, {"category": "grpc"}],
    )

    # Mock tool executor
    async def mock_executor(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "tool": tool_name,
            "category": args.get("category"),
            "standards": [f"{args.get('category')}-standard-{i}" for i in range(3)],
        }

    # Warm the cache
    warm_results = await cache.warm_cache(mock_executor, ["api_standards"])
    print(f"Warmed {warm_results['api_standards']} entries for api_standards")

    # Verify warmed data
    for category in ["rest", "graphql", "grpc"]:
        cached = await cache.get("api_standards", {"category": category})
        print(f"  {category}: {len(cached['standards'])} standards cached")


# Example 6: Monitoring and Metrics
async def monitoring_example():
    """Demonstrate cache monitoring and metrics."""
    print("\n=== Monitoring Example ===")

    cache = MCPCache(enable_metrics=True)

    # Simulate some cache operations
    for i in range(10):
        await cache.set(f"tool_{i % 3}", {"id": i}, {"data": f"result_{i}"})

    # Some hits and misses
    for i in range(20):
        await cache.get(f"tool_{i % 5}", {"id": i % 10})

    # Get comprehensive metrics
    metrics = cache.get_metrics()

    print("Cache Metrics:")
    print(f"  Hit Rate: {metrics['overall']['hit_rate']:.2%}")
    print(f"  Total Hits: {metrics['overall']['hits']}")
    print(f"  Total Misses: {metrics['overall']['misses']}")
    print(f"  Avg Hit Time: {metrics['performance']['avg_hit_time_ms']:.2f}ms")
    print(f"  Avg Miss Time: {metrics['performance']['avg_miss_time_ms']:.2f}ms")

    print("\nPer-Tool Metrics:")
    for tool, stats in metrics["by_tool"].items():
        print(f"  {tool}: {stats}")

    # Health check
    health = await cache.health_check()
    print(f"\nHealth Status: {health['status']}")


# Example 7: Advanced Integration
class MockMCPServer:
    """Mock MCP server for demonstration."""

    def __init__(self):
        self.tools_executed = 0

    async def _execute_tool(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate tool execution."""
        self.tools_executed += 1
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "tool": tool_name,
            "arguments": arguments,
            "result": f"Executed at call #{self.tools_executed}",
        }


async def integration_example():
    """Demonstrate full MCP server integration."""
    print("\n=== MCP Server Integration Example ===")

    # Create mock server
    MockMCPServer()

    # Integration configuration

    # Note: In a real scenario, you would use:
    # middleware = integrate_cache_with_mcp_server(server, cache_config)

    print("Cache integration configured with custom tool settings")
    print("- search_tool: 2 minute cache")
    print("- details_tool: 1 hour cache")


async def main():
    """Run all examples."""
    examples = [
        basic_cache_example,
        decorator_example,
        custom_config_example,
        cache_management_example,
        cache_warming_example,
        monitoring_example,
        integration_example,
    ]

    for example in examples:
        try:
            await example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print()


if __name__ == "__main__":
    print("MCP Cache Usage Examples")
    print("=" * 50)
    asyncio.run(main())
