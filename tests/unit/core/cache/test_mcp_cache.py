"""Tests for MCP cache module."""

import asyncio

import pytest

from src.core.cache.mcp_cache import (
    CacheMetrics,
    CacheStrategy,
    MCPCache,
    ToolCacheConfig,
    cache_tool_response,
)


class MockRedisCache:
    """Mock Redis cache for testing."""

    def __init__(self):
        self.data = {}
        self.call_count = {"get": 0, "set": 0, "delete": 0}

    async def async_get(self, key: str):
        self.call_count["get"] += 1
        return self.data.get(key)

    async def async_set(self, key: str, value, ttl=None):
        self.call_count["set"] += 1
        self.data[key] = value
        return True

    async def async_delete(self, key: str):
        self.call_count["delete"] += 1
        if key in self.data:
            del self.data[key]
            return True
        return False

    async def async_delete_pattern(self, pattern: str):
        # Simple pattern matching for tests
        deleted = 0
        keys_to_delete = []
        for key in self.data:
            if pattern.endswith("*") and key.startswith(pattern[:-1]):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del self.data[key]
            deleted += 1

        return deleted

    async def async_health_check(self):
        return {"status": "healthy", "redis_connected": True}

    def get_metrics(self):
        return {"operations": self.call_count}


@pytest.fixture
def mock_redis():
    """Create mock Redis cache."""
    return MockRedisCache()


@pytest.fixture
def mcp_cache(mock_redis):
    """Create MCP cache with mock Redis."""
    # Add custom config for test tools
    custom_configs = {
        "test_tool": ToolCacheConfig(
            tool_name="test_tool",
            strategy=CacheStrategy.MEDIUM_TTL,
            include_in_key=["arg1", "arg2", "id"],
        ),
        "tool1": ToolCacheConfig(
            tool_name="tool1", strategy=CacheStrategy.MEDIUM_TTL, include_in_key=["id"]
        ),
        "tool2": ToolCacheConfig(
            tool_name="tool2", strategy=CacheStrategy.MEDIUM_TTL, include_in_key=["id"]
        ),
        "get_test_data": ToolCacheConfig(
            tool_name="get_test_data",
            strategy=CacheStrategy.SHORT_TTL,
            include_in_key=["key"],
        ),
        "decorated_tool": ToolCacheConfig(
            tool_name="decorated_tool",
            strategy=CacheStrategy.MEDIUM_TTL,
            include_in_key=["value"],
        ),
    }
    cache = MCPCache(
        redis_cache=mock_redis, enable_metrics=True, custom_configs=custom_configs
    )
    # Ensure metrics are initialized
    cache.metrics = CacheMetrics()
    return cache


class TestMCPCache:
    """Test MCP cache functionality."""

    def test_cache_key_generation(self, mcp_cache):
        """Test cache key generation."""
        # Basic key generation
        key = mcp_cache.generate_cache_key("test_tool", {"arg1": "value1", "arg2": 123})
        assert key.startswith("mcp:tool:test_tool:")

        # Same arguments should generate same key
        key2 = mcp_cache.generate_cache_key(
            "test_tool", {"arg2": 123, "arg1": "value1"}  # Different order
        )
        assert key == key2

        # Different arguments should generate different keys
        key3 = mcp_cache.generate_cache_key(
            "test_tool", {"arg1": "value2", "arg2": 123}
        )
        assert key != key3

    def test_cache_key_with_config(self, mcp_cache):
        """Test cache key generation with custom config."""
        config = ToolCacheConfig(tool_name="test_tool", include_in_key=["arg1"])

        # Only arg1 should be included
        key1 = mcp_cache.generate_cache_key(
            "test_tool", {"arg1": "value1", "arg2": 123}, config
        )

        key2 = mcp_cache.generate_cache_key(
            "test_tool", {"arg1": "value1", "arg2": 456}, config  # Different arg2
        )

        assert key1 == key2  # Should be same since arg2 is not included

    @pytest.mark.asyncio
    async def test_get_set_basic(self, mcp_cache):
        """Test basic get/set operations."""
        tool_name = "test_tool"
        args = {"arg1": "value1"}
        response = {"result": "success"}

        # Configure the tool for caching
        mcp_cache.configure_tool(tool_name, strategy=CacheStrategy.SHORT_TTL)

        # Cache miss
        result = await mcp_cache.get(tool_name, args)
        assert result is None
        assert mcp_cache.metrics.misses == 1
        assert mcp_cache.metrics.hits == 0

        # Set value
        success = await mcp_cache.set(tool_name, args, response)
        assert success

        # Cache hit
        result = await mcp_cache.get(tool_name, args)
        assert result == response
        assert mcp_cache.metrics.hits == 1
        assert mcp_cache.metrics.misses == 1

    @pytest.mark.asyncio
    async def test_no_cache_strategy(self, mcp_cache):
        """Test NO_CACHE strategy."""
        # Configure tool with NO_CACHE
        mcp_cache.configure_tool("no_cache_tool", strategy=CacheStrategy.NO_CACHE)

        args = {"arg1": "value1"}
        response = {"result": "success"}

        # Should not cache
        success = await mcp_cache.set("no_cache_tool", args, response)
        assert not success

        # Should not retrieve
        result = await mcp_cache.get("no_cache_tool", args)
        assert result is None

    @pytest.mark.asyncio
    async def test_compression(self, mcp_cache):
        """Test response compression."""
        # Create large response
        large_response = {"data": "x" * 2000}

        # Configure with low compression threshold
        mcp_cache.configure_tool("compress_tool", compress_threshold=100)

        args = {"arg1": "value1"}

        # Set should compress
        success = await mcp_cache.set("compress_tool", args, large_response)
        assert success
        assert mcp_cache.metrics.compressed_saves == 1

        # Get should decompress
        result = await mcp_cache.get("compress_tool", args)
        assert result == large_response

    @pytest.mark.asyncio
    async def test_invalidation(self, mcp_cache):
        """Test cache invalidation."""
        tool_name = "test_tool"
        args1 = {"arg1": "value1"}
        args2 = {"arg1": "value2"}
        response = {"result": "success"}

        # Configure the tool for caching
        mcp_cache.configure_tool(tool_name, strategy=CacheStrategy.SHORT_TTL)

        # Cache two entries
        await mcp_cache.set(tool_name, args1, response)
        await mcp_cache.set(tool_name, args2, response)

        # Invalidate specific entry
        success = await mcp_cache.invalidate(tool_name, args1)
        assert success

        # First should be gone, second should remain
        assert await mcp_cache.get(tool_name, args1) is None
        assert await mcp_cache.get(tool_name, args2) == response

        # Invalidate all entries for tool
        success = await mcp_cache.invalidate(tool_name)
        assert success
        assert await mcp_cache.get(tool_name, args2) is None

    @pytest.mark.asyncio
    async def test_invalidation_cascade(self, mcp_cache):
        """Test invalidation cascade."""
        # Configure tools with invalidation relationships
        mcp_cache.configure_tool("reader_tool", invalidate_on=["writer_tool"])
        # Configure writer_tool to be cacheable (needed for invalidation to trigger)
        mcp_cache.configure_tool("writer_tool", strategy=CacheStrategy.SHORT_TTL)
        mcp_cache._build_invalidation_map()

        # Cache some data
        await mcp_cache.set("reader_tool", {"id": 1}, {"data": "cached"})

        # Writing should invalidate reader
        await mcp_cache.set("writer_tool", {"id": 1}, {"status": "written"})

        # Reader cache should be invalidated
        assert await mcp_cache.get("reader_tool", {"id": 1}) is None

    @pytest.mark.asyncio
    async def test_cache_warming(self, mcp_cache):
        """Test cache warming."""
        # Configure tool for warming
        mcp_cache.configure_tool(
            "warm_tool", warm_on_startup=True, warm_args=[{"id": 1}, {"id": 2}]
        )

        # Mock executor
        async def mock_executor(tool_name, args):
            return {"result": f"data_for_{args['id']}"}

        # Warm cache
        results = await mcp_cache.warm_cache(mock_executor, ["warm_tool"])
        assert results["warm_tool"] == 2

        # Check warmed entries
        assert await mcp_cache.get("warm_tool", {"id": 1}) == {"result": "data_for_1"}
        assert await mcp_cache.get("warm_tool", {"id": 2}) == {"result": "data_for_2"}

    def test_metrics(self, mcp_cache):
        """Test metrics collection."""
        metrics = mcp_cache.get_metrics()

        assert "overall" in metrics
        assert "performance" in metrics
        assert "compression" in metrics
        assert "by_tool" in metrics

        # Test hit rate calculation
        assert metrics["overall"]["hit_rate"] == 0.0  # No hits/misses yet

    @pytest.mark.asyncio
    async def test_ttl_override(self, mcp_cache):
        """Test TTL override."""
        tool_name = "test_tool"
        args = {"arg1": "value1"}
        response = {"result": "success"}

        # Set with custom TTL
        success = await mcp_cache.set(tool_name, args, response, ttl_override=60)
        assert success

        # Should be cached
        result = await mcp_cache.get(tool_name, args)
        assert result == response

    @pytest.mark.asyncio
    async def test_clear_all(self, mcp_cache, mock_redis):
        """Test clearing all caches."""
        # Add some cached data
        await mcp_cache.set("tool1", {"id": 1}, {"data": 1})
        await mcp_cache.set("tool2", {"id": 2}, {"data": 2})

        # Clear all
        count = await mcp_cache.clear_all()
        assert count == 2

        # Everything should be gone
        assert await mcp_cache.get("tool1", {"id": 1}) is None
        assert await mcp_cache.get("tool2", {"id": 2}) is None

    @pytest.mark.asyncio
    async def test_health_check(self, mcp_cache):
        """Test health check."""
        health = await mcp_cache.health_check()

        assert health["status"] == "healthy"
        assert health["cache_enabled"] is True
        assert "redis" in health


class TestCacheDecorator:
    """Test cache decorator functionality."""

    @pytest.mark.asyncio
    async def test_cache_decorator(self, mcp_cache):
        """Test cache_tool_response decorator."""
        call_count = 0

        @cache_tool_response(mcp_cache, "decorated_tool")
        async def expensive_operation(args):
            nonlocal call_count
            call_count += 1
            return {"result": args["value"] * 2}

        # First call - cache miss
        result1 = await expensive_operation({"value": 5})
        assert result1 == {"result": 10}
        assert call_count == 1

        # Second call - cache hit
        result2 = await expensive_operation({"value": 5})
        assert result2 == {"result": 10}
        assert call_count == 1  # Not called again

        # Different args - cache miss
        result3 = await expensive_operation({"value": 10})
        assert result3 == {"result": 20}
        assert call_count == 2


class TestDefaultConfigurations:
    """Test default tool configurations."""

    def test_default_configs_loaded(self, mcp_cache):
        """Test that default configurations are loaded."""
        # Check some key tools are configured
        assert "get_standard_details" in mcp_cache.tool_configs
        assert "search_standards" in mcp_cache.tool_configs
        assert "sync_standards" in mcp_cache.tool_configs

        # Check NO_CACHE tools
        sync_config = mcp_cache.tool_configs["sync_standards"]
        assert sync_config.strategy == CacheStrategy.NO_CACHE

        # Check cached tools
        details_config = mcp_cache.tool_configs["get_standard_details"]
        assert details_config.strategy == CacheStrategy.LONG_TTL
        assert details_config.warm_on_startup is True

    def test_custom_config_override(self):
        """Test custom configuration override."""
        custom_configs = {
            "get_standard_details": ToolCacheConfig(
                tool_name="get_standard_details",
                strategy=CacheStrategy.SHORT_TTL,
                ttl_seconds=60,
            )
        }

        cache = MCPCache(redis_cache=MockRedisCache(), custom_configs=custom_configs)

        config = cache.tool_configs["get_standard_details"]
        assert config.strategy == CacheStrategy.SHORT_TTL
        assert config.ttl_seconds == 60


@pytest.mark.asyncio
async def test_background_warming():
    """Test background cache warming."""
    mock_redis = MockRedisCache()
    cache = MCPCache(redis_cache=mock_redis)

    # Configure tool for warming
    cache.configure_tool("background_tool", warm_on_startup=True, warm_args=[{"id": 1}])

    call_count = 0

    async def mock_executor(tool_name, args):
        nonlocal call_count
        call_count += 1
        return {"result": "warmed"}

    # Start background warming with short interval
    await cache.start_background_warming(mock_executor, interval_seconds=0.1)

    # Wait for warming to occur
    await asyncio.sleep(0.15)

    # Should have warmed at least once
    assert call_count >= 1

    # Stop warming
    cache.stop_background_warming()

    # Wait a bit
    await asyncio.sleep(0.2)

    # Count should not increase much more
    final_count = call_count
    await asyncio.sleep(0.1)
    assert call_count == final_count  # No more warming
