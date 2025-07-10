"""Tests for cache decorators."""

from typing import Any
from unittest.mock import Mock

import pytest

from src.core.cache.decorators import (
    CacheKeyConfig,
    CacheManager,
    cache_key,
    cache_result,
    generate_cache_key,
    invalidate_cache,
)
from src.core.cache.redis_client import RedisCache


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_simple_function_key(self):
        """Test key generation for simple function."""

        def simple_func(a: int, b: str) -> str:
            return f"{a}:{b}"

        config = CacheKeyConfig(prefix="test")
        key = generate_cache_key(simple_func, (1, "hello"), {}, config)

        assert key.startswith("test:v1:simple_func:")
        assert len(key.split(":")) == 4

    def test_method_key_without_self(self):
        """Test key generation for method excluding self."""

        class TestClass:
            def method(self, value: str) -> str:
                return value

        obj = TestClass()
        config = CacheKeyConfig(prefix="test", include_self=False)
        key = generate_cache_key(obj.method, (obj, "data"), {}, config)

        # Should not include self in key
        assert "TestClass" not in key

    def test_method_key_with_self(self):
        """Test key generation for method including self."""

        class TestClass:
            def __init__(self, id: int):
                self.id = id

            def __repr__(self):
                return f"TestClass({self.id})"

            def method(self, value: str) -> str:
                return value

        obj = TestClass(123)
        config = CacheKeyConfig(prefix="test", include_self=True)
        key = generate_cache_key(obj.method, (obj, "data"), {}, config)

        # Key should be influenced by self
        assert key != generate_cache_key(
            TestClass(456).method, (TestClass(456), "data"), {}, config
        )

    def test_excluded_args(self):
        """Test excluding specific arguments from key."""

        def func(query: str, limit: int, offset: int) -> list[str]:
            return []

        config = CacheKeyConfig(prefix="search", exclude_args={"limit", "offset"})

        # Keys with different limit/offset should be same
        key1 = generate_cache_key(func, ("test",), {"limit": 10, "offset": 0}, config)
        key2 = generate_cache_key(func, ("test",), {"limit": 20, "offset": 10}, config)
        assert key1 == key2

        # Keys with different query should be different
        key3 = generate_cache_key(func, ("other",), {"limit": 10, "offset": 0}, config)
        assert key1 != key3

    def test_custom_key_function(self):
        """Test custom key generation function."""

        def custom_key_func(func, args, kwargs):
            return f"custom:{func.__name__}:{args[0] if args else 'none'}"

        config = CacheKeyConfig(prefix="test", custom_key_func=custom_key_func)

        def test_func(value: str) -> str:
            return value

        key = generate_cache_key(test_func, ("hello",), {}, config)
        assert key == "custom:test_func:hello"


class TestCacheResultDecorator:
    """Test cache_result decorator."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = Mock(spec=RedisCache)
        cache.get.return_value = None
        cache.set.return_value = True

        # Create proper async coroutines for async methods
        async def async_get(key):
            return None

        async def async_set(key, value, ttl=None):
            return True

        cache.async_get = async_get
        cache.async_set = async_set
        return cache

    def test_sync_function_cache_miss(self, mock_cache):
        """Test sync function with cache miss."""

        @cache_result("test", ttl=60, cache=mock_cache)
        def expensive_func(value: str) -> str:
            return f"computed:{value}"

        result = expensive_func("hello")
        assert result == "computed:hello"

        # Check cache interactions
        assert mock_cache.get.called
        assert mock_cache.set.called

        # Verify set was called with correct value
        set_args = mock_cache.set.call_args
        assert set_args[0][1] == "computed:hello"
        assert set_args[1]["ttl"] == 60

    def test_sync_function_cache_hit(self, mock_cache):
        """Test sync function with cache hit."""
        mock_cache.get.return_value = "cached:hello"

        @cache_result("test", cache=mock_cache)
        def expensive_func(value: str) -> str:
            return f"computed:{value}"

        result = expensive_func("hello")
        assert result == "cached:hello"

        # Should not call set on cache hit
        assert mock_cache.get.called
        assert not mock_cache.set.called

    @pytest.mark.asyncio
    async def test_async_function_cache_miss(self, mock_cache):
        """Test async function with cache miss."""
        # Track cache calls
        get_called = False
        set_called = False
        set_args = None

        async def async_get_track(key):
            nonlocal get_called
            get_called = True
            return None  # Cache miss

        async def async_set_track(key, value, ttl=None):
            nonlocal set_called, set_args
            set_called = True
            set_args = (key, value, ttl)
            return True

        mock_cache.async_get = async_get_track
        mock_cache.async_set = async_set_track

        @cache_result("test", ttl=120, cache=mock_cache)
        async def async_expensive_func(value: str) -> str:
            return f"async_computed:{value}"

        result = await async_expensive_func("world")
        assert result == "async_computed:world"

        # Check async cache interactions
        assert get_called
        assert set_called
        assert set_args is not None
        assert set_args[1] == "async_computed:world"  # value
        assert set_args[2] == 120  # ttl

    @pytest.mark.asyncio
    async def test_async_function_cache_hit(self, mock_cache):
        """Test async function with cache hit."""

        # Override the async_get to return cached value
        async def async_get_cached(key):
            return "async_cached:world"

        mock_cache.async_get = async_get_cached

        # Track if async_set was called
        set_called = False

        async def async_set_track(key, value, ttl=None):
            nonlocal set_called
            set_called = True
            return True

        mock_cache.async_set = async_set_track

        @cache_result("test", cache=mock_cache)
        async def async_expensive_func(value: str) -> str:
            return f"async_computed:{value}"

        result = await async_expensive_func("world")
        assert result == "async_cached:world"

        # Should not call set on cache hit
        assert not set_called

    def test_condition_function(self, mock_cache):
        """Test conditional caching."""

        @cache_result(
            "test", cache=mock_cache, condition=lambda value, use_cache=True: use_cache
        )
        def conditional_func(value: str, use_cache: bool = True) -> str:
            return f"result:{value}"

        # With cache enabled (default)
        conditional_func("test1")
        assert mock_cache.get.called

        # With cache disabled
        mock_cache.reset_mock()
        conditional_func("test2", use_cache=False)
        assert not mock_cache.get.called
        assert not mock_cache.set.called

    def test_callbacks(self, mock_cache):
        """Test on_cached and on_computed callbacks."""
        cached_calls = []
        computed_calls = []

        @cache_result(
            "test",
            cache=mock_cache,
            on_cached=lambda k, v: cached_calls.append((k, v)),
            on_computed=lambda k, v: computed_calls.append((k, v)),
        )
        def func_with_callbacks(value: str) -> str:
            return f"result:{value}"

        # First call - computed
        func_with_callbacks("test")
        assert len(computed_calls) == 1
        assert len(cached_calls) == 0

        # Second call - cached
        mock_cache.get.return_value = "result:test"
        func_with_callbacks("test")
        assert len(computed_calls) == 1
        assert len(cached_calls) == 1

    def test_cache_key_with_complex_args(self, mock_cache):
        """Test caching with complex arguments."""

        @cache_result("complex", cache=mock_cache)
        def complex_func(
            data: dict[str, Any], items: list[str], flag: bool = True
        ) -> dict[str, Any]:
            return {"processed": data, "items": items, "flag": flag}

        complex_func({"key": "value"}, ["item1", "item2"], flag=False)

        # Verify cache was used
        assert mock_cache.get.called
        assert mock_cache.set.called

    def test_error_handling(self, mock_cache):
        """Test decorator handles cache errors gracefully."""
        # Simulate cache error
        mock_cache.get.side_effect = Exception("Cache error")

        @cache_result("test", cache=mock_cache)
        def func_with_error(value: str) -> str:
            return f"computed:{value}"

        # Should still compute result despite cache error
        result = func_with_error("test")
        assert result == "computed:test"


class TestInvalidateCacheDecorator:
    """Test invalidate_cache decorator."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = Mock(spec=RedisCache)
        cache.delete.return_value = True
        cache.delete_pattern.return_value = 5

        # Create proper async coroutines for async methods
        async def async_delete(key):
            return True

        async def async_delete_pattern(pattern):
            return 5

        cache.async_delete = async_delete
        cache.async_delete_pattern = async_delete_pattern
        return cache

    def test_invalidate_specific_keys(self, mock_cache):
        """Test invalidating specific cache keys."""

        @invalidate_cache(keys=["key1", "key2"], cache=mock_cache)
        def update_func():
            return "updated"

        result = update_func()
        assert result == "updated"

        # Check keys were deleted
        assert mock_cache.delete.call_count == 2
        mock_cache.delete.assert_any_call("key1")
        mock_cache.delete.assert_any_call("key2")

    def test_invalidate_by_pattern(self, mock_cache):
        """Test invalidating by pattern."""

        @invalidate_cache(pattern="user:*", cache=mock_cache)
        def delete_all_users():
            return "deleted"

        result = delete_all_users()
        assert result == "deleted"

        mock_cache.delete_pattern.assert_called_once_with("user:*")

    def test_invalidate_by_prefix(self, mock_cache):
        """Test invalidating by prefix."""

        @invalidate_cache(prefix="search", cache=mock_cache)
        def clear_search_cache():
            return "cleared"

        result = clear_search_cache()
        assert result == "cleared"

        mock_cache.delete_pattern.assert_called_once_with("search:*")

    def test_pattern_substitution(self, mock_cache):
        """Test pattern substitution with function arguments."""

        @invalidate_cache(pattern="user:{user_id}:*", cache=mock_cache)
        def update_user(user_id: str, data: dict):
            return f"updated user {user_id}"

        result = update_user("123", {"name": "Test"})
        assert result == "updated user 123"

        mock_cache.delete_pattern.assert_called_once_with("user:123:*")

    @pytest.mark.asyncio
    async def test_async_invalidation(self, mock_cache):
        """Test async function invalidation."""

        @invalidate_cache(prefix="async", cache=mock_cache)
        async def async_update():
            return "async updated"

        result = await async_update()
        assert result == "async updated"

    def test_invalidation_error_handling(self, mock_cache):
        """Test invalidation continues despite errors."""
        mock_cache.delete_pattern.side_effect = Exception("Delete failed")

        @invalidate_cache(prefix="error", cache=mock_cache)
        def func_with_error():
            return "completed"

        # Should not raise exception
        result = func_with_error()
        assert result == "completed"


class TestCacheManager:
    """Test CacheManager context manager."""

    @pytest.fixture
    def mock_cache(self):
        """Create mock cache."""
        cache = Mock(spec=RedisCache)

        # Setup async methods
        async def async_get(key):
            return f"value_{key}"

        async def async_set(key, value, ttl=None):
            return True

        async def async_mget(keys):
            return {k: f"value_{k}" for k in keys}

        async def async_mset(mapping, ttl=None):
            return True

        cache.async_get = async_get
        cache.async_set = async_set
        cache.async_mget = async_mget
        cache.async_mset = async_mset

        # Setup sync methods
        cache.mset.return_value = True

        return cache

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_cache):
        """Test async context manager usage."""
        async with CacheManager(cache=mock_cache) as cm:
            # Test get
            value = await cm.get("key1")
            assert value == "value_key1"

            # Test set
            await cm.set("key2", "data2")

            # Test mget
            values = await cm.mget(["key3", "key4"])
            assert values == {"key3": "value_key3", "key4": "value_key4"}

            # Test mset
            await cm.mset({"key5": "data5", "key6": "data6"})

    def test_sync_context_manager(self, mock_cache):
        """Test sync context manager usage."""
        with CacheManager(cache=mock_cache) as cm:
            # Add to batch
            cm.batch_sets["key1"] = "value1"
            cm.batch_sets["key2"] = "value2"

        # Check batch was executed on exit
        mock_cache.mset.assert_called_once_with({"key1": "value1", "key2": "value2"})


class TestUtilityFunctions:
    """Test utility functions."""

    def test_cache_key_generation(self):
        """Test cache_key utility function."""
        key = cache_key("user", 123, role="admin", active=True)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hash

        # Same arguments should produce same key
        key2 = cache_key("user", 123, role="admin", active=True)
        assert key == key2

        # Different arguments should produce different key
        key3 = cache_key("user", 456, role="admin", active=True)
        assert key != key3
