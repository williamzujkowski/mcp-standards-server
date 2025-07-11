"""Tests for Redis cache client."""

import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import redis
import redis.asyncio as aioredis

from src.core.cache.redis_client import (
    CacheConfig,
    CircuitBreaker,
    RedisCache,
    get_cache,
    init_cache,
)

# Mock Redis connections globally for tests
original_redis = redis.Redis
original_aioredis = aioredis.Redis


def mock_redis_factory(*args, **kwargs):
    """Factory for mocked Redis instances."""
    mock_instance = Mock()
    mock_instance.__enter__ = Mock(return_value=mock_instance)
    mock_instance.__exit__ = Mock(return_value=None)
    return mock_instance


def mock_aioredis_factory(*args, **kwargs):
    """Factory for mocked async Redis instances."""
    mock_instance = Mock()
    mock_instance.__aenter__ = Mock(return_value=mock_instance)
    mock_instance.__aexit__ = Mock(return_value=None)
    return mock_instance


# Patch Redis at module level
redis.Redis = mock_redis_factory
aioredis.Redis = mock_aioredis_factory


def create_redis_mock(**method_configs):
    """Create a Redis mock with specified method behaviors."""

    def custom_mock_factory(*args, **kwargs):
        mock_instance = Mock()
        mock_instance.__enter__ = Mock(return_value=mock_instance)
        mock_instance.__exit__ = Mock(return_value=None)

        # Configure methods based on input
        for method_name, return_value in method_configs.items():
            if callable(return_value):
                setattr(mock_instance, method_name, return_value)
            else:
                setattr(mock_instance, method_name, Mock(return_value=return_value))

        return mock_instance

    return custom_mock_factory


def create_async_redis_mock(**method_configs):
    """Create an async Redis mock with specified method behaviors."""

    def custom_mock_factory(*args, **kwargs):
        mock_instance = Mock()

        # Create async context manager methods
        async def async_enter(self):
            return mock_instance

        async def async_exit(self, *args):
            return None

        mock_instance.__aenter__ = async_enter
        mock_instance.__aexit__ = async_exit

        # Configure methods based on input
        for method_name, return_value in method_configs.items():
            if callable(return_value):
                setattr(mock_instance, method_name, return_value)
            else:
                # Create async mock with proper closure
                def make_async_method(rv):
                    async def async_method(*args, **kwargs):
                        return rv

                    return async_method

                setattr(mock_instance, method_name, make_async_method(return_value))

        return mock_instance

    return custom_mock_factory


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        cb = CircuitBreaker(threshold=3, timeout=10)
        assert cb.state == "closed"
        assert cb.can_attempt() is True

    def test_failure_threshold(self):
        """Test circuit breaker opens after threshold."""
        cb = CircuitBreaker(threshold=3, timeout=10)

        # Record failures
        for _ in range(3):
            cb.record_failure()

        assert cb.state == "open"
        assert cb.can_attempt() is False

    def test_half_open_after_timeout(self):
        """Test circuit breaker goes half-open after timeout."""
        cb = CircuitBreaker(threshold=1, timeout=0.1)

        cb.record_failure()
        assert cb.state == "open"

        time.sleep(0.2)
        assert cb.can_attempt() is True
        assert cb.state == "half-open"

    def test_close_on_success(self):
        """Test circuit breaker closes on success."""
        cb = CircuitBreaker(threshold=1, timeout=0.1)

        cb.record_failure()
        time.sleep(0.2)
        cb.record_success()

        assert cb.state == "closed"
        assert cb.failure_count == 0


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.default_ttl == 300
        assert config.key_prefix == "mcp"

    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            host="redis.example.com",
            port=6380,
            password="secret",
            default_ttl=600,
            key_prefix="test",
        )

        assert config.host == "redis.example.com"
        assert config.port == 6380
        assert config.password == "secret"
        assert config.default_ttl == 600
        assert config.key_prefix == "test"


class TestRedisCache:
    """Test Redis cache client."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        with (
            patch("redis.connection.ConnectionPool") as mock_sync_pool_cls,
            patch("redis.asyncio.ConnectionPool") as mock_async_pool_cls,
        ):
            mock_sync_pool = MagicMock()
            mock_async_pool = MagicMock()
            mock_sync_pool_cls.return_value = mock_sync_pool
            mock_async_pool_cls.return_value = mock_async_pool

            # Disable circuit breaker and retries for tests
            config = CacheConfig(
                key_prefix="test",
                enable_metrics=False,
                max_retries=1,  # Reduce retries to speed up tests
                circuit_breaker_threshold=100,  # High threshold to prevent opening
            )
            return RedisCache(config)

    def test_key_building(self, cache):
        """Test cache key building."""
        assert cache._build_key("mykey") == "test:mykey"
        assert cache._build_key("user:123") == "test:user:123"

    def test_serialization(self, cache):
        """Test value serialization."""
        # Test simple types
        data = {"key": "value", "number": 42}
        serialized = cache._serialize(data)
        assert isinstance(serialized, bytes)
        assert serialized[0:1] in (b"U", b"Z")  # Uncompressed or compressed

        deserialized = cache._deserialize(serialized)
        assert deserialized == data

    def test_compression(self, cache):
        """Test compression for large values."""
        # Create large data that should be compressed
        large_data = {"data": "x" * 2000}
        serialized = cache._serialize(large_data)

        # Should be compressed
        assert serialized[0:1] == b"Z"

        # Test decompression
        deserialized = cache._deserialize(serialized)
        assert deserialized == large_data

    def test_sync_get_hit(self, cache):
        """Test sync get with cache hit."""
        test_data = {"result": "data"}
        serialized_data = cache._serialize(test_data)

        with patch.object(
            redis, "Redis", side_effect=create_redis_mock(get=serialized_data)
        ):
            # Test get
            result = cache.get("test_key")
            assert result == {"result": "data"}

            # Check L1 cache
            assert "test:test_key" in cache._l1_cache

    def test_sync_get_miss(self, cache):
        """Test sync get with cache miss."""
        with patch.object(redis, "Redis", side_effect=create_redis_mock(get=None)):
            result = cache.get("missing_key")
            assert result is None

    def test_sync_set(self, cache):
        """Test sync set operation."""
        with patch.object(redis, "Redis", side_effect=create_redis_mock(setex=True)):
            result = cache.set("test_key", {"data": "value"}, ttl=60)
            assert result is True

            # Check L1 cache
            assert cache._l1_cache["test:test_key"] == {"data": "value"}

    def test_l1_cache_hit(self, cache):
        """Test L1 cache hit without Redis call."""
        # Pre-populate L1 cache
        cache._l1_cache["test:cached_key"] = {"cached": "data"}

        # Get should not call Redis
        result = cache.get("cached_key")
        assert result == {"cached": "data"}

    def test_delete(self, cache):
        """Test delete operation."""
        # Pre-populate L1 cache
        cache._l1_cache["test:del_key"] = {"data": "value"}

        with patch.object(redis, "Redis", side_effect=create_redis_mock(delete=1)):
            result = cache.delete("del_key")
            assert result is True

            # Check L1 cache cleared
            assert "test:del_key" not in cache._l1_cache

    def test_mget(self, cache):
        """Test multi-get operation."""
        # Pre-populate some L1 cache entries
        cache._l1_cache["test:key1"] = "value1"

        # Mock mget response for L1 misses (missing, key2, key3)
        mget_response = [
            None,  # missing key
            cache._serialize("value2"),  # key2
            cache._serialize("value3"),  # key3
        ]

        with patch.object(
            redis, "Redis", side_effect=create_redis_mock(mget=mget_response)
        ):
            result = cache.mget(["key1", "missing", "key2", "key3"])

            assert result == {
                "key1": "value1",  # From L1
                "key2": "value2",  # From L2
                "key3": "value3",  # From L2
            }

    @pytest.mark.asyncio
    async def test_async_get(self, cache):
        """Test async get operation."""
        test_data = {"async": "data"}
        serialized_data = cache._serialize(test_data)

        # Create async get function
        async def mock_get(key):
            return serialized_data

        with patch.object(
            aioredis, "Redis", side_effect=create_async_redis_mock(get=mock_get)
        ):
            result = await cache.async_get("async_key")
            assert result == {"async": "data"}

    @pytest.mark.asyncio
    async def test_async_set(self, cache):
        """Test async set operation."""

        # Create async setex function
        async def mock_setex(k, t, v):
            return True

        with patch.object(
            aioredis, "Redis", side_effect=create_async_redis_mock(setex=mock_setex)
        ):
            result = await cache.async_set("async_key", {"async": "value"})
            assert result is True

    def test_circuit_breaker_integration(self, cache):
        """Test circuit breaker prevents operations when open."""
        # Force circuit breaker open
        cache._circuit_breaker.state = "open"
        cache._circuit_breaker.last_failure_time = time.time()

        # When circuit breaker is open, get returns None, not raises exception
        result = cache.get("any_key")
        assert result is None

    def test_retry_logic(self, cache):
        """Test retry logic on Redis errors."""
        # Prepare serialized success data
        success_data = cache._serialize({"retry": "success"})

        # Counter to track attempts
        attempt_count = 0

        def get_with_retries(key):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise redis.ConnectionError("Connection failed")
            return success_data

        # Adjust max_retries for this test
        cache.config.max_retries = 3

        with patch.object(
            redis, "Redis", side_effect=create_redis_mock(get=get_with_retries)
        ):
            result = cache.get("retry_key")
            assert result == {"retry": "success"}
            assert attempt_count == 3

    def test_delete_pattern(self, cache):
        """Test pattern-based deletion."""
        # Pre-populate L1 cache
        cache._l1_cache["test:user:1"] = "data1"
        cache._l1_cache["test:user:2"] = "data2"
        cache._l1_cache["test:other:1"] = "other"

        # Mock keys and delete to return integers
        with patch.object(
            redis,
            "Redis",
            side_effect=create_redis_mock(
                keys=[b"test:user:1", b"test:user:2"],
                delete=2,  # Number of keys deleted
            ),
        ):
            result = cache.delete_pattern("user:*")
            assert result == 2

            # Check L1 cache cleared correctly
            assert "test:user:1" not in cache._l1_cache
            assert "test:user:2" not in cache._l1_cache
            assert "test:other:1" in cache._l1_cache

    def test_health_check(self, cache):
        """Test health check functionality."""
        with patch.object(redis, "Redis", side_effect=create_redis_mock(ping=True)):
            health = cache.health_check()

            assert health["status"] == "healthy"
            assert health["redis_connected"] is True
            assert health["latency_ms"] is not None
            assert "metrics" in health

    def test_metrics_collection(self, cache):
        """Test metrics collection."""
        # Simulate some operations
        cache._metrics["l1_hits"] = 70
        cache._metrics["l1_misses"] = 30
        cache._metrics["l2_hits"] = 40
        cache._metrics["l2_misses"] = 10

        metrics = cache.get_metrics()

        assert metrics["l1_hit_rate"] == 0.7
        assert metrics["l2_hit_rate"] == 0.8
        assert metrics["l1_hits"] == 70
        assert metrics["l1_misses"] == 30

    def test_generate_cache_key(self):
        """Test cache key generation."""
        key = RedisCache.generate_cache_key("search", "query", k=10, threshold=0.7)
        assert isinstance(key, str)
        assert len(key) == 64  # SHA-256 hash length

        # Same args should generate same key
        key2 = RedisCache.generate_cache_key("search", "query", k=10, threshold=0.7)
        assert key == key2

        # Different args should generate different key
        key3 = RedisCache.generate_cache_key("search", "query", k=20, threshold=0.7)
        assert key != key3


class TestGlobalCache:
    """Test global cache instance management."""

    def test_get_cache_singleton(self):
        """Test get_cache returns singleton."""
        cache1 = get_cache()
        cache2 = get_cache()
        assert cache1 is cache2

    def test_init_cache(self):
        """Test cache initialization with config."""
        config = CacheConfig(key_prefix="custom")
        cache = init_cache(config)

        assert cache.config.key_prefix == "custom"
        assert get_cache() is cache
