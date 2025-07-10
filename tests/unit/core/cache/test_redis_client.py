"""Tests for Redis cache client."""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest
import redis

from src.core.cache.redis_client import (
    CacheConfig,
    CircuitBreaker,
    RedisCache,
    get_cache,
    init_cache,
)


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
            key_prefix="test"
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
        return RedisCache(CacheConfig(key_prefix="test"))

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        with patch('redis.Redis') as mock:
            yield mock

    @pytest.fixture
    def mock_aioredis(self):
        """Mock async Redis client."""
        with patch('redis.asyncio.Redis') as mock:
            yield mock

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
        assert serialized[0:1] in (b'U', b'Z')  # Uncompressed or compressed

        deserialized = cache._deserialize(serialized)
        assert deserialized == data

    def test_compression(self, cache):
        """Test compression for large values."""
        # Create large data that should be compressed
        large_data = {"data": "x" * 2000}
        serialized = cache._serialize(large_data)

        # Should be compressed
        assert serialized[0:1] == b'Z'

        # Test decompression
        deserialized = cache._deserialize(serialized)
        assert deserialized == large_data

    def test_sync_get_hit(self, cache, mock_redis):
        """Test sync get with cache hit."""
        # Mock Redis response
        mock_client = MagicMock()
        mock_client.get.return_value = cache._serialize({"result": "data"})
        mock_redis.return_value.__enter__.return_value = mock_client

        # Test get
        result = cache.get("test_key")
        assert result == {"result": "data"}

        # Check Redis was called
        mock_client.get.assert_called_once_with("test:test_key")

        # Check L1 cache
        assert "test:test_key" in cache._l1_cache

    def test_sync_get_miss(self, cache, mock_redis):
        """Test sync get with cache miss."""
        mock_client = MagicMock()
        mock_client.get.return_value = None
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.get("missing_key")
        assert result is None

    def test_sync_set(self, cache, mock_redis):
        """Test sync set operation."""
        mock_client = MagicMock()
        mock_client.setex.return_value = True
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.set("test_key", {"data": "value"}, ttl=60)
        assert result is True

        # Check Redis was called
        mock_client.setex.assert_called_once()
        args = mock_client.setex.call_args[0]
        assert args[0] == "test:test_key"
        assert args[1] == 60

        # Check L1 cache
        assert cache._l1_cache["test:test_key"] == {"data": "value"}

    def test_l1_cache_hit(self, cache, mock_redis):
        """Test L1 cache hit without Redis call."""
        # Pre-populate L1 cache
        cache._l1_cache["test:cached_key"] = {"cached": "data"}

        # Get should not call Redis
        result = cache.get("cached_key")
        assert result == {"cached": "data"}

        # Redis should not be called
        mock_redis.assert_not_called()

    def test_delete(self, cache, mock_redis):
        """Test delete operation."""
        # Pre-populate L1 cache
        cache._l1_cache["test:del_key"] = {"data": "value"}

        mock_client = MagicMock()
        mock_client.delete.return_value = 1
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.delete("del_key")
        assert result is True

        # Check L1 cache cleared
        assert "test:del_key" not in cache._l1_cache

        # Check Redis delete called
        mock_client.delete.assert_called_once_with("test:del_key")

    def test_mget(self, cache, mock_redis):
        """Test multi-get operation."""
        # Pre-populate some L1 cache entries
        cache._l1_cache["test:key1"] = "value1"

        mock_client = MagicMock()
        mock_client.mget.return_value = [
            None,
            cache._serialize("value2"),
            cache._serialize("value3")
        ]
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.mget(["key1", "missing", "key2", "key3"])

        assert result == {
            "key1": "value1",  # From L1
            "key2": "value2",  # From L2
            "key3": "value3"   # From L2
        }

        # Check Redis mget was called only for L1 misses
        mock_client.mget.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_get(self, cache, mock_aioredis):
        """Test async get operation."""
        mock_client = MagicMock()
        mock_client.get = asyncio.coroutine(
            lambda key: cache._serialize({"async": "data"})
        )
        mock_aioredis.return_value.__aenter__.return_value = mock_client

        result = await cache.async_get("async_key")
        assert result == {"async": "data"}

    @pytest.mark.asyncio
    async def test_async_set(self, cache, mock_aioredis):
        """Test async set operation."""
        mock_client = MagicMock()
        mock_client.setex = asyncio.coroutine(lambda k, t, v: True)
        mock_aioredis.return_value.__aenter__.return_value = mock_client

        result = await cache.async_set("async_key", {"async": "value"})
        assert result is True

    def test_circuit_breaker_integration(self, cache, mock_redis):
        """Test circuit breaker prevents operations when open."""
        # Force circuit breaker open
        cache._circuit_breaker.state = "open"
        cache._circuit_breaker.last_failure_time = time.time()

        with pytest.raises(redis.ConnectionError):
            cache.get("any_key")

    def test_retry_logic(self, cache, mock_redis):
        """Test retry logic on Redis errors."""
        mock_client = MagicMock()

        # Fail twice, then succeed
        mock_client.get.side_effect = [
            redis.ConnectionError("Connection failed"),
            redis.ConnectionError("Connection failed"),
            cache._serialize({"retry": "success"})
        ]
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.get("retry_key")
        assert result == {"retry": "success"}
        assert mock_client.get.call_count == 3

    def test_delete_pattern(self, cache, mock_redis):
        """Test pattern-based deletion."""
        # Pre-populate L1 cache
        cache._l1_cache["test:user:1"] = "data1"
        cache._l1_cache["test:user:2"] = "data2"
        cache._l1_cache["test:other:1"] = "other"

        mock_client = MagicMock()
        mock_client.keys.return_value = [b"test:user:1", b"test:user:2"]
        mock_client.delete.return_value = 2
        mock_redis.return_value.__enter__.return_value = mock_client

        result = cache.delete_pattern("user:*")
        assert result == 2

        # Check L1 cache cleared correctly
        assert "test:user:1" not in cache._l1_cache
        assert "test:user:2" not in cache._l1_cache
        assert "test:other:1" in cache._l1_cache

    def test_health_check(self, cache, mock_redis):
        """Test health check functionality."""
        mock_client = MagicMock()
        mock_client.ping.return_value = True
        mock_redis.return_value.__enter__.return_value = mock_client

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
        assert len(key) == 32  # MD5 hash length

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
