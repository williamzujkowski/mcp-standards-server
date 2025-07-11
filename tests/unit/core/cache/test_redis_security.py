"""Security tests for Redis cache client."""

import json
import pickle
from unittest.mock import MagicMock, patch

import pytest

from src.core.cache.redis_client import CacheConfig, RedisCache


class TestRedisCacheSecurity:
    """Test security features of Redis cache client."""

    @pytest.fixture
    def cache(self):
        """Create cache instance."""
        return RedisCache(CacheConfig(key_prefix="security_test"))

    def test_pickle_deserialization_disabled(self, cache):
        """Test that pickle deserialization is properly disabled."""
        # Create some data that would be pickle-serialized
        test_data = {"key": "value", "number": 42}

        # Simulate old pickle-serialized data
        pickled_data = pickle.dumps(test_data)
        old_format = b"Up" + pickled_data  # 'p' for pickle serializer

        # Test that attempting to deserialize pickle data raises an error
        with pytest.raises(ValueError, match="Pickle deserialization is disabled"):
            cache._deserialize(old_format)

    def test_pickle_deserialization_logging(self, cache, caplog):
        """Test that pickle deserialization attempts are logged."""
        test_data = {"key": "value"}
        pickled_data = pickle.dumps(test_data)
        old_format = b"Up" + pickled_data

        with pytest.raises(ValueError):
            cache._deserialize(old_format)

        # Check that error was logged
        assert "Attempted to deserialize pickled data" in caplog.text
        assert "security reasons" in caplog.text

    def test_msgpack_serialization_preferred(self, cache):
        """Test that msgpack is used for serialization when possible."""
        test_data = {"key": "value", "number": 42}

        serialized = cache._serialize(test_data)

        # Should use msgpack ('m' serializer)
        assert serialized[1:2] == b"m"

        # Should be able to deserialize
        deserialized = cache._deserialize(serialized)
        assert deserialized == test_data

    def test_json_fallback_serialization(self, cache):
        """Test JSON fallback for msgpack-incompatible data."""
        # Create data that msgpack can't handle but JSON can
        test_data = {"set": {"a", "b", "c"}}

        # Mock msgpack to fail
        with patch("msgpack.packb", side_effect=TypeError("Mock msgpack failure")):
            serialized = cache._serialize(test_data)

            # Should fall back to JSON ('j' serializer)
            assert serialized[1:2] == b"j"

            # Should be able to deserialize
            deserialized = cache._deserialize(serialized)
            # The custom JSON encoder preserves sets
            assert deserialized == {"set": {"a", "b", "c"}}

    def test_unsafe_serialization_rejected(self, cache):
        """Test that unsafe/unserializable data is rejected."""

        # Create an object that can't be serialized by msgpack or JSON
        class UnsafeObject:
            def __init__(self):
                self.data = "dangerous"

        unsafe_obj = UnsafeObject()

        # Should raise TypeError for unserializable objects
        with pytest.raises(TypeError, match="Cannot serialize object"):
            cache._serialize(unsafe_obj)

    def test_custom_json_encoder_security(self, cache):
        """Test that custom JSON encoder handles safe types only."""
        encoder = cache._get_json_encoder()

        # Test safe types
        safe_data = {"set": {1, 2, 3}, "bytes": b"test data"}

        json_str = json.dumps(safe_data, cls=encoder)
        assert "__type__" in json_str

        # Test that arbitrary objects are not serialized
        class DangerousObject:
            pass

        dangerous_data = {"obj": DangerousObject()}

        with pytest.raises(TypeError):
            json.dumps(dangerous_data, cls=encoder)

    def test_sha256_cache_key_generation(self, cache):
        """Test that SHA-256 is used for cache key generation."""
        key = cache.generate_cache_key("test", "data", param=123)

        # SHA-256 produces 64 character hex string
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

        # Should be deterministic
        key2 = cache.generate_cache_key("test", "data", param=123)
        assert key == key2

        # Different inputs should produce different keys
        key3 = cache.generate_cache_key("test", "data", param=124)
        assert key != key3

    def test_cache_key_salting(self, cache):
        """Test that cache keys are salted to prevent rainbow table attacks."""
        # The salt should be included in the key generation
        key = cache.generate_cache_key("test")

        # Generate a key with the same input but different salt
        import hashlib

        raw_key = hashlib.sha256(b"test").hexdigest()

        # Should be different due to salting
        assert key != raw_key

    def test_compression_security(self, cache):
        """Test that compression doesn't introduce security vulnerabilities."""
        # Test with large data that triggers compression
        large_data = {"data": "x" * 2000}

        serialized = cache._serialize(large_data)

        # Should be compressed (starts with 'Z')
        assert serialized[0:1] == b"Z"

        # Should decompress safely
        deserialized = cache._deserialize(serialized)
        assert deserialized == large_data

    def test_input_validation_in_serialization(self, cache):
        """Test input validation in serialization methods."""
        # Test empty data
        empty_result = cache._deserialize(b"")
        assert empty_result is None

        # Test invalid serializer format
        with pytest.raises(ValueError, match="Unknown serializer type"):
            cache._deserialize(b"Ux")  # 'x' is not a valid serializer

    def test_no_code_execution_in_deserialization(self, cache):
        """Test that deserialization cannot execute arbitrary code."""
        # Test that even if someone tries to inject code, it's not executed
        malicious_json = (
            '{"__type__": "exec", "code": "import os; os.system(\'rm -rf /\')"}'
        )

        # Should not execute the code, just return the data
        result = cache._json_object_hook(json.loads(malicious_json))
        assert result == {
            "__type__": "exec",
            "code": "import os; os.system('rm -rf /')",
        }

        # Should not have executed anything dangerous
        assert "__type__" in result

    def test_size_limits_in_serialization(self, cache):
        """Test that extremely large data is handled safely."""
        # Create data that's large but not too large to cause memory issues
        large_data = {"data": "x" * 100000}  # 100KB

        serialized = cache._serialize(large_data)
        deserialized = cache._deserialize(serialized)

        assert deserialized == large_data

    def test_circuit_breaker_security(self, cache):
        """Test that circuit breaker prevents abuse."""
        # Circuit breaker should prevent rapid failures from overwhelming the system
        cb = cache._circuit_breaker

        # Simulate many failures
        for _ in range(10):
            cb.record_failure()

        assert cb.state == "open"
        assert not cb.can_attempt()

        # Circuit breaker doesn't raise exceptions, it just logs and returns None
        # The get operation should return None when circuit breaker is open
        result = cache.get("test_key")
        assert result is None

    def test_key_prefix_isolation(self, cache):
        """Test that key prefixes provide namespace isolation."""
        key1 = cache._build_key("test")

        # Different cache instance with different prefix
        cache2 = RedisCache(CacheConfig(key_prefix="other"))
        key2 = cache2._build_key("test")

        assert key1 != key2
        assert key1.startswith("security_test:")
        assert key2.startswith("other:")

    def test_metrics_dont_leak_sensitive_data(self, cache):
        """Test that metrics don't expose sensitive information."""
        metrics = cache.get_metrics()

        # Should only contain numeric metrics, not actual data
        expected_keys = [
            "l1_hits",
            "l1_misses",
            "l2_hits",
            "l2_misses",
            "errors",
            "slow_queries",
            "connection_errors",
            "pool_exhausted",
            "pipeline_operations",
            "l1_hit_rate",
            "l2_hit_rate",
            "l1_cache_size",
            "circuit_breaker_state",
            "connection_health",
            "pool_stats",
            "health_check_success_rate",
        ]

        for key, value in metrics.items():
            # Check that values are safe types (not actual cached data)
            assert isinstance(value, int | float | str | dict)
            # Check that key is expected
            assert key in expected_keys, f"Unexpected metric key: {key}"

    def test_health_check_security(self, cache):
        """Test that health check doesn't expose sensitive information."""
        with patch("redis.Redis") as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_redis.return_value.__enter__.return_value = mock_client

            health = cache.health_check()

            # Should only contain safe health information
            safe_keys = {
                "status",
                "l1_cache_size",
                "circuit_breaker_state",
                "metrics",
                "redis_connected",
                "latency_ms",
                "connection_health",
                "pool_stats",
            }

            assert set(health.keys()) == safe_keys
            assert health["status"] in ["healthy", "degraded", "unhealthy"]
            assert isinstance(health["redis_connected"], bool)
