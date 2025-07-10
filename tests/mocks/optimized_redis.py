"""Optimized Redis mock for faster test execution."""

import asyncio
from typing import Any


class OptimizedMockRedis:
    """Lightweight Redis mock optimized for test performance."""

    __slots__ = ("_data", "_expires", "_lock")

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._expires: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        """Get value with minimal overhead."""
        return self._data.get(key)

    async def set(self, key: str, value: Any, ex: int | None = None) -> bool:
        """Set value with minimal overhead."""
        async with self._lock:
            self._data[key] = value
            if ex:
                self._expires[key] = asyncio.get_event_loop().time() + ex
        return True

    async def delete(self, *keys: str) -> int:
        """Delete keys with minimal overhead."""
        deleted = 0
        async with self._lock:
            for key in keys:
                if key in self._data:
                    del self._data[key]
                    self._expires.pop(key, None)
                    deleted += 1
        return deleted

    async def exists(self, *keys: str) -> int:
        """Check key existence."""
        return sum(1 for k in keys if k in self._data)

    async def flushdb(self) -> bool:
        """Clear all data."""
        async with self._lock:
            self._data.clear()
            self._expires.clear()
        return True

    async def ping(self) -> str:
        """Health check."""
        return "PONG"

    # Async context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    # Sync methods for compatibility
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OptimizedRedisCache:
    """Optimized Redis cache wrapper for tests."""

    __slots__ = ("_redis", "connected")

    def __init__(self, redis_instance=None):
        self._redis = redis_instance or OptimizedMockRedis()
        self.connected = True

    async def async_get(self, key: str) -> Any | None:
        """Async get."""
        return await self._redis.get(key)

    async def async_set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Async set."""
        return await self._redis.set(key, value, ex=ttl)

    async def async_delete(self, key: str) -> bool:
        """Async delete."""
        result = await self._redis.delete(key)
        return result > 0

    async def async_delete_pattern(self, pattern: str) -> int:
        """Delete by pattern - simplified for tests."""
        # Simple pattern matching for tests
        deleted = 0
        if "*" in pattern:
            prefix = pattern.split("*")[0]
            keys_to_delete = [k for k in self._redis._data if k.startswith(prefix)]
            if keys_to_delete:
                deleted = await self._redis.delete(*keys_to_delete)
        return deleted

    async def async_health_check(self) -> dict[str, Any]:
        """Health check."""
        return {"status": "healthy", "redis_connected": True}

    def get_metrics(self) -> dict[str, Any]:
        """Get metrics."""
        return {"operations": {"total": 0}}
