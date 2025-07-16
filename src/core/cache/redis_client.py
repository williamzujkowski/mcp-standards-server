"""Redis client implementation with connection pooling and retry logic."""

import asyncio
import base64
import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, cast

import msgpack
import redis
from cachetools import TTLCache
from redis import asyncio as aioredis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Redis cache configuration."""

    # Connection settings
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    ssl: bool = False

    # Pool settings
    max_connections: int = 50
    max_connections_per_process: int = 10
    socket_keepalive: bool = True
    socket_keepalive_options: dict[str, Any] = field(default_factory=dict)

    # Enhanced connection pool settings
    connection_pool_size: int = 20
    connection_pool_max_overflow: int = 10
    connection_pool_timeout: float = 30.0
    connection_pool_recycle: int = 3600  # seconds
    connection_pool_pre_ping: bool = True

    # Connection health monitoring
    health_check_interval: float = 30.0  # seconds
    connection_timeout: float = 5.0
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 0.1
    retry_backoff: float = 2.0

    # Cache settings
    default_ttl: int = 300  # 5 minutes
    key_prefix: str = "mcp"
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes

    # L1 cache settings
    l1_max_size: int = 1000
    l1_ttl: int = 30  # seconds

    # Circuit breaker settings
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 30

    # Monitoring
    enable_metrics: bool = True
    slow_query_threshold: float = 0.1  # seconds

    # Performance optimization
    enable_pipeline: bool = True
    pipeline_batch_size: int = 100
    enable_async_operations: bool = True


class CircuitBreaker:
    """Simple circuit breaker implementation."""

    def __init__(self, threshold: int = 5, timeout: int = 30) -> None:
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: float = 0
        self.state = "closed"  # closed, open, half-open

    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.threshold:
            self.state = "open"

    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == "closed":
            return True

        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                return True
            return False

        return True  # half-open state


class RedisCache:
    """Redis cache client with L1/L2 caching and resilience features."""

    def __init__(self, config: CacheConfig | None = None) -> None:
        self.config = config or CacheConfig()
        self._sync_pool: redis.ConnectionPool | None = None
        self._async_pool: aioredis.ConnectionPool | None = None
        self._sync_client: redis.Redis | None = None
        self._async_client: aioredis.Redis | None = None
        self._l1_cache: TTLCache[str, Any] = TTLCache(
            maxsize=self.config.l1_max_size, ttl=self.config.l1_ttl
        )
        self._circuit_breaker = CircuitBreaker(
            threshold=self.config.circuit_breaker_threshold,
            timeout=self.config.circuit_breaker_timeout,
        )
        self._metrics = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "errors": 0,
            "slow_queries": 0,
            "connection_errors": 0,
            "pool_exhausted": 0,
            "pipeline_operations": 0,
        }

        # Health monitoring
        self._health_check_task: threading.Thread | None = None
        self._connection_health: dict[str, Any] = {
            "is_healthy": True,
            "last_check": None,
            "consecutive_failures": 0,
            "total_checks": 0,
            "failed_checks": 0,
        }

        # Pipeline support
        self._pipeline_queue: list[tuple[str, Any]] = []
        self._pipeline_lock = threading.Lock()

        # Connection pool monitoring
        self._pool_stats = {
            "active_connections": 0,
            "idle_connections": 0,
            "total_connections": 0,
            "pool_exhaustion_count": 0,
        }

    def _get_json_encoder(self) -> type[json.JSONEncoder]:
        """Get JSON encoder that handles common Python types."""

        class ExtendedJSONEncoder(json.JSONEncoder):
            def default(self, obj: Any) -> Any:
                if isinstance(obj, set | frozenset):
                    return {"__type__": "set", "values": list(obj)}
                elif isinstance(obj, bytes):
                    return {
                        "__type__": "bytes",
                        "data": base64.b64encode(obj).decode("ascii"),
                    }
                # Don't try to serialize arbitrary objects for security reasons
                return super().default(obj)

        return ExtendedJSONEncoder

    def _json_object_hook(self, obj: Any) -> Any:
        """JSON decoder hook to reconstruct special types."""
        if isinstance(obj, dict) and "__type__" in obj:
            type_name = obj["__type__"]
            if type_name == "set":
                return set(obj["values"])
            elif type_name == "bytes":
                return base64.b64decode(obj["data"].encode("ascii"))
        return obj

    @property
    def sync_pool(self) -> redis.ConnectionPool:
        """Get or create sync connection pool."""
        if self._sync_pool is None:
            self._sync_pool = redis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.connection_pool_size,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,  # We'll handle encoding/decoding
                retry_on_timeout=True,
                health_check_interval=self.config.health_check_interval,
            )

            # Start health monitoring
            if self.config.enable_metrics:
                self._start_health_monitoring()

        return self._sync_pool

    @property
    def async_pool(self) -> aioredis.ConnectionPool:
        """Get or create async connection pool."""
        if self._async_pool is None:
            self._async_pool = aioredis.ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.connection_pool_size,
                socket_keepalive=self.config.socket_keepalive,
                socket_keepalive_options=self.config.socket_keepalive_options,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False,
                retry_on_timeout=True,
                health_check_interval=self.config.health_check_interval,
            )

            # Start health monitoring
            if self.config.enable_metrics:
                self._start_health_monitoring()

        return self._async_pool

    @property
    def sync_client(self) -> redis.Redis:
        """Get or create sync Redis client instance."""
        if self._sync_client is None:
            self._sync_client = redis.Redis(connection_pool=self.sync_pool)
        return self._sync_client

    @property
    def async_client(self) -> aioredis.Redis:
        """Get or create async Redis client instance."""
        if self._async_client is None:
            self._async_client = aioredis.Redis(connection_pool=self.async_pool)
        return self._async_client

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage with secure methods."""
        try:
            # Try msgpack first (faster and safer)
            data = msgpack.packb(value, use_bin_type=True)
            serializer = "m"  # msgpack
        except (TypeError, msgpack.exceptions.PackException):
            # Fall back to JSON for all types
            try:
                # Convert to JSON-serializable format
                json_data = json.dumps(value, cls=self._get_json_encoder())
                data = json_data.encode("utf-8")
                serializer = "j"  # json
            except (TypeError, ValueError) as e:
                # For types that can't be serialized, raise an error
                raise TypeError(
                    f"Cannot serialize object of type {type(value).__name__}. "
                    f"Only msgpack and JSON serializable types are supported. "
                    f"Consider implementing a custom serialization method. Error: {e}"
                )

        # Compress if needed
        if (
            self.config.enable_compression
            and len(data) > self.config.compression_threshold
        ):
            import zlib

            data = zlib.compress(data)
            result = bytearray(b"Z")
            result.extend(serializer.encode("utf-8"))
            result.extend(data)
            return bytes(result)
        else:
            result = bytearray(b"U")
            result.extend(serializer.encode("utf-8"))
            result.extend(data)
            return bytes(result)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage with security checks."""
        if not data:
            return None

        compression = data[0:1]
        serializer = data[1:2].decode()
        payload = data[2:]

        # Decompress if needed
        if compression == b"Z":
            import zlib

            payload = zlib.decompress(payload)

        # Deserialize based on serializer type
        if serializer == "m":
            return msgpack.unpackb(payload, raw=False)
        elif serializer == "j":
            # JSON deserialization with object hook
            return json.loads(
                payload.decode("utf-8"), object_hook=self._json_object_hook
            )
        elif serializer == "p":
            # Pickle is no longer supported for security reasons
            logger.error(
                "Attempted to deserialize pickled data. This is no longer supported for security reasons. "
                "Please re-cache the data using msgpack or JSON serialization."
            )
            raise ValueError(
                "Pickle deserialization is disabled for security reasons. "
                "Please clear the cache and re-populate it with secure serialization."
            )
        else:
            raise ValueError(f"Unknown serializer type: {serializer}")

    def _build_key(self, key: str) -> str:
        """Build full cache key with prefix."""
        return f"{self.config.key_prefix}:{key}"

    def _with_retry(self, func: Callable) -> Callable:
        """Decorator for retry logic."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            delay = self.config.retry_delay

            for attempt in range(self.config.max_retries):
                try:
                    if not self._circuit_breaker.can_attempt():
                        raise RedisConnectionError("Circuit breaker is open")

                    result = func(*args, **kwargs)
                    self._circuit_breaker.record_success()
                    return result

                except RedisError as e:
                    last_error = e
                    self._circuit_breaker.record_failure()
                    self._metrics["errors"] += 1

                    if attempt < self.config.max_retries - 1:
                        time.sleep(delay)
                        delay *= self.config.retry_backoff

            logger.error(
                f"Redis operation failed after {self.config.max_retries} retries: {last_error}"
            )
            if last_error:
                raise last_error
            else:
                raise RedisConnectionError("Operation failed with unknown error")

        return wrapper

    def _with_metrics(self, operation: str) -> Callable:
        """Decorator for metrics collection."""

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = func(self, *args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if duration > self.config.slow_query_threshold:
                        self._metrics["slow_queries"] += 1
                        logger.warning(f"Slow {operation} operation: {duration:.3f}s")

            @wraps(func)
            async def async_wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
                start_time = time.time()
                try:
                    result = await func(self, *args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    if duration > self.config.slow_query_threshold:
                        self._metrics["slow_queries"] += 1
                        logger.warning(f"Slow {operation} operation: {duration:.3f}s")

            return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper

        return decorator

    def _start_health_monitoring(self) -> None:
        """Start health monitoring task."""
        if self._health_check_task is None:
            self._health_check_task = threading.Thread(
                target=self._health_monitor_worker, daemon=True
            )
            self._health_check_task.start()

    def _health_monitor_worker(self) -> None:
        """Health monitoring worker thread."""
        while True:
            try:
                self._perform_health_check()
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.config.health_check_interval)

    def _perform_health_check(self) -> None:
        """Perform health check on connections."""
        time.time()

        try:
            # Check sync pool
            self.sync_client.ping()

            # Update health status
            self._connection_health["is_healthy"] = True
            self._connection_health["consecutive_failures"] = 0
            self._connection_health["last_check"] = datetime.now().isoformat()
            self._connection_health["total_checks"] = (
                self._connection_health.get("total_checks", 0) + 1
            )

            # Update pool stats
            self._update_pool_stats()

        except Exception as e:
            self._connection_health["is_healthy"] = False
            self._connection_health["consecutive_failures"] = (
                self._connection_health.get("consecutive_failures", 0) + 1
            )
            self._connection_health["failed_checks"] = (
                self._connection_health.get("failed_checks", 0) + 1
            )
            self._connection_health["total_checks"] = (
                self._connection_health.get("total_checks", 0) + 1
            )
            self._metrics["connection_errors"] += 1

            logger.warning(f"Health check failed: {e}")

    def _update_pool_stats(self) -> None:
        """Update connection pool statistics."""
        try:
            if self._sync_pool:
                # Get pool info (this is a simplified version)
                self._pool_stats.update(
                    {
                        "active_connections": getattr(
                            self._sync_pool, "_available_connections", 0
                        ),
                        "total_connections": self._sync_pool.max_connections,
                    }
                )
        except Exception as e:
            logger.debug(f"Could not update pool stats: {e}")

    def execute_pipeline(self, operations: list[dict[str, Any]]) -> list[Any]:
        """Execute multiple operations in a pipeline."""
        if not self.config.enable_pipeline:
            # Execute operations individually
            results = []
            for op in operations:
                method = getattr(self, op["method"])
                result = method(*op.get("args", []), **op.get("kwargs", {}))
                results.append(result)
            return results

        try:
            pipe = self.sync_client.pipeline()

            for op in operations:
                method_name = op["method"]
                args = op.get("args", [])
                kwargs = op.get("kwargs", {})

                # Map our method names to Redis methods
                if method_name == "set":
                    pipe.set(args[0], self._serialize(args[1]), **kwargs)
                elif method_name == "get":
                    pipe.get(args[0])
                elif method_name == "delete":
                    pipe.delete(args[0])
                else:
                    logger.warning(f"Unknown pipeline method: {method_name}")

            results = pipe.execute()

            # Deserialize results where needed
            processed_results = []
            for _i, (result, op) in enumerate(zip(results, operations, strict=False)):
                if op["method"] == "get" and result:
                    processed_results.append(self._deserialize(result))
                else:
                    processed_results.append(result)

            self._metrics["pipeline_operations"] += 1
            return processed_results

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise

    async def execute_pipeline_async(
        self, operations: list[dict[str, Any]]
    ) -> list[Any]:
        """Execute multiple operations in an async pipeline."""
        if not self.config.enable_pipeline:
            # Execute operations individually
            results = []
            for op in operations:
                method = getattr(self, f"async_{op['method']}")
                result = await method(*op.get("args", []), **op.get("kwargs", {}))
                results.append(result)
            return results

        try:
            pipe = self.async_client.pipeline()

            for op in operations:
                method_name = op["method"]
                args = op.get("args", [])
                kwargs = op.get("kwargs", {})

                # Map our method names to Redis methods
                if method_name == "set":
                    await pipe.set(args[0], self._serialize(args[1]), **kwargs)
                elif method_name == "get":
                    await pipe.get(args[0])
                elif method_name == "delete":
                    await pipe.delete(args[0])
                else:
                    logger.warning(f"Unknown pipeline method: {method_name}")

            results = await pipe.execute()

            # Deserialize results where needed
            processed_results = []
            for _i, (result, op) in enumerate(zip(results, operations, strict=False)):
                if op["method"] == "get" and result:
                    processed_results.append(self._deserialize(result))
                else:
                    processed_results.append(result)

            self._metrics["pipeline_operations"] += 1
            return processed_results

        except Exception as e:
            logger.error(f"Async pipeline execution failed: {e}")
            raise

    # Sync methods

    def get(self, key: str) -> Any | None:
        """Get value from cache (sync)."""
        start_time = time.time()
        full_key = self._build_key(key)

        # Check L1 cache first
        if full_key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            return self._l1_cache[full_key]

        self._metrics["l1_misses"] += 1

        # Check L2 cache
        try:

            @self._with_retry
            def _get() -> Any:
                return self.sync_client.get(full_key)

            data = _get()
            if data:
                self._metrics["l2_hits"] += 1
                value = self._deserialize(data)
                self._l1_cache[full_key] = value
                return value
            else:
                self._metrics["l2_misses"] += 1
                return None

        except RedisError:
            logger.warning(f"Redis get failed for key: {key}, using L1 only")
            return None
        finally:
            duration = time.time() - start_time
            if duration > self.config.slow_query_threshold:
                self._metrics["slow_queries"] += 1
                logger.warning(f"Slow get operation: {duration:.3f}s")

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache (sync)."""
        start_time = time.time()
        full_key = self._build_key(key)
        ttl = ttl or self.config.default_ttl

        # Set in L1 cache
        self._l1_cache[full_key] = value

        # Set in L2 cache
        try:

            @self._with_retry
            def _set() -> Any:
                return self.sync_client.setex(full_key, ttl, self._serialize(value))

            return bool(_set())

        except RedisError:
            logger.warning(f"Redis set failed for key: {key}, value cached in L1 only")
            return False
        finally:
            duration = time.time() - start_time
            if duration > self.config.slow_query_threshold:
                self._metrics["slow_queries"] += 1
                logger.warning(f"Slow set operation: {duration:.3f}s")

    def delete(self, key: str) -> bool:
        """Delete value from cache (sync)."""
        full_key = self._build_key(key)

        # Delete from L1 cache
        self._l1_cache.pop(full_key, None)

        # Delete from L2 cache
        try:

            @self._with_retry
            def _delete() -> Any:
                return self.sync_client.delete(full_key)

            return bool(_delete())

        except RedisError:
            logger.warning(f"Redis delete failed for key: {key}")
            return False

    def mget(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple values from cache (sync)."""
        result = {}
        l2_keys = []

        # Check L1 cache first
        for key in keys:
            full_key = self._build_key(key)
            if full_key in self._l1_cache:
                result[key] = self._l1_cache[full_key]
                self._metrics["l1_hits"] += 1
            else:
                l2_keys.append(key)
                self._metrics["l1_misses"] += 1

        # Check L2 cache for remaining keys
        if l2_keys:
            try:

                @self._with_retry
                def _mget() -> Any:
                    full_keys = [self._build_key(k) for k in l2_keys]
                    return self.sync_client.mget(full_keys)

                values = _mget()
                for key, value in zip(l2_keys, values, strict=False):
                    if value:
                        self._metrics["l2_hits"] += 1
                        deserialized = self._deserialize(value)
                        result[key] = deserialized
                        self._l1_cache[self._build_key(key)] = deserialized
                    else:
                        self._metrics["l2_misses"] += 1

            except RedisError:
                logger.warning("Redis mget failed, partial results from L1 only")

        return result

    def mset(self, mapping: dict[str, Any], ttl: int | None = None) -> bool:
        """Set multiple values in cache (sync)."""
        ttl = ttl or self.config.default_ttl

        # Set in L1 cache
        for key, value in mapping.items():
            self._l1_cache[self._build_key(key)] = value

        # Set in L2 cache
        try:

            @self._with_retry
            def _mset() -> Any:
                pipe = self.sync_client.pipeline()
                for key, value in mapping.items():
                    full_key = self._build_key(key)
                    pipe.setex(full_key, ttl, self._serialize(value))
                return pipe.execute()

            return all(_mset())

        except RedisError:
            logger.warning("Redis mset failed, values cached in L1 only")
            return False

    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern (sync)."""
        full_pattern = self._build_key(pattern)

        # Clear matching keys from L1 cache
        l1_cleared = 0
        for key in list(self._l1_cache.keys()):
            if self._match_pattern(key, full_pattern):
                del self._l1_cache[key]
                l1_cleared += 1

        # Delete from L2 cache
        try:

            @self._with_retry
            def _delete_pattern() -> int:
                keys = cast(list[str], self.sync_client.keys(full_pattern))
                if keys:
                    return cast(int, self.sync_client.delete(*keys))
                return 0

            l2_cleared = _delete_pattern()
            return cast(int, max(l1_cleared, l2_cleared))

        except RedisError:
            logger.warning(f"Redis delete_pattern failed for pattern: {pattern}")
            return l1_cleared

    # Async methods

    async def async_get(self, key: str) -> Any | None:
        """Get value from cache (async)."""
        start_time = time.time()
        full_key = self._build_key(key)

        # Check L1 cache first
        if full_key in self._l1_cache:
            self._metrics["l1_hits"] += 1
            return self._l1_cache[full_key]

        self._metrics["l1_misses"] += 1

        # Check L2 cache
        try:
            if not self._circuit_breaker.can_attempt():
                raise RedisConnectionError("Circuit breaker is open")

            data = await self.async_client.get(full_key)

            self._circuit_breaker.record_success()

            if data:
                self._metrics["l2_hits"] += 1
                value = self._deserialize(data)
                self._l1_cache[full_key] = value
                return value
            else:
                self._metrics["l2_misses"] += 1
                return None

        except RedisError:
            self._circuit_breaker.record_failure()
            self._metrics["errors"] += 1
            logger.warning(f"Redis async_get failed for key: {key}, using L1 only")
            return None
        finally:
            duration = time.time() - start_time
            if duration > self.config.slow_query_threshold:
                self._metrics["slow_queries"] += 1
                logger.warning(f"Slow async_get operation: {duration:.3f}s")

    async def async_set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache (async)."""
        start_time = time.time()
        full_key = self._build_key(key)
        ttl = ttl or self.config.default_ttl

        # Set in L1 cache
        self._l1_cache[full_key] = value

        # Set in L2 cache
        try:
            if not self._circuit_breaker.can_attempt():
                raise RedisConnectionError("Circuit breaker is open")

            result = await self.async_client.setex(
                full_key, ttl, self._serialize(value)
            )

            self._circuit_breaker.record_success()
            return bool(result)

        except RedisError:
            self._circuit_breaker.record_failure()
            self._metrics["errors"] += 1
            logger.warning(
                f"Redis async_set failed for key: {key}, value cached in L1 only"
            )
            return False
        finally:
            duration = time.time() - start_time
            if duration > self.config.slow_query_threshold:
                self._metrics["slow_queries"] += 1
                logger.warning(f"Slow async_set operation: {duration:.3f}s")

    async def async_delete(self, key: str) -> bool:
        """Delete value from cache (async)."""
        full_key = self._build_key(key)

        # Delete from L1 cache
        self._l1_cache.pop(full_key, None)

        # Delete from L2 cache
        try:
            if not self._circuit_breaker.can_attempt():
                raise RedisConnectionError("Circuit breaker is open")

            result = await self.async_client.delete(full_key)

            self._circuit_breaker.record_success()
            return bool(result)

        except RedisError:
            self._circuit_breaker.record_failure()
            self._metrics["errors"] += 1
            logger.warning(f"Redis async_delete failed for key: {key}")
            return False

    async def async_delete_pattern(self, pattern: str) -> int:
        """Delete keys matching pattern (async)."""
        full_pattern = self._build_key(pattern)

        # Clear matching keys from L1 cache
        l1_cleared = 0
        for key in list(self._l1_cache.keys()):
            if self._match_pattern(key, full_pattern):
                del self._l1_cache[key]
                l1_cleared += 1

        # Delete from L2 cache
        try:
            if not self._circuit_breaker.can_attempt():
                raise RedisConnectionError("Circuit breaker is open")

            keys = []
            async for key in self.async_client.scan_iter(match=full_pattern):
                keys.append(key)

            if keys:
                result = await self.async_client.delete(*keys)
                self._circuit_breaker.record_success()
                return max(l1_cleared, cast(int, result))
            else:
                self._circuit_breaker.record_success()
                return l1_cleared

        except RedisError:
            self._circuit_breaker.record_failure()
            self._metrics["errors"] += 1
            logger.warning(
                f"Redis async_delete_pattern failed for pattern: {pattern}, using L1 only"
            )
            return l1_cleared

    # Health and monitoring

    def health_check(self) -> dict[str, Any]:
        """Perform health check."""
        health = {
            "status": "healthy",
            "l1_cache_size": len(self._l1_cache),
            "circuit_breaker_state": self._circuit_breaker.state,
            "metrics": self._metrics.copy(),
            "redis_connected": False,
            "latency_ms": None,
            "connection_health": self._connection_health.copy(),
            "pool_stats": self._pool_stats.copy(),
        }

        # Check Redis connection
        try:
            start_time = time.time()
            self.sync_client.ping()
            health["redis_connected"] = True
            health["latency_ms"] = (time.time() - start_time) * 1000

        except RedisError:
            health["status"] = (
                "degraded" if self._circuit_breaker.state != "open" else "unhealthy"
            )

        # Overall health assessment
        if not self._connection_health["is_healthy"]:
            health["status"] = "unhealthy"
        elif self._connection_health.get("consecutive_failures", 0) > 0:
            health["status"] = "degraded"

        return health

    async def async_health_check(self) -> dict[str, Any]:
        """Perform async health check."""
        health = {
            "status": "healthy",
            "l1_cache_size": len(self._l1_cache),
            "circuit_breaker_state": self._circuit_breaker.state,
            "metrics": self._metrics.copy(),
            "redis_connected": False,
            "latency_ms": None,
            "connection_health": self._connection_health.copy(),
            "pool_stats": self._pool_stats.copy(),
        }

        # Check Redis connection
        try:
            start_time = time.time()
            await self.async_client.ping()
            health["redis_connected"] = True
            health["latency_ms"] = (time.time() - start_time) * 1000

        except RedisError:
            health["status"] = (
                "degraded" if self._circuit_breaker.state != "open" else "unhealthy"
            )

        # Overall health assessment
        if not self._connection_health["is_healthy"]:
            health["status"] = "unhealthy"
        elif self._connection_health.get("consecutive_failures", 0) > 0:
            health["status"] = "degraded"

        return health

    def get_metrics(self) -> dict[str, Any]:
        """Get cache metrics."""
        total_l1 = self._metrics["l1_hits"] + self._metrics["l1_misses"]
        total_l2 = self._metrics["l2_hits"] + self._metrics["l2_misses"]
        total_health_checks = self._connection_health.get("total_checks", 0)
        failed_checks = self._connection_health.get("failed_checks", 0)

        return {
            **self._metrics,
            "l1_hit_rate": self._metrics["l1_hits"] / total_l1 if total_l1 > 0 else 0,
            "l2_hit_rate": self._metrics["l2_hits"] / total_l2 if total_l2 > 0 else 0,
            "l1_cache_size": len(self._l1_cache),
            "circuit_breaker_state": self._circuit_breaker.state,
            "connection_health": self._connection_health.copy(),
            "pool_stats": self._pool_stats.copy(),
            "health_check_success_rate": (
                (total_health_checks - failed_checks) / total_health_checks
                if total_health_checks > 0
                else 1.0
            ),
        }

    def clear_l1_cache(self) -> None:
        """Clear L1 cache."""
        self._l1_cache.clear()

    def close(self) -> None:
        """Close connection pools and clients."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._sync_pool:
            self._sync_pool.disconnect()

        # Handle async cleanup
        async def _async_cleanup() -> None:
            if self._async_client:
                await self._async_client.aclose()
            if self._async_pool:
                await self._async_pool.disconnect()

        if self._async_client or self._async_pool:
            try:
                # Try to get running loop
                loop = asyncio.get_running_loop()
                loop.create_task(_async_cleanup())
            except RuntimeError:
                # No running loop - try to close synchronously
                try:
                    asyncio.run(_async_cleanup())
                except Exception as e:
                    # If all else fails, just clear references without calling async methods
                    logger.warning(
                        "Failed to close Redis connections gracefully during cleanup: %s",
                        str(e),
                    )
            self._async_client = None
            self._async_pool = None

    async def async_close(self) -> None:
        """Close connection pools and clients (async)."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
        if self._sync_pool:
            self._sync_pool.disconnect()
        if self._async_pool:
            await self._async_pool.disconnect()

    @staticmethod
    def _match_pattern(key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple glob matching)."""
        import fnmatch

        return fnmatch.fnmatch(key, pattern)

    @staticmethod
    def generate_cache_key(*args: Any, **kwargs: Any) -> str:
        """Generate a cache key from arguments using SHA-256."""
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        key_string = ":".join(key_parts)
        # Use SHA-256 with application-specific salt
        salted_key = f"mcp_cache_v1:{key_string}"
        return hashlib.sha256(salted_key.encode()).hexdigest()


# Global cache instance (can be initialized in app startup)
_global_cache: RedisCache | None = None


def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = RedisCache()
    return _global_cache


def init_cache(config: CacheConfig) -> RedisCache:
    """Initialize global cache with config."""
    global _global_cache
    _global_cache = RedisCache(config)
    return _global_cache


def get_redis_client() -> RedisCache:
    """Get Redis client (alias for get_cache for backward compatibility)."""
    return get_cache()
