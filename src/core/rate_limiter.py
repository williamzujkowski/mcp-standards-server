"""
Rate limiting module for MCP server.

Implements token bucket algorithm for rate limiting with Redis backend.
Enhanced with async support and request queuing for high-concurrency scenarios.
"""

import asyncio
import logging
import time
from typing import Any

from src.core.cache.redis_client import get_redis_client

logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter with Redis backend."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_prefix: str = "mcp:ratelimit",
    ):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
            redis_prefix: Redis key prefix for rate limit data
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.redis_prefix = redis_prefix
        self.redis_client = get_redis_client()

    def check_rate_limit(
        self, identifier: str
    ) -> tuple[bool, dict[str, int | str] | None]:
        """
        Check if a request is allowed under rate limit.

        Args:
            identifier: Unique identifier (e.g., user ID, API key, IP)

        Returns:
            Tuple of (is_allowed, limit_info)
            limit_info contains: remaining, limit, reset_time
        """
        if not self.redis_client:
            # Redis not available, allow request
            return True, None

        key = f"{self.redis_prefix}:{identifier}"
        current_time = int(time.time())
        window_start = current_time - self.window_seconds

        try:
            # Simple sliding window using hash storage
            request_times = self.redis_client.get(key) or []

            # Remove old entries outside the window
            request_times = [t for t in request_times if t > window_start]

            if len(request_times) >= self.max_requests:
                # Get oldest request time to calculate reset
                oldest_request = min(request_times) if request_times else current_time
                reset_time = oldest_request + self.window_seconds

                limit_info = {
                    "remaining": 0,
                    "limit": self.max_requests,
                    "reset_time": reset_time,
                    "retry_after": reset_time - current_time,
                }
                return False, limit_info

            # Add current request
            request_times.append(current_time)
            self.redis_client.set(key, request_times, ttl=self.window_seconds + 60)

            limit_info = {
                "remaining": self.max_requests - len(request_times),
                "limit": self.max_requests,
                "reset_time": current_time + self.window_seconds,
            }
            return True, limit_info

        except Exception as e:
            # Redis error, allow request but log
            print(f"Rate limit check failed: {e}")
            return True, None

    def reset_limit(self, identifier: str) -> None:
        """Reset rate limit for an identifier."""
        if self.redis_client:
            key = f"{self.redis_prefix}:{identifier}"
            self.redis_client.delete(key)


class MultiTierRateLimiter:
    """Rate limiter with multiple tiers (per-minute, per-hour, per-day)."""

    def __init__(self, redis_prefix: str = "mcp:ratelimit") -> None:
        """Initialize multi-tier rate limiter."""
        self.tiers = {
            "minute": RateLimiter(100, 60, f"{redis_prefix}:minute"),
            "hour": RateLimiter(5000, 3600, f"{redis_prefix}:hour"),
            "day": RateLimiter(50000, 86400, f"{redis_prefix}:day"),
        }

    def check_all_limits(self, identifier: str) -> tuple[bool, dict[str, Any] | None]:
        """
        Check all rate limit tiers.

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        for tier_name, limiter in self.tiers.items():
            is_allowed, limit_info = limiter.check_rate_limit(identifier)

            if not is_allowed:
                # Add tier information
                if limit_info is not None:
                    limit_info["tier"] = tier_name
                    limit_info["window"] = tier_name
                return False, limit_info

        # All tiers passed
        all_limits = {}
        for tier_name, limiter in self.tiers.items():
            _, info = limiter.check_rate_limit(identifier)
            if info:
                all_limits[tier_name] = info

        return True, all_limits

    def reset_all_limits(self, identifier: str) -> None:
        """Reset all rate limits for an identifier."""
        for limiter in self.tiers.values():
            limiter.reset_limit(identifier)


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on user behavior and system load."""

    def __init__(
        self,
        base_limit: int = 100,
        window_seconds: int = 60,
        redis_prefix: str = "mcp:adaptive",
    ):
        """Initialize adaptive rate limiter."""
        self.base_limit = base_limit
        self.window_seconds = window_seconds
        self.redis_prefix = redis_prefix
        self.redis_client = get_redis_client()

    def get_user_limit(self, identifier: str) -> int:
        """Get adapted limit for a specific user."""
        if not self.redis_client:
            return self.base_limit

        # Check user reputation score
        reputation_key = f"{self.redis_prefix}:reputation:{identifier}"
        reputation = self.redis_client.get(reputation_key)

        if reputation:
            reputation = float(reputation)
            # Good reputation gets higher limits
            if reputation > 0.8:
                return int(self.base_limit * 1.5)
            elif reputation < 0.3:
                return int(self.base_limit * 0.5)

        return self.base_limit

    def update_reputation(self, identifier: str, is_good_request: bool) -> None:
        """Update user reputation based on request behavior."""
        if not self.redis_client:
            return

        reputation_key = f"{self.redis_prefix}:reputation:{identifier}"
        current = self.redis_client.get(reputation_key)

        if current:
            reputation = float(current)
        else:
            reputation = 0.5  # Start neutral

        # Simple exponential moving average
        alpha = 0.1
        new_value = 1.0 if is_good_request else 0.0
        reputation = (1 - alpha) * reputation + alpha * new_value

        self.redis_client.set(reputation_key, str(reputation), ttl=86400 * 7)  # 7 days


class AsyncRateLimiter:
    """
    Async rate limiter with request queuing and circuit breaker capabilities.

    Designed for high-concurrency scenarios in async applications.
    """

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        enable_queuing: bool = True,
        max_queue_size: int = 1000,
        queue_timeout_seconds: float = 30.0,
        redis_prefix: str = "mcp:async_ratelimit",
    ):
        """
        Initialize async rate limiter.

        Args:
            max_requests: Maximum requests allowed in the window
            window_seconds: Time window in seconds
            enable_queuing: Whether to queue rate-limited requests
            max_queue_size: Maximum queue size per user
            queue_timeout_seconds: Timeout for queued requests
            redis_prefix: Redis key prefix for rate limit data
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.enable_queuing = enable_queuing
        self.max_queue_size = max_queue_size
        self.queue_timeout_seconds = queue_timeout_seconds
        self.redis_prefix = redis_prefix
        self.redis_client = get_redis_client()

        # Request queues per identifier
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._queue_processors: dict[str, asyncio.Task] = {}
        self._queue_locks: dict[str, asyncio.Lock] = {}

        # Metrics
        self._metrics = {
            "total_requests": 0,
            "allowed_requests": 0,
            "rate_limited_requests": 0,
            "queued_requests": 0,
            "rejected_requests": 0,
            "queue_timeouts": 0,
        }

        # Circuit breaker state
        self._circuit_breaker_failures: dict[str, int] = {}
        self._circuit_breaker_last_failure: dict[str, float] = {}
        self._circuit_breaker_state: dict[str, str] = (
            {}
        )  # "closed", "open", "half-open"

        # Global lock for thread safety
        self._global_lock = asyncio.Lock()

    async def check_rate_limit(
        self, identifier: str, priority: str = "normal"
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Check if a request is allowed under rate limit with async support.

        Args:
            identifier: Unique identifier (e.g., user ID, API key, IP)
            priority: Request priority ("high", "normal", "low")

        Returns:
            Tuple of (is_allowed, limit_info)
        """
        async with self._global_lock:
            self._metrics["total_requests"] += 1

            # Check circuit breaker
            if self._is_circuit_breaker_open(identifier):
                self._metrics["rejected_requests"] += 1
                logger.warning(
                    f"Request rejected - circuit breaker open for {identifier}"
                )
                return False, {"error": "circuit_breaker_open", "retry_after": 60}

            # Check standard rate limit
            is_allowed, limit_info = await self._check_standard_rate_limit(identifier)

            if is_allowed:
                self._metrics["allowed_requests"] += 1
                self._record_success(identifier)
                return True, limit_info

            # Rate limit exceeded
            self._metrics["rate_limited_requests"] += 1

            # Try to queue the request if queuing is enabled
            if self.enable_queuing:
                queued = await self._try_queue_request(identifier, priority)
                if queued:
                    self._metrics["queued_requests"] += 1
                    return True, {
                        "queued": True,
                        "estimated_wait": self._estimate_wait_time(identifier),
                    }
                else:
                    self._metrics["rejected_requests"] += 1
                    return False, limit_info

            return False, limit_info

    async def _check_standard_rate_limit(
        self, identifier: str
    ) -> tuple[bool, dict[str, Any] | None]:
        """Check standard rate limit using Redis."""
        if not self.redis_client:
            # Redis not available, allow request
            return True, None

        key = f"{self.redis_prefix}:{identifier}"
        current_time = int(time.time())
        window_start = current_time - self.window_seconds

        try:
            # Use async Redis operations
            request_times = await self.redis_client.async_get(key) or []

            # Remove old entries outside the window
            request_times = [t for t in request_times if t > window_start]

            if len(request_times) >= self.max_requests:
                # Get oldest request time to calculate reset
                oldest_request = min(request_times) if request_times else current_time
                reset_time = oldest_request + self.window_seconds

                limit_info = {
                    "remaining": 0,
                    "limit": self.max_requests,
                    "reset_time": reset_time,
                    "retry_after": reset_time - current_time,
                }
                return False, limit_info

            # Add current request
            request_times.append(current_time)
            await self.redis_client.async_set(
                key, request_times, ttl=self.window_seconds + 60
            )

            limit_info = {
                "remaining": self.max_requests - len(request_times),
                "limit": self.max_requests,
                "reset_time": current_time + self.window_seconds,
            }
            return True, limit_info

        except Exception as e:
            # Redis error, allow request but log
            logger.error(f"Async rate limit check failed for {identifier}: {e}")
            return True, None

    async def _try_queue_request(self, identifier: str, priority: str) -> bool:
        """Try to queue a rate-limited request."""
        # Initialize queue if needed
        if identifier not in self._request_queues:
            self._request_queues[identifier] = asyncio.Queue(
                maxsize=self.max_queue_size
            )
            self._queue_locks[identifier] = asyncio.Lock()

            # Start queue processor for this identifier
            processor = asyncio.create_task(self._process_queue(identifier))
            self._queue_processors[identifier] = processor

        queue = self._request_queues[identifier]

        # Check if queue is full
        if queue.qsize() >= self.max_queue_size:
            logger.warning(f"Request queue full for {identifier}, rejecting request")
            return False

        # Add to queue
        try:
            request_item = {
                "timestamp": time.time(),
                "priority": priority,
                "identifier": identifier,
            }
            queue.put_nowait(request_item)
            logger.debug(
                f"Request queued for {identifier}, queue size: {queue.qsize()}"
            )
            return True
        except asyncio.QueueFull:
            return False

    async def _process_queue(self, identifier: str) -> None:
        """Process queued requests for a specific identifier."""
        queue = self._request_queues[identifier]
        lock = self._queue_locks[identifier]

        logger.debug(f"Started queue processor for {identifier}")

        while True:
            try:
                # Wait for a request in the queue
                request_item = await asyncio.wait_for(
                    queue.get(), timeout=self.queue_timeout_seconds
                )

                # Check if request has expired
                if time.time() - request_item["timestamp"] > self.queue_timeout_seconds:
                    self._metrics["queue_timeouts"] += 1
                    logger.debug(f"Request expired in queue for {identifier}")
                    queue.task_done()
                    continue

                # Wait until rate limit allows the request
                while True:
                    async with lock:
                        is_allowed, _ = await self._check_standard_rate_limit(
                            identifier
                        )
                        if is_allowed:
                            break

                    # Wait a bit before checking again
                    await asyncio.sleep(0.1)

                # Mark task as done
                queue.task_done()
                logger.debug(f"Processed queued request for {identifier}")

            except asyncio.TimeoutError:
                # No requests in queue for a while, keep waiting
                continue
            except Exception as e:
                logger.error(f"Error processing queue for {identifier}: {e}")
                await asyncio.sleep(1)

    def _estimate_wait_time(self, identifier: str) -> float:
        """Estimate wait time for queued requests."""
        queue = self._request_queues.get(identifier)
        if not queue:
            return 0.0

        # Simple estimation: queue_size * average_processing_time
        queue_size = queue.qsize()
        avg_processing_time = self.window_seconds / self.max_requests
        return queue_size * avg_processing_time

    def _is_circuit_breaker_open(self, identifier: str) -> bool:
        """Check if circuit breaker is open for an identifier."""
        state = self._circuit_breaker_state.get(identifier, "closed")

        if state == "closed":
            return False
        elif state == "open":
            # Check if timeout has elapsed
            last_failure = self._circuit_breaker_last_failure.get(identifier, 0)
            if time.time() - last_failure > 60:  # 60 second timeout
                self._circuit_breaker_state[identifier] = "half-open"
                return False
            return True
        elif state == "half-open":
            return False

        return False

    def _record_success(self, identifier: str) -> None:
        """Record a successful request for circuit breaker."""
        if identifier in self._circuit_breaker_failures:
            self._circuit_breaker_failures[identifier] = 0

        state = self._circuit_breaker_state.get(identifier, "closed")
        if state == "half-open":
            self._circuit_breaker_state[identifier] = "closed"
            logger.info(f"Circuit breaker closed for {identifier}")

    def record_failure(self, identifier: str) -> None:
        """Record a failed request for circuit breaker."""
        failures = self._circuit_breaker_failures.get(identifier, 0) + 1
        self._circuit_breaker_failures[identifier] = failures
        self._circuit_breaker_last_failure[identifier] = time.time()

        if failures >= 10:  # Open circuit after 10 failures
            self._circuit_breaker_state[identifier] = "open"
            logger.warning(
                f"Circuit breaker opened for {identifier} after {failures} failures"
            )

    def get_metrics(self) -> dict[str, Any]:
        """Get rate limiting metrics."""
        return {
            **self._metrics,
            "active_queues": len(self._request_queues),
            "total_queue_size": sum(q.qsize() for q in self._request_queues.values()),
        }

    async def cleanup(self) -> None:
        """Clean up resources and stop queue processors."""
        # Cancel all queue processors
        for processor in self._queue_processors.values():
            processor.cancel()

        # Wait for cancellation
        await asyncio.gather(*self._queue_processors.values(), return_exceptions=True)

        self._request_queues.clear()
        self._queue_processors.clear()
        self._queue_locks.clear()

        logger.info("Async rate limiter cleaned up")


# Default rate limiter instances
_default_rate_limiter: MultiTierRateLimiter | None = None
_default_async_rate_limiter: AsyncRateLimiter | None = None


def get_rate_limiter() -> MultiTierRateLimiter:
    """Get the default rate limiter instance."""
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = MultiTierRateLimiter()
    return _default_rate_limiter


async def get_async_rate_limiter(
    max_requests: int = 100,
    window_seconds: int = 60,
    enable_queuing: bool = True,
) -> AsyncRateLimiter:
    """Get the default async rate limiter instance."""
    global _default_async_rate_limiter
    if _default_async_rate_limiter is None:
        _default_async_rate_limiter = AsyncRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            enable_queuing=enable_queuing,
        )
    return _default_async_rate_limiter
