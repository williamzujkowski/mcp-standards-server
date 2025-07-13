"""
Connection retry logic with exponential backoff for MCP server.

Provides resilient connection handling with configurable retry strategies.
"""

import asyncio
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeVar, cast

from src.core.errors import ErrorCode, MCPError
from src.core.metrics import get_mcp_metrics

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryStrategy(str, Enum):
    """Retry strategy types."""

    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    CONSTANT = "constant"


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    exponential_base: float = 2.0
    jitter: bool = True
    jitter_factor: float = 0.1
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    retry_on: tuple | None = None  # Exception types to retry on

    def __post_init__(self) -> None:
        """Set default retry exceptions if not provided."""
        if self.retry_on is None:
            self.retry_on = (
                ConnectionError,
                TimeoutError,
                asyncio.TimeoutError,
                OSError,
            )


class RetryManager:
    """Manages retry logic for operations."""

    def __init__(self, config: RetryConfig | None = None) -> None:
        """Initialize retry manager with configuration."""
        self.config = config or RetryConfig()
        self.metrics = get_mcp_metrics()

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the next retry attempt.

        Args:
            attempt: Current attempt number (0-based)

        Returns:
            Delay in seconds
        """
        if self.config.strategy == RetryStrategy.CONSTANT:
            delay = self.config.initial_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.initial_delay * (attempt + 1)
        else:  # EXPONENTIAL
            delay = self.config.initial_delay * (self.config.exponential_base**attempt)

        # Apply max delay cap
        delay = min(delay, self.config.max_delay)

        # Add jitter if enabled
        if self.config.jitter:
            jitter_range = delay * self.config.jitter_factor
            jitter = random.uniform(-jitter_range, jitter_range)  # nosec B311
            delay += jitter

        return max(0, delay)  # Ensure non-negative

    async def retry_async(
        self,
        func: Callable[..., T],
        *args: Any,
        operation_name: str = "operation",
        **kwargs: Any,
    ) -> T:
        """
        Retry an async operation with exponential backoff.

        Args:
            func: Async function to retry
            args: Positional arguments for func
            operation_name: Name for logging/metrics
            kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Record attempt
                self.metrics.collector.increment(
                    "mcp_retry_attempts_total",
                    labels={"operation": operation_name, "attempt": str(attempt)},
                )

                # Call the function
                result = await func(*args, **kwargs)  # type: ignore[misc]

                # Success - record and return
                if attempt > 0:
                    logger.info(
                        f"Operation '{operation_name}' succeeded after {attempt} retries"
                    )
                    self.metrics.collector.increment(
                        "mcp_retry_success_total",
                        labels={"operation": operation_name, "attempts": str(attempt)},
                    )

                return cast(T, result)
            except self.config.retry_on as e:  # type: ignore[misc]
                last_exception = e

                if attempt >= self.config.max_retries:
                    # No more retries
                    logger.error(
                        f"Operation '{operation_name}' failed after {attempt + 1} attempts: {e}"
                    )
                    self.metrics.collector.increment(
                        "mcp_retry_failures_total",
                        labels={"operation": operation_name, "error": type(e).__name__},
                    )
                    raise

                # Calculate delay
                delay = self.calculate_delay(attempt)

                logger.warning(
                    f"Operation '{operation_name}' failed (attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}. Retrying in {delay:.2f}s..."
                )

                # Wait before retry
                await asyncio.sleep(delay)

            except Exception as e:
                # Non-retryable exception
                logger.error(
                    f"Operation '{operation_name}' failed with non-retryable error: {e}"
                )
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation '{operation_name}' failed without exception")

    def retry_sync(
        self,
        func: Callable[..., T],
        *args: Any,
        operation_name: str = "operation",
        **kwargs: Any,
    ) -> T:
        """
        Retry a sync operation with exponential backoff.

        Args:
            func: Sync function to retry
            args: Positional arguments for func
            operation_name: Name for logging/metrics
            kwargs: Keyword arguments for func

        Returns:
            Result from successful function call

        Raises:
            Last exception if all retries fail
        """
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                # Record attempt
                self.metrics.collector.increment(
                    "mcp_retry_attempts_total",
                    labels={"operation": operation_name, "attempt": str(attempt)},
                )

                # Call the function
                result = func(*args, **kwargs)

                # Success - record and return
                if attempt > 0:
                    logger.info(
                        f"Operation '{operation_name}' succeeded after {attempt} retries"
                    )
                    self.metrics.collector.increment(
                        "mcp_retry_success_total",
                        labels={"operation": operation_name, "attempts": str(attempt)},
                    )

                return result
            except self.config.retry_on as e:  # type: ignore[misc]
                last_exception = e

                if attempt >= self.config.max_retries:
                    # No more retries
                    logger.error(
                        f"Operation '{operation_name}' failed after {attempt + 1} attempts: {e}"
                    )
                    self.metrics.collector.increment(
                        "mcp_retry_failures_total",
                        labels={"operation": operation_name, "error": type(e).__name__},
                    )
                    raise

                # Calculate delay
                delay = self.calculate_delay(attempt)

                logger.warning(
                    f"Operation '{operation_name}' failed (attempt {attempt + 1}/"
                    f"{self.config.max_retries + 1}): {e}. Retrying in {delay:.2f}s..."
                )

                # Wait before retry
                time.sleep(delay)

            except Exception as e:
                # Non-retryable exception
                logger.error(
                    f"Operation '{operation_name}' failed with non-retryable error: {e}"
                )
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception
        else:
            raise RuntimeError(f"Operation '{operation_name}' failed without exception")


def with_retry(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
    operation_name: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to add retry logic to functions.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        strategy: Retry strategy to use
        operation_name: Name for logging (defaults to function name)
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        config = RetryConfig(
            max_retries=max_retries, initial_delay=initial_delay, strategy=strategy
        )
        retry_manager = RetryManager(config)

        if asyncio.iscoroutinefunction(func):

            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = operation_name or func.__name__
                return await retry_manager.retry_async(
                    func, *args, operation_name=name, **kwargs
                )

            async_wrapper.__name__ = func.__name__
            async_wrapper.__doc__ = func.__doc__
            return async_wrapper

        else:

            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                name = operation_name or func.__name__
                return retry_manager.retry_sync(
                    func, *args, operation_name=name, **kwargs
                )

            sync_wrapper.__name__ = func.__name__
            sync_wrapper.__doc__ = func.__doc__
            return sync_wrapper

    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for preventing cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            expected_exception: Exception type to count as failure
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._state = "closed"  # closed, open, half-open

    @property
    def state(self) -> str:
        """Get current circuit breaker state."""
        if self._state == "open" and self._last_failure_time:
            # Check if recovery timeout has passed
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half-open"
        return self._state

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Call function through circuit breaker.

        Args:
            func: Function to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Function result

        Raises:
            Exception from function or CircuitOpenError
        """
        if self.state == "open":
            raise MCPError(
                code=ErrorCode.SYSTEM_UNAVAILABLE,
                message="Circuit breaker is open - service temporarily unavailable",
                details={
                    "recovery_time": (self._last_failure_time or 0.0)
                    + self.recovery_timeout
                },
            )

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise

    async def call_async(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """Async version of call."""
        if self.state == "open":
            raise MCPError(
                code=ErrorCode.SYSTEM_UNAVAILABLE,
                message="Circuit breaker is open - service temporarily unavailable",
                details={
                    "recovery_time": (self._last_failure_time or 0.0)
                    + self.recovery_timeout
                },
            )

        try:
            result = await func(*args, **kwargs)  # type: ignore[misc]
            self._on_success()
            return cast(T, result)
        except Exception as e:
            if isinstance(e, self.expected_exception):
                self._on_failure()
            raise

    def _on_success(self) -> None:
        """Handle successful call."""
        if self._state == "half-open":
            self._state = "closed"
        self._failure_count = 0

    def _on_failure(self) -> None:
        """Handle failed call."""
        self._failure_count += 1
        self._last_failure_time = time.time()

        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            logger.warning(
                f"Circuit breaker opened after {self._failure_count} failures"
            )


# Global retry manager instance
_default_retry_manager: RetryManager | None = None


def get_retry_manager() -> RetryManager:
    """Get the default retry manager instance."""
    global _default_retry_manager
    if _default_retry_manager is None:
        _default_retry_manager = RetryManager()
    return _default_retry_manager
