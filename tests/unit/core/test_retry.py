"""
Unit tests for retry logic module.
"""

import time
from unittest.mock import AsyncMock, Mock

import pytest

from src.core.errors import ErrorCode, MCPError
from src.core.retry import (
    CircuitBreaker,
    RetryConfig,
    RetryManager,
    RetryStrategy,
    with_retry,
)


class TestRetryConfig:
    """Test retry configuration."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()

        assert config.max_retries == 3
        assert config.initial_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert config.strategy == RetryStrategy.EXPONENTIAL
        assert config.retry_on is not None
        assert ConnectionError in config.retry_on

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_retries=5,
            initial_delay=0.5,
            strategy=RetryStrategy.LINEAR,
            retry_on=(ValueError, KeyError),
        )

        assert config.max_retries == 5
        assert config.initial_delay == 0.5
        assert config.strategy == RetryStrategy.LINEAR
        assert config.retry_on is not None
        assert ValueError in config.retry_on
        assert KeyError in config.retry_on


class TestRetryManager:
    """Test retry manager functionality."""

    @pytest.fixture
    def retry_manager(self):
        """Create retry manager with fast delays for testing."""
        config = RetryConfig(
            max_retries=3,
            initial_delay=0.01,  # 10ms for fast tests
            max_delay=0.1,
            jitter=False,  # Disable for predictable tests
        )
        return RetryManager(config)

    def test_calculate_delay_exponential(self, retry_manager):
        """Test exponential backoff delay calculation."""
        delays = [retry_manager.calculate_delay(i) for i in range(4)]

        assert delays[0] == 0.01  # initial_delay
        assert delays[1] == 0.02  # initial_delay * 2^1
        assert delays[2] == 0.04  # initial_delay * 2^2
        assert delays[3] == 0.08  # initial_delay * 2^3

    def test_calculate_delay_linear(self):
        """Test linear backoff delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.LINEAR, initial_delay=0.01, jitter=False
        )
        manager = RetryManager(config)

        delays = [manager.calculate_delay(i) for i in range(4)]

        assert delays[0] == 0.01  # initial_delay * 1
        assert delays[1] == 0.02  # initial_delay * 2
        assert delays[2] == 0.03  # initial_delay * 3
        assert delays[3] == 0.04  # initial_delay * 4

    def test_calculate_delay_constant(self):
        """Test constant delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.CONSTANT, initial_delay=0.05, jitter=False
        )
        manager = RetryManager(config)

        delays = [manager.calculate_delay(i) for i in range(4)]

        assert all(d == 0.05 for d in delays)

    def test_calculate_delay_max_cap(self, retry_manager):
        """Test that delays are capped at max_delay."""
        # Set low max delay
        retry_manager.config.max_delay = 0.05

        delay = retry_manager.calculate_delay(10)  # Would be 10.24s without cap
        assert delay == 0.05

    def test_calculate_delay_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(initial_delay=1.0, jitter=True, jitter_factor=0.1)
        manager = RetryManager(config)

        # Generate multiple delays for same attempt
        delays = [manager.calculate_delay(0) for _ in range(10)]

        # Should have variation due to jitter
        assert len(set(delays)) > 1
        # All should be within 10% of base delay
        assert all(0.9 <= d <= 1.1 for d in delays)

    @pytest.mark.asyncio
    async def test_retry_async_success_first_try(self, retry_manager):
        """Test async retry succeeds on first try."""
        async_func = AsyncMock(return_value="success")

        result = await retry_manager.retry_async(
            async_func, 1, 2, operation_name="test_op", kwarg="value"
        )

        assert result == "success"
        async_func.assert_called_once_with(1, 2, kwarg="value")

    @pytest.mark.asyncio
    async def test_retry_async_success_after_retries(self, retry_manager):
        """Test async retry succeeds after failures."""
        async_func = AsyncMock(
            side_effect=[
                ConnectionError("Network error"),
                TimeoutError("Timeout"),
                "success",
            ]
        )

        result = await retry_manager.retry_async(async_func, operation_name="test_op")

        assert result == "success"
        assert async_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_async_all_failures(self, retry_manager):
        """Test async retry fails after max attempts."""
        async_func = AsyncMock(side_effect=ConnectionError("Network error"))

        with pytest.raises(ConnectionError):
            await retry_manager.retry_async(async_func, operation_name="test_op")

        assert async_func.call_count == 4  # initial + 3 retries

    @pytest.mark.asyncio
    async def test_retry_async_non_retryable_error(self, retry_manager):
        """Test async retry doesn't retry non-retryable errors."""
        async_func = AsyncMock(side_effect=ValueError("Bad value"))

        with pytest.raises(ValueError):
            await retry_manager.retry_async(async_func, operation_name="test_op")

        assert async_func.call_count == 1  # No retries

    def test_retry_sync_success_after_retries(self, retry_manager):
        """Test sync retry succeeds after failures."""
        sync_func = Mock(side_effect=[ConnectionError("Network error"), "success"])

        result = retry_manager.retry_sync(sync_func, operation_name="test_op")

        assert result == "success"
        assert sync_func.call_count == 2


class TestRetryDecorator:
    """Test retry decorator functionality."""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test retry decorator on async function."""
        call_count = 0

        @with_retry(max_retries=2, initial_delay=0.01)
        async def flaky_async_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network issue")
            return "success"

        result = await flaky_async_func()

        assert result == "success"
        assert call_count == 3

    def test_sync_decorator(self):
        """Test retry decorator on sync function."""
        call_count = 0

        @with_retry(max_retries=1, initial_delay=0.01)
        def flaky_sync_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"

        result = flaky_sync_func()

        assert result == "success"
        assert call_count == 2

    def test_decorator_preserves_metadata(self):
        """Test decorator preserves function metadata."""

        @with_retry()
        def example_func():
            """Example function docstring."""
            pass

        assert example_func.__name__ == "example_func"
        assert example_func.__doc__ == "Example function docstring."


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.fixture
    def circuit_breaker(self):
        """Create circuit breaker with low thresholds for testing."""
        return CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,  # 100ms for fast tests
            expected_exception=ConnectionError,
        )

    def test_initial_state(self, circuit_breaker):
        """Test circuit breaker starts closed."""
        assert circuit_breaker.state == "closed"
        assert circuit_breaker._failure_count == 0

    def test_success_keeps_closed(self, circuit_breaker):
        """Test successful calls keep circuit closed."""
        func = Mock(return_value="success")

        for _ in range(5):
            result = circuit_breaker.call(func)
            assert result == "success"

        assert circuit_breaker.state == "closed"
        assert circuit_breaker._failure_count == 0

    def test_failures_open_circuit(self, circuit_breaker):
        """Test failures open the circuit."""
        func = Mock(side_effect=ConnectionError("Network error"))

        # First failure
        with pytest.raises(ConnectionError):
            circuit_breaker.call(func)
        assert circuit_breaker.state == "closed"

        # Second failure opens circuit
        with pytest.raises(ConnectionError):
            circuit_breaker.call(func)
        assert circuit_breaker.state == "open"

    def test_open_circuit_blocks_calls(self, circuit_breaker):
        """Test open circuit blocks calls."""
        func = Mock(side_effect=ConnectionError("Network error"))

        # Open the circuit
        for _ in range(2):
            try:
                circuit_breaker.call(func)
            except ConnectionError:
                pass

        # Circuit is open
        assert circuit_breaker.state == "open"

        # Next call should be blocked
        with pytest.raises(MCPError) as exc_info:
            circuit_breaker.call(func)

        assert exc_info.value.error_detail.code == ErrorCode.SYSTEM_UNAVAILABLE
        assert "Circuit breaker is open" in str(exc_info.value)

    def test_half_open_after_timeout(self, circuit_breaker):
        """Test circuit goes to half-open after timeout."""
        func = Mock(side_effect=ConnectionError("Network error"))

        # Open the circuit
        for _ in range(2):
            try:
                circuit_breaker.call(func)
            except ConnectionError:
                pass

        assert circuit_breaker.state == "open"

        # Wait for recovery timeout
        time.sleep(0.15)

        # Should be half-open now
        assert circuit_breaker.state == "half-open"

    def test_half_open_success_closes_circuit(self, circuit_breaker):
        """Test successful call in half-open state closes circuit."""
        # Open the circuit
        circuit_breaker._failure_count = 2
        circuit_breaker._state = "open"
        circuit_breaker._last_failure_time = time.time() - 1  # Past recovery timeout

        assert circuit_breaker.state == "half-open"

        # Successful call
        func = Mock(return_value="success")
        result = circuit_breaker.call(func)

        assert result == "success"
        assert circuit_breaker.state == "closed"
        assert circuit_breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_async_circuit_breaker(self, circuit_breaker):
        """Test circuit breaker with async calls."""
        async_func = AsyncMock(side_effect=ConnectionError("Network error"))

        # Open the circuit
        for _ in range(2):
            with pytest.raises(ConnectionError):
                await circuit_breaker.call_async(async_func)

        # Circuit is open
        with pytest.raises(MCPError) as exc_info:
            await circuit_breaker.call_async(async_func)

        assert "Circuit breaker is open" in str(exc_info.value)
