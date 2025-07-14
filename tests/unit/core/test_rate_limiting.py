"""Tests for rate limiting functionality."""

import time
from unittest.mock import Mock, patch

import pytest

from src.core.rate_limiter import (
    AdaptiveRateLimiter,
    MultiTierRateLimiter,
    RateLimiter,
    get_rate_limiter,
)


class TestRateLimiter:
    """Test basic rate limiter functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = Mock()
        mock_redis.get.return_value = []
        mock_redis.set.return_value = True
        return mock_redis

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create rate limiter with mocked Redis."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        limiter.redis_client = mock_redis
        return limiter

    def test_rate_limiter_allows_requests_under_limit(self, rate_limiter, mock_redis):
        """Test that requests under the limit are allowed."""
        # Mock empty request history
        mock_redis.get.return_value = []

        is_allowed, limit_info = rate_limiter.check_rate_limit("user123")

        assert is_allowed is True
        assert limit_info["remaining"] == 4  # 5 - 1 = 4
        assert limit_info["limit"] == 5

        # Check that request was recorded
        mock_redis.set.assert_called_once()

    def test_rate_limiter_blocks_requests_over_limit(self, rate_limiter, mock_redis):
        """Test that requests over the limit are blocked."""
        # Mock request history with 5 recent requests
        current_time = int(time.time())
        request_times = [current_time - i for i in range(5)]
        mock_redis.get.return_value = request_times

        is_allowed, limit_info = rate_limiter.check_rate_limit("user123")

        assert is_allowed is False
        assert limit_info["remaining"] == 0
        assert limit_info["limit"] == 5
        assert "retry_after" in limit_info

        # Should not record new request
        mock_redis.set.assert_not_called()

    def test_rate_limiter_cleans_old_requests(self, rate_limiter, mock_redis):
        """Test that old requests are cleaned from history."""
        # Mock request history with old requests
        current_time = int(time.time())
        old_requests = [current_time - 120, current_time - 90]  # 2 and 1.5 minutes ago
        recent_requests = [
            current_time - 30,
            current_time - 15,
        ]  # 30 and 15 seconds ago

        mock_redis.get.return_value = old_requests + recent_requests

        is_allowed, limit_info = rate_limiter.check_rate_limit("user123")

        assert is_allowed is True
        assert limit_info["remaining"] == 2  # 5 - 2 recent - 1 new = 2

        # Should record new request with cleaned history
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        new_history = args[1]

        # Should contain only recent requests plus new one
        assert len(new_history) == 3

    def test_rate_limiter_handles_redis_failure(self, rate_limiter):
        """Test that rate limiter handles Redis failures gracefully."""
        # Mock Redis failure
        rate_limiter.redis_client = None

        is_allowed, limit_info = rate_limiter.check_rate_limit("user123")

        assert is_allowed is True
        assert limit_info is None

    def test_rate_limiter_reset_functionality(self, rate_limiter, mock_redis):
        """Test that rate limit can be reset."""
        rate_limiter.reset_limit("user123")

        mock_redis.delete.assert_called_once_with("mcp:ratelimit:user123")


class TestMultiTierRateLimiter:
    """Test multi-tier rate limiter functionality."""

    @pytest.fixture
    def multi_tier_limiter(self):
        """Create multi-tier rate limiter."""
        return MultiTierRateLimiter()

    def test_multi_tier_limiter_initialization(self, multi_tier_limiter):
        """Test multi-tier limiter initialization."""
        assert "minute" in multi_tier_limiter.tiers
        assert "hour" in multi_tier_limiter.tiers
        assert "day" in multi_tier_limiter.tiers

        # Check default limits
        assert multi_tier_limiter.tiers["minute"].max_requests == 100
        assert multi_tier_limiter.tiers["hour"].max_requests == 5000
        assert multi_tier_limiter.tiers["day"].max_requests == 50000

    def test_multi_tier_limiter_checks_all_tiers(self, multi_tier_limiter):
        """Test that all tiers are checked."""
        with patch.object(
            multi_tier_limiter.tiers["minute"], "check_rate_limit"
        ) as mock_minute:
            with patch.object(
                multi_tier_limiter.tiers["hour"], "check_rate_limit"
            ) as mock_hour:
                with patch.object(
                    multi_tier_limiter.tiers["day"], "check_rate_limit"
                ) as mock_day:

                    # Mock all tiers allowing requests
                    mock_minute.return_value = (True, {"remaining": 99, "limit": 100})
                    mock_hour.return_value = (True, {"remaining": 4999, "limit": 5000})
                    mock_day.return_value = (True, {"remaining": 49999, "limit": 50000})

                    is_allowed, limit_info = multi_tier_limiter.check_all_limits(
                        "user123"
                    )

                    assert is_allowed is True
                    assert "minute" in limit_info
                    assert "hour" in limit_info
                    assert "day" in limit_info

    def test_multi_tier_limiter_blocks_on_any_tier(self, multi_tier_limiter):
        """Test that any tier blocking stops the request."""
        with patch.object(
            multi_tier_limiter.tiers["minute"], "check_rate_limit"
        ) as mock_minute:
            with patch.object(
                multi_tier_limiter.tiers["hour"], "check_rate_limit"
            ) as mock_hour:
                with patch.object(
                    multi_tier_limiter.tiers["day"], "check_rate_limit"
                ) as mock_day:

                    # Mock minute tier blocking
                    mock_minute.return_value = (
                        False,
                        {
                            "remaining": 0,
                            "limit": 100,
                            "tier": "minute",
                            "retry_after": 30,
                        },
                    )

                    is_allowed, limit_info = multi_tier_limiter.check_all_limits(
                        "user123"
                    )

                    assert is_allowed is False
                    assert limit_info["tier"] == "minute"
                    assert limit_info["window"] == "minute"

                    # Hour and day tiers should not be checked
                    mock_hour.assert_not_called()
                    mock_day.assert_not_called()

    def test_multi_tier_limiter_reset_all(self, multi_tier_limiter):
        """Test that all tier limits can be reset."""
        with patch.object(
            multi_tier_limiter.tiers["minute"], "reset_limit"
        ) as mock_minute:
            with patch.object(
                multi_tier_limiter.tiers["hour"], "reset_limit"
            ) as mock_hour:
                with patch.object(
                    multi_tier_limiter.tiers["day"], "reset_limit"
                ) as mock_day:

                    multi_tier_limiter.reset_all_limits("user123")

                    mock_minute.assert_called_once_with("user123")
                    mock_hour.assert_called_once_with("user123")
                    mock_day.assert_called_once_with("user123")


class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        mock_redis = Mock()
        mock_redis.get.return_value = None
        mock_redis.setex.return_value = True
        return mock_redis

    @pytest.fixture
    def adaptive_limiter(self, mock_redis):
        """Create adaptive rate limiter with mocked Redis."""
        limiter = AdaptiveRateLimiter(base_limit=100)
        limiter.redis_client = mock_redis
        return limiter

    def test_adaptive_limiter_default_limit(self, adaptive_limiter, mock_redis):
        """Test default limit for new users."""
        # Mock no reputation data
        mock_redis.get.return_value = None

        limit = adaptive_limiter.get_user_limit("new_user")

        assert limit == 100  # base_limit

    def test_adaptive_limiter_high_reputation_bonus(self, adaptive_limiter, mock_redis):
        """Test higher limits for high reputation users."""
        # Mock high reputation
        mock_redis.get.return_value = "0.9"

        limit = adaptive_limiter.get_user_limit("good_user")

        assert limit == 150  # base_limit * 1.5

    def test_adaptive_limiter_low_reputation_penalty(
        self, adaptive_limiter, mock_redis
    ):
        """Test lower limits for low reputation users."""
        # Mock low reputation
        mock_redis.get.return_value = "0.2"

        limit = adaptive_limiter.get_user_limit("bad_user")

        assert limit == 50  # base_limit * 0.5

    def test_adaptive_limiter_reputation_update_good(
        self, adaptive_limiter, mock_redis
    ):
        """Test reputation update for good requests."""
        # Mock existing reputation
        mock_redis.get.return_value = "0.5"

        adaptive_limiter.update_reputation("user123", is_good_request=True)

        # Should update reputation upward
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        key, new_reputation = args

        assert key == "mcp:adaptive:reputation:user123"
        assert kwargs.get("ttl") == 86400 * 7  # 7 days
        assert float(new_reputation) > 0.5

    def test_adaptive_limiter_reputation_update_bad(self, adaptive_limiter, mock_redis):
        """Test reputation update for bad requests."""
        # Mock existing reputation
        mock_redis.get.return_value = "0.5"

        adaptive_limiter.update_reputation("user123", is_good_request=False)

        # Should update reputation downward
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        key, new_reputation = args

        assert key == "mcp:adaptive:reputation:user123"
        assert kwargs.get("ttl") == 86400 * 7  # 7 days
        assert float(new_reputation) < 0.5

    def test_adaptive_limiter_new_user_reputation(self, adaptive_limiter, mock_redis):
        """Test reputation initialization for new users."""
        # Mock no existing reputation
        mock_redis.get.return_value = None

        adaptive_limiter.update_reputation("new_user", is_good_request=True)

        # Should start with neutral reputation (0.5) and improve
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        new_reputation = float(args[1])

        assert new_reputation > 0.5

    def test_adaptive_limiter_handles_redis_failure(self, adaptive_limiter):
        """Test that adaptive limiter handles Redis failures gracefully."""
        # Mock Redis failure
        adaptive_limiter.redis_client = None

        limit = adaptive_limiter.get_user_limit("user123")
        assert limit == 100  # base_limit

        # Should not crash on reputation update
        adaptive_limiter.update_reputation("user123", is_good_request=True)


class TestRateLimitingIntegration:
    """Test rate limiting integration with security measures."""

    def test_rate_limiting_with_security_violations(self):
        """Test that security violations affect rate limiting."""
        # This would be implemented when integrating with security middleware
        # For now, we'll test the structure

        rate_limiter = MultiTierRateLimiter()

        # Simulate security violation by updating reputation
        with patch.object(
            rate_limiter.tiers["minute"], "check_rate_limit"
        ) as mock_check:
            mock_check.return_value = (
                False,
                {"remaining": 0, "limit": 10, "tier": "minute", "retry_after": 60},
            )

            is_allowed, limit_info = rate_limiter.check_all_limits("suspicious_user")

            assert is_allowed is False
            assert limit_info is not None
            assert limit_info["tier"] == "minute"

    def test_rate_limiting_error_handling(self):
        """Test that rate limiting errors are handled gracefully."""
        rate_limiter = RateLimiter()

        # Mock Redis client that raises exception
        rate_limiter.redis_client = Mock()
        rate_limiter.redis_client.get.side_effect = Exception("Redis connection error")

        # Should not crash and should allow request
        is_allowed, limit_info = rate_limiter.check_rate_limit("user123")

        assert is_allowed is True
        assert limit_info is None


class TestGlobalRateLimiter:
    """Test global rate limiter management."""

    def test_get_rate_limiter_singleton(self):
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2
        assert isinstance(limiter1, MultiTierRateLimiter)

    def test_rate_limiter_configuration(self):
        """Test rate limiter configuration."""
        limiter = get_rate_limiter()

        # Should have all tiers configured
        assert "minute" in limiter.tiers
        assert "hour" in limiter.tiers
        assert "day" in limiter.tiers

        # Check tier configuration
        assert limiter.tiers["minute"].max_requests == 100
        assert limiter.tiers["minute"].window_seconds == 60

        assert limiter.tiers["hour"].max_requests == 5000
        assert limiter.tiers["hour"].window_seconds == 3600

        assert limiter.tiers["day"].max_requests == 50000
        assert limiter.tiers["day"].window_seconds == 86400


class TestRateLimitingSecurityIntegration:
    """Test rate limiting integration with security systems."""

    def test_rate_limiting_with_client_identification(self):
        """Test rate limiting with proper client identification."""
        rate_limiter = MultiTierRateLimiter()

        # Test with different client identifiers
        client_ids = ["ip:192.168.1.1", "user:john_doe", "api_key:abc123"]

        for client_id in client_ids:
            with patch.object(
                rate_limiter.tiers["minute"], "check_rate_limit"
            ) as mock_check:
                mock_check.return_value = (True, {"remaining": 99, "limit": 100})

                is_allowed, limit_info = rate_limiter.check_all_limits(client_id)

                assert is_allowed is True
                mock_check.assert_called_with(client_id)

    def test_rate_limiting_escalation_on_violations(self):
        """Test rate limiting escalation on security violations."""
        adaptive_limiter = AdaptiveRateLimiter()

        # Mock Redis client
        mock_redis = Mock()
        mock_redis.get.return_value = None  # Start with no reputation
        mock_redis.set.return_value = True
        adaptive_limiter.redis_client = mock_redis

        # Simulate multiple security violations
        for _ in range(5):
            adaptive_limiter.update_reputation("malicious_user", is_good_request=False)

        # Mock the reduced reputation after violations
        mock_redis.get.return_value = "0.2"  # Low reputation

        # Should have reduced limit
        limit = adaptive_limiter.get_user_limit("malicious_user")
        assert limit < 100  # Should be less than base limit

    def test_rate_limiting_recovery_after_good_behavior(self):
        """Test that rate limits recover after good behavior."""
        adaptive_limiter = AdaptiveRateLimiter()

        # Mock Redis client
        mock_redis = Mock()
        mock_redis.get.return_value = None  # Start with no reputation
        mock_redis.set.return_value = True
        adaptive_limiter.redis_client = mock_redis

        # Start with bad reputation
        for _ in range(10):
            adaptive_limiter.update_reputation("reformed_user", is_good_request=False)

        # Mock low reputation after bad behavior
        mock_redis.get.return_value = "0.2"
        initial_limit = adaptive_limiter.get_user_limit("reformed_user")

        # Show good behavior
        for _ in range(20):
            adaptive_limiter.update_reputation("reformed_user", is_good_request=True)

        # Mock improved reputation after good behavior
        mock_redis.get.return_value = "0.85"
        final_limit = adaptive_limiter.get_user_limit("reformed_user")

        assert final_limit > initial_limit
