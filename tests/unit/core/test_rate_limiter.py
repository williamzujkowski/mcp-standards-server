"""
Unit tests for rate limiting module.
"""

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
        """Create mock Redis client."""
        mock = Mock()
        mock.zremrangebyscore = Mock(return_value=0)
        mock.zcard = Mock(return_value=0)
        mock.zadd = Mock(return_value=1)
        mock.expire = Mock(return_value=True)
        mock.zrange = Mock(return_value=[])
        mock.delete = Mock(return_value=1)
        return mock

    @pytest.fixture
    def rate_limiter(self, mock_redis):
        """Create rate limiter with mock Redis."""
        with patch("src.core.rate_limiter.get_redis_client", return_value=mock_redis):
            limiter = RateLimiter(max_requests=10, window_seconds=60)
            limiter.redis_client = mock_redis
            return limiter

    def test_allow_request_under_limit(self, rate_limiter, mock_redis):
        """Test allowing request when under limit."""
        # Mock previous request times (5 requests in the past 30 seconds)
        current_time = int(time.time())
        previous_requests = [
            current_time - i for i in range(5, 30, 5)
        ]  # 5 requests within window
        mock_redis.get.return_value = previous_requests

        is_allowed, limit_info = rate_limiter.check_rate_limit("test_user")

        assert is_allowed
        assert limit_info["remaining"] == 4  # 10 - 6 (5 previous + 1 current)
        assert limit_info["limit"] == 10

    def test_block_request_at_limit(self, rate_limiter, mock_redis):
        """Test blocking request when at limit."""
        # Mock 10 requests already in the current window
        current_time = int(time.time())
        previous_requests = [
            current_time - i for i in range(1, 11)
        ]  # 10 recent requests
        mock_redis.get.return_value = previous_requests

        is_allowed, limit_info = rate_limiter.check_rate_limit("test_user")

        assert not is_allowed
        assert limit_info["remaining"] == 0
        assert limit_info["retry_after"] > 0

    def test_cleanup_old_entries(self, rate_limiter, mock_redis):
        """Test that old entries are cleaned up."""
        # Mock some old and new requests
        current_time = int(time.time())
        old_requests = [
            current_time - 120,
            current_time - 100,
        ]  # Old (outside 60s window)
        new_requests = [current_time - 30, current_time - 10]  # Recent
        all_requests = old_requests + new_requests
        mock_redis.get.return_value = all_requests

        is_allowed, limit_info = rate_limiter.check_rate_limit("test_user")

        # Verify set was called with cleaned data (only new requests + current)
        assert mock_redis.set.called
        saved_data = mock_redis.set.call_args[0][1]
        # Should have 3 items: 2 new + 1 current
        assert len(saved_data) == 3
        assert all(t >= current_time - 60 for t in saved_data)

    def test_no_redis_allows_all(self):
        """Test that missing Redis allows all requests."""
        with patch("src.core.rate_limiter.get_redis_client", return_value=None):
            limiter = RateLimiter()

        is_allowed, limit_info = limiter.check_rate_limit("test_user")

        assert is_allowed
        assert limit_info is None

    def test_redis_error_allows_request(self, rate_limiter, mock_redis):
        """Test that Redis errors don't block requests."""
        mock_redis.get.side_effect = Exception("Redis connection error")

        is_allowed, limit_info = rate_limiter.check_rate_limit("test_user")

        assert is_allowed
        assert limit_info is None

    def test_reset_limit(self, rate_limiter, mock_redis):
        """Test resetting rate limit."""
        rate_limiter.reset_limit("test_user")

        mock_redis.delete.assert_called_once_with("mcp:ratelimit:test_user")


class TestMultiTierRateLimiter:
    """Test multi-tier rate limiter."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = Mock()
        # Set up for the actual implementation which uses get/set
        mock.get = Mock(return_value=[])  # Empty list by default
        mock.set = Mock(return_value=True)
        mock.delete = Mock(return_value=1)
        return mock

    @pytest.fixture
    def multi_limiter(self, mock_redis):
        """Create multi-tier rate limiter with mock Redis."""
        with patch("src.core.rate_limiter.get_redis_client", return_value=mock_redis):
            limiter = MultiTierRateLimiter()
            # Patch Redis for all tiers
            for tier_limiter in limiter.tiers.values():
                tier_limiter.redis_client = mock_redis
            return limiter

    def test_all_tiers_pass(self, multi_limiter, mock_redis):
        """Test when all rate limit tiers pass."""
        mock_redis.get.return_value = []  # No requests yet

        is_allowed, limit_info = multi_limiter.check_all_limits("test_user")

        assert is_allowed
        assert "minute" in limit_info
        assert "hour" in limit_info
        assert "day" in limit_info

    def test_minute_tier_blocks(self, multi_limiter, mock_redis):
        """Test when minute tier blocks request."""

        # Make minute tier fail by returning more than 100 requests in the current window
        def get_side_effect(key):
            if "minute" in key:
                current_time = int(time.time())
                # Generate 105 requests within the 60-second window (need more than 100 to exceed limit)
                requests = []
                for i in range(105):
                    # Distribute across 60 seconds but with some overlap
                    time_offset = (
                        i % 60
                    )  # This will create multiple requests per second
                    requests.append(current_time - time_offset)
                return requests
            return []

        mock_redis.get.side_effect = get_side_effect

        is_allowed, limit_info = multi_limiter.check_all_limits("test_user")

        assert not is_allowed
        assert limit_info["tier"] == "minute"
        assert limit_info["window"] == "minute"

    def test_reset_all_limits(self, multi_limiter, mock_redis):
        """Test resetting all tier limits."""
        multi_limiter.reset_all_limits("test_user")

        # Should delete keys for all tiers
        assert mock_redis.delete.call_count == 3  # minute, hour, day


class TestAdaptiveRateLimiter:
    """Test adaptive rate limiter."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        mock = Mock()
        mock.get = Mock(return_value=None)
        mock.setex = Mock(return_value=True)
        return mock

    @pytest.fixture
    def adaptive_limiter(self, mock_redis):
        """Create adaptive rate limiter with mock Redis."""
        with patch("src.core.rate_limiter.get_redis_client", return_value=mock_redis):
            limiter = AdaptiveRateLimiter(base_limit=100)
            limiter.redis_client = mock_redis
            return limiter

    def test_default_limit_for_new_user(self, adaptive_limiter, mock_redis):
        """Test default limit for new user."""
        mock_redis.get.return_value = None

        limit = adaptive_limiter.get_user_limit("new_user")

        assert limit == 100  # Base limit

    def test_increased_limit_for_good_reputation(self, adaptive_limiter, mock_redis):
        """Test increased limit for good reputation."""
        mock_redis.get.return_value = "0.9"  # High reputation

        limit = adaptive_limiter.get_user_limit("good_user")

        assert limit == 150  # 1.5x base limit

    def test_decreased_limit_for_bad_reputation(self, adaptive_limiter, mock_redis):
        """Test decreased limit for bad reputation."""
        mock_redis.get.return_value = "0.2"  # Low reputation

        limit = adaptive_limiter.get_user_limit("bad_user")

        assert limit == 50  # 0.5x base limit

    def test_update_reputation_good_request(self, adaptive_limiter, mock_redis):
        """Test reputation update for good request."""
        mock_redis.get.return_value = "0.5"  # Neutral starting

        adaptive_limiter.update_reputation("test_user", is_good_request=True)

        # Should update reputation upward
        mock_redis.set.assert_called()
        args = mock_redis.set.call_args[0]
        new_reputation = float(args[1])  # Second argument is the value
        assert new_reputation > 0.5

    def test_update_reputation_bad_request(self, adaptive_limiter, mock_redis):
        """Test reputation update for bad request."""
        mock_redis.get.return_value = "0.5"  # Neutral starting

        adaptive_limiter.update_reputation("test_user", is_good_request=False)

        # Should update reputation downward
        mock_redis.set.assert_called()
        args = mock_redis.set.call_args[0]
        new_reputation = float(args[1])  # Second argument is the value
        assert new_reputation < 0.5

    def test_singleton_instance(self):
        """Test that get_rate_limiter returns singleton."""
        limiter1 = get_rate_limiter()
        limiter2 = get_rate_limiter()

        assert limiter1 is limiter2
