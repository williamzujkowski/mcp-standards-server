"""
Rate limiting module for MCP server.

Implements token bucket algorithm for rate limiting with Redis backend.
"""

import time
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta

from src.core.cache.redis_client import get_redis_client
from src.core.errors import RateLimitError


class RateLimiter:
    """Token bucket rate limiter with Redis backend."""
    
    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: int = 60,
        redis_prefix: str = "mcp:ratelimit"
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
        
    def check_rate_limit(self, identifier: str) -> Tuple[bool, Optional[Dict[str, int]]]:
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
                    "retry_after": reset_time - current_time
                }
                return False, limit_info
                
            # Add current request
            request_times.append(current_time)
            self.redis_client.set(key, request_times, ttl=self.window_seconds + 60)
            
            limit_info = {
                "remaining": self.max_requests - len(request_times),
                "limit": self.max_requests,
                "reset_time": current_time + self.window_seconds
            }
            return True, limit_info
            
        except Exception as e:
            # Redis error, allow request but log
            print(f"Rate limit check failed: {e}")
            return True, None
            
    def reset_limit(self, identifier: str):
        """Reset rate limit for an identifier."""
        if self.redis_client:
            key = f"{self.redis_prefix}:{identifier}"
            self.redis_client.delete(key)


class MultiTierRateLimiter:
    """Rate limiter with multiple tiers (per-minute, per-hour, per-day)."""
    
    def __init__(self, redis_prefix: str = "mcp:ratelimit"):
        """Initialize multi-tier rate limiter."""
        self.tiers = {
            "minute": RateLimiter(100, 60, f"{redis_prefix}:minute"),
            "hour": RateLimiter(5000, 3600, f"{redis_prefix}:hour"),
            "day": RateLimiter(50000, 86400, f"{redis_prefix}:day")
        }
        
    def check_all_limits(self, identifier: str) -> Tuple[bool, Optional[Dict[str, any]]]:
        """
        Check all rate limit tiers.
        
        Returns:
            Tuple of (is_allowed, limit_info)
        """
        for tier_name, limiter in self.tiers.items():
            is_allowed, limit_info = limiter.check_rate_limit(identifier)
            
            if not is_allowed:
                # Add tier information
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
        
    def reset_all_limits(self, identifier: str):
        """Reset all rate limits for an identifier."""
        for limiter in self.tiers.values():
            limiter.reset_limit(identifier)


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on user behavior and system load."""
    
    def __init__(
        self,
        base_limit: int = 100,
        window_seconds: int = 60,
        redis_prefix: str = "mcp:adaptive"
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
        
    def update_reputation(self, identifier: str, is_good_request: bool):
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
        
        self.redis_client.set(
            reputation_key,
            str(reputation),
            ttl=86400 * 7  # 7 days
        )


# Default rate limiter instance
_default_rate_limiter: Optional[MultiTierRateLimiter] = None


def get_rate_limiter() -> MultiTierRateLimiter:
    """Get the default rate limiter instance."""
    global _default_rate_limiter
    if _default_rate_limiter is None:
        _default_rate_limiter = MultiTierRateLimiter()
    return _default_rate_limiter