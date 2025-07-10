"""
MCP Caching Module.

Provides intelligent caching for MCP tool responses with Redis backend.
"""

from .decorators import cache_key, cache_result, invalidate_cache
from .mcp_cache import (
    CacheMetrics,
    CacheStrategy,
    MCPCache,
    ToolCacheConfig,
    cache_tool_response,
)
from .mcp_cache_integration import MCPCacheMiddleware, integrate_cache_with_mcp_server
from .redis_client import CacheConfig, CircuitBreaker, RedisCache

__all__ = [
    # Original exports
    "RedisCache",
    "CacheConfig",
    "cache_result",
    "invalidate_cache",
    "cache_key",
    # Core cache
    "MCPCache",
    "CacheStrategy",
    "ToolCacheConfig",
    "CacheMetrics",
    "cache_tool_response",
    # Integration
    "MCPCacheMiddleware",
    "integrate_cache_with_mcp_server",
    # Redis backend
    "CircuitBreaker",
]

# Version info
__version__ = "1.0.0"
