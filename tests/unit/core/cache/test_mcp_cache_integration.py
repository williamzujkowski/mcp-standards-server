"""Tests for MCP cache integration."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock

from src.core.cache.mcp_cache_integration import (
    MCPCacheMiddleware,
    integrate_cache_with_mcp_server
)
from src.core.cache.mcp_cache import CacheStrategy, ToolCacheConfig


class MockMCPServer:
    """Mock MCP server for testing."""
    
    def __init__(self):
        self.server = Mock()
        self.server._tools = []
        self.server.call_tool = Mock()
        self._execute_tool = AsyncMock()
        self._cache_middleware = None
        
        # Mock some tool responses
        self._execute_tool.side_effect = self._mock_execute
    
    async def _mock_execute(self, tool_name, arguments):
        """Mock tool execution."""
        if tool_name == "get_standard_details":
            return {"id": arguments.get("standard_id"), "name": "Test Standard"}
        elif tool_name == "list_templates":
            return {"templates": ["template1", "template2"]}
        else:
            return {"result": "mock_response"}


@pytest.fixture
def mock_server():
    """Create mock MCP server."""
    return MockMCPServer()


@pytest.fixture
def cache_middleware():
    """Create cache middleware."""
    return MCPCacheMiddleware()


class TestMCPCacheMiddleware:
    """Test cache middleware functionality."""
    
    @pytest.mark.asyncio
    async def test_wrap_tool_executor(self, cache_middleware, mock_server):
        """Test wrapping tool executor with caching."""
        # Wrap the executor
        wrapped = cache_middleware.wrap_tool_executor(mock_server._execute_tool)
        
        # First call - cache miss
        result1 = await wrapped(mock_server, "get_standard_details", {"standard_id": "test-std"})
        assert result1["id"] == "test-std"
        assert result1.get("_cache_hit") is False
        assert mock_server._execute_tool.call_count == 1
        
        # Second call - cache hit
        result2 = await wrapped(mock_server, "get_standard_details", {"standard_id": "test-std"})
        assert result2["id"] == "test-std"
        assert result2.get("_cache_hit") is True
        assert mock_server._execute_tool.call_count == 1  # Not called again
    
    @pytest.mark.asyncio
    async def test_warm_standard_caches(self, cache_middleware, mock_server):
        """Test warming standard caches."""
        results = await cache_middleware.warm_standard_caches(mock_server)
        
        # Should have warmed some tools
        assert "list_templates" in results
        assert results["list_templates"] > 0
    
    def test_get_cache_stats_tool(self, cache_middleware):
        """Test cache stats tool definition."""
        tool_def = cache_middleware.get_cache_stats_tool()
        
        assert tool_def["name"] == "get_cache_stats"
        assert "inputSchema" in tool_def
        assert "include_redis" in tool_def["inputSchema"]["properties"]
    
    @pytest.mark.asyncio
    async def test_handle_cache_stats(self, cache_middleware):
        """Test handling cache stats request."""
        stats = await cache_middleware.handle_cache_stats({"include_redis": False})
        
        assert "overall" in stats
        assert "health" in stats
        assert "redis_metrics" not in stats  # Excluded
    
    def test_get_cache_management_tools(self, cache_middleware):
        """Test cache management tool definitions."""
        tools = cache_middleware.get_cache_management_tools()
        
        tool_names = [t["name"] for t in tools]
        assert "cache_invalidate" in tool_names
        assert "cache_warm" in tool_names
        assert "cache_clear_all" in tool_names
        assert "cache_configure" in tool_names
    
    @pytest.mark.asyncio
    async def test_handle_cache_invalidate(self, cache_middleware):
        """Test cache invalidation handling."""
        # Add some cached data first
        await cache_middleware.cache.set("test_tool", {"arg": 1}, {"result": "cached"})
        
        # Invalidate
        result = await cache_middleware.handle_cache_management(
            "cache_invalidate",
            {"tool_name": "test_tool"}
        )
        
        assert result["success"] is True
        assert result["tool"] == "test_tool"
        
        # Check cache is empty
        cached = await cache_middleware.cache.get("test_tool", {"arg": 1})
        assert cached is None
    
    @pytest.mark.asyncio
    async def test_handle_cache_warm(self, cache_middleware, mock_server):
        """Test cache warming handling."""
        result = await cache_middleware.handle_cache_management(
            "cache_warm",
            {"tools": ["list_templates"]},
            mock_server
        )
        
        assert result["success"] is True
        assert "warmed_counts" in result
    
    @pytest.mark.asyncio
    async def test_handle_cache_clear_all(self, cache_middleware):
        """Test clearing all caches."""
        # Add some data
        await cache_middleware.cache.set("tool1", {"arg": 1}, {"result": 1})
        await cache_middleware.cache.set("tool2", {"arg": 2}, {"result": 2})
        
        # Try without confirmation
        result = await cache_middleware.handle_cache_management(
            "cache_clear_all",
            {"confirm": False}
        )
        assert result["success"] is False
        
        # Try with confirmation
        result = await cache_middleware.handle_cache_management(
            "cache_clear_all",
            {"confirm": True}
        )
        assert result["success"] is True
        assert result["cleared_count"] >= 0
    
    @pytest.mark.asyncio
    async def test_handle_cache_configure(self, cache_middleware):
        """Test cache configuration handling."""
        result = await cache_middleware.handle_cache_management(
            "cache_configure",
            {
                "tool_name": "test_tool",
                "strategy": "short_ttl",
                "ttl_seconds": 120
            }
        )
        
        assert result["success"] is True
        assert result["strategy"] == "short_ttl"
        assert result["ttl_seconds"] == 120


class TestIntegration:
    """Test full integration with MCP server."""
    
    def test_integrate_cache_with_mcp_server(self, mock_server):
        """Test integrating cache with MCP server."""
        # Basic integration
        middleware = integrate_cache_with_mcp_server(mock_server)
        
        # Should have added cache management tools
        tool_names = [t["name"] for t in mock_server.server._tools]
        assert "get_cache_stats" in tool_names
        assert "cache_invalidate" in tool_names
        
        # Should have stored middleware reference
        assert mock_server._cache_middleware is not None
    
    def test_integrate_with_config(self, mock_server):
        """Test integration with custom configuration."""
        config = {
            "redis": {
                "host": "localhost",
                "port": 6379,
                "default_ttl": 600
            },
            "tools": {
                "custom_tool": {
                    "strategy": "long_ttl",
                    "ttl_seconds": 3600,
                    "warm_on_startup": True
                }
            },
            "warm_on_startup": False
        }
        
        middleware = integrate_cache_with_mcp_server(mock_server, config)
        
        # Check custom tool config was applied
        tool_config = middleware.cache.tool_configs.get("custom_tool")
        assert tool_config is not None
        assert tool_config.strategy == CacheStrategy.LONG_TTL
        assert tool_config.ttl_seconds == 3600
    
    @pytest.mark.asyncio
    async def test_cached_tool_execution(self, mock_server):
        """Test that tool execution uses cache after integration."""
        # Integrate cache
        middleware = integrate_cache_with_mcp_server(mock_server)
        
        # Execute tool (should be cached)
        result1 = await mock_server._execute_tool("get_standard_details", {"standard_id": "test"})
        
        # Reset mock to track calls
        original_execute = mock_server._mock_execute
        mock_server._mock_execute = AsyncMock(side_effect=original_execute)
        
        # Execute again (should hit cache)
        result2 = await mock_server._execute_tool("get_standard_details", {"standard_id": "test"})
        
        # Original executor should not have been called
        assert mock_server._mock_execute.call_count == 0
        assert result1 == result2