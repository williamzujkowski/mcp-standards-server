"""
MCP Cache Integration Module.

Provides seamless integration of caching with the MCP server.
"""

import logging
from functools import wraps
from typing import Any

from .mcp_cache import CacheStrategy, MCPCache, ToolCacheConfig
from .redis_client import CacheConfig, RedisCache

logger = logging.getLogger(__name__)


class MCPCacheMiddleware:
    """Middleware to integrate caching with MCP server."""

    def __init__(
        self,
        cache: MCPCache | None = None,
        redis_config: CacheConfig | None = None,
        custom_tool_configs: dict[str, ToolCacheConfig] | None = None,
    ):
        """Initialize cache middleware.

        Args:
            cache: Pre-configured MCPCache instance
            redis_config: Redis configuration (if cache not provided)
            custom_tool_configs: Custom tool cache configurations
        """
        if cache:
            self.cache = cache
        else:
            redis_cache = RedisCache(redis_config) if redis_config else RedisCache()
            self.cache = MCPCache(
                redis_cache=redis_cache, custom_configs=custom_tool_configs
            )

    def wrap_tool_executor(self, executor_func: Any) -> Any:
        """Wrap the tool executor function with caching logic.

        Args:
            executor_func: Original _execute_tool function

        Returns:
            Wrapped function with caching
        """

        @wraps(executor_func)
        async def wrapped_executor(
            self_ref: Any, tool_name: str, arguments: dict[str, Any]
        ) -> dict[str, Any]:
            # Try to get from cache first
            cached_response = await self.cache.get(tool_name, arguments)
            if cached_response is not None:
                logger.debug(f"Cache hit for tool: {tool_name}")
                # Add cache metadata to response
                if isinstance(cached_response, dict):
                    cached_response["_cache_hit"] = True
                return cached_response  # type: ignore[no-any-return]

            # Execute the tool
            logger.debug(f"Cache miss for tool: {tool_name}, executing...")
            response = await executor_func(self_ref, tool_name, arguments)

            # Cache the response
            await self.cache.set(tool_name, arguments, response)

            # Add cache metadata
            if isinstance(response, dict):
                response["_cache_hit"] = False

            return response  # type: ignore[no-any-return]

        return wrapped_executor

    async def warm_standard_caches(self, mcp_server: Any) -> dict[str, int]:
        """Warm caches for commonly used standards.

        Args:
            mcp_server: MCPStandardsServer instance

        Returns:
            Dictionary of warmed cache counts by tool
        """
        # Define common warming scenarios
        warm_configs = {
            "list_templates": [
                {},  # List all templates
                {"domain": "api"},
                {"domain": "security"},
                {"domain": "frontend"},
            ],
            "list_available_standards": [
                {},  # List all standards
                {"limit": 50},
                {"category": "security", "limit": 100},
                {"category": "api", "limit": 100},
            ],
            "get_standard_details": [
                {"standard_id": "secure-api-design"},
                {"standard_id": "react-best-practices"},
                {"standard_id": "python-coding-standards"},
            ],
        }

        # Create executor function
        async def executor(tool_name: str, args: dict[str, Any]) -> Any:
            return await mcp_server._execute_tool(tool_name, args)

        # Update warm configurations in cache
        for tool_name, args_list in warm_configs.items():
            if tool_name in self.cache.tool_configs:
                self.cache.tool_configs[tool_name].warm_args = args_list  # type: ignore[assignment]

        # Perform warming
        return await self.cache.warm_cache(executor)

    def get_cache_stats_tool(self) -> dict[str, Any]:
        """Get a tool definition for cache statistics.

        Returns:
            Tool definition that can be added to MCP server
        """
        return {
            "name": "get_cache_stats",
            "description": "Get cache statistics and metrics",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "include_redis": {
                        "type": "boolean",
                        "description": "Include Redis-level metrics",
                        "default": True,
                    }
                },
            },
        }

    async def handle_cache_stats(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Handle cache statistics tool call."""
        metrics = self.cache.get_metrics()

        if not arguments.get("include_redis", True):
            metrics.pop("redis_metrics", None)

        # Add health status
        health = await self.cache.health_check()
        metrics["health"] = health

        return metrics

    def get_cache_management_tools(self) -> list[dict[str, Any]]:
        """Get tool definitions for cache management.

        Returns:
            List of tool definitions for cache management
        """
        return [
            {
                "name": "cache_invalidate",
                "description": "Invalidate cache for a specific tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to invalidate cache for",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Specific arguments to invalidate (omit to clear all)",
                        },
                    },
                    "required": ["tool_name"],
                },
            },
            {
                "name": "cache_warm",
                "description": "Warm cache for specific tools",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of tools to warm (omit for all configured)",
                        }
                    },
                },
            },
            {
                "name": "cache_clear_all",
                "description": "Clear all MCP tool caches",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "confirm": {
                            "type": "boolean",
                            "description": "Confirm clearing all caches",
                            "default": False,
                        }
                    },
                },
            },
            {
                "name": "cache_configure",
                "description": "Configure caching for a specific tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "description": "Name of the tool to configure",
                        },
                        "strategy": {
                            "type": "string",
                            "enum": [
                                "no_cache",
                                "short_ttl",
                                "medium_ttl",
                                "long_ttl",
                                "permanent",
                            ],
                            "description": "Caching strategy",
                        },
                        "ttl_seconds": {
                            "type": "integer",
                            "description": "Custom TTL in seconds",
                        },
                    },
                    "required": ["tool_name"],
                },
            },
        ]

    async def handle_cache_management(
        self, tool_name: str, arguments: dict[str, Any], mcp_server: Any = None
    ) -> dict[str, Any]:
        """Handle cache management tool calls."""

        if tool_name == "cache_invalidate":
            success = await self.cache.invalidate(
                arguments["tool_name"], arguments.get("arguments")
            )
            return {
                "success": success,
                "tool": arguments["tool_name"],
                "message": "Cache invalidated" if success else "Invalidation failed",
            }

        elif tool_name == "cache_warm":
            if mcp_server:

                async def executor(tool: str, args: dict[str, Any]) -> Any:
                    return await mcp_server._execute_tool(tool, args)

                results = await self.cache.warm_cache(executor, arguments.get("tools"))
                return {"success": True, "warmed_counts": results}
            else:
                return {
                    "success": False,
                    "error": "MCP server instance required for warming",
                }

        elif tool_name == "cache_clear_all":
            if arguments.get("confirm", False):
                count = await self.cache.clear_all()
                return {"success": True, "cleared_count": count}
            else:
                return {
                    "success": False,
                    "error": "Confirmation required to clear all caches",
                }

        elif tool_name == "cache_configure":
            tool = arguments["tool_name"]
            strategy_str = arguments.get("strategy")
            ttl = arguments.get("ttl_seconds")

            if strategy_str:
                strategy = CacheStrategy(strategy_str)
                self.cache.configure_tool(tool, strategy=strategy, ttl_seconds=ttl)
            elif ttl is not None:
                self.cache.configure_tool(tool, ttl_seconds=ttl)

            # Return current configuration
            config = self.cache.tool_configs.get(tool)
            if config:
                return {
                    "success": True,
                    "tool": tool,
                    "strategy": config.strategy.value,
                    "ttl_seconds": config.ttl_seconds
                    or self.cache.DEFAULT_TTLS.get(config.strategy),
                }
            else:
                return {
                    "success": False,
                    "error": f"Tool {tool} not found in cache configuration",
                }

        else:
            return {
                "success": False,
                "error": f"Unknown cache management tool: {tool_name}",
            }


def integrate_cache_with_mcp_server(
    mcp_server: Any, cache_config: dict[str, Any] | None = None
) -> MCPCacheMiddleware:
    """Integrate caching with an MCP server instance.

    Args:
        mcp_server: MCPStandardsServer instance
        cache_config: Optional cache configuration

    This function:
    1. Creates cache middleware
    2. Wraps the tool executor
    3. Adds cache management tools
    4. Optionally warms the cache
    """
    # Create cache configuration
    redis_config = None
    if cache_config and "redis" in cache_config:
        redis_config = CacheConfig(**cache_config["redis"])

    # Create custom tool configs if provided
    custom_configs = None
    if cache_config and "tools" in cache_config:
        custom_configs = {}
        for tool_name, tool_cfg in cache_config["tools"].items():
            if isinstance(tool_cfg, dict):
                strategy = CacheStrategy(tool_cfg.get("strategy", "medium_ttl"))
                custom_configs[tool_name] = ToolCacheConfig(
                    tool_name=tool_name,
                    strategy=strategy,
                    ttl_seconds=tool_cfg.get("ttl_seconds"),
                    compress_threshold=tool_cfg.get("compress_threshold", 1024),
                    warm_on_startup=tool_cfg.get("warm_on_startup", False),
                )

    # Create middleware
    middleware = MCPCacheMiddleware(
        redis_config=redis_config, custom_tool_configs=custom_configs
    )

    # Store original executor
    original_executor = mcp_server._execute_tool

    # Create a wrapper that matches the expected signature
    async def executor_adapter(
        self_ref: Any, tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        # Call the original executor as a method (self is already bound)
        result = await original_executor(tool_name, arguments)
        return result  # type: ignore[no-any-return]

    # Wrap the adapter with caching
    wrapped_executor = middleware.wrap_tool_executor(executor_adapter)

    # Create the final cached executor
    async def cached_executor(
        tool_name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        result = await wrapped_executor(mcp_server, tool_name, arguments)
        return result  # type: ignore[no-any-return]

    # Replace executor
    mcp_server._execute_tool = cached_executor

    # Add cache management tools to the tools list
    if hasattr(mcp_server.server, "_tools"):
        # Add cache stats tool
        mcp_server.server._tools.append(middleware.get_cache_stats_tool())

        # Add management tools
        for tool in middleware.get_cache_management_tools():
            mcp_server.server._tools.append(tool)

    # Store middleware reference
    mcp_server._cache_middleware = middleware

    # Add cache tool handlers
    original_call_tool = mcp_server.server.call_tool

    @mcp_server.server.call_tool()
    async def enhanced_call_tool(
        name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> Any:
        # Handle cache-specific tools
        if name == "get_cache_stats":
            return await middleware.handle_cache_stats(arguments)
        elif name in [
            "cache_invalidate",
            "cache_warm",
            "cache_clear_all",
            "cache_configure",
        ]:
            return await middleware.handle_cache_management(name, arguments, mcp_server)
        else:
            # Delegate to original handler
            return await original_call_tool(name, arguments, **kwargs)

    # Warm cache if configured
    if cache_config and cache_config.get("warm_on_startup", False):
        import asyncio

        asyncio.create_task(middleware.warm_standard_caches(mcp_server))

    logger.info("Cache integration completed successfully")

    return middleware
