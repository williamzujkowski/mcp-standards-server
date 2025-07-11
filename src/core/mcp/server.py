"""MCP Server implementation."""

import logging
from typing import Any

from ..errors import get_secure_error_handler
from ..rate_limiter import get_rate_limiter
from ..security import get_security_middleware
from .handlers import StandardsHandler

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol server for standards management."""

    def __init__(self, standards_engine: Any = None) -> None:
        self.standards_engine = standards_engine
        self.handlers = {}
        self.running = False
        self.security_middleware = get_security_middleware()
        self.rate_limiter = get_rate_limiter()
        self.error_handler = get_secure_error_handler()

        # Register default handlers
        if standards_engine:
            self.handlers["standards"] = StandardsHandler(standards_engine)

    async def start(self) -> None:
        """Start the MCP server."""
        logger.info("Starting MCP server...")
        self.running = True

        # Initialize handlers
        for handler in self.handlers.values():
            if hasattr(handler, "initialize"):
                await handler.initialize()

        logger.info("MCP server started successfully")

    async def stop(self) -> None:
        """Stop the MCP server."""
        logger.info("Stopping MCP server...")
        self.running = False

        # Cleanup handlers
        for handler in self.handlers.values():
            if hasattr(handler, "cleanup"):
                await handler.cleanup()

        logger.info("MCP server stopped")

    async def handle_request(
        self, request: dict[str, Any], client_id: str = "default"
    ) -> dict[str, Any]:
        """Handle an incoming MCP request."""
        if not self.running:
            return {"error": "Server not running"}

        try:
            # Apply security middleware
            request = self.security_middleware.validate_and_sanitize_request(request)

            # Apply rate limiting
            is_allowed, limit_info = self.rate_limiter.check_all_limits(client_id)
            if not is_allowed:
                return {"error": "Rate limit exceeded", "rate_limit": limit_info}

            method = request.get("method")
            params = request.get("params", {})

            if method == "list_tools":
                return await self._list_tools()
            elif method == "call_tool":
                return await self._call_tool(params)
            else:
                return {"error": f"Unknown method: {method}"}

        except Exception as e:
            logger.error(f"Request handling error: {str(e)}")
            return self.error_handler.handle_exception(
                e, context={"method": "handle_request"}
            )

    async def _list_tools(self) -> dict[str, Any]:
        """List available tools."""
        tools = []

        for handler in self.handlers.values():
            if hasattr(handler, "get_tools"):
                tools.extend(await handler.get_tools())

        return {"tools": tools}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Call a specific tool."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if tool_name is None:
            return {"error": "Tool name is required"}

        try:
            # Validate tool arguments
            tool_args = self.security_middleware.validate_and_sanitize_request(
                tool_args
            )

            # Find appropriate handler
            for handler in self.handlers.values():
                if hasattr(handler, "handle_tool"):
                    result = await handler.handle_tool(tool_name, tool_args)
                    if result is not None:
                        # Add security headers to response
                        if isinstance(result, dict):
                            result.setdefault("headers", {})
                            result["headers"] = (
                                self.security_middleware.add_security_headers(
                                    result["headers"]
                                )
                            )
                        return result

            return {"error": f"Tool not found: {tool_name}"}

        except Exception as e:
            logger.error(f"Tool call error: {str(e)}")
            return self.error_handler.handle_exception(
                e, context={"method": "_call_tool", "tool_name": tool_name}
            )
