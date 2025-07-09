"""MCP Server implementation."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from .handlers import StandardsHandler

logger = logging.getLogger(__name__)


class MCPServer:
    """Model Context Protocol server for standards management."""
    
    def __init__(self, standards_engine=None):
        self.standards_engine = standards_engine
        self.handlers = {}
        self.running = False
        
        # Register default handlers
        if standards_engine:
            self.handlers['standards'] = StandardsHandler(standards_engine)
    
    async def start(self):
        """Start the MCP server."""
        logger.info("Starting MCP server...")
        self.running = True
        
        # Initialize handlers
        for handler in self.handlers.values():
            if hasattr(handler, 'initialize'):
                await handler.initialize()
        
        logger.info("MCP server started successfully")
    
    async def stop(self):
        """Stop the MCP server."""
        logger.info("Stopping MCP server...")
        self.running = False
        
        # Cleanup handlers
        for handler in self.handlers.values():
            if hasattr(handler, 'cleanup'):
                await handler.cleanup()
        
        logger.info("MCP server stopped")
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP request."""
        if not self.running:
            return {"error": "Server not running"}
        
        method = request.get('method')
        params = request.get('params', {})
        
        if method == 'list_tools':
            return await self._list_tools()
        elif method == 'call_tool':
            return await self._call_tool(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    async def _list_tools(self) -> Dict[str, Any]:
        """List available tools."""
        tools = []
        
        for handler in self.handlers.values():
            if hasattr(handler, 'get_tools'):
                tools.extend(await handler.get_tools())
        
        return {"tools": tools}
    
    async def _call_tool(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific tool."""
        tool_name = params.get('name')
        tool_args = params.get('arguments', {})
        
        # Find appropriate handler
        for handler in self.handlers.values():
            if hasattr(handler, 'handle_tool'):
                result = await handler.handle_tool(tool_name, tool_args)
                if result is not None:
                    return result
        
        return {"error": f"Tool not found: {tool_name}"}