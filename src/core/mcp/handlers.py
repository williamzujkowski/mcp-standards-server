"""MCP handlers for standards operations."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class StandardsHandler:
    """Handler for standards-related MCP operations."""

    def __init__(self, standards_engine):
        self.standards_engine = standards_engine

    async def initialize(self):
        """Initialize the handler."""
        if self.standards_engine and hasattr(self.standards_engine, 'initialize'):
            await self.standards_engine.initialize()

    async def cleanup(self):
        """Cleanup handler resources."""
        if self.standards_engine and hasattr(self.standards_engine, 'close'):
            await self.standards_engine.close()

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools."""
        return [
            {
                "name": "get_applicable_standards",
                "description": "Get standards applicable to a project context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_context": {
                            "type": "object",
                            "description": "Project context for standard selection"
                        }
                    },
                    "required": ["project_context"]
                }
            },
            {
                "name": "search_standards",
                "description": "Search for standards using semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_standard",
                "description": "Get a specific standard by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "standard_id": {
                            "type": "string",
                            "description": "Standard ID"
                        },
                        "version": {
                            "type": "string",
                            "description": "Standard version (optional)"
                        }
                    },
                    "required": ["standard_id"]
                }
            }
        ]

    async def handle_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Handle a tool call."""
        if not self.standards_engine:
            return {"error": "Standards engine not available"}

        try:
            if tool_name == "get_applicable_standards":
                result = await self.standards_engine.get_applicable_standards(
                    args.get("project_context", {})
                )
                return {"result": result}

            elif tool_name == "search_standards":
                result = await self.standards_engine.search_standards(
                    query=args.get("query", ""),
                    limit=args.get("limit", 10)
                )
                return {"result": result}

            elif tool_name == "get_standard":
                result = await self.standards_engine.get_standard(
                    standard_id=args.get("standard_id"),
                    version=args.get("version")
                )
                return {"result": result}

            else:
                return None  # Tool not handled by this handler

        except Exception as e:
            logger.error(f"Error handling tool {tool_name}: {e}")
            return {"error": str(e)}
