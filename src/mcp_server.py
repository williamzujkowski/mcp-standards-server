"""
MCP Standards Server implementation.

Provides Model Context Protocol interface to standards functionality.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool
from pydantic import ValidationError as PydanticValidationError

# Import authentication, validation and error handling modules
from .core.auth import AuthConfig, get_auth_manager
from .core.errors import (
    AuthenticationError,
    AuthorizationError,
    ErrorCode,
    MCPError,
    RateLimitError,
    ToolNotFoundError,
    ValidationError,
)

# Import metrics module
from .core.metrics import get_mcp_metrics

# Import privacy filtering
from .core.privacy import PrivacyConfig, get_privacy_filter
from .core.standards.analytics import StandardsAnalytics
from .core.standards.cross_referencer import CrossReferencer
from .core.standards.rule_engine import RuleEngine
from .core.standards.sync import StandardsSynchronizer
from .core.standards.token_optimizer import (
    DynamicLoader,
    ModelType,
    StandardFormat,
    TokenBudget,
    create_token_optimizer,
)
from .core.validation import get_input_validator
from .generators import StandardsGenerator

logger = logging.getLogger(__name__)


class MCPStandardsServer:
    """MCP server for standards management."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.server = Server("mcp-standards-server")

        # Initialize metrics collector
        self.metrics = get_mcp_metrics()
        self._active_connections = 0

        # Initialize auth manager
        auth_config = AuthConfig(**self.config.get("auth", {}))
        self.auth_manager = get_auth_manager()
        self.auth_manager.config = auth_config

        # Initialize input validator
        self.input_validator = get_input_validator()

        # Initialize privacy filter
        privacy_config = PrivacyConfig(**self.config.get("privacy", {}))
        self.privacy_filter = get_privacy_filter()
        self.privacy_filter.config = privacy_config

        # Initialize rate limiting (simple in-memory implementation)
        self._rate_limit_store: dict[str, list[float]] = {}
        self.rate_limit_window = self.config.get("rate_limit_window", 60)  # seconds
        self.rate_limit_max_requests = self.config.get("rate_limit_max_requests", 100)

        # Initialize components
        # Get data directory
        data_dir = Path(os.environ.get("MCP_STANDARDS_DATA_DIR", "data"))
        rules_file = data_dir / "standards" / "meta" / "enhanced-selection-rules.json"

        self.rule_engine = RuleEngine(
            Path(self.config.get("rules_file", str(rules_file)))
        )
        sync_config_path = self.config.get(
            "sync_config", data_dir / "standards" / "sync_config.yaml"
        )
        self.synchronizer = StandardsSynchronizer(
            config_path=sync_config_path, cache_dir=data_dir / "standards" / "cache"
        )

        # Initialize cross-referencer and analytics
        self.cross_referencer = CrossReferencer(data_dir / "standards")
        self.analytics = StandardsAnalytics(data_dir / "standards" / "analytics")

        # Initialize standards generator
        templates_dir = Path("templates")
        self.generator = StandardsGenerator(templates_dir)

        # Initialize search only if enabled
        self.search = None
        if os.environ.get("MCP_DISABLE_SEARCH") != "true" and self.config.get(
            "search", {}
        ).get("enabled", True):
            try:
                from .core.standards.semantic_search import SemanticSearch

                self.search = SemanticSearch(
                    embedding_model=self.config.get("search_model", "all-MiniLM-L6-v2")
                )
            except ImportError:
                logger.warning(
                    "Semantic search disabled: sentence-transformers not installed"
                )

        # Initialize token optimizer
        model_type = ModelType(self.config.get("token_model", "gpt-4"))
        default_budget = self.config.get("default_token_budget", 8000)
        self.token_optimizer = create_token_optimizer(
            model_type=model_type, default_budget=default_budget
        )
        self.dynamic_loader = DynamicLoader(self.token_optimizer)

        # Register tools
        self._register_tools()

    def _register_tools(self):
        """Register all MCP tools."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """Return list of available tools."""
            return [
                Tool(
                    name="get_applicable_standards",
                    description="Get applicable standards based on project context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "context": {
                                "type": "object",
                                "description": "Project context information",
                            },
                            "include_resolution_details": {
                                "type": "boolean",
                                "description": "Include detailed resolution information",
                                "default": False,
                            },
                        },
                        "required": ["context"],
                    },
                ),
                Tool(
                    name="validate_against_standard",
                    description="Validate code against a specific standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to validate",
                            },
                            "standard": {
                                "type": "string",
                                "description": "Standard ID",
                            },
                            "language": {
                                "type": "string",
                                "description": "Programming language",
                            },
                        },
                        "required": ["code", "standard"],
                    },
                ),
                Tool(
                    name="search_standards",
                    description="Search standards using semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results",
                                "default": 10,
                            },
                            "min_relevance": {
                                "type": "number",
                                "description": "Minimum relevance score",
                                "default": 0.0,
                            },
                            "filters": {
                                "type": "object",
                                "description": "Additional filters",
                            },
                        },
                        "required": ["query"],
                    },
                ),
                Tool(
                    name="get_standard_details",
                    description="Get detailed information about a specific standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {
                                "type": "string",
                                "description": "Standard identifier",
                            }
                        },
                        "required": ["standard_id"],
                    },
                ),
                Tool(
                    name="list_available_standards",
                    description="List all available standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "Filter by category",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum results",
                                "default": 100,
                            },
                        },
                    },
                ),
                Tool(
                    name="suggest_improvements",
                    description="Suggest improvements based on applicable standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Code to analyze",
                            },
                            "context": {
                                "type": "object",
                                "description": "Project context",
                            },
                        },
                        "required": ["code", "context"],
                    },
                ),
                Tool(
                    name="sync_standards",
                    description="Synchronize standards from repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force": {
                                "type": "boolean",
                                "description": "Force sync",
                                "default": False,
                            }
                        },
                    },
                ),
                Tool(
                    name="get_optimized_standard",
                    description="Get a token-optimized version of a standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {
                                "type": "string",
                                "description": "Standard ID",
                            },
                            "format_type": {
                                "type": "string",
                                "description": "Format type",
                                "default": "condensed",
                            },
                            "token_budget": {
                                "type": "integer",
                                "description": "Token budget",
                            },
                            "required_sections": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context",
                            },
                        },
                        "required": ["standard_id"],
                    },
                ),
                Tool(
                    name="auto_optimize_standards",
                    description="Automatically optimize multiple standards within a token budget",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "total_token_budget": {
                                "type": "integer",
                                "description": "Total token budget",
                            },
                            "context": {
                                "type": "object",
                                "description": "Additional context",
                            },
                        },
                        "required": ["standard_ids", "total_token_budget"],
                    },
                ),
                Tool(
                    name="progressive_load_standard",
                    description="Get a progressive loading plan for a standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {
                                "type": "string",
                                "description": "Standard ID",
                            },
                            "initial_sections": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum depth",
                                "default": 3,
                            },
                        },
                        "required": ["standard_id", "initial_sections"],
                    },
                ),
                Tool(
                    name="estimate_token_usage",
                    description="Estimate token usage for standards in different formats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "format_types": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["standard_ids"],
                    },
                ),
                Tool(
                    name="get_sync_status",
                    description="Get current synchronization status",
                    inputSchema={"type": "object", "properties": {}},
                ),
                Tool(
                    name="generate_standard",
                    description="Generate a new standard based on template and context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_name": {
                                "type": "string",
                                "description": "Template name",
                            },
                            "context": {
                                "type": "object",
                                "description": "Generation context",
                            },
                            "domain": {
                                "type": "string",
                                "description": "Domain/category",
                            },
                            "title": {
                                "type": "string",
                                "description": "Standard title",
                            },
                        },
                        "required": ["template_name", "context", "title"],
                    },
                ),
                Tool(
                    name="validate_standard",
                    description="Validate a standard document for completeness and quality",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_content": {
                                "type": "string",
                                "description": "Standard content to validate",
                            },
                            "format": {
                                "type": "string",
                                "description": "Content format (yaml/json)",
                                "default": "yaml",
                            },
                        },
                        "required": ["standard_content"],
                    },
                ),
                Tool(
                    name="list_templates",
                    description="List available standard templates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {
                                "type": "string",
                                "description": "Filter by domain",
                            }
                        },
                    },
                ),
                Tool(
                    name="get_cross_references",
                    description="Get cross-references for a standard or concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {
                                "type": "string",
                                "description": "Standard ID",
                            },
                            "concept": {
                                "type": "string",
                                "description": "Concept to find references for",
                            },
                            "max_depth": {
                                "type": "integer",
                                "description": "Maximum reference depth",
                                "default": 2,
                            },
                        },
                    },
                ),
                Tool(
                    name="generate_cross_references",
                    description="Generate cross-references between standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {
                                "type": "boolean",
                                "description": "Force refresh of references",
                                "default": False,
                            }
                        },
                    },
                ),
                Tool(
                    name="get_standards_analytics",
                    description="Get analytics and usage statistics for standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric_type": {
                                "type": "string",
                                "description": "Type of metrics (usage/popularity/gaps)",
                                "default": "usage",
                            },
                            "time_range": {
                                "type": "string",
                                "description": "Time range for metrics",
                                "default": "30d",
                            },
                            "standard_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Specific standards",
                            },
                        },
                    },
                ),
                Tool(
                    name="track_standards_usage",
                    description="Track usage of specific standards or sections",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {
                                "type": "string",
                                "description": "Standard ID",
                            },
                            "section_id": {
                                "type": "string",
                                "description": "Section ID",
                            },
                            "usage_type": {
                                "type": "string",
                                "description": "Usage type (view/apply/reference)",
                            },
                            "context": {
                                "type": "object",
                                "description": "Usage context",
                            },
                        },
                        "required": ["standard_id", "usage_type"],
                    },
                ),
                Tool(
                    name="get_recommendations",
                    description="Get recommendations for standards improvement or gaps",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_type": {
                                "type": "string",
                                "description": "Type of analysis (gaps/quality/usage)",
                                "default": "gaps",
                            },
                            "context": {
                                "type": "object",
                                "description": "Analysis context",
                            },
                        },
                    },
                ),
                Tool(
                    name="get_metrics_dashboard",
                    description="Get performance metrics dashboard",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]

        # Capture self in closure for nested function
        server_instance = self

        @self.server.call_tool()
        async def call_tool(
            name: str, arguments: dict[str, Any], **kwargs
        ) -> list[TextContent]:
            """Handle tool calls with authentication and validation."""
            logger.info(f"call_tool invoked with name={name}, arguments={arguments}")

            # Track request size
            request_size = len(json.dumps(arguments).encode("utf-8"))
            server_instance.metrics.record_request_size(request_size, name)

            # Start timing the tool call
            start_time = time.time()

            try:
                # Extract authentication from request context if available
                headers = kwargs.get("headers", {})
                auth_type, credential = (
                    server_instance.auth_manager.extract_auth_from_headers(headers)
                )

                # Verify authentication if enabled
                if server_instance.auth_manager.is_enabled():
                    if not credential:
                        server_instance.metrics.record_auth_attempt("none", False)
                        raise AuthenticationError("Authentication required")

                    if auth_type == "bearer":
                        is_valid, payload, error_msg = (
                            server_instance.auth_manager.verify_token(credential)
                        )
                        server_instance.metrics.record_auth_attempt("bearer", is_valid)
                        if not is_valid:
                            raise AuthenticationError(error_msg or "Invalid token")

                        # Check tool permission
                        if not server_instance.auth_manager.check_permission(
                            payload, "mcp:tools"
                        ):
                            raise AuthorizationError(
                                "Insufficient permissions for tool access",
                                required_scope="mcp:tools",
                            )
                    elif auth_type == "api_key":
                        is_valid, user_id, error_msg = (
                            server_instance.auth_manager.verify_api_key(credential)
                        )
                        server_instance.metrics.record_auth_attempt("api_key", is_valid)
                        if not is_valid:
                            raise AuthenticationError(error_msg or "Invalid API key")

                    # Check rate limits
                    user_key = credential if credential else "anonymous"
                    if not server_instance._check_rate_limit(user_key):
                        server_instance.metrics.record_rate_limit_hit(
                            user_key, "standard"
                        )
                        raise RateLimitError(
                            limit=server_instance.rate_limit_max_requests,
                            window=f"{server_instance.rate_limit_window}s",
                            retry_after=server_instance.rate_limit_window,
                        )

                # Validate tool exists
                if name not in [
                    "get_applicable_standards",
                    "validate_against_standard",
                    "search_standards",
                    "get_standard_details",
                    "list_available_standards",
                    "suggest_improvements",
                    "sync_standards",
                    "get_optimized_standard",
                    "auto_optimize_standards",
                    "progressive_load_standard",
                    "estimate_token_usage",
                    "get_sync_status",
                    "generate_standard",
                    "validate_standard",
                    "list_templates",
                    "get_cross_references",
                    "generate_cross_references",
                    "get_standards_analytics",
                    "track_standards_usage",
                    "get_recommendations",
                    "get_metrics_dashboard",
                ]:
                    raise ToolNotFoundError(name)

                # Validate input arguments
                try:
                    validated_args = (
                        server_instance.input_validator.validate_tool_input(
                            name, arguments
                        )
                    )
                except ValidationError:
                    # Re-raise our custom validation errors
                    raise
                except PydanticValidationError as e:
                    # Convert Pydantic validation errors to our format
                    errors = e.errors()
                    first_error = errors[0] if errors else {}
                    field = ".".join(str(loc) for loc in first_error.get("loc", []))
                    raise ValidationError(
                        message=first_error.get("msg", "Validation error"),
                        field=field or "unknown",
                    )

                # Execute tool
                result = await server_instance._execute_tool(name, validated_args)

                # Log successful execution
                logger.debug(f"Tool {name} executed successfully")

                # Record successful tool call
                duration = time.time() - start_time
                server_instance.metrics.record_tool_call(name, duration, True)

                # Apply privacy filtering if enabled
                if server_instance.privacy_filter.config.detect_pii:
                    # Generate privacy report for monitoring
                    privacy_report = server_instance.privacy_filter.get_privacy_report(
                        result
                    )
                    if privacy_report["has_pii"]:
                        logger.warning(
                            f"PII detected in response for tool {name}: "
                            f"{privacy_report['pii_count']} instances of "
                            f"{', '.join(privacy_report['pii_types_found'])}"
                        )
                        # Filter the response
                        filtered_result, _ = server_instance.privacy_filter.filter_dict(
                            result
                        )
                        result = filtered_result

                # Track response size
                response_text = json.dumps(result, indent=2)
                response_size = len(response_text.encode("utf-8"))
                server_instance.metrics.record_response_size(response_size, name)

                # Return successful result
                return [TextContent(type="text", text=response_text)]

            except MCPError as e:
                # Handle structured errors
                logger.error(f"MCP error in tool {name}: {e}", exc_info=True)
                duration = time.time() - start_time
                server_instance.metrics.record_tool_call(
                    name, duration, False, e.code.value
                )
                return [
                    TextContent(type="text", text=json.dumps(e.to_dict(), indent=2))
                ]
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error in tool {name}: {e}", exc_info=True)
                duration = time.time() - start_time
                server_instance.metrics.record_tool_call(
                    name, duration, False, type(e).__name__
                )
                error = MCPError(
                    code=ErrorCode.SYSTEM_INTERNAL_ERROR,
                    message=f"Internal error: {str(e)}",
                    details={"tool": name},
                )
                return [
                    TextContent(type="text", text=json.dumps(error.to_dict(), indent=2))
                ]

    async def _execute_tool(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute a specific tool with validated arguments."""
        try:
            if name == "get_applicable_standards":
                return await self._get_applicable_standards(
                    arguments["context"],
                    arguments.get("include_resolution_details", False),
                )
            elif name == "validate_against_standard":
                return await self._validate_against_standard(
                    arguments["code"], arguments["standard"], arguments.get("language")
                )
            elif name == "search_standards":
                return await self._search_standards(
                    arguments["query"],
                    arguments.get("limit", 10),
                    arguments.get("min_relevance", 0.0),
                    arguments.get("filters"),
                )
            elif name == "get_standard_details":
                return await self._get_standard_details(arguments["standard_id"])
            elif name == "list_available_standards":
                return await self._list_available_standards(
                    arguments.get("category"), arguments.get("limit", 100)
                )
            elif name == "suggest_improvements":
                return await self._suggest_improvements(
                    arguments["code"], arguments["context"]
                )
            elif name == "sync_standards":
                return await self._sync_standards(arguments.get("force", False))
            elif name == "get_optimized_standard":
                return await self._get_optimized_standard(
                    arguments["standard_id"],
                    arguments.get("format_type", "condensed"),
                    arguments.get("token_budget"),
                    arguments.get("required_sections"),
                    arguments.get("context"),
                )
            elif name == "auto_optimize_standards":
                return await self._auto_optimize_standards(
                    arguments["standard_ids"],
                    arguments["total_token_budget"],
                    arguments.get("context"),
                )
            elif name == "progressive_load_standard":
                return await self._progressive_load_standard(
                    arguments["standard_id"],
                    arguments["initial_sections"],
                    arguments.get("max_depth", 3),
                )
            elif name == "estimate_token_usage":
                return await self._estimate_token_usage(
                    arguments["standard_ids"], arguments.get("format_types")
                )
            elif name == "get_sync_status":
                return await self._get_sync_status()
            elif name == "generate_standard":
                return await self._generate_standard(
                    arguments["template_name"],
                    arguments["context"],
                    arguments["title"],
                    arguments.get("domain"),
                )
            elif name == "validate_standard":
                return await self._validate_standard(
                    arguments["standard_content"], arguments.get("format", "yaml")
                )
            elif name == "list_templates":
                return await self._list_templates(arguments.get("domain"))
            elif name == "get_cross_references":
                return await self._get_cross_references(
                    arguments.get("standard_id"),
                    arguments.get("concept"),
                    arguments.get("max_depth", 2),
                )
            elif name == "generate_cross_references":
                return await self._generate_cross_references(
                    arguments.get("force_refresh", False)
                )
            elif name == "get_standards_analytics":
                return await self._get_standards_analytics(
                    arguments.get("metric_type", "usage"),
                    arguments.get("time_range", "30d"),
                    arguments.get("standard_ids"),
                )
            elif name == "track_standards_usage":
                return await self._track_standards_usage(
                    arguments["standard_id"],
                    arguments["usage_type"],
                    arguments.get("section_id"),
                    arguments.get("context"),
                )
            elif name == "get_recommendations":
                return await self._get_recommendations(
                    arguments.get("analysis_type", "gaps"), arguments.get("context")
                )
            elif name == "get_metrics_dashboard":
                return await self._get_metrics_dashboard()
            else:
                raise ToolNotFoundError(name)

        except MCPError:
            # Re-raise MCP errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise MCPError(
                code=ErrorCode.TOOL_EXECUTION_FAILED,
                message=f"Tool execution failed: {str(e)}",
                details={"tool": name, "error_type": type(e).__name__},
            )

    async def _get_applicable_standards(
        self, context: dict[str, Any], include_resolution_details: bool = False
    ) -> dict[str, Any]:
        """Get applicable standards based on project context."""
        result = self.rule_engine.evaluate(context)

        response = {
            "standards": result["resolved_standards"],
            "evaluation_path": result.get("evaluation_path", []),
        }

        if include_resolution_details:
            response["resolution_details"] = {
                "matched_rules": result.get("matched_rules", []),
                "conflicts_resolved": result.get("conflicts_resolved", 0),
                "final_priority_order": result.get("priority_order", []),
            }

        return response

    async def _validate_against_standard(
        self, code: str, standard: str, language: str | None = None
    ) -> dict[str, Any]:
        """Validate code against a specific standard."""
        # This is a placeholder - actual implementation would analyze code
        violations = []

        # Simple validation checks
        if standard == "react-18-patterns" and language == "javascript":
            if "class " in code and "extends React.Component" in code:
                violations.append(
                    {
                        "line": 1,
                        "message": "Prefer functional components over class components",
                        "severity": "warning",
                    }
                )

        return {
            "standard": standard,
            "passed": len(violations) == 0,
            "violations": violations,
        }

    async def _search_standards(
        self,
        query: str,
        limit: int = 10,
        min_relevance: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Search standards using semantic search."""
        if self.search is None:
            # Fallback to simple keyword search
            return {"results": [], "warning": "Semantic search is disabled"}

        results = self.search.search(query, top_k=limit, filters=filters)

        # Filter by minimum relevance and convert to dict format
        filtered_results = []
        for result in results:
            if result.score >= min_relevance:
                filtered_results.append(
                    {
                        "standard": result.id,
                        "relevance_score": result.score,
                        "content": result.content,
                        "metadata": result.metadata,
                        "highlights": result.highlights,
                    }
                )

        return {"results": filtered_results}

    async def _get_standard_details(self, standard_id: str) -> dict[str, Any]:
        """Get detailed information about a specific standard."""
        # Load standard from cache or repository
        standard_path = self.synchronizer.cache_dir / f"{standard_id}.json"

        if standard_path.exists():
            # Record cache hit
            self.metrics.record_cache_access("get_standard_details", True)
            with open(standard_path) as f:
                return json.load(f)
        else:
            # Record cache miss
            self.metrics.record_cache_access("get_standard_details", False)
            raise MCPError(
                code=ErrorCode.STANDARDS_NOT_FOUND,
                message=f"Standard '{standard_id}' not found",
                details={"standard_id": standard_id},
            )

    async def _list_available_standards(
        self, category: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        """List all available standards."""
        standards = []

        # List cached standards
        cache_dir = self.synchronizer.cache_dir
        if not cache_dir.exists():
            raise MCPError(
                code=ErrorCode.RESOURCE_NOT_FOUND,
                message="Standards cache directory not found",
                suggestion="Run sync_standards to populate the cache",
            )

        for file_path in cache_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    standard = json.load(f)

                if category is None or standard.get("category") == category:
                    standards.append(
                        {
                            "id": standard.get("id", file_path.stem),
                            "name": standard.get("name", "Unknown"),
                            "category": standard.get("category"),
                            "tags": standard.get("tags", []),
                        }
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in {file_path}: {e}")
                continue
            except KeyError as e:
                logger.warning(f"Missing required field in {file_path}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Failed to load standard from {file_path}: {e}")
                continue

        return {"standards": standards[:limit]}

    async def _suggest_improvements(
        self, code: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """Suggest improvements based on applicable standards."""
        # Get applicable standards
        standards_result = await self._get_applicable_standards(context)
        applicable_standards = standards_result["standards"]

        suggestions = []

        # Generate suggestions based on standards
        for standard_id in applicable_standards:
            # This is a placeholder - actual implementation would analyze code
            if "javascript" in context.get("language", "").lower():
                if "var " in code:
                    suggestions.append(
                        {
                            "description": "Use const/let instead of var",
                            "priority": "high",
                            "standard_reference": standard_id,
                        }
                    )
                if "function" in code and "async" not in code:
                    suggestions.append(
                        {
                            "description": "Consider using async/await for asynchronous operations",
                            "priority": "medium",
                            "standard_reference": standard_id,
                        }
                    )
                if ".then(" in code:
                    suggestions.append(
                        {
                            "description": "Consider using async/await instead of promise chains",
                            "priority": "medium",
                            "standard_reference": standard_id,
                        }
                    )

        return {"suggestions": suggestions}

    async def _sync_standards(self, force: bool = False) -> dict[str, Any]:
        """Synchronize standards from repository."""
        try:
            result = await self.synchronizer.sync(force=force)
            return {
                "status": result.status.value,
                "synced_files": [f.path for f in result.synced_files],
                "failed_files": result.failed_files,
                "message": result.message,
            }
        except Exception as e:
            raise MCPError(
                code=ErrorCode.STANDARDS_SYNC_FAILED,
                message=f"Standards synchronization failed: {str(e)}",
                details={"force": force},
            )

    async def _get_optimized_standard(
        self,
        standard_id: str,
        format_type: str = "condensed",
        token_budget: int | None = None,
        required_sections: list[str] | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get a token-optimized version of a standard."""
        # Load standard
        standard = await self._get_standard_details(standard_id)

        # Parse format type
        try:
            format_enum = StandardFormat(format_type)
        except ValueError:
            format_enum = StandardFormat.CONDENSED

        # Create budget
        budget = None
        if token_budget:
            budget = TokenBudget(total=token_budget)

        # Optimize standard
        content, result = self.token_optimizer.optimize_standard(
            standard,
            format_type=format_enum,
            budget=budget,
            required_sections=required_sections,
            context=context,
        )

        return {
            "standard_id": standard_id,
            "content": content,
            "format": result.format_used.value,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "compression_ratio": result.compression_ratio,
            "sections_included": result.sections_included,
            "sections_excluded": result.sections_excluded,
            "warnings": result.warnings,
        }

    async def _auto_optimize_standards(
        self,
        standard_ids: list[str],
        total_token_budget: int,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Automatically optimize multiple standards within a token budget."""
        # Estimate token usage for each standard
        standards = []
        for std_id in standard_ids:
            try:
                standard = await self._get_standard_details(std_id)
                standards.append(standard)
            except Exception as e:
                logger.warning(f"Failed to load standard {std_id}: {e}")
                continue

        # Estimate token distribution
        estimates = self.token_optimizer.estimate_tokens(
            standards, format_type=StandardFormat.CONDENSED
        )

        # Allocate budget proportionally
        total_original = estimates["total_original"]
        results = []

        for i, standard in enumerate(standards):
            std_estimate = estimates["standards"][i]

            # Calculate proportional budget
            proportion = std_estimate["original_tokens"] / total_original
            allocated_budget = int(total_token_budget * proportion)

            # Create budget with some buffer
            budget = TokenBudget(total=allocated_budget)

            # Auto-select format
            selected_format = self.token_optimizer.auto_select_format(
                standard, budget, context
            )

            # Optimize
            content, result = self.token_optimizer.optimize_standard(
                standard, format_type=selected_format, budget=budget, context=context
            )

            results.append(
                {
                    "standard_id": standard["id"],
                    "content": content,
                    "format": result.format_used.value,
                    "tokens_used": result.compressed_tokens,
                    "allocated_budget": allocated_budget,
                }
            )

        total_used = sum(r["tokens_used"] for r in results)

        return {
            "results": results,
            "total_tokens_used": total_used,
            "total_budget": total_token_budget,
            "budget_utilization": (
                total_used / total_token_budget if total_token_budget > 0 else 0
            ),
        }

    async def _progressive_load_standard(
        self, standard_id: str, initial_sections: list[str], max_depth: int = 3
    ) -> dict[str, Any]:
        """Get a progressive loading plan for a standard."""
        # Load standard
        standard = await self._get_standard_details(standard_id)

        # Generate loading plan
        loading_plan = self.token_optimizer.progressive_load(
            standard, initial_sections=initial_sections, max_depth=max_depth
        )

        # Format plan for response
        formatted_plan = []
        for batch_idx, batch in enumerate(loading_plan):
            formatted_batch = {
                "batch": batch_idx + 1,
                "sections": [
                    {"id": section_id, "estimated_tokens": tokens}
                    for section_id, tokens in batch
                ],
                "batch_total_tokens": sum(tokens for _, tokens in batch),
            }
            formatted_plan.append(formatted_batch)

        return {
            "standard_id": standard_id,
            "loading_plan": formatted_plan,
            "total_batches": len(loading_plan),
            "total_sections": sum(len(batch) for batch in loading_plan),
            "estimated_total_tokens": sum(
                batch["batch_total_tokens"] for batch in formatted_plan
            ),
        }

    async def _estimate_token_usage(
        self, standard_ids: list[str], format_types: list[str] | None = None
    ) -> dict[str, Any]:
        """Estimate token usage for standards in different formats."""
        if not format_types:
            format_types = ["full", "condensed", "reference", "summary"]

        # Convert to enums
        formats = []
        for fmt in format_types:
            try:
                formats.append(StandardFormat(fmt))
            except ValueError:
                continue

        # Load standards
        standards = []
        for std_id in standard_ids:
            try:
                standard = await self._get_standard_details(std_id)
                standards.append(standard)
            except Exception as e:
                logger.warning(f"Failed to load standard {std_id}: {e}")
                continue

        # Estimate for each format
        results = {}
        for format_enum in formats:
            estimates = self.token_optimizer.estimate_tokens(
                standards, format_type=format_enum
            )
            results[format_enum.value] = estimates

        return {
            "estimates": results,
            "recommendations": {
                "tight_budget": "Use 'summary' or 'reference' format",
                "normal_budget": "Use 'condensed' format",
                "large_budget": "Use 'full' format",
            },
        }

    async def _get_sync_status(self) -> dict[str, Any]:
        """Get current synchronization status."""
        status = self.synchronizer.get_sync_status()
        updates = self.synchronizer.check_updates()

        # Get the most recent sync time from all files
        last_sync = None
        if status.get("last_sync_times"):
            sync_times = [t for t in status["last_sync_times"].values() if t]
            if sync_times:
                last_sync = max(sync_times)

        return {
            "last_sync": last_sync,
            "total_standards": status["total_files"],
            "outdated_standards": len(updates.get("outdated_files", [])),
            "cache_size_mb": status["total_size_mb"],
            "rate_limit": status["rate_limit"],
        }

    async def _generate_standard(
        self,
        template_name: str,
        context: dict[str, Any],
        title: str,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Generate a new standard based on template and context."""
        try:
            result = self.generator.generate_standard(
                template_name=template_name, context=context, title=title, domain=domain
            )
            return {
                "standard": result.standard,
                "metadata": result.metadata,
                "warnings": result.warnings,
                "quality_score": result.quality_score,
            }
        except Exception as e:
            return {"error": str(e), "template_name": template_name}

    async def _validate_standard(
        self, standard_content: str, format: str = "yaml"
    ) -> dict[str, Any]:
        """Validate a standard document for completeness and quality."""
        try:
            from .generators.validator import StandardValidator

            validator = StandardValidator()

            # Parse content based on format
            if format.lower() == "yaml":
                import yaml

                content = yaml.safe_load(standard_content)
            else:
                content = json.loads(standard_content)

            validation_result = validator.validate(content)

            return {
                "valid": validation_result.is_valid,
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
                "quality_metrics": validation_result.quality_metrics,
                "completeness_score": validation_result.completeness_score,
                "suggestions": validation_result.suggestions,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def _list_templates(self, domain: str | None = None) -> dict[str, Any]:
        """List available standard templates."""
        try:
            templates = self.generator.list_templates(domain=domain)
            return {
                "templates": [
                    {
                        "name": template.name,
                        "domain": template.domain,
                        "description": template.description,
                        "variables": template.variables,
                        "features": template.features,
                    }
                    for template in templates
                ]
            }
        except Exception as e:
            return {"error": str(e), "templates": []}

    async def _get_cross_references(
        self,
        standard_id: str | None = None,
        concept: str | None = None,
        max_depth: int = 2,
    ) -> dict[str, Any]:
        """Get cross-references for a standard or concept."""
        try:
            if standard_id:
                refs = self.cross_referencer.get_references_for_standard(
                    standard_id, max_depth=max_depth
                )
            elif concept:
                refs = self.cross_referencer.find_concept_references(
                    concept, max_depth=max_depth
                )
            else:
                return {"error": "Either standard_id or concept must be provided"}

            return {"references": refs, "depth": max_depth, "total_found": len(refs)}
        except Exception as e:
            return {"error": str(e), "references": []}

    async def _generate_cross_references(
        self, force_refresh: bool = False
    ) -> dict[str, Any]:
        """Generate cross-references between standards."""
        try:
            result = self.cross_referencer.generate_cross_references(
                force_refresh=force_refresh
            )
            return {
                "status": "completed",
                "processed_standards": result.processed_count,
                "new_references": result.new_references_count,
                "updated_references": result.updated_references_count,
                "processing_time": result.processing_time,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _get_standards_analytics(
        self,
        metric_type: str = "usage",
        time_range: str = "30d",
        standard_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Get analytics and usage statistics for standards."""
        try:
            if metric_type == "usage":
                result = self.analytics.get_usage_metrics(
                    time_range=time_range, standard_ids=standard_ids
                )
            elif metric_type == "popularity":
                result = self.analytics.get_popularity_metrics(
                    time_range=time_range, standard_ids=standard_ids
                )
            elif metric_type == "gaps":
                result = self.analytics.analyze_coverage_gaps(standard_ids=standard_ids)
            else:
                return {"error": f"Unknown metric type: {metric_type}"}

            return {
                "metric_type": metric_type,
                "time_range": time_range,
                "data": result,
            }
        except Exception as e:
            return {"error": str(e), "metric_type": metric_type}

    async def _track_standards_usage(
        self,
        standard_id: str,
        usage_type: str,
        section_id: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Track usage of specific standards or sections."""
        try:
            self.analytics.track_usage(
                standard_id=standard_id,
                usage_type=usage_type,
                section_id=section_id,
                context=context or {},
            )
            return {
                "status": "tracked",
                "standard_id": standard_id,
                "usage_type": usage_type,
                "section_id": section_id,
            }
        except Exception as e:
            return {"status": "failed", "error": str(e)}

    async def _get_recommendations(
        self, analysis_type: str = "gaps", context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Get recommendations for standards improvement or gaps."""
        try:
            if analysis_type == "gaps":
                recommendations = self.analytics.get_gap_recommendations(
                    context=context
                )
            elif analysis_type == "quality":
                recommendations = self.analytics.get_quality_recommendations(
                    context=context
                )
            elif analysis_type == "usage":
                recommendations = self.analytics.get_usage_recommendations(
                    context=context
                )
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}

            return {
                "analysis_type": analysis_type,
                "recommendations": recommendations,
                "generated_at": self.analytics.get_current_timestamp(),
            }
        except Exception as e:
            return {"error": str(e), "analysis_type": analysis_type}

    async def _get_metrics_dashboard(self) -> dict[str, Any]:
        """Get performance metrics dashboard."""
        return self.metrics.get_dashboard_metrics()

    def _get_semantic_search_engine(self):
        """Get semantic search engine instance."""
        return self.search

    def _validate_input(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate input data."""
        return self.input_validator.validate(data)

    def _filter_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Filter response for privacy."""
        return self.privacy_filter.filter(response)

    async def _cross_reference_standards(self, standard_id: str) -> dict[str, Any]:
        """Get cross-references for a standard."""
        return await self.cross_referencer.get_references(standard_id)

    async def _get_analytics(self) -> dict[str, Any]:
        """Get analytics data."""
        return await self.analytics.get_analytics()

    async def _get_compliance_mapping(
        self, standard_id: str, framework: str = "nist"
    ) -> dict[str, Any]:
        """Get compliance mapping for a standard."""
        # This is a placeholder implementation
        return {"standard_id": standard_id, "framework": framework, "mappings": []}

    @property
    def rate_limiter(self):
        """Get rate limiter for backwards compatibility."""
        return self

    def _check_rate_limit(self, user_key: str) -> bool:
        """Check if user is within rate limits."""
        now = time.time()

        # Clean up old entries
        if user_key in self._rate_limit_store:
            self._rate_limit_store[user_key] = [
                timestamp
                for timestamp in self._rate_limit_store[user_key]
                if now - timestamp < self.rate_limit_window
            ]
        else:
            self._rate_limit_store[user_key] = []

        # Check if limit exceeded
        if len(self._rate_limit_store[user_key]) >= self.rate_limit_max_requests:
            return False

        # Add current request
        self._rate_limit_store[user_key].append(now)
        return True

    async def run(self):
        """Run the MCP server."""
        # Start metrics export task
        await self.metrics.collector.start_export_task()

        # Track connection
        self._active_connections += 1
        self.metrics.update_active_connections(self._active_connections)

        try:
            async with stdio_server() as (read_stream, write_stream):
                init_options = self.server.create_initialization_options()
                await self.server.run(read_stream, write_stream, init_options)
        finally:
            # Stop metrics export task
            await self.metrics.collector.stop_export_task()

            # Update active connections
            self._active_connections = max(0, self._active_connections - 1)
            self.metrics.update_active_connections(self._active_connections)


async def main():
    """Main entry point for MCP server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Load configuration
    config = {}
    config_path = os.environ.get("MCP_CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)

    # Create and run server
    server = MCPStandardsServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())
