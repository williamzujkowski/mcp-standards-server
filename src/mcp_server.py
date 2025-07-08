"""
MCP Standards Server implementation.

Provides Model Context Protocol interface to standards functionality.
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, CallToolResult, InitializeRequest

from .core.standards.rule_engine import RuleEngine
from .core.standards.sync import StandardsSynchronizer
from .core.standards.token_optimizer import (
    TokenOptimizer,
    TokenBudget,
    StandardFormat,
    ModelType,
    DynamicLoader,
    create_token_optimizer
)
from .core.standards.cross_referencer import CrossReferencer
from .core.standards.analytics import StandardsAnalytics
from .generators.engine import StandardsGenerator


logger = logging.getLogger(__name__)


class MCPStandardsServer:
    """MCP server for standards management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.server = Server("mcp-standards-server")
        
        # Initialize components
        # Get data directory
        data_dir = Path(os.environ.get("MCP_STANDARDS_DATA_DIR", "data"))
        rules_file = data_dir / "standards" / "meta" / "enhanced-selection-rules.json"
        
        self.rule_engine = RuleEngine(
            Path(self.config.get("rules_file", str(rules_file)))
        )
        sync_config_path = self.config.get("sync_config", data_dir / "standards" / "sync_config.yaml")
        self.synchronizer = StandardsSynchronizer(
            config_path=sync_config_path,
            cache_dir=data_dir / "standards" / "cache"
        )
        
        # Initialize cross-referencer and analytics
        self.cross_referencer = CrossReferencer(data_dir / "standards")
        self.analytics = StandardsAnalytics(data_dir / "standards" / "analytics")
        
        # Initialize standards generator
        templates_dir = Path("templates")
        self.generator = StandardsGenerator(templates_dir)
        
        # Initialize search only if enabled
        self.search = None
        if os.environ.get("MCP_DISABLE_SEARCH") != "true" and self.config.get("search", {}).get("enabled", True):
            try:
                from .core.standards.semantic_search import SemanticSearch
                self.search = SemanticSearch(
                    model_name=self.config.get("search_model", "sentence-transformers/all-MiniLM-L6-v2")
                )
            except ImportError:
                logger.warning("Semantic search disabled: sentence-transformers not installed")
        
        # Initialize token optimizer
        model_type = ModelType(self.config.get("token_model", "gpt-4"))
        default_budget = self.config.get("default_token_budget", 8000)
        self.token_optimizer = create_token_optimizer(
            model_type=model_type,
            default_budget=default_budget
        )
        self.dynamic_loader = DynamicLoader(self.token_optimizer)
        
        # Register tools
        self._register_tools()
        
    def _register_tools(self):
        """Register all MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
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
                                "description": "Project context information"
                            },
                            "include_resolution_details": {
                                "type": "boolean",
                                "description": "Include detailed resolution information",
                                "default": False
                            }
                        },
                        "required": ["context"]
                    }
                ),
                Tool(
                    name="validate_against_standard",
                    description="Validate code against a specific standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to validate"},
                            "standard": {"type": "string", "description": "Standard ID"},
                            "language": {"type": "string", "description": "Programming language"}
                        },
                        "required": ["code", "standard"]
                    }
                ),
                Tool(
                    name="search_standards",
                    description="Search standards using semantic search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Maximum results", "default": 10},
                            "min_relevance": {"type": "number", "description": "Minimum relevance score", "default": 0.0},
                            "filters": {"type": "object", "description": "Additional filters"}
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="get_standard_details",
                    description="Get detailed information about a specific standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {"type": "string", "description": "Standard identifier"}
                        },
                        "required": ["standard_id"]
                    }
                ),
                Tool(
                    name="list_available_standards",
                    description="List all available standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "category": {"type": "string", "description": "Filter by category"},
                            "limit": {"type": "integer", "description": "Maximum results", "default": 100}
                        }
                    }
                ),
                Tool(
                    name="suggest_improvements",
                    description="Suggest improvements based on applicable standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Code to analyze"},
                            "context": {"type": "object", "description": "Project context"}
                        },
                        "required": ["code", "context"]
                    }
                ),
                Tool(
                    name="sync_standards",
                    description="Synchronize standards from repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force": {"type": "boolean", "description": "Force sync", "default": False}
                        }
                    }
                ),
                Tool(
                    name="get_optimized_standard",
                    description="Get a token-optimized version of a standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {"type": "string", "description": "Standard ID"},
                            "format_type": {"type": "string", "description": "Format type", "default": "condensed"},
                            "token_budget": {"type": "integer", "description": "Token budget"},
                            "required_sections": {"type": "array", "items": {"type": "string"}},
                            "context": {"type": "object", "description": "Additional context"}
                        },
                        "required": ["standard_id"]
                    }
                ),
                Tool(
                    name="auto_optimize_standards",
                    description="Automatically optimize multiple standards within a token budget",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_ids": {"type": "array", "items": {"type": "string"}},
                            "total_token_budget": {"type": "integer", "description": "Total token budget"},
                            "context": {"type": "object", "description": "Additional context"}
                        },
                        "required": ["standard_ids", "total_token_budget"]
                    }
                ),
                Tool(
                    name="progressive_load_standard",
                    description="Get a progressive loading plan for a standard",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {"type": "string", "description": "Standard ID"},
                            "initial_sections": {"type": "array", "items": {"type": "string"}},
                            "max_depth": {"type": "integer", "description": "Maximum depth", "default": 3}
                        },
                        "required": ["standard_id", "initial_sections"]
                    }
                ),
                Tool(
                    name="estimate_token_usage",
                    description="Estimate token usage for standards in different formats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_ids": {"type": "array", "items": {"type": "string"}},
                            "format_types": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["standard_ids"]
                    }
                ),
                Tool(
                    name="get_sync_status",
                    description="Get current synchronization status",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="generate_standard",
                    description="Generate a new standard based on template and context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "template_name": {"type": "string", "description": "Template name"},
                            "context": {"type": "object", "description": "Generation context"},
                            "domain": {"type": "string", "description": "Domain/category"},
                            "title": {"type": "string", "description": "Standard title"}
                        },
                        "required": ["template_name", "context", "title"]
                    }
                ),
                Tool(
                    name="validate_standard",
                    description="Validate a standard document for completeness and quality",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_content": {"type": "string", "description": "Standard content to validate"},
                            "format": {"type": "string", "description": "Content format (yaml/json)", "default": "yaml"}
                        },
                        "required": ["standard_content"]
                    }
                ),
                Tool(
                    name="list_templates",
                    description="List available standard templates",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "domain": {"type": "string", "description": "Filter by domain"}
                        }
                    }
                ),
                Tool(
                    name="get_cross_references",
                    description="Get cross-references for a standard or concept",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {"type": "string", "description": "Standard ID"},
                            "concept": {"type": "string", "description": "Concept to find references for"},
                            "max_depth": {"type": "integer", "description": "Maximum reference depth", "default": 2}
                        }
                    }
                ),
                Tool(
                    name="generate_cross_references",
                    description="Generate cross-references between standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "force_refresh": {"type": "boolean", "description": "Force refresh of references", "default": false}
                        }
                    }
                ),
                Tool(
                    name="get_standards_analytics",
                    description="Get analytics and usage statistics for standards",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "metric_type": {"type": "string", "description": "Type of metrics (usage/popularity/gaps)", "default": "usage"},
                            "time_range": {"type": "string", "description": "Time range for metrics", "default": "30d"},
                            "standard_ids": {"type": "array", "items": {"type": "string"}, "description": "Specific standards"}
                        }
                    }
                ),
                Tool(
                    name="track_standards_usage",
                    description="Track usage of specific standards or sections",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "standard_id": {"type": "string", "description": "Standard ID"},
                            "section_id": {"type": "string", "description": "Section ID"},
                            "usage_type": {"type": "string", "description": "Usage type (view/apply/reference)"},
                            "context": {"type": "object", "description": "Usage context"}
                        },
                        "required": ["standard_id", "usage_type"]
                    }
                ),
                Tool(
                    name="get_recommendations",
                    description="Get recommendations for standards improvement or gaps",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "analysis_type": {"type": "string", "description": "Type of analysis (gaps/quality/usage)", "default": "gaps"},
                            "context": {"type": "object", "description": "Analysis context"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls."""
            try:
                if name == "get_applicable_standards":
                    result = await self._get_applicable_standards(
                        arguments["context"],
                        arguments.get("include_resolution_details", False)
                    )
                elif name == "validate_against_standard":
                    result = await self._validate_against_standard(
                        arguments["code"],
                        arguments["standard"],
                        arguments.get("language")
                    )
                elif name == "search_standards":
                    result = await self._search_standards(
                        arguments["query"],
                        arguments.get("limit", 10),
                        arguments.get("min_relevance", 0.0),
                        arguments.get("filters")
                    )
                elif name == "get_standard_details":
                    result = await self._get_standard_details(
                        arguments["standard_id"]
                    )
                elif name == "list_available_standards":
                    result = await self._list_available_standards(
                        arguments.get("category"),
                        arguments.get("limit", 100)
                    )
                elif name == "suggest_improvements":
                    result = await self._suggest_improvements(
                        arguments["code"],
                        arguments["context"]
                    )
                elif name == "sync_standards":
                    result = await self._sync_standards(
                        arguments.get("force", False)
                    )
                elif name == "get_optimized_standard":
                    result = await self._get_optimized_standard(
                        arguments["standard_id"],
                        arguments.get("format_type", "condensed"),
                        arguments.get("token_budget"),
                        arguments.get("required_sections"),
                        arguments.get("context")
                    )
                elif name == "auto_optimize_standards":
                    result = await self._auto_optimize_standards(
                        arguments["standard_ids"],
                        arguments["total_token_budget"],
                        arguments.get("context")
                    )
                elif name == "progressive_load_standard":
                    result = await self._progressive_load_standard(
                        arguments["standard_id"],
                        arguments["initial_sections"],
                        arguments.get("max_depth", 3)
                    )
                elif name == "estimate_token_usage":
                    result = await self._estimate_token_usage(
                        arguments["standard_ids"],
                        arguments.get("format_types")
                    )
                elif name == "get_sync_status":
                    result = await self._get_sync_status()
                elif name == "generate_standard":
                    result = await self._generate_standard(
                        arguments["template_name"],
                        arguments["context"],
                        arguments["title"],
                        arguments.get("domain")
                    )
                elif name == "validate_standard":
                    result = await self._validate_standard(
                        arguments["standard_content"],
                        arguments.get("format", "yaml")
                    )
                elif name == "list_templates":
                    result = await self._list_templates(
                        arguments.get("domain")
                    )
                elif name == "get_cross_references":
                    result = await self._get_cross_references(
                        arguments.get("standard_id"),
                        arguments.get("concept"),
                        arguments.get("max_depth", 2)
                    )
                elif name == "generate_cross_references":
                    result = await self._generate_cross_references(
                        arguments.get("force_refresh", False)
                    )
                elif name == "get_standards_analytics":
                    result = await self._get_standards_analytics(
                        arguments.get("metric_type", "usage"),
                        arguments.get("time_range", "30d"),
                        arguments.get("standard_ids")
                    )
                elif name == "track_standards_usage":
                    result = await self._track_standards_usage(
                        arguments["standard_id"],
                        arguments["usage_type"],
                        arguments.get("section_id"),
                        arguments.get("context")
                    )
                elif name == "get_recommendations":
                    result = await self._get_recommendations(
                        arguments.get("analysis_type", "gaps"),
                        arguments.get("context")
                    )
                else:
                    raise ValueError(f"Unknown tool: {name}")
                
                return [TextContent(
                    type="text",
                    text=json.dumps(result, indent=2)
                )]
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "tool": name
                    })
                )]
    
    async def _get_applicable_standards(
        self,
        context: Dict[str, Any],
        include_resolution_details: bool = False
    ) -> Dict[str, Any]:
        """Get applicable standards based on project context."""
        result = self.rule_engine.evaluate(context)
        
        response = {
            "standards": result["resolved_standards"],
            "evaluation_path": result.get("evaluation_path", [])
        }
        
        if include_resolution_details:
            response["resolution_details"] = {
                "matched_rules": result.get("matched_rules", []),
                "conflicts_resolved": result.get("conflicts_resolved", 0),
                "final_priority_order": result.get("priority_order", [])
            }
            
        return response
        
    async def _validate_against_standard(
        self,
        code: str,
        standard: str,
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate code against a specific standard."""
        # This is a placeholder - actual implementation would analyze code
        violations = []
        
        # Simple validation checks
        if standard == "react-18-patterns" and language == "javascript":
            if "class " in code and "extends React.Component" in code:
                violations.append({
                    "line": 1,
                    "message": "Prefer functional components over class components",
                    "severity": "warning"
                })
                
        return {
            "standard": standard,
            "passed": len(violations) == 0,
            "violations": violations
        }
        
    async def _search_standards(
        self,
        query: str,
        limit: int = 10,
        min_relevance: float = 0.0,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Search standards using semantic search."""
        if self.search is None:
            # Fallback to simple keyword search
            return {"results": [], "warning": "Semantic search is disabled"}
        
        results = await self.search.search(
            query, 
            limit=limit,
            filters=filters
        )
        
        # Filter by minimum relevance
        filtered_results = [
            r for r in results 
            if r.get("relevance_score", 0) >= min_relevance
        ]
        
        return {"results": filtered_results}
        
    async def _get_standard_details(
        self,
        standard_id: str
    ) -> Dict[str, Any]:
        """Get detailed information about a specific standard."""
        # Load standard from cache or repository
        standard_path = self.synchronizer.cache_dir / f"{standard_id}.json"
        
        if standard_path.exists():
            with open(standard_path, 'r') as f:
                return json.load(f)
        else:
            raise ValueError(f"Standard '{standard_id}' not found")
            
    async def _list_available_standards(
        self,
        category: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """List all available standards."""
        standards = []
        
        # List cached standards
        for file_path in self.synchronizer.cache_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    standard = json.load(f)
                    
                if category is None or standard.get("category") == category:
                    standards.append({
                        "id": standard["id"],
                        "name": standard["name"],
                        "category": standard.get("category"),
                        "tags": standard.get("tags", [])
                    })
            except Exception:
                continue
                
        return {"standards": standards[:limit]}
        
    async def _suggest_improvements(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    suggestions.append({
                        "description": "Use const/let instead of var",
                        "priority": "high",
                        "standard_reference": standard_id
                    })
                if "function" in code and not "async" in code:
                    suggestions.append({
                        "description": "Consider using async/await for asynchronous operations",
                        "priority": "medium",
                        "standard_reference": standard_id
                    })
                if ".then(" in code:
                    suggestions.append({
                        "description": "Consider using async/await instead of promise chains",
                        "priority": "medium",
                        "standard_reference": standard_id
                    })
                    
        return {"suggestions": suggestions}
        
    async def _sync_standards(
        self,
        force: bool = False
    ) -> Dict[str, Any]:
        """Synchronize standards from repository."""
        try:
            result = await self.synchronizer.sync(force=force)
            return {
                "status": result.status.value,
                "synced_files": [f.path for f in result.synced_files],
                "failed_files": result.failed_files,
                "message": result.message
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
            
    async def _get_optimized_standard(
        self,
        standard_id: str,
        format_type: str = "condensed",
        token_budget: Optional[int] = None,
        required_sections: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
            context=context
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
            "warnings": result.warnings
        }
        
    async def _auto_optimize_standards(
        self,
        standard_ids: List[str],
        total_token_budget: int,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Automatically optimize multiple standards within a token budget."""
        # Estimate token usage for each standard
        standards = []
        for std_id in standard_ids:
            try:
                standard = await self._get_standard_details(std_id)
                standards.append(standard)
            except Exception:
                continue
        
        # Estimate token distribution
        estimates = self.token_optimizer.estimate_tokens(
            standards,
            format_type=StandardFormat.CONDENSED
        )
        
        # Allocate budget proportionally
        total_original = estimates['total_original']
        results = []
        
        for i, standard in enumerate(standards):
            std_estimate = estimates['standards'][i]
            
            # Calculate proportional budget
            proportion = std_estimate['original_tokens'] / total_original
            allocated_budget = int(total_token_budget * proportion)
            
            # Create budget with some buffer
            budget = TokenBudget(total=allocated_budget)
            
            # Auto-select format
            selected_format = self.token_optimizer.auto_select_format(
                standard, budget, context
            )
            
            # Optimize
            content, result = self.token_optimizer.optimize_standard(
                standard,
                format_type=selected_format,
                budget=budget,
                context=context
            )
            
            results.append({
                "standard_id": standard['id'],
                "content": content,
                "format": result.format_used.value,
                "tokens_used": result.compressed_tokens,
                "allocated_budget": allocated_budget
            })
        
        total_used = sum(r['tokens_used'] for r in results)
        
        return {
            "results": results,
            "total_tokens_used": total_used,
            "total_budget": total_token_budget,
            "budget_utilization": total_used / total_token_budget if total_token_budget > 0 else 0
        }
        
    async def _progressive_load_standard(
        self,
        standard_id: str,
        initial_sections: List[str],
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Get a progressive loading plan for a standard."""
        # Load standard
        standard = await self._get_standard_details(standard_id)
        
        # Generate loading plan
        loading_plan = self.token_optimizer.progressive_load(
            standard,
            initial_sections=initial_sections,
            max_depth=max_depth
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
                "batch_total_tokens": sum(tokens for _, tokens in batch)
            }
            formatted_plan.append(formatted_batch)
        
        return {
            "standard_id": standard_id,
            "loading_plan": formatted_plan,
            "total_batches": len(loading_plan),
            "total_sections": sum(len(batch) for batch in loading_plan),
            "estimated_total_tokens": sum(
                batch["batch_total_tokens"] for batch in formatted_plan
            )
        }
        
    async def _estimate_token_usage(
        self,
        standard_ids: List[str],
        format_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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
            except Exception:
                continue
        
        # Estimate for each format
        results = {}
        for format_enum in formats:
            estimates = self.token_optimizer.estimate_tokens(
                standards,
                format_type=format_enum
            )
            results[format_enum.value] = estimates
        
        return {
            "estimates": results,
            "recommendations": {
                "tight_budget": "Use 'summary' or 'reference' format",
                "normal_budget": "Use 'condensed' format",
                "large_budget": "Use 'full' format"
            }
        }
            
    async def _get_sync_status(self) -> Dict[str, Any]:
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
            "rate_limit": status["rate_limit"]
        }
        
    async def _generate_standard(
        self,
        template_name: str,
        context: Dict[str, Any],
        title: str,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a new standard based on template and context."""
        try:
            result = self.generator.generate_standard(
                template_name=template_name,
                context=context,
                title=title,
                domain=domain
            )
            return {
                "standard": result.standard,
                "metadata": result.metadata,
                "warnings": result.warnings,
                "quality_score": result.quality_score
            }
        except Exception as e:
            return {
                "error": str(e),
                "template_name": template_name
            }
    
    async def _validate_standard(
        self,
        standard_content: str,
        format: str = "yaml"
    ) -> Dict[str, Any]:
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
                "suggestions": validation_result.suggestions
            }
        except Exception as e:
            return {
                "valid": False,
                "error": str(e)
            }
    
    async def _list_templates(
        self,
        domain: Optional[str] = None
    ) -> Dict[str, Any]:
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
                        "features": template.features
                    }
                    for template in templates
                ]
            }
        except Exception as e:
            return {
                "error": str(e),
                "templates": []
            }
    
    async def _get_cross_references(
        self,
        standard_id: Optional[str] = None,
        concept: Optional[str] = None,
        max_depth: int = 2
    ) -> Dict[str, Any]:
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
            
            return {
                "references": refs,
                "depth": max_depth,
                "total_found": len(refs)
            }
        except Exception as e:
            return {
                "error": str(e),
                "references": []
            }
    
    async def _generate_cross_references(
        self,
        force_refresh: bool = False
    ) -> Dict[str, Any]:
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
                "processing_time": result.processing_time
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_standards_analytics(
        self,
        metric_type: str = "usage",
        time_range: str = "30d",
        standard_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get analytics and usage statistics for standards."""
        try:
            if metric_type == "usage":
                result = self.analytics.get_usage_metrics(
                    time_range=time_range,
                    standard_ids=standard_ids
                )
            elif metric_type == "popularity":
                result = self.analytics.get_popularity_metrics(
                    time_range=time_range,
                    standard_ids=standard_ids
                )
            elif metric_type == "gaps":
                result = self.analytics.analyze_coverage_gaps(
                    standard_ids=standard_ids
                )
            else:
                return {"error": f"Unknown metric type: {metric_type}"}
            
            return {
                "metric_type": metric_type,
                "time_range": time_range,
                "data": result
            }
        except Exception as e:
            return {
                "error": str(e),
                "metric_type": metric_type
            }
    
    async def _track_standards_usage(
        self,
        standard_id: str,
        usage_type: str,
        section_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track usage of specific standards or sections."""
        try:
            self.analytics.track_usage(
                standard_id=standard_id,
                usage_type=usage_type,
                section_id=section_id,
                context=context or {}
            )
            return {
                "status": "tracked",
                "standard_id": standard_id,
                "usage_type": usage_type,
                "section_id": section_id
            }
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_recommendations(
        self,
        analysis_type: str = "gaps",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
                "generated_at": self.analytics.get_current_timestamp()
            }
        except Exception as e:
            return {
                "error": str(e),
                "analysis_type": analysis_type
            }
            
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            init_options = self.server.create_initialization_options()
            await self.server.run(read_stream, write_stream, init_options)


async def main():
    """Main entry point for MCP server."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config = {}
    config_path = os.environ.get("MCP_CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
    # Create and run server
    server = MCPStandardsServer(config)
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())