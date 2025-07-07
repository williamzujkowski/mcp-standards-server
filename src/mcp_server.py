"""
MCP Standards Server implementation.

Provides Model Context Protocol interface to standards functionality.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

from mcp import Tool, Resource, Server
from mcp.server.stdio import stdio_server

from .core.standards.rule_engine import RuleEngine
from .core.standards.sync import StandardsSynchronizer
from .core.standards.semantic_search import SemanticSearch
from .core.standards.token_optimizer import (
    TokenOptimizer,
    TokenBudget,
    StandardFormat,
    ModelType,
    DynamicLoader,
    create_token_optimizer
)


logger = logging.getLogger(__name__)


class MCPStandardsServer:
    """MCP server for standards management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.server = Server("mcp-standards-server")
        
        # Initialize components
        self.rule_engine = RuleEngine(
            self.config.get("rules_file", "data/standards/meta/standard-selection-rules.json")
        )
        self.synchronizer = StandardsSynchronizer(
            config_path=self.config.get("sync_config")
        )
        self.search = SemanticSearch(
            model_name=self.config.get("search_model", "sentence-transformers/all-MiniLM-L6-v2")
        )
        
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
        
        @self.server.tool()
        async def get_applicable_standards(
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
            
        @self.server.tool()
        async def validate_against_standard(
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
            
        @self.server.tool()
        async def search_standards(
            query: str,
            limit: int = 10,
            min_relevance: float = 0.0,
            filters: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Search standards using semantic search."""
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
            
        @self.server.tool()
        async def get_standard_details(
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
                
        @self.server.tool()
        async def list_available_standards(
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
            
        @self.server.tool()
        async def suggest_improvements(
            code: str,
            context: Dict[str, Any]
        ) -> Dict[str, Any]:
            """Suggest improvements based on applicable standards."""
            # Get applicable standards
            standards_result = await get_applicable_standards(context)
            applicable_standards = standards_result["standards"]
            
            suggestions = []
            
            # Generate suggestions based on standards
            for standard_id in applicable_standards:
                # This is a placeholder - actual implementation would analyze code
                if "react" in standard_id.lower() and "javascript" in context.get("language", "").lower():
                    if "var " in code:
                        suggestions.append({
                            "description": "Use const/let instead of var",
                            "priority": "high",
                            "standard_reference": standard_id
                        })
                        
            return {"suggestions": suggestions}
            
        @self.server.tool()
        async def sync_standards(
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
                
        @self.server.tool()
        async def get_optimized_standard(
            standard_id: str,
            format_type: str = "condensed",
            token_budget: Optional[int] = None,
            required_sections: Optional[List[str]] = None,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Get a token-optimized version of a standard."""
            # Load standard
            standard = await get_standard_details(standard_id)
            
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
            
        @self.server.tool()
        async def auto_optimize_standards(
            standard_ids: List[str],
            total_token_budget: int,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """Automatically optimize multiple standards within a token budget."""
            # Estimate token usage for each standard
            standards = []
            for std_id in standard_ids:
                try:
                    standard = await get_standard_details(std_id)
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
            
        @self.server.tool()
        async def progressive_load_standard(
            standard_id: str,
            initial_sections: List[str],
            max_depth: int = 3
        ) -> Dict[str, Any]:
            """Get a progressive loading plan for a standard."""
            # Load standard
            standard = await get_standard_details(standard_id)
            
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
            
        @self.server.tool()
        async def estimate_token_usage(
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
                    standard = await get_standard_details(std_id)
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
                
        @self.server.tool()
        async def get_sync_status() -> Dict[str, Any]:
            """Get current synchronization status."""
            status = self.synchronizer.get_sync_status()
            updates = self.synchronizer.check_updates()
            
            return {
                "last_sync": status.get("last_sync_time"),
                "total_standards": status["total_files"],
                "outdated_standards": len(updates["outdated_files"]),
                "cache_size_mb": status["total_size_mb"],
                "rate_limit": status["rate_limit"]
            }
            
    async def run(self):
        """Run the MCP server."""
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)


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