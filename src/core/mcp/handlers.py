"""MCP handlers for standards operations."""

import logging
import tempfile
from typing import Any

logger = logging.getLogger(__name__)


class StandardsHandler:
    """Handler for standards-related MCP operations."""

    def __init__(self, standards_engine: Any) -> None:
        self.standards_engine = standards_engine

    async def initialize(self) -> None:
        """Initialize the handler."""
        if self.standards_engine and hasattr(self.standards_engine, "initialize"):
            await self.standards_engine.initialize()

    async def cleanup(self) -> None:
        """Cleanup handler resources."""
        if self.standards_engine and hasattr(self.standards_engine, "close"):
            await self.standards_engine.close()

    async def get_tools(self) -> list[dict[str, Any]]:
        """Get available tools."""
        return [
            {
                "name": "list_available_standards",
                "description": "List all available standards with optional filtering",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "Filter by category",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter by tags",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 100,
                        },
                    },
                },
            },
            {
                "name": "get_applicable_standards",
                "description": "Get standards applicable to a project context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_context": {
                            "type": "object",
                            "description": "Project context for standard selection",
                        }
                    },
                    "required": ["project_context"],
                },
            },
            {
                "name": "search_standards",
                "description": "Search for standards using semantic search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 10,
                        },
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_standard",
                "description": "Get a specific standard by ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "standard_id": {"type": "string", "description": "Standard ID"},
                        "version": {
                            "type": "string",
                            "description": "Standard version (optional)",
                        },
                    },
                    "required": ["standard_id"],
                },
            },
            {
                "name": "get_optimized_standard",
                "description": "Get an optimized/condensed version of a standard",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "standard_id": {"type": "string", "description": "Standard ID"},
                        "format": {
                            "type": "string",
                            "description": "Format type (full, condensed, reference)",
                            "default": "condensed",
                        },
                    },
                    "required": ["standard_id"],
                },
            },
            {
                "name": "validate_against_standard",
                "description": "Validate code or configuration against a standard",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "standard_id": {"type": "string", "description": "Standard ID"},
                        "code": {"type": "string", "description": "Code to validate"},
                        "file_path": {
                            "type": "string",
                            "description": "Path to file to validate",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (auto-detected if not provided)",
                            "enum": [
                                "python",
                                "javascript",
                                "typescript",
                                "go",
                                "java",
                                "rust",
                            ],
                        },
                    },
                    "required": ["standard_id"],
                },
            },
            {
                "name": "get_compliance_mapping",
                "description": "Get NIST control mappings for standards",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "standard_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of standard IDs",
                        },
                        "control_id": {
                            "type": "string",
                            "description": "Filter by specific NIST control ID",
                        },
                    },
                },
            },
        ]

    async def handle_tool(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a tool call."""
        if not self.standards_engine:
            return {"error": "Standards engine not available"}

        try:
            if tool_name == "list_available_standards":
                result = await self.standards_engine.list_standards(
                    category=args.get("category"),
                    tags=args.get("tags"),
                    limit=args.get("limit", 100),
                )
                return {"result": result}

            elif tool_name == "get_applicable_standards":
                result = await self.standards_engine.get_applicable_standards(
                    args.get("project_context", {})
                )
                return {"result": result}

            elif tool_name == "search_standards":
                result = await self.standards_engine.search_standards(
                    query=args.get("query", ""), limit=args.get("limit", 10)
                )
                return {"result": result}

            elif tool_name == "get_standard":
                result = await self.standards_engine.get_standard(
                    standard_id=args.get("standard_id"), version=args.get("version")
                )
                return {"result": result}

            elif tool_name == "get_optimized_standard":
                standard = await self.standards_engine.get_standard(
                    standard_id=args.get("standard_id")
                )
                if not standard:
                    return {"error": f"Standard not found: {args.get('standard_id')}"}

                # Apply token optimization if available
                if (
                    hasattr(self.standards_engine, "token_optimizer")
                    and self.standards_engine.token_optimizer
                ):
                    format_type_str = args.get("format", "condensed")

                    # Import and convert to enum
                    from ..standards.token_optimizer import StandardFormat

                    # Map string to enum
                    format_map = {
                        "full": StandardFormat.FULL,
                        "condensed": StandardFormat.CONDENSED,
                        "reference": StandardFormat.REFERENCE,
                        "summary": StandardFormat.SUMMARY,
                    }
                    format_type = format_map.get(
                        format_type_str, StandardFormat.CONDENSED
                    )

                    # Convert Standard to dict for optimizer
                    standard_dict = {
                        "id": standard.id,
                        "title": standard.title,
                        "description": standard.description,
                        "content": standard.content,
                        "category": standard.category,
                        "tags": standard.tags,
                        "version": standard.version,
                        "metadata": (
                            standard.metadata.__dict__
                            if hasattr(standard.metadata, "__dict__")
                            else {}
                        ),
                    }
                    optimized_content, compression_result = (
                        self.standards_engine.token_optimizer.optimize_standard(
                            standard_dict, format_type
                        )
                    )
                    return {
                        "result": {
                            "id": standard.id,
                            "title": standard.title,
                            "content": optimized_content,
                            "format": format_type_str,
                            "compression": (
                                compression_result.__dict__
                                if hasattr(compression_result, "__dict__")
                                else compression_result
                            ),
                        }
                    }
                else:
                    # Return standard as-is if no optimizer
                    return {"result": standard}

            elif tool_name == "validate_against_standard":
                standard_id = args.get("standard_id")
                code = args.get("code")
                file_path = args.get("file_path")
                language = args.get("language")

                if not standard_id:
                    return {"error": "standard_id is required"}

                # Validate code using analyzer infrastructure
                try:
                    from pathlib import Path

                    from ...analyzers.base import AnalyzerPlugin

                    if code and not file_path:
                        # Create temporary file for code analysis
                        extension_map = {
                            "python": ".py",
                            "typescript": ".ts",
                            "javascript": ".js",
                            "go": ".go",
                            "java": ".java",
                            "rust": ".rs",
                        }

                        # Auto-detect language if not provided
                        if not language and file_path:
                            ext = Path(file_path).suffix
                            for lang, lang_ext in extension_map.items():
                                if ext == lang_ext:
                                    language = lang
                                    break

                        if not language:
                            language = "typescript"  # Default fallback

                        extension = extension_map.get(language, ".txt")
                        with tempfile.NamedTemporaryFile(
                            mode="w", suffix=extension, delete=False
                        ) as f:
                            f.write(code)
                            file_path = f.name

                    if not file_path:
                        return {
                            "error": "Either 'code' or 'file_path' must be provided"
                        }

                    # Auto-detect language from file extension if not provided
                    if not language:
                        ext = Path(file_path).suffix
                        for lang in AnalyzerPlugin.list_languages():
                            analyzer = AnalyzerPlugin.get_analyzer(lang)
                            if analyzer and ext in analyzer.file_extensions:
                                language = lang
                                break

                        if not language:
                            return {
                                "error": f"Could not detect language for file: {file_path}"
                            }

                    # Get analyzer for language
                    analyzer = AnalyzerPlugin.get_analyzer(language)
                    if not analyzer:
                        return {
                            "error": f"No analyzer available for language: {language}"
                        }

                    # Run analysis
                    if code and hasattr(analyzer, "analyze_code"):
                        # Direct code analysis
                        analysis_result = analyzer.analyze_code(code, file_path)
                    else:
                        # File-based analysis
                        analysis_result = analyzer.analyze_file(Path(file_path))

                    # Format results for MCP response
                    issues = []
                    warnings = []

                    for issue in analysis_result.issues:
                        issue_dict = {
                            "type": issue.type.value,
                            "severity": issue.severity.value,
                            "message": issue.message,
                            "line": issue.line_number,
                            "column": issue.column_number,
                            "recommendation": issue.recommendation,
                        }

                        if hasattr(issue, "cwe_id") and issue.cwe_id:
                            issue_dict["cwe_id"] = issue.cwe_id
                        if hasattr(issue, "owasp_category") and issue.owasp_category:
                            issue_dict["owasp_category"] = issue.owasp_category

                        if issue.severity.value in ["critical", "high"]:
                            issues.append(issue_dict)
                        else:
                            warnings.append(issue_dict)

                    result = {
                        "standard_id": standard_id,
                        "language": language,
                        "valid": len(issues) == 0,  # Valid if no critical/high issues
                        "issues": issues,
                        "warnings": warnings,
                        "metrics": analysis_result.metrics,
                        "analysis_time": analysis_result.analysis_time,
                        "summary": {
                            "total_issues": len(analysis_result.issues),
                            "critical_issues": len(
                                [
                                    i
                                    for i in analysis_result.issues
                                    if i.severity.value == "critical"
                                ]
                            ),
                            "high_issues": len(
                                [
                                    i
                                    for i in analysis_result.issues
                                    if i.severity.value == "high"
                                ]
                            ),
                            "medium_issues": len(
                                [
                                    i
                                    for i in analysis_result.issues
                                    if i.severity.value == "medium"
                                ]
                            ),
                            "low_issues": len(
                                [
                                    i
                                    for i in analysis_result.issues
                                    if i.severity.value == "low"
                                ]
                            ),
                        },
                    }

                    # Clean up temporary file if created
                    if (
                        code
                        and file_path
                        and file_path.startswith(tempfile.gettempdir())
                    ):
                        try:
                            Path(file_path).unlink()
                        except OSError:
                            pass

                    return {"result": result}

                except Exception as e:
                    logger.error(f"Error during code validation: {e}")
                    return {"error": f"Validation failed: {str(e)}"}

            elif tool_name == "get_compliance_mapping":
                standard_ids = args.get("standard_ids", [])
                control_id = args.get("control_id")

                # Get compliance mappings from standards metadata
                mappings = []
                for std_id in standard_ids:
                    standard = await self.standards_engine.get_standard(std_id)
                    if standard and hasattr(standard, "metadata") and standard.metadata:
                        if hasattr(standard.metadata, "nist_controls"):
                            for control in standard.metadata.nist_controls:
                                if not control_id or control == control_id:
                                    mappings.append(
                                        {
                                            "standard_id": std_id,
                                            "control_id": control,
                                            "standard_title": standard.title,
                                        }
                                    )

                return {"result": mappings}

            else:
                return None  # Tool not handled by this handler

        except Exception as e:
            logger.error(f"Error handling tool {tool_name}: {e}")
            return {"error": str(e)}
