"""MCP integration for language analyzers."""

from pathlib import Path
from typing import Any

from ..core.mcp.models import MCPTool, ToolResult
from .base import AnalyzerPlugin, AnalyzerResult


class AnalyzerMCPTools:
    """MCP tools for code analysis."""

    @staticmethod
    def get_tools() -> list[MCPTool]:
        """Get available analyzer tools."""
        return [
            MCPTool(
                name="analyze_code",
                description="Analyze code for security, performance, and best practices",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "Programming language (optional, auto-detected if not provided)",
                            "enum": AnalyzerPlugin.list_languages(),
                        },
                        "checks": {
                            "type": "array",
                            "description": "Specific checks to run (default: all)",
                            "items": {
                                "type": "string",
                                "enum": ["security", "performance", "best_practices"],
                            },
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            MCPTool(
                name="analyze_directory",
                description="Analyze all code files in a directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to analyze",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language (optional)",
                            "enum": AnalyzerPlugin.list_languages(),
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "Analyze subdirectories recursively (default: true)",
                        },
                    },
                    "required": ["directory_path"],
                },
            ),
            MCPTool(
                name="list_analyzers",
                description="List available language analyzers",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    @staticmethod
    async def analyze_code(
        file_path: str, language: str | None = None, checks: list[str] | None = None
    ) -> ToolResult:
        """Analyze a single code file."""
        try:
            path = Path(file_path)
            if not path.exists():
                return ToolResult(
                    content=[{"type": "text", "text": f"File not found: {file_path}"}],
                    is_error=True,
                )

            # Auto-detect language if not provided
            if not language:
                extension = path.suffix
                for lang in AnalyzerPlugin.list_languages():
                    analyzer = AnalyzerPlugin.get_analyzer(lang)
                    if analyzer and extension in analyzer.file_extensions:
                        language = lang
                        break

            if not language:
                return ToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": f"Could not detect language for {file_path}",
                        }
                    ],
                    is_error=True,
                )

            # Get analyzer
            analyzer = AnalyzerPlugin.get_analyzer(language)
            if not analyzer:
                return ToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": f"No analyzer available for {language}",
                        }
                    ],
                    is_error=True,
                )

            # Run analysis
            result = analyzer.analyze_file(path)

            # Filter checks if specified
            if checks:
                filtered_issues = []
                for issue in result.issues:
                    if "security" in checks and issue.type.value == "security":
                        filtered_issues.append(issue)
                    elif "performance" in checks and issue.type.value == "performance":
                        filtered_issues.append(issue)
                    elif "best_practices" in checks and issue.type.value in [
                        "best_practice",
                        "code_quality",
                    ]:
                        filtered_issues.append(issue)
                result.issues = filtered_issues

            # Format output
            output = _format_analysis_result(result)

            return ToolResult(content=[{"type": "text", "text": output}])

        except Exception as e:
            return ToolResult(
                content=[{"type": "text", "text": f"Analysis failed: {str(e)}"}],
                is_error=True,
            )

    @staticmethod
    async def analyze_directory(
        directory_path: str, language: str | None = None, recursive: bool = True
    ) -> ToolResult:
        """Analyze all code files in a directory."""
        try:
            path = Path(directory_path)
            if not path.exists() or not path.is_dir():
                return ToolResult(
                    content=[
                        {
                            "type": "text",
                            "text": f"Directory not found: {directory_path}",
                        }
                    ],
                    is_error=True,
                )

            all_results = []

            if language:
                # Analyze specific language
                analyzer = AnalyzerPlugin.get_analyzer(language)
                if analyzer:
                    results = analyzer.analyze_directory(path)
                    all_results.extend(results)
            else:
                # Analyze all supported languages
                for lang in AnalyzerPlugin.list_languages():
                    analyzer = AnalyzerPlugin.get_analyzer(lang)
                    if analyzer:
                        results = analyzer.analyze_directory(path)
                        all_results.extend(results)

            # Format summary
            output = _format_directory_analysis(all_results)

            return ToolResult(content=[{"type": "text", "text": output}])

        except Exception as e:
            return ToolResult(
                content=[
                    {"type": "text", "text": f"Directory analysis failed: {str(e)}"}
                ],
                is_error=True,
            )

    @staticmethod
    async def list_analyzers() -> ToolResult:
        """List available language analyzers."""
        languages = AnalyzerPlugin.list_languages()

        output = "Available Language Analyzers:\n\n"
        for lang in sorted(languages):
            analyzer = AnalyzerPlugin.get_analyzer(lang)
            if analyzer:
                output += f"- **{lang}**: {', '.join(analyzer.file_extensions)}\n"

        return ToolResult(content=[{"type": "text", "text": output}])


def _format_analysis_result(result: AnalyzerResult) -> str:
    """Format analysis result for display."""
    output = f"# Analysis Results: {result.file_path}\n\n"
    output += f"**Language**: {result.language}\n"
    output += f"**Analysis Time**: {result.analysis_time:.2f}s\n\n"

    # Summary
    output += "## Summary\n\n"
    summary = result.to_dict()["summary"]
    output += f"- Total Issues: {summary['total_issues']}\n"
    output += f"- Critical: {summary['by_severity']['critical']}\n"
    output += f"- High: {summary['by_severity']['high']}\n"
    output += f"- Medium: {summary['by_severity']['medium']}\n"
    output += f"- Low: {summary['by_severity']['low']}\n"
    output += f"- Info: {summary['by_severity']['info']}\n\n"

    # Metrics
    if result.metrics:
        output += "## Code Metrics\n\n"
        for key, value in result.metrics.items():
            output += f"- {key.replace('_', ' ').title()}: {value}\n"
        output += "\n"

    # Issues by severity
    if result.issues:
        output += "## Issues\n\n"

        for severity in ["critical", "high", "medium", "low", "info"]:
            severity_issues = [i for i in result.issues if i.severity.value == severity]
            if severity_issues:
                output += f"### {severity.upper()} ({len(severity_issues)})\n\n"

                for issue in severity_issues:
                    output += f"**{issue.type.value.replace('_', ' ').title()}**: {issue.message}\n"
                    output += f"- Location: Line {issue.line_number}, Column {issue.column_number}\n"

                    if issue.code_snippet:
                        output += f"- Code: `{issue.code_snippet}`\n"

                    if issue.recommendation:
                        output += f"- Recommendation: {issue.recommendation}\n"

                    if hasattr(issue, "cwe_id") and issue.cwe_id:
                        output += f"- CWE: {issue.cwe_id}\n"

                    if hasattr(issue, "owasp_category") and issue.owasp_category:
                        output += f"- OWASP: {issue.owasp_category}\n"

                    output += "\n"
    else:
        output += "## No Issues Found! ✨\n\n"

    return output


def _format_directory_analysis(results: list[AnalyzerResult]) -> str:
    """Format directory analysis results."""
    output = "# Directory Analysis Results\n\n"
    output += f"**Files Analyzed**: {len(results)}\n\n"

    # Overall summary
    total_issues = sum(len(r.issues) for r in results)
    output += "## Overall Summary\n\n"
    output += f"- Total Issues: {total_issues}\n"

    # Issues by severity
    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}
    for result in results:
        for issue in result.issues:
            severity_counts[issue.severity.value] += 1

    for severity, count in severity_counts.items():
        if count > 0:
            output += f"- {severity.title()}: {count}\n"

    output += "\n"

    # Language breakdown
    language_stats = {}
    for result in results:
        if result.language not in language_stats:
            language_stats[result.language] = {"files": 0, "issues": 0}
        language_stats[result.language]["files"] += 1
        language_stats[result.language]["issues"] += len(result.issues)

    output += "## Language Breakdown\n\n"
    for lang, stats in sorted(language_stats.items()):
        output += f"- **{lang}**: {stats['files']} files, {stats['issues']} issues\n"

    output += "\n"

    # Files with most issues
    files_with_issues = [(r.file_path, len(r.issues)) for r in results if r.issues]
    files_with_issues.sort(key=lambda x: x[1], reverse=True)

    if files_with_issues:
        output += "## Files with Most Issues\n\n"
        for file_path, issue_count in files_with_issues[:10]:  # Top 10
            output += f"- {file_path}: {issue_count} issues\n"
        output += "\n"

    # Critical and high severity issues
    critical_high = []
    for result in results:
        for issue in result.issues:
            if issue.severity.value in ["critical", "high"]:
                critical_high.append((result.file_path, issue))

    if critical_high:
        output += "## Critical and High Severity Issues\n\n"
        for file_path, issue in critical_high[:20]:  # Top 20
            output += f"- **{file_path}** (Line {issue.line_number}): {issue.message}\n"

    return output


# Register with MCP server
def register_analyzer_tools(mcp_server: Any) -> None:
    """Register analyzer tools with MCP server."""
    tools = AnalyzerMCPTools.get_tools()

    for tool in tools:
        if tool.name == "analyze_code":
            mcp_server.register_tool(tool, AnalyzerMCPTools.analyze_code)
        elif tool.name == "analyze_directory":
            mcp_server.register_tool(tool, AnalyzerMCPTools.analyze_directory)
        elif tool.name == "list_analyzers":
            mcp_server.register_tool(tool, AnalyzerMCPTools.list_analyzers)
