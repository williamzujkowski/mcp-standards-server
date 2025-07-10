"""Python code analyzer implementation."""

import ast
from typing import Any

from .base import AnalyzerResult, BaseAnalyzer, Issue, IssueType, Severity


class PythonAnalyzer(BaseAnalyzer):
    """Python-specific code analyzer."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def language(self) -> str:
        """Return the language this analyzer supports."""
        return "python"

    @property
    def file_extensions(self) -> list[str]:
        """Return file extensions this analyzer handles."""
        return [".py"]

    def analyze_code(self, code: str, file_path: str | None = None) -> AnalyzerResult:
        """Analyze Python code for issues."""
        issues = []
        metrics = {
            "lines_of_code": len(code.splitlines()),
            "complexity": 1,
            "functions": 0,
            "classes": 0,
        }

        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree, file_path or ""))
            metrics.update(self._calculate_metrics(tree))
        except SyntaxError as e:
            issues.append(
                Issue(
                    type=IssueType.CODE_QUALITY,
                    severity=Severity.HIGH,
                    message=f"Syntax error: {e.msg}",
                    file_path=file_path or "",
                    line_number=e.lineno or 1,
                    column_number=e.offset or 1,
                )
            )

        # Store additional metrics
        metrics["total_lines"] = len(code.splitlines())
        metrics["analyzed_files"] = 1 if file_path else 0
        
        return AnalyzerResult(
            file_path=file_path or "",
            language=self.language,
            issues=issues,
            metrics=metrics,
        )

    def _analyze_ast(self, tree: ast.AST, file_path: str) -> list[Issue]:
        """Analyze AST for issues."""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for class-based React components
                if any(
                    base.id == "Component"
                    for base in node.bases
                    if isinstance(base, ast.Name)
                ):
                    issues.append(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message="Prefer functional components over class components",
                            file_path=file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                        )
                    )

        return issues

    def _calculate_metrics(self, tree: ast.AST) -> dict[str, Any]:
        """Calculate code metrics."""
        metrics = {"functions": 0, "classes": 0, "complexity": 1}

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1

        return metrics
