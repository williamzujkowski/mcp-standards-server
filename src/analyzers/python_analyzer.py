"""Python code analyzer implementation."""

import ast
from typing import Any

from .base import (
    AnalyzerPlugin,
    AnalyzerResult,
    BaseAnalyzer,
    Issue,
    IssueType,
    SecurityIssue,
    Severity,
)


@AnalyzerPlugin.register("python")
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

            # Create a temporary result to collect issues
            temp_result = AnalyzerResult(
                file_path=file_path or "", language=self.language
            )

            # Run all analysis methods
            self.analyze_security(tree, temp_result)
            self.analyze_performance(tree, temp_result)
            self.analyze_best_practices(tree, temp_result)

            # Add issues from temp result
            issues.extend(temp_result.issues)

            # Also run the old _analyze_ast method for React-specific checks
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

    def parse_ast(self, content: str) -> ast.AST:
        """Parse Python source code into AST."""
        return ast.parse(content)

    def analyze_security(self, tree: ast.AST, result: AnalyzerResult) -> None:
        """Analyze security issues in Python code."""
        for node in ast.walk(tree):
            # Check for hardcoded secrets
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id.lower()
                        if any(
                            keyword in var_name
                            for keyword in ["password", "secret", "key", "token"]
                        ):
                            if isinstance(node.value, ast.Constant) and isinstance(
                                node.value.value, str
                            ):
                                if len(node.value.value) > 8:  # Likely a real secret
                                    result.add_issue(
                                        SecurityIssue(
                                            type=IssueType.SECURITY,
                                            severity=Severity.CRITICAL,
                                            message=f"Hardcoded {var_name} found",
                                            file_path=result.file_path,
                                            line_number=node.lineno,
                                            column_number=node.col_offset,
                                            recommendation="Use environment variables for secrets",
                                            cwe_id="CWE-798",
                                        )
                                    )

            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name == "exec":
                        result.add_issue(
                            SecurityIssue(
                                type=IssueType.SECURITY,
                                severity=Severity.CRITICAL,
                                message="Use of exec() function is dangerous",
                                file_path=result.file_path,
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                recommendation="Avoid dynamic code execution",
                                cwe_id="CWE-94",
                            )
                        )
                    elif func_name == "eval":
                        result.add_issue(
                            SecurityIssue(
                                type=IssueType.SECURITY,
                                severity=Severity.HIGH,
                                message="Use of eval() function is dangerous",
                                file_path=result.file_path,
                                line_number=node.lineno,
                                column_number=node.col_offset,
                                recommendation="Use ast.literal_eval() for safe evaluation",
                                cwe_id="CWE-94",
                            )
                        )

    def analyze_performance(self, tree: ast.AST, result: AnalyzerResult) -> None:
        """Analyze performance issues in Python code."""
        for node in ast.walk(tree):
            # Check for inefficient string concatenation in loops
            if isinstance(node, ast.For | ast.While):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.AugAssign)
                        and isinstance(child.op, ast.Add)
                        and isinstance(child.target, ast.Name)
                    ):
                        result.add_issue(
                            Issue(
                                type=IssueType.PERFORMANCE,
                                severity=Severity.MEDIUM,
                                message="String concatenation in loop detected",
                                file_path=result.file_path,
                                line_number=child.lineno,
                                column_number=child.col_offset,
                                recommendation="Use join() or list comprehension instead",
                            )
                        )

    def analyze_best_practices(self, tree: ast.AST, result: AnalyzerResult) -> None:
        """Analyze best practice violations in Python code."""
        for node in ast.walk(tree):
            # Check for bare except clauses
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message="Bare except clause found",
                            file_path=result.file_path,
                            line_number=node.lineno,
                            column_number=node.col_offset,
                            recommendation="Catch specific exceptions instead of using bare except",
                        )
                    )

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, ast.If | ast.For | ast.While | ast.Try | ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        return complexity

    def _extract_dependencies(self, tree: ast.AST) -> list[str]:
        """Extract external dependencies from imports."""
        dependencies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if not alias.name.startswith("."):
                        dependencies.append(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith("."):
                    dependencies.append(node.module.split(".")[0])
        return list(set(dependencies))
