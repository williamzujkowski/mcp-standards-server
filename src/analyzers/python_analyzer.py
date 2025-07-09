"""Python code analyzer implementation."""

import ast
from typing import List, Dict, Any, Optional
from pathlib import Path

from .base import BaseAnalyzer, AnalyzerResult, Issue, IssueType, Severity


class PythonAnalyzer(BaseAnalyzer):
    """Python-specific code analyzer."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.language = "python"
        self.file_extensions = [".py"]
    
    def analyze_code(self, code: str, file_path: Optional[str] = None) -> AnalyzerResult:
        """Analyze Python code for issues."""
        issues = []
        metrics = {
            "lines_of_code": len(code.splitlines()),
            "complexity": 1,
            "functions": 0,
            "classes": 0
        }
        
        try:
            tree = ast.parse(code)
            issues.extend(self._analyze_ast(tree, file_path or ""))
            metrics.update(self._calculate_metrics(tree))
        except SyntaxError as e:
            issues.append(Issue(
                type=IssueType.CODE_QUALITY,
                severity=Severity.HIGH,
                message=f"Syntax error: {e.msg}",
                file_path=file_path or "",
                line_number=e.lineno or 1,
                column_number=e.offset or 1
            ))
        
        return AnalyzerResult(
            issues=issues,
            metrics=metrics,
            total_lines=len(code.splitlines()),
            analyzed_files=1 if file_path else 0,
            language=self.language
        )
    
    def _analyze_ast(self, tree: ast.AST, file_path: str) -> List[Issue]:
        """Analyze AST for issues."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check for class-based React components
                if any(base.id == "Component" for base in node.bases if isinstance(base, ast.Name)):
                    issues.append(Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message="Prefer functional components over class components",
                        file_path=file_path,
                        line_number=node.lineno,
                        column_number=node.col_offset
                    ))
        
        return issues
    
    def _calculate_metrics(self, tree: ast.AST) -> Dict[str, Any]:
        """Calculate code metrics."""
        metrics = {
            "functions": 0,
            "classes": 0,
            "complexity": 1
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics["functions"] += 1
            elif isinstance(node, ast.ClassDef):
                metrics["classes"] += 1
        
        return metrics