"""Base analyzer class with common patterns and plugin system."""

import hashlib
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, cast


class IssueType(Enum):
    """Types of issues that can be detected."""

    SECURITY = "security"
    PERFORMANCE = "performance"
    BEST_PRACTICE = "best_practice"
    CODE_QUALITY = "code_quality"
    MEMORY_SAFETY = "memory_safety"
    TYPE_SAFETY = "type_safety"
    ERROR_HANDLING = "error_handling"
    CONCURRENCY = "concurrency"


class Severity(Enum):
    """Severity levels for detected issues."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Issue:
    """Base class for detected issues."""

    type: IssueType
    severity: Severity
    message: str
    file_path: str
    line_number: int
    column_number: int
    code_snippet: str | None = None
    recommendation: str | None = None
    references: list[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to dictionary format."""
        return {
            "type": self.type.value,
            "severity": self.severity.value,
            "message": self.message,
            "location": {
                "file": self.file_path,
                "line": self.line_number,
                "column": self.column_number,
            },
            "code_snippet": self.code_snippet,
            "recommendation": self.recommendation,
            "references": self.references,
            "confidence": self.confidence,
        }


@dataclass
class SecurityIssue(Issue):
    """Security-specific issue."""

    cwe_id: str | None = None
    owasp_category: str | None = None

    def __post_init__(self) -> None:
        self.type = IssueType.SECURITY


@dataclass
class PerformanceIssue(Issue):
    """Performance-specific issue."""

    impact: str | None = None

    def __post_init__(self) -> None:
        self.type = IssueType.PERFORMANCE


@dataclass
class AnalyzerResult:
    """Result of code analysis."""

    file_path: str
    language: str
    issues: list[Issue] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    ast_hash: str | None = None
    analysis_time: float = 0.0

    def add_issue(self, issue: Issue) -> None:
        """Add an issue to the results."""
        self.issues.append(issue)

    def get_issues_by_severity(self, severity: Severity) -> list[Issue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]

    def get_issues_by_type(self, issue_type: IssueType) -> list[Issue]:
        """Get all issues of a specific type."""
        return [issue for issue in self.issues if issue.type == issue_type]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "file_path": self.file_path,
            "language": self.language,
            "issues": [issue.to_dict() for issue in self.issues],
            "metrics": self.metrics,
            "ast_hash": self.ast_hash,
            "analysis_time": self.analysis_time,
            "summary": {
                "total_issues": len(self.issues),
                "by_severity": {
                    severity.value: len(self.get_issues_by_severity(severity))
                    for severity in Severity
                },
                "by_type": {
                    issue_type.value: len(self.get_issues_by_type(issue_type))
                    for issue_type in IssueType
                },
            },
        }


class BaseAnalyzer(ABC):
    """Base analyzer class with common patterns and utilities."""

    def __init__(self) -> None:
        self.patterns = self._load_patterns()
        self._cache: dict[str, Any] = {}

    @property
    @abstractmethod
    def language(self) -> str:
        """Return the language this analyzer supports."""
        pass

    @property
    @abstractmethod
    def file_extensions(self) -> list[str]:
        """Return supported file extensions."""
        pass

    @abstractmethod
    def parse_ast(self, content: str) -> Any:
        """Parse source code into AST."""
        pass

    @abstractmethod
    def analyze_security(self, ast: Any, result: AnalyzerResult) -> None:
        """Analyze security issues in the AST."""
        pass

    @abstractmethod
    def analyze_performance(self, ast: Any, result: AnalyzerResult) -> None:
        """Analyze performance issues in the AST."""
        pass

    @abstractmethod
    def analyze_best_practices(self, ast: Any, result: AnalyzerResult) -> None:
        """Analyze best practice violations in the AST."""
        pass

    def _load_patterns(self) -> dict[str, Any]:
        """Load security and performance patterns for the language."""
        return {
            "security": self._get_security_patterns(),
            "performance": self._get_performance_patterns(),
            "best_practices": self._get_best_practice_patterns(),
        }

    def _get_security_patterns(self) -> dict[str, Any]:
        """Get common security patterns."""
        return {
            "sql_injection": [
                r'(SELECT|INSERT|UPDATE|DELETE).*\+.*(%s|{}|f"|f\')',
                r"(execute|query)\s*\(.*\%.*\)",
                r"(execute|query)\s*\(.*\+.*\)",
            ],
            "xss": [
                r"innerHTML\s*=",
                r"document\.write\(",
                r"eval\s*\(",
                r'setTimeout\s*\([\'"].*[\'"]',
                r'setInterval\s*\([\'"].*[\'"]',
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"(open|read|write).*\+.*user_input",
            ],
            "command_injection": [
                r"(exec|system|popen|subprocess).*\+",
                r"os\.(system|exec|popen).*\%",
                r"`.*\$.*`",  # Shell command substitution
            ],
            "hardcoded_secrets": [
                r'(password|passwd|pwd|secret|key|token|api_key)\s*=\s*[\'"][^\'"]{8,}[\'"]',
                r'(AWS|aws|AZURE|azure|GCP|gcp).*=\s*[\'"][^\'"]+[\'"]',
            ],
        }

    def _get_performance_patterns(self) -> dict[str, Any]:
        """Get common performance patterns."""
        return {
            "n_plus_one": [],  # Language-specific
            "memory_leak": [],  # Language-specific
            "inefficient_algorithm": [],  # Language-specific
            "blocking_io": [],  # Language-specific
        }

    def _get_best_practice_patterns(self) -> dict[str, Any]:
        """Get best practice patterns."""
        return {
            "error_handling": [],  # Language-specific
            "naming_convention": [],  # Language-specific
            "code_duplication": [],  # Language-specific
        }

    def analyze_file(self, file_path: Path) -> AnalyzerResult:
        """Analyze a single file."""
        content = file_path.read_text()
        result = AnalyzerResult(file_path=str(file_path), language=self.language)

        try:
            # Parse AST
            ast = self.parse_ast(content)
            result.ast_hash = self._calculate_ast_hash(ast)

            # Run analysis
            self.analyze_security(ast, result)
            self.analyze_performance(ast, result)
            self.analyze_best_practices(ast, result)

            # Collect metrics
            result.metrics = self._collect_metrics(ast, content)

        except Exception as e:
            result.add_issue(
                Issue(
                    type=IssueType.CODE_QUALITY,
                    severity=Severity.HIGH,
                    message=f"Failed to parse file: {str(e)}",
                    file_path=str(file_path),
                    line_number=1,
                    column_number=1,
                )
            )

        return result

    def analyze_directory(self, directory: Path) -> list[AnalyzerResult]:
        """Analyze all supported files in a directory."""
        results = []

        for ext in self.file_extensions:
            for file_path in directory.rglob(f"*{ext}"):
                if not self._should_skip_file(file_path):
                    results.append(self.analyze_file(file_path))

        return results

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped."""
        skip_patterns = [
            "node_modules",
            "vendor",
            "venv",
            ".git",
            "dist",
            "build",
            "__pycache__",
            ".pytest_cache",
        ]

        for pattern in skip_patterns:
            if pattern in str(file_path):
                return True

        return False

    def _calculate_ast_hash(self, ast: Any) -> str:
        """Calculate hash of AST for caching."""
        ast_str = str(ast)
        return hashlib.sha256(ast_str.encode()).hexdigest()

    def _collect_metrics(self, ast: Any, content: str) -> dict[str, Any]:
        """Collect code metrics."""
        lines = content.split("\n")
        return {
            "lines_of_code": len(lines),
            "lines_of_comments": sum(1 for line in lines if self._is_comment(line)),
            "complexity": self._calculate_complexity(ast),
            "dependencies": self._extract_dependencies(ast),
        }

    def _is_comment(self, line: str) -> bool:
        """Check if line is a comment."""
        line = line.strip()
        return line.startswith("#") or line.startswith("//")

    @abstractmethod
    def _calculate_complexity(self, ast: Any) -> int:
        """Calculate cyclomatic complexity."""
        pass

    @abstractmethod
    def _extract_dependencies(self, ast: Any) -> list[str]:
        """Extract external dependencies."""
        pass

    def find_pattern_matches(
        self, content: str, patterns: list[str]
    ) -> list[tuple[str, int, int]]:
        """Find all pattern matches in content."""
        matches = []
        lines = content.split("\n")

        for pattern in patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            for line_num, line in enumerate(lines, 1):
                for match in regex.finditer(line):
                    matches.append((match.group(), line_num, match.start()))

        return matches


class AnalyzerPlugin:
    """Plugin system for easy language addition."""

    _analyzers: dict[str, type] = {}

    @classmethod
    def register(cls, language: str) -> Callable[[type], type]:
        """Decorator to register an analyzer."""

        def decorator(analyzer_class: type) -> type:
            cls._analyzers[language.lower()] = analyzer_class
            return analyzer_class

        return decorator

    @classmethod
    def get_analyzer(cls, language: str) -> BaseAnalyzer | None:
        """Get analyzer instance for a language."""
        analyzer_class = cls._analyzers.get(language.lower())
        if analyzer_class:
            return cast(BaseAnalyzer, analyzer_class())
        return None

    @classmethod
    def list_languages(cls) -> list[str]:
        """List all supported languages."""
        return list(cls._analyzers.keys())
