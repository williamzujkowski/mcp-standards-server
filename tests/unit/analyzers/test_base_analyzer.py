"""Tests for base analyzer functionality."""

from src.analyzers.base import (
    AnalyzerPlugin,
    AnalyzerResult,
    BaseAnalyzer,
    Issue,
    IssueType,
    PerformanceIssue,
    SecurityIssue,
    Severity,
)


class MockAnalyzer(BaseAnalyzer):
    """Mock analyzer for testing base functionality."""

    @property
    def language(self) -> str:
        return "mock"

    @property
    def file_extensions(self) -> list:
        return [".mock"]

    def parse_ast(self, content: str):
        return {"type": "mock", "content": content}

    def analyze_security(self, ast, result: AnalyzerResult):
        if "vulnerability" in ast.get("content", ""):
            result.add_issue(
                SecurityIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.HIGH,
                    message="Mock vulnerability detected",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    cwe_id="CWE-000",
                )
            )

    def analyze_performance(self, ast, result: AnalyzerResult):
        if "slow" in ast.get("content", ""):
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message="Mock performance issue",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    impact="Slow operation",
                )
            )

    def analyze_best_practices(self, ast, result: AnalyzerResult):
        if "bad_practice" in ast.get("content", ""):
            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.LOW,
                    message="Mock best practice violation",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                )
            )

    def _calculate_complexity(self, ast) -> int:
        return 5

    def _extract_dependencies(self, ast) -> list:
        return ["mock-dep"]


class TestAnalyzerResult:
    """Test AnalyzerResult class."""

    def test_result_creation(self):
        """Test creating analyzer result."""
        result = AnalyzerResult(file_path="/test/file.py", language="python")

        assert result.file_path == "/test/file.py"
        assert result.language == "python"
        assert len(result.issues) == 0
        assert result.metrics == {}

    def test_add_issue(self):
        """Test adding issues to result."""
        result = AnalyzerResult(file_path="/test/file.py", language="python")

        issue = Issue(
            type=IssueType.SECURITY,
            severity=Severity.HIGH,
            message="Test issue",
            file_path="/test/file.py",
            line_number=10,
            column_number=5,
        )

        result.add_issue(issue)
        assert len(result.issues) == 1
        assert result.issues[0] == issue

    def test_get_issues_by_severity(self):
        """Test filtering issues by severity."""
        result = AnalyzerResult(file_path="/test/file.py", language="python")

        high_issue = Issue(
            type=IssueType.SECURITY,
            severity=Severity.HIGH,
            message="High severity",
            file_path="/test/file.py",
            line_number=1,
            column_number=1,
        )

        low_issue = Issue(
            type=IssueType.CODE_QUALITY,
            severity=Severity.LOW,
            message="Low severity",
            file_path="/test/file.py",
            line_number=2,
            column_number=1,
        )

        result.add_issue(high_issue)
        result.add_issue(low_issue)

        high_issues = result.get_issues_by_severity(Severity.HIGH)
        assert len(high_issues) == 1
        assert high_issues[0] == high_issue

        low_issues = result.get_issues_by_severity(Severity.LOW)
        assert len(low_issues) == 1
        assert low_issues[0] == low_issue

    def test_to_dict(self):
        """Test converting result to dictionary."""
        result = AnalyzerResult(
            file_path="/test/file.py",
            language="python",
            ast_hash="abc123",
            analysis_time=1.5,
        )

        result.add_issue(
            Issue(
                type=IssueType.SECURITY,
                severity=Severity.HIGH,
                message="Test issue",
                file_path="/test/file.py",
                line_number=1,
                column_number=1,
            )
        )

        result.metrics = {"lines_of_code": 100}

        data = result.to_dict()

        assert data["file_path"] == "/test/file.py"
        assert data["language"] == "python"
        assert data["ast_hash"] == "abc123"
        assert data["analysis_time"] == 1.5
        assert len(data["issues"]) == 1
        assert data["metrics"]["lines_of_code"] == 100
        assert data["summary"]["total_issues"] == 1
        assert data["summary"]["by_severity"][Severity.HIGH.value] == 1


class TestSecurityIssue:
    """Test SecurityIssue class."""

    def test_security_issue_creation(self):
        """Test creating security issue."""
        issue = SecurityIssue(
            type=IssueType.SECURITY,
            severity=Severity.HIGH,
            message="SQL injection detected",
            file_path="/test/file.py",
            line_number=10,
            column_number=5,
            cwe_id="CWE-89",
            owasp_category="A03:2021",
        )

        assert issue.type == IssueType.SECURITY
        assert issue.severity == Severity.HIGH
        assert issue.cwe_id == "CWE-89"
        assert issue.owasp_category == "A03:2021"


class TestBaseAnalyzer:
    """Test BaseAnalyzer functionality."""

    def test_analyze_file(self, tmp_path):
        """Test analyzing a single file."""
        # Create test file
        test_file = tmp_path / "test.mock"
        test_file.write_text("This contains a vulnerability")

        analyzer = MockAnalyzer()
        result = analyzer.analyze_file(test_file)

        assert result.file_path == str(test_file)
        assert result.language == "mock"
        assert len(result.issues) == 1
        assert result.issues[0].message == "Mock vulnerability detected"
        assert result.metrics["lines_of_code"] == 1
        assert result.metrics["complexity"] == 5
        assert result.metrics["dependencies"] == ["mock-dep"]

    def test_analyze_directory(self, tmp_path):
        """Test analyzing a directory."""
        # Create test files
        (tmp_path / "file1.mock").write_text("slow code")
        (tmp_path / "file2.mock").write_text("bad_practice here")
        (tmp_path / "file3.other").write_text("not analyzed")

        analyzer = MockAnalyzer()
        results = analyzer.analyze_directory(tmp_path)

        assert len(results) == 2

        # Check that correct files were analyzed
        file_paths = {r.file_path for r in results}
        assert str(tmp_path / "file1.mock") in file_paths
        assert str(tmp_path / "file2.mock") in file_paths

    def test_skip_patterns(self, tmp_path):
        """Test that certain directories are skipped."""
        # Create files in directories that should be skipped
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "test.mock").write_text("vulnerability")

        vendor = tmp_path / "vendor"
        vendor.mkdir()
        (vendor / "test.mock").write_text("vulnerability")

        # Create file that should be analyzed
        (tmp_path / "test.mock").write_text("vulnerability")

        analyzer = MockAnalyzer()
        results = analyzer.analyze_directory(tmp_path)

        # Only the file in the root should be analyzed
        assert len(results) == 1
        assert "node_modules" not in results[0].file_path
        assert "vendor" not in results[0].file_path

    def test_find_pattern_matches(self):
        """Test pattern matching functionality."""
        analyzer = MockAnalyzer()
        content = """Line 1
Line with pattern match
Another line
pattern appears here too"""

        matches = analyzer.find_pattern_matches(content, [r"pattern"])

        assert len(matches) == 2
        assert matches[0] == ("pattern", 2, 10)
        assert matches[1] == ("pattern", 4, 0)


class TestAnalyzerPlugin:
    """Test AnalyzerPlugin system."""

    def test_register_analyzer(self):
        """Test registering an analyzer."""
        # Clear existing analyzers
        AnalyzerPlugin._analyzers.clear()

        @AnalyzerPlugin.register("test")
        class TestAnalyzer(MockAnalyzer):
            pass

        assert "test" in AnalyzerPlugin._analyzers
        assert AnalyzerPlugin._analyzers["test"] == TestAnalyzer

    def test_get_analyzer(self):
        """Test getting analyzer instance."""
        AnalyzerPlugin._analyzers.clear()

        @AnalyzerPlugin.register("test")
        class TestAnalyzer(MockAnalyzer):
            pass

        analyzer = AnalyzerPlugin.get_analyzer("test")
        assert isinstance(analyzer, TestAnalyzer)

        # Test non-existent analyzer
        assert AnalyzerPlugin.get_analyzer("nonexistent") is None

    def test_list_languages(self):
        """Test listing supported languages."""
        AnalyzerPlugin._analyzers.clear()

        @AnalyzerPlugin.register("lang1")
        class Lang1Analyzer(MockAnalyzer):
            pass

        @AnalyzerPlugin.register("lang2")
        class Lang2Analyzer(MockAnalyzer):
            pass

        languages = AnalyzerPlugin.list_languages()
        assert set(languages) == {"lang1", "lang2"}
