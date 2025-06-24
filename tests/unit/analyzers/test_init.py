"""
Test analyzers __init__ module
@nist-controls: SA-11, CA-7
@evidence: Unit tests for analyzer initialization
"""

from pathlib import Path

from src.analyzers import (
    BaseAnalyzer,
    CodeAnnotation,
    GoAnalyzer,
    JavaAnalyzer,
    JavaScriptAnalyzer,
    PythonAnalyzer,
    SecurityPattern,
    get_analyzer_for_file,
)


class TestAnalyzersInit:
    """Test analyzers initialization module"""

    def test_imports(self):
        """Test that all classes are importable"""
        assert BaseAnalyzer is not None
        assert CodeAnnotation is not None
        assert SecurityPattern is not None
        assert PythonAnalyzer is not None
        assert JavaScriptAnalyzer is not None
        assert GoAnalyzer is not None
        assert JavaAnalyzer is not None

    def test_get_analyzer_for_file_python(self):
        """Test getting analyzer for Python file"""
        analyzer = get_analyzer_for_file(Path("test.py"))
        assert isinstance(analyzer, PythonAnalyzer)

    def test_get_analyzer_for_file_javascript(self):
        """Test getting analyzer for JavaScript file"""
        analyzer = get_analyzer_for_file(Path("test.js"))
        assert isinstance(analyzer, JavaScriptAnalyzer)

    def test_get_analyzer_for_file_jsx(self):
        """Test getting analyzer for JSX file"""
        analyzer = get_analyzer_for_file(Path("test.jsx"))
        assert isinstance(analyzer, JavaScriptAnalyzer)

    def test_get_analyzer_for_file_typescript(self):
        """Test getting analyzer for TypeScript file"""
        analyzer = get_analyzer_for_file(Path("test.ts"))
        assert isinstance(analyzer, JavaScriptAnalyzer)

    def test_get_analyzer_for_file_tsx(self):
        """Test getting analyzer for TSX file"""
        analyzer = get_analyzer_for_file(Path("test.tsx"))
        assert isinstance(analyzer, JavaScriptAnalyzer)

    def test_get_analyzer_for_file_go(self):
        """Test getting analyzer for Go file"""
        analyzer = get_analyzer_for_file(Path("test.go"))
        assert isinstance(analyzer, GoAnalyzer)

    def test_get_analyzer_for_file_java(self):
        """Test getting analyzer for Java file"""
        analyzer = get_analyzer_for_file(Path("test.java"))
        assert isinstance(analyzer, JavaAnalyzer)

    def test_get_analyzer_for_file_unknown(self):
        """Test getting analyzer for unknown file type"""
        analyzer = get_analyzer_for_file(Path("test.unknown"))
        assert analyzer is None

    def test_get_analyzer_for_file_no_extension(self):
        """Test getting analyzer for file without extension"""
        analyzer = get_analyzer_for_file(Path("test"))
        assert analyzer is None
