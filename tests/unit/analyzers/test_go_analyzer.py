"""Tests for Go language analyzer."""

import pytest

from src.analyzers.base import IssueType
from src.analyzers.go_analyzer import GoAnalyzer


class TestGoAnalyzer:
    """Test Go analyzer functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create Go analyzer instance."""
        return GoAnalyzer()

    def test_language_properties(self, analyzer):
        """Test language and file extension properties."""
        assert analyzer.language == "go"
        assert analyzer.file_extensions == [".go"]

    def test_sql_injection_detection(self, analyzer, tmp_path):
        """Test SQL injection vulnerability detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

import "database/sql"

func vulnerable(userInput string) {
    query := "SELECT * FROM users WHERE id = " + userInput
    db.Query(query)
}

func safe(userInput string) {
    db.Query("SELECT * FROM users WHERE id = ?", userInput)
}
"""
        )

        result = analyzer.analyze_file(test_file)

        # Should find SQL injection in vulnerable function
        sql_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-89"
        ]
        assert len(sql_issues) >= 1
        assert any("SQL injection" in issue.message for issue in sql_issues)

    def test_command_injection_detection(self, analyzer, tmp_path):
        """Test command injection detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

import "os/exec"

func vulnerable(userInput string) {
    cmd := exec.Command("sh", "-c", "echo " + userInput)
    cmd.Run()
}
"""
        )

        result = analyzer.analyze_file(test_file)

        cmd_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-78"
        ]
        assert len(cmd_issues) >= 1

    def test_race_condition_detection(self, analyzer, tmp_path):
        """Test race condition detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

var counter int

func increment() {
    go func() {
        counter = counter + 1
    }()
}
"""
        )

        result = analyzer.analyze_file(test_file)

        race_issues = [
            i for i in result.issues if "race condition" in i.message.lower()
        ]
        assert len(race_issues) >= 1

    def test_error_handling_detection(self, analyzer, tmp_path):
        """Test error handling issue detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

import "os"

func ignoreError() {
    _, _ = os.Open("file.txt")
}

func panicWithoutRecover() {
    panic("something went wrong")
}
"""
        )

        result = analyzer.analyze_file(test_file)

        error_issues = [i for i in result.issues if i.type == IssueType.ERROR_HANDLING]
        assert len(error_issues) >= 1

    def test_crypto_issues(self, analyzer, tmp_path):
        """Test cryptographic issue detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

import (
    "crypto/md5"
    "crypto/sha1"
)

const apiKey = "hardcoded-secret-key-12345"

func weakHash(data []byte) {
    md5.Sum(data)
    sha1.Sum(data)
}
"""
        )

        result = analyzer.analyze_file(test_file)

        # Check for weak crypto
        crypto_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-327"
        ]
        assert len(crypto_issues) >= 2  # MD5 and SHA1

        # Check for hardcoded secrets
        secret_issues = [
            i for i in result.issues if hasattr(i, "cwe_id") and i.cwe_id == "CWE-798"
        ]
        assert len(secret_issues) >= 1

    def test_performance_string_concat(self, analyzer, tmp_path):
        """Test string concatenation performance issue."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

func badStringConcat(items []string) string {
    result := ""
    for _, item := range items {
        result += item
    }
    return result
}
"""
        )

        result = analyzer.analyze_file(test_file)

        perf_issues = [i for i in result.issues if i.type == IssueType.PERFORMANCE]
        assert any(
            "string concatenation" in issue.message.lower() for issue in perf_issues
        )

    def test_defer_in_loop(self, analyzer, tmp_path):
        """Test defer in loop detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

func deferInLoop(files []string) {
    for _, file := range files {
        f, _ := os.Open(file)
        defer f.Close()
    }
}
"""
        )

        result = analyzer.analyze_file(test_file)

        defer_issues = [
            i for i in result.issues if "defer in loop" in i.message.lower()
        ]
        assert len(defer_issues) >= 1

    def test_slice_append_performance(self, analyzer, tmp_path):
        """Test slice append without pre-allocation."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

func inefficientAppend(n int) []int {
    var result []int
    for i := 0; i < n; i++ {
        result = append(result, i)
    }
    return result
}
"""
        )

        result = analyzer.analyze_file(test_file)

        append_issues = [
            i
            for i in result.issues
            if "append without pre-allocation" in i.message.lower()
        ]
        assert len(append_issues) >= 1

    def test_naming_conventions(self, analyzer, tmp_path):
        """Test Go naming convention checks."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

// Missing package comment

func publicFunction() {
    // Should be PublicFunction
}

type privateStruct struct {
    // This is fine
}
"""
        )

        result = analyzer.analyze_file(test_file)

        naming_issues = [i for i in result.issues if i.type == IssueType.BEST_PRACTICE]
        assert any("uppercase" in issue.message for issue in naming_issues)
        assert any("package" in issue.message.lower() for issue in naming_issues)

    def test_interface_complexity(self, analyzer, tmp_path):
        """Test interface complexity detection."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

type ComplexInterface interface {
    Method1()
    Method2()
    Method3()
    Method4()
    Method5()
    Method6()
    Method7()
}
"""
        )

        result = analyzer.analyze_file(test_file)

        interface_issues = [
            i
            for i in result.issues
            if "interface" in i.message.lower() and "methods" in i.message.lower()
        ]
        assert len(interface_issues) >= 1

    def test_context_usage(self, analyzer, tmp_path):
        """Test context.Context usage recommendations."""
        test_file = tmp_path / "test.go"
        test_file.write_text(
            """
package main

import (
    "database/sql"
    "net/http"
)

func DoHTTPRequest(url string) (*http.Response, error) {
    return http.Get(url)
}

func QueryDatabase(query string) (*sql.Rows, error) {
    // Missing context parameter
    return db.Query(query)
}
"""
        )

        result = analyzer.analyze_file(test_file)

        context_issues = [i for i in result.issues if "context.Context" in i.message]
        assert len(context_issues) >= 1
