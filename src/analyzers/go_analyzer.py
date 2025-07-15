"""Go language analyzer for security, performance, and best practices."""

import re
from pathlib import Path

from .ast_utils import ASTNode
from .base import (
    AnalyzerPlugin,
    AnalyzerResult,
    BaseAnalyzer,
    Issue,
    IssueType,
    PerformanceIssue,
    SecurityIssue,
    Severity,
)


@AnalyzerPlugin.register("go")
class GoAnalyzer(BaseAnalyzer):
    """Analyzer for Go language."""

    @property
    def language(self) -> str:
        return "go"

    @property
    def file_extensions(self) -> list[str]:
        return [".go"]

    def parse_ast(self, content: str) -> ASTNode:
        """Parse Go source code into AST using go/ast."""
        # For production, we'd use tree-sitter-go or call go/ast via subprocess
        # This is a simplified version
        root = ASTNode("file")

        # Parse imports
        import_regex = re.compile(
            r'import\s+(?:"([^"]+)"|(\w+)\s+"([^"]+)"|\((.*?)\))', re.DOTALL
        )
        for match in import_regex.finditer(content):
            import_node = ASTNode("import", match.group())
            root.add_child(import_node)

        # Parse functions
        func_regex = re.compile(
            r"func\s+(?:\(.*?\)\s+)?(\w+)\s*\((.*?)\)\s*(?:\((.*?)\))?\s*{", re.DOTALL
        )
        for match in func_regex.finditer(content):
            func_name = match.group(1)
            func_node = ASTNode("function", func_name)
            func_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(func_node)

        # Parse structs
        struct_regex = re.compile(r"type\s+(\w+)\s+struct\s*{([^}]*)}", re.DOTALL)
        for match in struct_regex.finditer(content):
            struct_name = match.group(1)
            struct_node = ASTNode("struct", struct_name)
            root.add_child(struct_node)

        return root

    def analyze_security(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Go-specific security issues."""
        content = Path(result.file_path).read_text()

        # SQL Injection - look for string concatenation with SQL keywords
        lines = content.split("\n")
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE"]

        for i, line in enumerate(lines, 1):
            # Check for string concatenation with SQL keywords
            if any(keyword in line.upper() for keyword in sql_keywords) and "+" in line:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message="SQL injection via string concatenation",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("+"),
                        code_snippet=line.strip(),
                        recommendation="Use parameterized queries with placeholders",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

            # Check for fmt.Sprintf with SQL keywords
            if "fmt.Sprintf" in line and any(
                keyword in line.upper() for keyword in sql_keywords
            ):
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message="SQL injection via fmt.Sprintf",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("fmt.Sprintf"),
                        code_snippet=line.strip(),
                        recommendation="Use parameterized queries with placeholders",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

        # Command Injection - look for command execution with concatenation
        for i, line in enumerate(lines, 1):
            # Check for exec.Command with string concatenation
            if "exec.Command" in line and "+" in line:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="Command injection via string concatenation",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("exec.Command"),
                        code_snippet=line.strip(),
                        recommendation="Validate and sanitize all user input "
                        "before using in commands",
                        cwe_id="CWE-78",
                        owasp_category="A03:2021 - Injection",
                    )
                )

            # Check for os.system with concatenation
            if "os.system" in line and "+" in line:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="Command injection in os.system",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("os.system"),
                        code_snippet=line.strip(),
                        recommendation="Validate and sanitize all user input "
                        "before using in commands",
                        cwe_id="CWE-78",
                        owasp_category="A03:2021 - Injection",
                    )
                )

        # Goroutine Race Conditions
        self._analyze_race_conditions(content, result)

        # Error Handling
        self._analyze_error_handling(content, result)

        # Cryptography Issues
        self._analyze_crypto_issues(content, result)

    def _analyze_race_conditions(self, content: str, result: AnalyzerResult) -> None:
        """Detect potential race conditions in goroutines."""
        lines = content.split("\n")

        # Look for goroutines with shared variable access
        for i, line in enumerate(lines, 1):
            # Check for goroutine with variable assignment
            if "go func()" in line or "go func(" in line:
                # Look for variable assignment in the goroutine
                for j in range(i, min(i + 10, len(lines) + 1)):
                    if j - 1 < len(lines):
                        next_line = lines[j - 1]
                        # Check for assignment to a shared variable
                        if (
                            re.search(r"\w+\s*=\s*\w+\s*\+", next_line)
                            and "sync" not in content
                        ):
                            result.add_issue(
                                SecurityIssue(
                                    type=IssueType.SECURITY,
                                    severity=Severity.HIGH,
                                    message="Potential race condition: shared variable access "
                                    "without synchronization",
                                    file_path=result.file_path,
                                    line_number=j,
                                    column_number=1,
                                    code_snippet=next_line.strip(),
                                    recommendation="Use sync.Mutex or channels for synchronization",
                                    cwe_id="CWE-362",
                                )
                            )
                            break

    def _analyze_error_handling(self, content: str, result: AnalyzerResult) -> None:
        """Analyze error handling patterns."""
        # Ignored errors
        ignored_error_pattern = r"_\s*(?:,\s*_\s*)?:?=\s*\w+\.[A-Z]\w*\("
        matches = self.find_pattern_matches(content, [ignored_error_pattern])

        for match_text, line_num, col in matches:
            result.add_issue(
                Issue(
                    type=IssueType.ERROR_HANDLING,
                    severity=Severity.MEDIUM,
                    message="Error return value ignored",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Always handle or explicitly ignore errors with a comment",
                )
            )

        # Panic without recover
        panic_pattern = r"panic\s*\("
        panic_matches = self.find_pattern_matches(content, [panic_pattern])

        for match_text, line, col in panic_matches:
            # Check if there's a defer recover nearby
            lines = content.split("\n")
            has_recover = False
            for i in range(max(0, line - 10), min(len(lines), line + 10)):
                if "defer" in lines[i] and "recover" in lines[i]:
                    has_recover = True
                    break

            if not has_recover:
                result.add_issue(
                    Issue(
                        type=IssueType.ERROR_HANDLING,
                        severity=Severity.MEDIUM,
                        message="Panic without recover mechanism",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Add defer recover() to handle panics gracefully",
                    )
                )

    def _analyze_crypto_issues(self, content: str, result: AnalyzerResult) -> None:
        """Analyze cryptographic issues."""
        # Weak crypto algorithms
        weak_crypto = {
            "crypto/md5": ("MD5", "CWE-327"),
            "crypto/sha1": ("SHA1", "CWE-327"),
            "crypto/des": ("DES", "CWE-327"),
            "crypto/rc4": ("RC4", "CWE-327"),
        }

        for import_name, (algo, cwe) in weak_crypto.items():
            if import_name in content:
                # Find the line number properly
                lines = content.split("\n")
                line_num = 1
                for i, line in enumerate(lines, 1):
                    if import_name in line:
                        line_num = i
                        break

                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=f"Use of weak cryptographic algorithm: {algo}",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation=(
                            f"Use stronger algorithms like SHA256 or AES instead of {algo}"
                        ),
                        cwe_id=cwe,
                    )
                )

        # Hardcoded keys/secrets
        secret_patterns = [
            (
                r'(?i)(key|secret|password|token)\s*:?=\s*"[^"]{8,}"',
                "Hardcoded secret detected",
            ),
            (r'(?i)api_?key\s*:?=\s*"[^"]+"', "Hardcoded API key detected"),
        ]

        for pattern, message in secret_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text[:50] + "...",  # Truncate sensitive data
                        recommendation="Use environment variables or secure key management systems",
                        cwe_id="CWE-798",
                    )
                )

    def analyze_performance(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Go-specific performance issues."""
        content = Path(result.file_path).read_text()

        # String concatenation in loops
        self._analyze_string_concat_loops(content, result)

        # Defer in loops
        self._analyze_defer_in_loops(content, result)

        # Unnecessary allocations
        self._analyze_allocations(content, result)

        # Channel operations
        self._analyze_channel_usage(content, result)

        # Map operations
        self._analyze_map_usage(content, result)

    def _analyze_string_concat_loops(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Detect inefficient string concatenation in loops."""
        lines = content.split("\n")
        in_loop = False

        for i, line in enumerate(lines, 1):
            # Check for loop start
            if re.search(r"for\s+.*?{", line):
                in_loop = True
            elif "}" in line and in_loop:
                in_loop = False
            elif in_loop and "+=" in line:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.MEDIUM,
                        message="String concatenation in loop - use strings.Builder",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("+="),
                        recommendation="Use strings.Builder for efficient string concatenation",
                        impact="O(n²) complexity due to string immutability",
                    )
                )

    def _analyze_defer_in_loops(self, content: str, result: AnalyzerResult) -> None:
        """Detect defer statements in loops."""
        lines = content.split("\n")
        in_loop = False

        for i, line in enumerate(lines, 1):
            # Check for loop start
            if re.search(r"for\s+.*?{", line):
                in_loop = True
            elif "}" in line and in_loop:
                in_loop = False
            elif in_loop and "defer" in line:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.HIGH,
                        message="Defer in loop can cause memory accumulation",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=line.find("defer"),
                        recommendation="Move defer outside the loop or use explicit cleanup",
                        impact="Memory usage grows with iteration count",
                    )
                )

    def _analyze_allocations(self, content: str, result: AnalyzerResult) -> None:
        """Detect unnecessary allocations."""
        lines = content.split("\n")
        slice_vars = {}

        for i, line in enumerate(lines, 1):
            # Track slice variable declarations
            slice_match = re.search(r"var\s+(\w+)\s+\[\]", line)
            if slice_match:
                var_name = slice_match.group(1)
                slice_vars[var_name] = i

            # Check for append in loops
            elif "append(" in line:
                for var_name, decl_line in slice_vars.items():
                    if (
                        var_name in line
                        and "for"
                        in content[
                            content.find("\n".join(lines[: i - 5])) : content.find(line)
                        ]
                    ):
                        result.add_issue(
                            PerformanceIssue(
                                type=IssueType.PERFORMANCE,
                                severity=Severity.MEDIUM,
                                message="Slice append without pre-allocation",
                                file_path=result.file_path,
                                line_number=decl_line,
                                column_number=1,
                                recommendation=(
                                    f"Pre-allocate slice with make({var_name}, 0, expectedSize)"
                                ),
                                impact="Multiple memory allocations and copies",
                            )
                        )
                        break

    def _analyze_channel_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze channel usage patterns."""
        # Unbuffered channels in high-throughput scenarios
        unbuffered_chan = r"make\s*\(\s*chan\s+\w+\s*\)"
        matches = self.find_pattern_matches(content, [unbuffered_chan])

        for match_text, line_num, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.LOW,
                    message="Consider using buffered channels for better performance",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Use make(chan Type, bufferSize) for async operations",
                    impact="Potential goroutine blocking",
                )
            )

    def _analyze_map_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze map usage patterns."""
        # Map without size hint
        map_pattern = r"make\s*\(\s*map\s*\[[^\]]+\][^\)]+\)"
        matches = self.find_pattern_matches(content, [map_pattern])

        for match_text, line_num, col in matches:
            if not re.search(r",\s*\d+\s*\)", match_text):
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message="Map created without size hint",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Provide size hint: make(map[K]V, expectedSize)",
                        impact="Potential map resizing during growth",
                    )
                )

    def analyze_best_practices(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Go best practices."""
        content = Path(result.file_path).read_text()

        # Naming conventions
        self._analyze_naming_conventions(content, result)

        # Package comments
        self._analyze_package_comments(content, result)

        # Interface usage
        self._analyze_interface_usage(content, result)

        # Context usage
        self._analyze_context_usage(content, result)

    def _analyze_naming_conventions(self, content: str, result: AnalyzerResult) -> None:
        """Check Go naming conventions."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            # Check for functions that should be exported (public)
            func_match = re.search(r"func\s+([a-z]\w*)\s*\(", line)
            if func_match:
                func_name = func_match.group(1)
                if (
                    func_name[0].islower()
                    and not func_name.startswith("test")
                    and func_name not in ["main", "init"]
                ):
                    # Check if this looks like it should be exported (contains words like "public")
                    if "public" in func_name.lower() or len(func_name) > 6:
                        result.add_issue(
                            Issue(
                                type=IssueType.BEST_PRACTICE,
                                severity=Severity.LOW,
                                message=(
                                    f"Function '{func_name}' should start with "
                                    "uppercase if exported"
                                ),
                                file_path=result.file_path,
                                line_number=i,
                                column_number=line.find("func"),
                                recommendation="Use CamelCase for exported identifiers",
                            )
                        )

    def _analyze_package_comments(self, content: str, result: AnalyzerResult) -> None:
        """Check for package documentation."""
        lines = content.split("\n")
        package_line = None
        has_package_comment = False

        for i, line in enumerate(lines):
            if line.strip().startswith("package "):
                package_line = i + 1
                break
            elif line.strip().startswith("// Package") or line.strip().startswith(
                "/* Package"
            ):
                has_package_comment = True

        if package_line and not has_package_comment:
            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.LOW,
                    message="Package lacks documentation comment",
                    file_path=result.file_path,
                    line_number=package_line,
                    column_number=1,
                    recommendation="Add '// Package <name> ...' comment before package declaration",
                )
            )

    def _analyze_interface_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze interface best practices."""
        # Large interfaces
        interface_pattern = r"type\s+\w+\s+interface\s*{([^}]+)}"
        interfaces = re.finditer(interface_pattern, content, re.DOTALL)

        for interface in interfaces:
            methods = interface.group(1).strip().split("\n")
            method_count = len([m for m in methods if m.strip()])

            if method_count > 5:
                line_num = content[: interface.start()].count("\n") + 1

                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message=f"Interface has {method_count} methods - consider splitting",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Keep interfaces small and focused (1-3 methods)",
                    )
                )

    def _analyze_context_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze context.Context usage."""
        # Functions that should accept context
        func_pattern = r"func\s+(?:\(.*?\)\s+)?(\w+)\s*\(([^)]*)\)"
        functions = re.finditer(func_pattern, content)

        for func in functions:
            func_name = func.group(1)
            params = func.group(2)

            # Check if function does I/O or is exported but doesn't accept context
            if func_name[0].isupper() and "context.Context" not in params:
                # Look for I/O operations in function
                func_start = func.end()
                func_end = content.find("\n}\n", func_start)
                if func_end == -1:
                    func_end = len(content)

                func_body = content[func_start:func_end]
                io_operations = ["http.", "sql.", "Read", "Write", "Query", "Exec"]

                if any(op in func_body for op in io_operations):
                    line_num = content[: func.start()].count("\n") + 1

                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message=(
                                f"Function '{func_name}' performs I/O but "
                                "doesn't accept context.Context"
                            ),
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=1,
                            recommendation="Add context.Context as first parameter for "
                            "cancellation/timeout support",
                        )
                    )

    def _calculate_complexity(self, ast: ASTNode) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity

        # Count decision points
        for node in ast.children:
            if node.type == "function":
                # Simple approximation based on function content
                complexity += 1

        return complexity

    def _extract_dependencies(self, ast: ASTNode) -> list[str]:
        """Extract external dependencies from imports."""
        dependencies = []

        for node in ast.children:
            if node.type == "import":
                import_path = node.value
                # Extract the actual import path
                match = re.search(r'"([^"]+)"', import_path)
                if match:
                    dependencies.append(match.group(1))

        return dependencies
