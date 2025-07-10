"""Java language analyzer for security, performance, and best practices."""

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


@AnalyzerPlugin.register("java")
class JavaAnalyzer(BaseAnalyzer):
    """Analyzer for Java language."""

    @property
    def language(self) -> str:
        return "java"

    @property
    def file_extensions(self) -> list[str]:
        return [".java"]

    def parse_ast(self, content: str) -> ASTNode:
        """Parse Java source code into AST."""
        # Simplified AST parsing - in production use javalang or tree-sitter
        root = ASTNode("compilation_unit")

        # Parse package declaration
        package_match = re.search(r"package\s+([\w.]+);", content)
        if package_match:
            package_node = ASTNode("package", package_match.group(1))
            root.add_child(package_node)

        # Parse imports
        import_pattern = re.compile(r"import\s+(?:static\s+)?([\w.*]+);")
        for match in import_pattern.finditer(content):
            import_node = ASTNode("import", match.group(1))
            root.add_child(import_node)

        # Parse classes
        class_pattern = re.compile(
            r"(?:public\s+)?(?:abstract\s+)?(?:final\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*{",
            re.MULTILINE,
        )
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            class_node = ASTNode("class", class_name)
            class_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(class_node)

        # Parse methods
        method_pattern = re.compile(
            r"(?:public|private|protected)?\s*(?:static\s+)?(?:final\s+)?(?:synchronized\s+)?(?:\w+(?:<[^>]+>)?)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*{",
            re.MULTILINE,
        )
        for match in method_pattern.finditer(content):
            method_name = match.group(1)
            method_node = ASTNode("method", method_name)
            method_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(method_node)

        return root

    def analyze_security(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Java-specific security issues (OWASP Top 10)."""
        content = Path(result.file_path).read_text()

        # A01:2021 - Broken Access Control
        self._analyze_access_control(content, result)

        # A02:2021 - Cryptographic Failures
        self._analyze_cryptographic_failures(content, result)

        # A03:2021 - Injection
        self._analyze_injection_vulnerabilities(content, result)

        # A04:2021 - Insecure Design
        self._analyze_insecure_design(content, result)

        # A05:2021 - Security Misconfiguration
        self._analyze_security_misconfiguration(content, result)

        # A06:2021 - Vulnerable Components
        self._analyze_vulnerable_components(ast, result)

        # A07:2021 - Authentication Failures
        self._analyze_authentication_failures(content, result)

        # A08:2021 - Integrity Failures
        self._analyze_integrity_failures(content, result)

        # A09:2021 - Logging Failures
        self._analyze_logging_failures(content, result)

        # A10:2021 - SSRF
        self._analyze_ssrf(content, result)

    def _analyze_access_control(self, content: str, result: AnalyzerResult) -> None:
        """Analyze broken access control issues."""
        # Missing authorization checks
        controller_pattern = r"@(RestController|Controller|RequestMapping)"
        if re.search(controller_pattern, content):
            # Check for missing security annotations
            method_pattern = r"@(GetMapping|PostMapping|PutMapping|DeleteMapping|RequestMapping)\s*\([^)]*\)\s*public"
            methods = list(re.finditer(method_pattern, content))

            for method in methods:
                method_start = method.start()
                # Check if there's a security annotation nearby
                before_method = content[max(0, method_start - 200) : method_start]
                security_annotations = ["@PreAuthorize", "@Secured", "@RolesAllowed"]

                if not any(ann in before_method for ann in security_annotations):
                    line_num = content[:method_start].count("\n") + 1

                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.HIGH,
                            message="Endpoint lacks authorization checks",
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=1,
                            recommendation="Add @PreAuthorize or @Secured annotation",
                            cwe_id="CWE-862",
                            owasp_category="A01:2021 - Broken Access Control",
                        )
                    )

    def _analyze_cryptographic_failures(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze cryptographic failures."""
        # Weak algorithms
        weak_algorithms = {
            "MD5": "Use SHA-256 or stronger",
            "SHA1": "Use SHA-256 or stronger",
            "DES": "Use AES with at least 128-bit keys",
            "RC4": "Use AES-GCM",
            "ECB": "Use CBC or GCM mode",
        }

        for algo, recommendation in weak_algorithms.items():
            # Use word boundaries to avoid false positives
            pattern = rf"\b({algo}|{algo.lower()})\b"
            matches = self.find_pattern_matches(content, [pattern])

            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=f"Use of weak cryptographic algorithm: {algo}",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation=recommendation,
                        cwe_id="CWE-327",
                        owasp_category="A02:2021 - Cryptographic Failures",
                    )
                )

        # Hardcoded secrets
        secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*=\s*"[^"]+"', "Hardcoded password"),
            (r'(?i)(api[_-]?key|apikey)\s*=\s*"[^"]+"', "Hardcoded API key"),
            (r'(?i)(secret|token)\s*=\s*"[^"]+"', "Hardcoded secret"),
            (
                r'(?i)private\s+static\s+final\s+String\s+\w*KEY\w*\s*=\s*"[^"]+"',
                "Hardcoded cryptographic key",
            ),
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
                        code_snippet=match_text[:50] + "...",
                        recommendation="Use environment variables or secure configuration management",
                        cwe_id="CWE-798",
                        owasp_category="A02:2021 - Cryptographic Failures",
                    )
                )

    def _analyze_injection_vulnerabilities(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze injection vulnerabilities."""
        lines = content.split("\n")

        # SQL Injection - line by line analysis
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Check for string concatenation in SQL statements
            if "+" in stripped_line and any(
                keyword in stripped_line.lower()
                for keyword in ["select", "insert", "update", "delete", "sql"]
            ):
                if any(
                    method in stripped_line
                    for method in [
                        "createQuery",
                        "createNativeQuery",
                        "prepareStatement",
                        "jdbcTemplate",
                        "query",
                        "execute",
                    ]
                ):
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.CRITICAL,
                            message="SQL injection via string concatenation",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find("+"),
                            code_snippet=stripped_line,
                            recommendation="Use parameterized queries or prepared statements",
                            cwe_id="CWE-89",
                            owasp_category="A03:2021 - Injection",
                        )
                    )

            # Check for SQL variable assignment with concatenation
            if (
                "String" in stripped_line
                and ("sql" in stripped_line.lower() or "query" in stripped_line.lower())
                and "+" in stripped_line
            ):
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="SQL injection via string concatenation in variable assignment",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("+"),
                        code_snippet=stripped_line,
                        recommendation="Use parameterized queries or prepared statements",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

            # Check for user input concatenation
            if "+" in stripped_line and any(
                input_source in stripped_line
                for input_source in ["request.get", "params.get", "getParameter"]
            ):
                if any(
                    keyword in stripped_line.lower()
                    for keyword in [
                        "select",
                        "insert",
                        "update",
                        "delete",
                        "sql",
                        "query",
                    ]
                ):
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.CRITICAL,
                            message="SQL injection from user input concatenation",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find("+"),
                            code_snippet=stripped_line,
                            recommendation="Use parameterized queries or prepared statements",
                            cwe_id="CWE-89",
                            owasp_category="A03:2021 - Injection",
                        )
                    )

        # LDAP Injection
        ldap_patterns = [
            (r"searchControls\.setSearchScope\([^)]*\+", "LDAP injection risk"),
            (
                r"new\s+SearchFilter\s*\([^)]*\+",
                "LDAP injection via filter concatenation",
            ),
        ]

        for pattern, message in ldap_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use parameterized LDAP queries",
                        cwe_id="CWE-90",
                        owasp_category="A03:2021 - Injection",
                    )
                )

        # Command Injection
        cmd_patterns = [
            (
                r"Runtime\.getRuntime\(\)\.exec\([^)]*\+",
                "Command injection via Runtime.exec",
            ),
            (r"ProcessBuilder.*?\([^)]*\+", "Command injection via ProcessBuilder"),
        ]

        for pattern, message in cmd_patterns:
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
                        code_snippet=match_text,
                        recommendation="Validate and sanitize all user input",
                        cwe_id="CWE-78",
                        owasp_category="A03:2021 - Injection",
                    )
                )

    def _analyze_insecure_design(self, content: str, result: AnalyzerResult) -> None:
        """Analyze insecure design patterns."""
        # Missing input validation
        if "@RestController" in content or "@Controller" in content:
            param_pattern = (
                r"@(RequestParam|PathVariable)\s*(?:\([^)]*\))?\s*\w+\s+(\w+)"
            )
            params = re.finditer(param_pattern, content)

            for param in params:
                param_name = param.group(2)
                line_num = content[: param.start()].count("\n") + 1

                # Check if there's validation nearby
                method_end = content.find("}", param.end())
                method_body = content[param.end() : method_end]

                validation_patterns = [
                    f"{param_name}\\s*==\\s*null",
                    f"validate.*{param_name}",
                    f"StringUtils.*{param_name}",
                    "@Valid",
                    "@Validated",
                ]

                if not any(re.search(pat, method_body) for pat in validation_patterns):
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.MEDIUM,
                            message=f"Missing validation for parameter: {param_name}",
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=1,
                            recommendation="Add input validation for all user inputs",
                            cwe_id="CWE-20",
                            owasp_category="A04:2021 - Insecure Design",
                        )
                    )

    def _analyze_security_misconfiguration(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze security misconfiguration."""
        # CORS misconfiguration
        cors_patterns = [
            (r'\.allowedOrigins\s*\(\s*"\*"\s*\)', "CORS allows all origins"),
            (r'@CrossOrigin\s*\(\s*origins\s*=\s*"\*"', "CORS allows all origins"),
            (r"Access-Control-Allow-Origin.*\*", "CORS header allows all origins"),
        ]

        for pattern, message in cors_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Specify allowed origins explicitly",
                        cwe_id="CWE-942",
                        owasp_category="A05:2021 - Security Misconfiguration",
                    )
                )

    def _analyze_vulnerable_components(
        self, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze use of vulnerable components."""
        # Check imports for known vulnerable libraries
        vulnerable_libs = {
            "org.apache.struts": "Struts has known vulnerabilities",
            "commons-collections": "Vulnerable to deserialization attacks",
            "org.springframework.boot:1.": "Outdated Spring Boot version",
        }

        for node in ast.children:
            if node.type == "import":
                for lib, message in vulnerable_libs.items():
                    if lib in node.value:
                        result.add_issue(
                            SecurityIssue(
                                type=IssueType.SECURITY,
                                severity=Severity.HIGH,
                                message=f"Potentially vulnerable dependency: {message}",
                                file_path=result.file_path,
                                line_number=1,
                                column_number=1,
                                recommendation="Update to latest secure version",
                                cwe_id="CWE-1104",
                                owasp_category="A06:2021 - Vulnerable Components",
                            )
                        )

    def _analyze_authentication_failures(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze authentication failures."""
        # Weak password requirements
        password_patterns = [
            (
                r"password\.length\(\)\s*[<>]=?\s*[1-7]\b",
                "Weak password length requirement",
            ),
            (r"@Size\s*\(\s*min\s*=\s*[1-7]\b", "Weak password length in validation"),
        ]

        for pattern, message in password_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.MEDIUM,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Require passwords of at least 8 characters",
                        cwe_id="CWE-521",
                        owasp_category="A07:2021 - Authentication Failures",
                    )
                )

    def _analyze_integrity_failures(self, content: str, result: AnalyzerResult) -> None:
        """Analyze software and data integrity failures."""
        lines = content.split("\n")

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Check for ObjectInputStream.readObject()
            if "ObjectInputStream" in stripped_line or "readObject()" in stripped_line:
                if "readObject(" in stripped_line:
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.CRITICAL,
                            message="Unsafe deserialization detected",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find("readObject"),
                            code_snippet=stripped_line,
                            recommendation="Avoid deserializing untrusted data",
                            cwe_id="CWE-502",
                            owasp_category="A08:2021 - Integrity Failures",
                        )
                    )

            # Check for @JsonTypeInfo with defaultImpl
            if "@JsonTypeInfo" in stripped_line and "defaultImpl" in stripped_line:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="Potentially unsafe JSON deserialization",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("@JsonTypeInfo"),
                        code_snippet=stripped_line,
                        recommendation="Avoid deserializing untrusted data",
                        cwe_id="CWE-502",
                        owasp_category="A08:2021 - Integrity Failures",
                    )
                )

            # Check for XMLDecoder.readObject
            if "XMLDecoder" in stripped_line and "readObject" in stripped_line:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="Unsafe XML deserialization",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("XMLDecoder"),
                        code_snippet=stripped_line,
                        recommendation="Avoid deserializing untrusted data",
                        cwe_id="CWE-502",
                        owasp_category="A08:2021 - Integrity Failures",
                    )
                )

    def _analyze_logging_failures(self, content: str, result: AnalyzerResult) -> None:
        """Analyze security logging and monitoring failures."""
        # Sensitive data in logs
        log_patterns = [
            (r"log\.(info|debug|error).*password", "Password logged in plain text"),
            (r"log\.(info|debug|error).*token", "Token logged in plain text"),
            (r"System\.out\.print.*password", "Sensitive data in console output"),
        ]

        for pattern, message in log_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Never log sensitive information",
                        cwe_id="CWE-532",
                        owasp_category="A09:2021 - Logging Failures",
                    )
                )

    def _analyze_ssrf(self, content: str, result: AnalyzerResult) -> None:
        """Analyze Server-Side Request Forgery vulnerabilities."""
        ssrf_patterns = [
            (r"new\s+URL\s*\([^)]*request\.get", "SSRF via URL constructor"),
            (
                r"RestTemplate.*\.(get|post)ForObject.*request\.get",
                "SSRF via RestTemplate",
            ),
            (r"HttpClient.*\.send.*request\.get", "SSRF via HttpClient"),
        ]

        for pattern, message in ssrf_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Validate and whitelist URLs before making requests",
                        cwe_id="CWE-918",
                        owasp_category="A10:2021 - SSRF",
                    )
                )

    def analyze_performance(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Java-specific performance issues."""
        content = Path(result.file_path).read_text()

        # Memory management issues
        self._analyze_memory_management(content, result)

        # Collection usage
        self._analyze_collection_usage(content, result)

        # String operations
        self._analyze_string_operations(content, result)

        # Thread safety
        self._analyze_thread_safety(content, result)

        # Database operations
        self._analyze_database_operations(content, result)

    def _analyze_memory_management(self, content: str, result: AnalyzerResult) -> None:
        """Analyze memory management issues."""
        lines = content.split("\n")

        # Memory leaks - unclosed resources (simple line-by-line check)
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Check for FileInputStream/FileOutputStream creation
            if (
                "new FileInputStream(" in stripped_line
                or "new FileOutputStream(" in stripped_line
            ):
                resource_type = (
                    "FileInputStream"
                    if "FileInputStream" in stripped_line
                    else "FileOutputStream"
                )
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.HIGH,
                        message=f"Resource leak: {resource_type} not closed - use try-with-resources",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("new"),
                        code_snippet=stripped_line,
                        recommendation="Use try-with-resources or ensure resources are closed",
                        impact="Memory leak and resource exhaustion",
                    )
                )

            # Check for database connection creation
            if "getConnection()" in stripped_line and (
                "Connection" in stripped_line or "conn" in stripped_line.lower()
            ):
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.HIGH,
                        message="Resource leak: Database connection not closed",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("getConnection"),
                        code_snippet=stripped_line,
                        recommendation="Use try-with-resources or ensure resources are closed",
                        impact="Memory leak and resource exhaustion",
                    )
                )

        # Large object allocations in loops
        loop_allocation = r"for\s*\([^)]*\)\s*{[^}]*new\s+\w+\[[0-9]{4,}\]"
        matches = self.find_pattern_matches(content, [loop_allocation])

        for match_text, line_num, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message="Large array allocation in loop",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Move allocation outside loop or use object pooling",
                    impact="Excessive garbage collection",
                )
            )

    def _analyze_collection_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze collection usage patterns."""
        lines = content.split("\n")

        # Track LinkedList variables and detect .get() usage
        linkedlist_vars = set()

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Find LinkedList variable declarations
            if "LinkedList" in stripped_line and "=" in stripped_line:
                # Extract variable name
                parts = stripped_line.split("=")[0].strip().split()
                if len(parts) >= 2:
                    var_name = parts[-1]
                    linkedlist_vars.add(var_name)

            # Check for .get() calls on LinkedList variables
            for var_name in linkedlist_vars:
                if f"{var_name}.get(" in stripped_line:
                    result.add_issue(
                        PerformanceIssue(
                            type=IssueType.PERFORMANCE,
                            severity=Severity.MEDIUM,
                            message="Random access on LinkedList is O(n)",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find(f"{var_name}.get"),
                            code_snippet=stripped_line,
                            recommendation="Use ArrayList for random access",
                            impact="O(n) time complexity for get operations",
                        )
                    )

        # HashMap without initial capacity - check line by line
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()
            if "new HashMap<" in stripped_line and "()" in stripped_line:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message="HashMap created without initial capacity",
                        file_path=result.file_path,
                        line_number=i,
                        column_number=stripped_line.find("new HashMap"),
                        code_snippet=stripped_line,
                        recommendation="Specify initial capacity to avoid resizing",
                        impact="Multiple resize operations during growth",
                    )
                )

    def _analyze_string_operations(self, content: str, result: AnalyzerResult) -> None:
        """Analyze string operation performance."""
        lines = content.split("\n")
        in_loop = False

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Track if we're in a loop
            if any(
                keyword in stripped_line for keyword in ["for (", "while (", "do {"]
            ):
                in_loop = True
            elif stripped_line.startswith("}"):
                in_loop = False

            # Check for string concatenation in loops (both with literals and variables)
            if in_loop and "+=" in stripped_line:
                # Check if it's likely string concatenation (variable names suggest strings)
                if any(
                    keyword in stripped_line.lower()
                    for keyword in [
                        "result",
                        "string",
                        "text",
                        "msg",
                        "message",
                        '"',
                        "'",
                    ]
                ):
                    result.add_issue(
                        PerformanceIssue(
                            type=IssueType.PERFORMANCE,
                            severity=Severity.MEDIUM,
                            message="String concatenation in loop",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find("+="),
                            code_snippet=stripped_line,
                            recommendation="Use StringBuilder for string concatenation in loops",
                            impact="O(n²) time complexity due to string immutability",
                        )
                    )

    def _analyze_thread_safety(self, content: str, result: AnalyzerResult) -> None:
        """Analyze thread safety issues."""
        # SimpleDateFormat as instance variable
        sdf_pattern = r"private\s+(?:static\s+)?SimpleDateFormat\s+"
        matches = self.find_pattern_matches(content, [sdf_pattern])

        for match_text, line_num, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.HIGH,
                    message="SimpleDateFormat is not thread-safe",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Use ThreadLocal<SimpleDateFormat> or DateTimeFormatter",
                    impact="Thread safety issues and potential data corruption",
                )
            )

    def _analyze_database_operations(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze database operation performance."""
        lines = content.split("\n")
        in_loop = False

        for i, line in enumerate(lines, 1):
            stripped_line = line.strip()

            # Track if we're in a loop
            if any(
                keyword in stripped_line for keyword in ["for (", "while (", "do {"]
            ):
                in_loop = True
            elif stripped_line.startswith("}"):
                in_loop = False

            # Check for repository/dao calls in loops
            if in_loop and any(
                pattern in stripped_line
                for pattern in ["repository.", "dao.", "Repository."]
            ):
                # Check if it's a query method (find, get, load, etc.)
                if any(
                    method in stripped_line
                    for method in ["find", "get", "load", "select", "query"]
                ):
                    result.add_issue(
                        PerformanceIssue(
                            type=IssueType.PERFORMANCE,
                            severity=Severity.HIGH,
                            message="Potential N+1 query problem",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=(
                                stripped_line.find("repository")
                                if "repository" in stripped_line
                                else stripped_line.find("dao")
                            ),
                            code_snippet=stripped_line,
                            recommendation="Use JOIN fetch or batch loading",
                            impact="Database performance degradation",
                        )
                    )

    def analyze_best_practices(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Java best practices and design patterns."""
        content = Path(result.file_path).read_text()

        # Code style
        self._analyze_code_style(content, result)

        # Design patterns
        self._analyze_design_patterns(content, result)

        # Spring Framework best practices
        self._analyze_spring_practices(content, result)

        # Exception handling
        self._analyze_exception_handling(content, result)

    def _analyze_code_style(self, content: str, result: AnalyzerResult) -> None:
        """Analyze code style issues."""
        # Class naming
        class_pattern = r"class\s+([a-z]\w*)"
        matches = re.finditer(class_pattern, content)

        for match in matches:
            class_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.LOW,
                    message=f"Class name '{class_name}' should start with uppercase",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=1,
                    recommendation="Use PascalCase for class names",
                )
            )

        # Constant naming
        const_pattern = r"static\s+final\s+\w+\s+([a-z]\w*)\s*="
        matches = re.finditer(const_pattern, content)

        for match in matches:
            const_name = match.group(1)
            if not const_name.isupper():
                line_num = content[: match.start()].count("\n") + 1

                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.LOW,
                        message=f"Constant '{const_name}' should be UPPER_CASE",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Use UPPER_SNAKE_CASE for constants",
                    )
                )

    def _analyze_design_patterns(self, content: str, result: AnalyzerResult) -> None:
        """Analyze design pattern usage."""
        # Singleton pattern issues
        singleton_pattern = r"private\s+static\s+\w+\s+instance\s*;"
        if re.search(singleton_pattern, content):
            # Check for thread safety
            if "synchronized" not in content and "volatile" not in content:
                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message="Singleton pattern without thread safety",
                        file_path=result.file_path,
                        line_number=1,
                        column_number=1,
                        recommendation="Use double-checked locking or enum singleton",
                    )
                )

    def _analyze_spring_practices(self, content: str, result: AnalyzerResult) -> None:
        """Analyze Spring Framework best practices."""
        if "@Component" in content or "@Service" in content or "@Repository" in content:
            lines = content.split("\n")
            autowired_next = False

            for i, line in enumerate(lines, 1):
                stripped_line = line.strip()

                # Check for @Autowired annotation
                if "@Autowired" in stripped_line:
                    autowired_next = True
                    continue

                # Check if the next line after @Autowired is a private field
                if (
                    autowired_next
                    and "private" in stripped_line
                    and (";" in stripped_line or "=" in stripped_line)
                ):
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message="Field injection detected",
                            file_path=result.file_path,
                            line_number=i,
                            column_number=stripped_line.find("private"),
                            code_snippet=stripped_line,
                            recommendation="Use constructor injection for better testability",
                        )
                    )
                    autowired_next = False
                elif stripped_line and not stripped_line.startswith("//"):
                    # Reset if we encounter any non-comment line that's not private field
                    autowired_next = False

    def _analyze_exception_handling(self, content: str, result: AnalyzerResult) -> None:
        """Analyze exception handling practices."""
        # Empty catch blocks
        empty_catch = r"catch\s*\([^)]+\)\s*{\s*}"
        matches = self.find_pattern_matches(content, [empty_catch])

        for match_text, line_num, col in matches:
            result.add_issue(
                Issue(
                    type=IssueType.ERROR_HANDLING,
                    severity=Severity.HIGH,
                    message="Empty catch block",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Log the exception or add appropriate error handling",
                )
            )

        # Catching generic Exception
        generic_catch = r"catch\s*\(\s*Exception\s+\w+\s*\)"
        matches = self.find_pattern_matches(content, [generic_catch])

        for match_text, line_num, col in matches:
            result.add_issue(
                Issue(
                    type=IssueType.ERROR_HANDLING,
                    severity=Severity.MEDIUM,
                    message="Catching generic Exception",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Catch specific exceptions",
                )
            )

    def _calculate_complexity(self, ast: ASTNode) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        # Count decision points
        decision_keywords = [
            "if",
            "else",
            "for",
            "while",
            "switch",
            "case",
            "&&",
            "||",
            "?",
        ]
        content = Path(ast.value).read_text() if ast.value else ""

        for keyword in decision_keywords:
            complexity += content.count(keyword)

        return complexity

    def _extract_dependencies(self, ast: ASTNode) -> list[str]:
        """Extract external dependencies from imports."""
        dependencies = []

        for node in ast.children:
            if node.type == "import":
                # Filter out java.* and javax.* standard library imports
                if not node.value.startswith(("java.", "javax.")):
                    dependencies.append(node.value)

        return dependencies
