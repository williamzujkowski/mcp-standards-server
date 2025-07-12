"""TypeScript language analyzer for type safety, security, and modern patterns."""

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


@AnalyzerPlugin.register("typescript")
@AnalyzerPlugin.register("javascript")
class TypeScriptAnalyzer(BaseAnalyzer):
    """Analyzer for TypeScript with focus on type safety and modern patterns."""

    @property
    def language(self) -> str:
        return "typescript"

    @property
    def file_extensions(self) -> list[str]:
        return [".ts", ".tsx", ".js", ".jsx"]

    def parse_ast(self, content: str) -> ASTNode:
        """Parse TypeScript source code into AST."""
        # Simplified AST parsing - in production use typescript compiler API or tree-sitter
        root = ASTNode("module")

        # Parse imports
        import_pattern = re.compile(
            r'import\s+(?:{[^}]+}|[\w*]+|\*\s+as\s+\w+)\s+from\s+[\'"]([^\'"]+)[\'"]'
        )
        for match in import_pattern.finditer(content):
            import_node = ASTNode("import", match.group(1))
            root.add_child(import_node)

        # Parse interfaces
        interface_pattern = re.compile(
            r"(?:export\s+)?interface\s+(\w+)(?:<[^>]+>)?\s*(?:extends\s+[^{]+)?\s*{"
        )
        for match in interface_pattern.finditer(content):
            interface_name = match.group(1)
            interface_node = ASTNode("interface", interface_name)
            interface_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(interface_node)

        # Parse types
        type_pattern = re.compile(r"(?:export\s+)?type\s+(\w+)(?:<[^>]+>)?\s*=")
        for match in type_pattern.finditer(content):
            type_name = match.group(1)
            type_node = ASTNode("type", type_name)
            type_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(type_node)

        # Parse classes
        class_pattern = re.compile(
            r"(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:<[^>]+>)?(?:\s+extends\s+[^{]+)?(?:\s+implements\s+[^{]+)?\s*{"
        )
        for match in class_pattern.finditer(content):
            class_name = match.group(1)
            class_node = ASTNode("class", class_name)
            class_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(class_node)

        # Parse functions
        fn_pattern = re.compile(
            r"(?:export\s+)?(?:async\s+)?function\s+(\w+)(?:<[^>]+>)?\s*\([^)]*\)(?:\s*:\s*[^{]+)?\s*{"
        )
        for match in fn_pattern.finditer(content):
            fn_name = match.group(1)
            fn_node = ASTNode("function", fn_name)
            fn_node.metadata["line"] = content[: match.start()].count("\n") + 1
            fn_node.metadata["is_async"] = "async" in match.group(0)
            root.add_child(fn_node)

        # Parse arrow functions assigned to variables
        arrow_pattern = re.compile(
            r"(?:export\s+)?const\s+(\w+)\s*(?::\s*[^=]+)?\s*=\s*(?:async\s+)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>"
        )
        for match in arrow_pattern.finditer(content):
            fn_name = match.group(1)
            # Check if this is a React component (starts with uppercase)
            if fn_name[0].isupper():
                fn_node = ASTNode("react_component", fn_name)
            else:
                fn_node = ASTNode("arrow_function", fn_name)
            fn_node.metadata["line"] = content[: match.start()].count("\n") + 1
            fn_node.metadata["is_async"] = "async" in match.group(0)
            root.add_child(fn_node)

        # Parse React components
        component_pattern = re.compile(
            r"(?:export\s+)?(?:function|const)\s+(\w+)(?:<[^>]+>)?\s*(?:\([^)]*\)|:\s*React\.FC)"
        )
        for match in component_pattern.finditer(content):
            comp_name = match.group(1)
            if comp_name[0].isupper():  # React components start with uppercase
                comp_node = ASTNode("react_component", comp_name)
                comp_node.metadata["line"] = content[: match.start()].count("\n") + 1
                root.add_child(comp_node)

        return root

    def analyze_security(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze TypeScript-specific security issues."""
        content = Path(result.file_path).read_text()

        # XSS vulnerabilities
        self._analyze_xss_vulnerabilities(content, result)

        # Injection vulnerabilities
        self._analyze_injection_vulnerabilities(content, result)

        # Unsafe type assertions
        self._analyze_unsafe_type_usage(content, result)

        # Authentication/Authorization
        self._analyze_auth_patterns(content, result)

        # Sensitive data exposure
        self._analyze_sensitive_data(content, result)

        # Third-party vulnerabilities
        self._analyze_dependencies(ast, result)

    def _analyze_xss_vulnerabilities(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze XSS vulnerabilities in TypeScript/React code."""
        # dangerouslySetInnerHTML
        dangerous_html_pattern = r"dangerouslySetInnerHTML\s*=\s*{\s*{[^}]+}\s*}"
        matches = self.find_pattern_matches(content, [dangerous_html_pattern])

        for match_text, line, col in matches:
            result.add_issue(
                SecurityIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.HIGH,
                    message="Use of dangerouslySetInnerHTML can lead to XSS",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Sanitize HTML content or use safer alternatives",
                    cwe_id="CWE-79",
                    owasp_category="A03:2021 - Injection",
                )
            )

        # Direct DOM manipulation
        dom_patterns = [
            (r"innerHTML\s*=", "Direct innerHTML assignment"),
            (r"outerHTML\s*=", "Direct outerHTML assignment"),
            (r"document\.write\s*\(", "Use of document.write"),
            (r"eval\s*\(", "Use of eval() is dangerous"),
            (r"new\s+Function\s*\(", "Dynamic function creation"),
        ]

        for pattern, message in dom_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message=f"{message} can lead to XSS",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use textContent or React state instead",
                        cwe_id="CWE-79",
                    )
                )

        # URL manipulation without validation
        url_patterns = [
            (r'window\.location\.href\s*=\s*[^\'"]', "Unvalidated URL redirect"),
            (r'window\.open\s*\([^\'"]', "Unvalidated window.open URL"),
        ]

        for pattern, message in url_patterns:
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
                        recommendation="Validate and sanitize URLs before use",
                        cwe_id="CWE-601",
                    )
                )

    def _analyze_injection_vulnerabilities(
        self, content: str, result: AnalyzerResult
    ) -> None:
        """Analyze injection vulnerabilities."""
        # Look for template literals and string concatenation in SQL-like strings
        lines = content.split("\n")

        # SQL injection patterns - look for template literals and concatenation in SQL strings
        for line_num, line in enumerate(lines, 1):
            # Template literals with SQL keywords
            if re.search(
                r"`[^`]*(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)[^`]*\$\{",
                line,
                re.IGNORECASE,
            ):
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="SQL injection via template literals",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        code_snippet=line.strip(),
                        recommendation="Use parameterized queries",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

            # String concatenation with SQL keywords
            if re.search(
                r'["\'][^"\']*(?:SELECT|INSERT|UPDATE|DELETE|FROM|WHERE)[^"\']*["\'][^;]*\+',
                line,
                re.IGNORECASE,
            ):
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="SQL injection via string concatenation",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        code_snippet=line.strip(),
                        recommendation="Use parameterized queries",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

        # Additional patterns for query/execute calls
        sql_patterns = [
            (
                r"\.query\s*\([^)]*\$\{",
                "SQL injection via template literals in query call",
            ),
            (
                r"\.execute\s*\([^)]*\$\{",
                "SQL injection via template literals in execute call",
            ),
        ]

        for pattern, message in sql_patterns:
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
                        recommendation="Use parameterized queries",
                        cwe_id="CWE-89",
                        owasp_category="A03:2021 - Injection",
                    )
                )

        # Command injection
        cmd_patterns = [
            (r"exec\s*\([^)]*\$\{", "Command injection risk"),
            (r"spawn\s*\([^,]+,\s*\[[^\]]*\$\{", "Command injection in spawn"),
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
                        recommendation="Validate and escape shell arguments",
                        cwe_id="CWE-78",
                    )
                )

    def _analyze_unsafe_type_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze unsafe TypeScript type usage."""
        # Use of 'any' type
        any_pattern = r":\s*any(?:\s*[,\)\]\}]|\s*=|\s*;)"
        any_matches = self.find_pattern_matches(content, [any_pattern])

        if len(any_matches) > 3:  # Allow some uses of any
            result.add_issue(
                Issue(
                    type=IssueType.TYPE_SAFETY,
                    severity=Severity.MEDIUM,
                    message=f"Excessive use of 'any' type ({len(any_matches)} occurrences)",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Use specific types or 'unknown' instead of 'any'",
                )
            )

        # Type assertions without validation
        assertion_patterns = [
            (r"as\s+any", "Type assertion to 'any'"),
            (r"<any>", "Type assertion to 'any'"),
            (r"!\.\w+", "Non-null assertion without check"),
        ]

        for pattern, message in assertion_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    Issue(
                        type=IssueType.TYPE_SAFETY,
                        severity=Severity.LOW,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Add proper type guards or validation",
                    )
                )

        # @ts-ignore usage
        tsignore_pattern = r"@ts-ignore"
        matches = self.find_pattern_matches(content, [tsignore_pattern])

        for _match_text, line, col in matches:
            result.add_issue(
                Issue(
                    type=IssueType.TYPE_SAFETY,
                    severity=Severity.MEDIUM,
                    message="Use of @ts-ignore suppresses type errors",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    recommendation="Fix the type error instead of ignoring it",
                )
            )

    def _analyze_auth_patterns(self, content: str, result: AnalyzerResult) -> None:
        """Analyze authentication and authorization patterns."""
        # JWT token in localStorage
        if "localStorage" in content and ("token" in content or "jwt" in content):
            storage_patterns = [
                (
                    r'localStorage\.setItem\s*\([\'"](?:token|jwt)',
                    "JWT stored in localStorage",
                ),
                (
                    r'sessionStorage\.setItem\s*\([\'"](?:token|jwt)',
                    "JWT stored in sessionStorage",
                ),
            ]

            for pattern, message in storage_patterns:
                matches = self.find_pattern_matches(content, [pattern])
                for match_text, line_num, col in matches:
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.MEDIUM,
                            message=f"{message} is vulnerable to XSS",
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=col,
                            code_snippet=match_text,
                            recommendation="Use httpOnly cookies for sensitive tokens",
                            cwe_id="CWE-522",
                        )
                    )

        # Missing authentication checks
        if (
            "router" in content.lower()
            or "route" in content.lower()
            or "app." in content.lower()
        ):
            # Look for route definitions without auth middleware
            route_patterns = [
                r'(?:router|app|express)\.(?:get|post|put|delete|patch)\s*\([\'"][^\'\"]+',
            ]
            routes = []
            for pattern in route_patterns:
                routes.extend(self.find_pattern_matches(content, [pattern]))

            auth_keywords = [
                "requireAuth",
                "isAuthenticated",
                "checkAuth",
                "authenticate",
                "authorize",
            ]

            for match_text, line, col in routes:
                # Check if auth middleware is mentioned in the route line (not in comments)
                route_line = content.split("\n")[line - 1]
                # Remove comments from the line
                route_line_no_comments = re.sub(r"//.*$", "", route_line)

                if not any(
                    keyword in route_line_no_comments for keyword in auth_keywords
                ):
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.LOW,
                            message="Route might lack authentication check",
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=col,
                            code_snippet=match_text,
                            recommendation="Ensure all protected routes have authentication",
                        )
                    )

    def _analyze_sensitive_data(self, content: str, result: AnalyzerResult) -> None:
        """Analyze sensitive data handling."""
        # Hardcoded secrets
        secret_patterns = [
            (
                r'(?:api[_-]?key|apikey)\s*[:=]\s*[\'"][^\'"]{20,}[\'"]',
                "Hardcoded API key",
            ),
            (
                r'(?:password|secret)\s*[:=]\s*[\'"][^\'"]+[\'"]',
                "Hardcoded password/secret",
            ),
            (
                r'(?:private[_-]?key)\s*[:=]\s*[\'"][^\'"]+[\'"]',
                "Hardcoded private key",
            ),
            (r'mongodb://[^\'"\s]+:[^\'"\s]+@', "Hardcoded database credentials"),
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
                        recommendation="Use environment variables for secrets",
                        cwe_id="CWE-798",
                    )
                )

        # Console.log with sensitive data
        console_patterns = [
            (
                r"console\.\w+\s*\([^)]*(?:password|token|secret|key)",
                "Logging sensitive data",
            )
        ]

        for pattern, message in console_patterns:
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
                        recommendation="Remove sensitive data from logs",
                        cwe_id="CWE-532",
                    )
                )

    def _analyze_dependencies(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze third-party dependencies for vulnerabilities."""
        vulnerable_packages = {
            "lodash": "Consider using lodash-es or native methods",
            "moment": "Deprecated - use date-fns or dayjs",
            "request": "Deprecated - use fetch or axios",
            "crypto-js": "Use Web Crypto API for better security",
        }

        for node in ast.children:
            if node.type == "import":
                for pkg, recommendation in vulnerable_packages.items():
                    if pkg in node.value:
                        result.add_issue(
                            Issue(
                                type=IssueType.BEST_PRACTICE,
                                severity=Severity.LOW,
                                message=f"Package '{pkg}' has known issues",
                                file_path=result.file_path,
                                line_number=node.metadata.get("line", 1),
                                column_number=1,
                                recommendation=recommendation,
                            )
                        )

    def analyze_performance(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze TypeScript/React performance issues."""
        content = Path(result.file_path).read_text()

        # React-specific performance
        if ".tsx" in result.file_path or "React" in content:
            self._analyze_react_performance(content, ast, result)

        # General JavaScript performance
        self._analyze_js_performance(content, result)

        # Async patterns
        self._analyze_async_patterns(content, result)

        # Bundle size issues
        self._analyze_bundle_size(ast, result)

    def _analyze_react_performance(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze React-specific performance issues."""
        # Missing React.memo for functional components
        components = [node for node in ast.children if node.type == "react_component"]

        for comp in components:
            comp_name = comp.value
            if not re.search(rf"memo\s*\(\s*{comp_name}\s*\)", content):
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message=f"Component '{comp_name}' could benefit from React.memo",
                        file_path=result.file_path,
                        line_number=comp.metadata["line"],
                        column_number=1,
                        recommendation="Use React.memo for components that re-render often",
                        impact="Unnecessary re-renders",
                    )
                )

        # Inline function definitions in JSX
        inline_fn_pattern = r"(?:onClick|onChange|onSubmit)\s*=\s*{\s*\(\)\s*=>"
        matches = self.find_pattern_matches(content, [inline_fn_pattern])

        for match_text, line, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.LOW,
                    message="Inline function in JSX prop",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Use useCallback or define function outside render",
                    impact="Creates new function on every render",
                )
            )

        # Missing dependency arrays in hooks
        hook_patterns = [
            (r"useEffect\s*\([^)]+\)(?!\s*,)", "useEffect without dependency array"),
            (r"useMemo\s*\([^)]+\)(?!\s*,)", "useMemo without dependency array"),
            (
                r"useCallback\s*\([^)]+\)(?!\s*,)",
                "useCallback without dependency array",
            ),
        ]

        for pattern, message in hook_patterns:
            # Use find_pattern_matches but with DOTALL for multi-line
            matches = []
            regex = re.compile(pattern, re.IGNORECASE | re.DOTALL)
            for match in regex.finditer(content):
                line_num = content[: match.start()].count("\n") + 1
                matches.append((match.group(), line_num, match.start()))

            for match_text, line_num, col in matches:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.MEDIUM,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text[:50] + "...",
                        recommendation="Add dependency array to optimize hook execution",
                        impact="Hook runs on every render",
                    )
                )

        # Large lists without virtualization
        if "map" in content and ("<li" in content or "<tr" in content):
            # Simple heuristic: if mapping over array to render list items
            list_pattern = r"\.map\s*\("
            matches = self.find_pattern_matches(content, [list_pattern])

            if (
                len(matches) > 0
                and "react-window" not in content
                and "react-virtual" not in content
            ):
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message="Consider virtualization for large lists",
                        file_path=result.file_path,
                        line_number=1,
                        column_number=1,
                        recommendation="Use react-window or react-virtual for large lists",
                        impact="Rendering performance with large datasets",
                    )
                )

    def _analyze_js_performance(self, content: str, result: AnalyzerResult) -> None:
        """Analyze general JavaScript performance issues."""
        # Array method chaining
        chain_pattern = r"\.filter\([^)]+\)\.map\([^)]+\)"
        matches = self.find_pattern_matches(content, [chain_pattern])

        for match_text, line, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.LOW,
                    message="Multiple array iterations can be combined",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Consider using reduce() or a single loop",
                    impact="Multiple iterations over the same array",
                )
            )

        # Inefficient array operations
        inefficient_patterns = [
            (r"\.find\([^)]+\)\.length", "Use .some() instead of .find().length"),
            (
                r"\.filter\([^)]+\)\.length\s*===?\s*0",
                "Use .every() or .some() instead",
            ),
            (r"JSON\.parse\s*\(\s*JSON\.stringify", "Inefficient deep clone"),
        ]

        for pattern, message in inefficient_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use more efficient alternatives",
                        impact="Unnecessary computation",
                    )
                )

    def _analyze_async_patterns(self, content: str, result: AnalyzerResult) -> None:
        """Analyze async/await patterns."""
        # Sequential awaits that could be parallel - search for consecutive await lines
        lines = content.split("\n")

        for i, line in enumerate(lines):
            if "await" in line and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                # Check if next line also contains await (possibly with variable assignment)
                if (
                    "await" in next_line
                    and "try" not in lines[max(0, i - 2) : i + 2]  # Not in try-catch
                    and ".catch" not in line
                    and ".catch" not in next_line
                ):  # Not using .catch

                    result.add_issue(
                        PerformanceIssue(
                            type=IssueType.PERFORMANCE,
                            severity=Severity.MEDIUM,
                            message="Sequential awaits could be parallelized",
                            file_path=result.file_path,
                            line_number=i + 1,
                            column_number=1,
                            code_snippet=line.strip() + "; " + next_line,
                            recommendation="Use Promise.all() for independent async operations",
                            impact="Increased latency",
                        )
                    )
                    break  # Only report the first instance per function

        # Use original pattern as fallback
        sequential_await = (
            r"await\s+\w+\([^)]*\);\s*\n\s*(?:const|let|var)?\s*\w*\s*=?\s*await"
        )
        matches = []
        regex = re.compile(sequential_await, re.IGNORECASE | re.DOTALL)
        for match in regex.finditer(content):
            line_num = content[: match.start()].count("\n") + 1
            matches.append((match.group(), line_num, match.start()))

        for match_text, line_num, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message="Sequential awaits could be parallelized",
                    file_path=result.file_path,
                    line_number=line_num,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Use Promise.all() for independent async operations",
                    impact="Increased latency",
                )
            )

        # Missing error handling in async functions
        async_pattern = r"async\s+(?:function\s+\w+|\([^)]*\)\s*=>)"
        async_fns = list(re.finditer(async_pattern, content))

        for fn_match in async_fns:
            fn_start = fn_match.end()
            # Find the function body
            brace_count = 0
            fn_end = fn_start
            started = False

            while fn_end < len(content):
                if content[fn_end] == "{":
                    if not started:
                        started = True
                    brace_count += 1
                elif content[fn_end] == "}":
                    brace_count -= 1
                    if brace_count == 0 and started:
                        break
                fn_end += 1

            fn_body = content[fn_start:fn_end]
            if "try" not in fn_body and ".catch" not in fn_body:
                line_num = content[: fn_match.start()].count("\n") + 1

                result.add_issue(
                    Issue(
                        type=IssueType.ERROR_HANDLING,
                        severity=Severity.MEDIUM,
                        message="Async function without error handling",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Add try-catch or .catch() for error handling",
                    )
                )

    def _analyze_bundle_size(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze bundle size issues."""
        # Large library imports
        large_lib_patterns = [
            (
                r'import\s+\*\s+as\s+_\s+from\s+[\'"]lodash[\'"]',
                "Import entire lodash library increases bundle size",
            ),
            (
                r'import\s+moment\s+from\s+[\'"]moment[\'"]',
                "Moment.js is large and increases bundle size (use date-fns)",
            ),
            (
                r'import\s+\*\s+(?:as\s+\w+\s+)?from\s+[\'"]rxjs[\'"]',
                "Import entire RxJS library increases bundle size",
            ),
        ]

        content = Path(result.file_path).read_text()
        for pattern, message in large_lib_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.MEDIUM,
                        message=message,
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use specific imports to reduce bundle size",
                        impact="Increased bundle size",
                    )
                )

    def analyze_best_practices(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze TypeScript best practices and modern patterns."""
        content = Path(result.file_path).read_text()

        # TypeScript configuration
        self._analyze_ts_config(content, result)

        # Code organization
        self._analyze_code_organization(content, ast, result)

        # Error handling
        self._analyze_error_handling(content, result)

        # Modern JavaScript features
        self._analyze_modern_features(content, result)

        # Testing patterns
        self._analyze_testing_patterns(content, result, ast)

    def _analyze_ts_config(self, content: str, result: AnalyzerResult) -> None:
        """Analyze TypeScript configuration best practices."""
        # Strict mode indicators
        if "tsconfig" not in result.file_path:
            # Check for indicators that strict mode might not be enabled
            loose_patterns = [
                (
                    r":\s*any\s*[,\)\]]",
                    "Use of 'any' type suggests strict mode may be off",
                ),
                (r"// @ts-nocheck", "File-level type checking disabled"),
            ]

            for pattern, message in loose_patterns:
                matches = self.find_pattern_matches(content, [pattern])
                if len(matches) > 2:
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message=message,
                            file_path=result.file_path,
                            line_number=1,
                            column_number=1,
                            recommendation="Enable TypeScript strict mode in tsconfig.json",
                        )
                    )
                    break

    def _analyze_code_organization(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze code organization patterns."""
        # File length
        lines = content.split("\n")
        if len(lines) > 300:
            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.LOW,
                    message=f"File is too long ({len(lines)} lines)",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Split into smaller, focused modules",
                )
            )

        # Mixed concerns (e.g., business logic in components)
        if ".tsx" in result.file_path:
            # Check for API calls in components
            api_patterns = ["fetch(", "axios.", "$.ajax", "XMLHttpRequest"]
            for pattern in api_patterns:
                if pattern in content:
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.MEDIUM,
                            message="API calls directly in component",
                            file_path=result.file_path,
                            line_number=1,
                            column_number=1,
                            recommendation="Move API calls to services or custom hooks",
                        )
                    )
                    break

    def _analyze_error_handling(self, content: str, result: AnalyzerResult) -> None:
        """Analyze error handling patterns."""
        # Empty catch blocks - search for catch blocks and check if they're effectively empty
        catch_pattern = r"catch\s*\([^)]*\)\s*\{[^}]*\}"
        catch_matches = list(re.finditer(catch_pattern, content, re.DOTALL))

        for match in catch_matches:
            # Extract the catch body and check if it's empty (only whitespace/comments)
            body = match.group().split("{", 1)[1].rsplit("}", 1)[0]
            clean_body = re.sub(
                r"//.*", "", body
            ).strip()  # Remove comments and whitespace

            if not clean_body:  # Empty catch block
                line_num = content[: match.start()].count("\n") + 1
                result.add_issue(
                    Issue(
                        type=IssueType.ERROR_HANDLING,
                        severity=Severity.HIGH,
                        message="Empty catch block",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        code_snippet=match.group(),
                        recommendation="Handle or log errors appropriately",
                    )
                )

        # Original pattern for simple cases
        empty_catch = r"catch\s*\([^)]*\)\s*{\s*}"
        matches = self.find_pattern_matches(content, [empty_catch])

        for match_text, line, col in matches:
            result.add_issue(
                Issue(
                    type=IssueType.ERROR_HANDLING,
                    severity=Severity.HIGH,
                    message="Empty catch block",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Handle or log errors appropriately",
                )
            )

        # Promise without error handling
        promise_patterns = [
            (r"\.then\([^)]+\)(?!.*\.catch)", "Promise without .catch()"),
            (
                r"new\s+Promise\s*\([^)]+\)(?!.*\.catch)",
                "Promise constructor without error handling",
            ),
        ]

        for pattern, message in promise_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line_num, col in matches:
                # Check if it's inside try-catch
                lines = content.split("\n")
                context_start = max(0, line_num - 5)
                context = "\n".join(lines[context_start:line_num])

                if "try" not in context:
                    result.add_issue(
                        Issue(
                            type=IssueType.ERROR_HANDLING,
                            severity=Severity.MEDIUM,
                            message=message,
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=col,
                            code_snippet=match_text[:50] + "...",
                            recommendation="Add .catch() or use try-catch with async/await",
                        )
                    )

    def _analyze_modern_features(self, content: str, result: AnalyzerResult) -> None:
        """Analyze use of modern JavaScript/TypeScript features."""
        # Old patterns
        old_patterns = [
            (r"var\s+\w+\s*=", "Use 'const' or 'let' instead of 'var'"),
            (r"function\s*\(\s*\)\s*{", "Consider arrow functions for callbacks"),
            (r"\.indexOf\([^)]+\)\s*!==?\s*-1", "Use .includes() instead of indexOf"),
            (r"Object\.assign\(\s*{}\s*,", "Use object spread syntax instead"),
            (r"Array\.prototype\.slice\.call", "Use Array.from() or spread syntax"),
        ]

        for pattern, message in old_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            if matches:
                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.LOW,
                        message=message,
                        file_path=result.file_path,
                        line_number=matches[0][1],
                        column_number=matches[0][2],
                        code_snippet=matches[0][0],
                        recommendation="Use modern ES6+ syntax",
                    )
                )

    def _analyze_testing_patterns(
        self, content: str, result: AnalyzerResult, ast: ASTNode | None = None
    ) -> None:
        """Analyze testing patterns."""
        # Check if this is a test file
        if ".test." in result.file_path or ".spec." in result.file_path:
            # Check for proper test structure
            if (
                "describe" not in content
                and "it(" not in content
                and "test(" not in content
            ):
                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message="Test file lacks proper test structure",
                        file_path=result.file_path,
                        line_number=1,
                        column_number=1,
                        recommendation="Use describe/it or test blocks",
                    )
                )

            # Check for assertions
            if "expect" not in content and "assert" not in content:
                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.HIGH,
                        message="Test file lacks assertions",
                        file_path=result.file_path,
                        line_number=1,
                        column_number=1,
                        recommendation="Add assertions to verify behavior",
                    )
                )
        else:
            # Check if non-test file has corresponding test
            base_name = Path(result.file_path).stem
            if not base_name.startswith("index") and not base_name.endswith(".d"):
                # This is a heuristic - in practice, you'd check if test file exists
                if ast and len(ast.children) >= 5:  # File has substantial content
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.LOW,
                            message="Consider adding test coverage for this module",
                            file_path=result.file_path,
                            line_number=1,
                            column_number=1,
                            recommendation=f"Create {base_name}.test.ts or {base_name}.spec.ts",
                        )
                    )

    def _calculate_complexity(self, ast: ASTNode) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        # Count decision points
        decision_keywords = [
            "if",
            "else",
            "switch",
            "case",
            "for",
            "while",
            "?",
            "&&",
            "||",
            "catch",
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
                # Skip relative imports and type imports
                if not node.value.startswith(".") and not node.value.startswith(
                    "@types/"
                ):
                    dependencies.append(node.value.split("/")[0])

        return list(set(dependencies))  # Remove duplicates
