"""Rust language analyzer for memory safety, performance, and best practices."""

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


@AnalyzerPlugin.register("rust")
class RustAnalyzer(BaseAnalyzer):
    """Analyzer for Rust language with focus on memory safety and performance."""

    @property
    def language(self) -> str:
        return "rust"

    @property
    def file_extensions(self) -> list[str]:
        return [".rs"]

    def parse_ast(self, content: str) -> ASTNode:
        """Parse Rust source code into AST."""
        # Simplified AST parsing - in production use tree-sitter-rust or syn
        root = ASTNode("crate")

        # Parse use statements
        use_pattern = re.compile(r"use\s+((?:\w+::)*\w+)(?:\s+as\s+\w+)?;")
        for match in use_pattern.finditer(content):
            use_node = ASTNode("use", match.group(1))
            root.add_child(use_node)

        # Parse functions
        fn_pattern = re.compile(
            r"(?:pub(?:\s*\(\s*\w+\s*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+(\w+)\s*(?:<[^>]+>)?\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*{",
            re.MULTILINE,
        )
        for match in fn_pattern.finditer(content):
            fn_name = match.group(1)
            fn_node = ASTNode("function", fn_name)
            fn_node.metadata["line"] = content[: match.start()].count("\n") + 1
            fn_node.metadata["is_unsafe"] = "unsafe" in match.group(0)
            fn_node.metadata["is_async"] = "async" in match.group(0)
            fn_node.metadata["is_pub"] = "pub" in match.group(0)
            root.add_child(fn_node)

        # Parse structs
        struct_pattern = re.compile(
            r"(?:pub\s+)?struct\s+(\w+)(?:<[^>]+>)?\s*(?:\{|;|\()"
        )
        for match in struct_pattern.finditer(content):
            struct_name = match.group(1)
            struct_node = ASTNode("struct", struct_name)
            struct_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(struct_node)

        # Parse impl blocks
        impl_pattern = re.compile(
            r"impl(?:<[^>]+>)?\s+(?:\w+\s+for\s+)?(\w+)(?:<[^>]+>)?\s*{"
        )
        for match in impl_pattern.finditer(content):
            impl_name = match.group(1)
            impl_node = ASTNode("impl", impl_name)
            impl_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(impl_node)

        # Parse traits
        trait_pattern = re.compile(
            r"(?:pub\s+)?trait\s+(\w+)(?:<[^>]+>)?\s*(?::\s*[^{]+)?\s*{"
        )
        for match in trait_pattern.finditer(content):
            trait_name = match.group(1)
            trait_node = ASTNode("trait", trait_name)
            trait_node.metadata["line"] = content[: match.start()].count("\n") + 1
            root.add_child(trait_node)

        return root

    def analyze_security(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Rust-specific security issues focusing on unsafe code."""
        content = Path(result.file_path).read_text()

        # Unsafe code analysis
        self._analyze_unsafe_code(content, ast, result)

        # Memory safety issues
        self._analyze_memory_safety(content, result)

        # Ownership and borrowing issues
        self._analyze_ownership_issues(content, result)

        # Concurrency safety
        self._analyze_concurrency_safety(content, ast, result)

        # Cryptographic issues
        self._analyze_crypto_usage(content, result)

        # Input validation
        self._analyze_input_validation(content, result)

    def _analyze_unsafe_code(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze unsafe code blocks and functions."""
        # Unsafe blocks
        unsafe_block_pattern = r"unsafe\s*\{"
        unsafe_blocks = list(re.finditer(unsafe_block_pattern, content))

        for match in unsafe_blocks:
            line_num = content[: match.start()].count("\n") + 1

            # Extract the unsafe block content
            block_start = match.end()
            brace_count = 1
            block_end = block_start

            while brace_count > 0 and block_end < len(content):
                if content[block_end] == "{":
                    brace_count += 1
                elif content[block_end] == "}":
                    brace_count -= 1
                block_end += 1

            unsafe_content = content[block_start : block_end - 1]

            # Check for common unsafe patterns
            if (
                "raw" in unsafe_content
                or "*const" in unsafe_content
                or "*mut" in unsafe_content
            ):
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message="Raw pointer manipulation in unsafe block",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        code_snippet=unsafe_content[:100] + "...",
                        recommendation="Ensure pointer validity and lifetime guarantees",
                        cwe_id="CWE-824",
                    )
                )

            if "transmute" in unsafe_content:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message="Use of std::mem::transmute is extremely dangerous",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Consider safer alternatives like From/Into traits",
                        cwe_id="CWE-843",
                    )
                )

            if "get_unchecked" in unsafe_content:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.HIGH,
                        message="Unchecked array/slice access can cause undefined behavior",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Ensure bounds are verified before unchecked access",
                        cwe_id="CWE-125",
                    )
                )

        # Unsafe functions
        unsafe_fns = [
            node
            for node in ast.children
            if node.type == "function" and node.metadata.get("is_unsafe")
        ]

        for fn_node in unsafe_fns:
            result.add_issue(
                Issue(
                    type=IssueType.MEMORY_SAFETY,
                    severity=Severity.MEDIUM,
                    message=f"Unsafe function '{fn_node.value}' requires careful usage",
                    file_path=result.file_path,
                    line_number=fn_node.metadata["line"],
                    column_number=1,
                    recommendation="Document safety requirements and invariants",
                )
            )

    def _analyze_memory_safety(self, content: str, result: AnalyzerResult) -> None:
        """Analyze memory safety issues."""
        # Use after free patterns
        drop_pattern = r"drop\s*\(\s*(\w+)\s*\)"
        drops = list(re.finditer(drop_pattern, content))

        for drop_match in drops:
            var_name = drop_match.group(1)
            line_num = content[: drop_match.start()].count("\n") + 1

            # Check if variable is used after drop
            after_drop = content[drop_match.end() :]
            if re.search(rf"\b{var_name}\b", after_drop[:500]):  # Check next 500 chars
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message=f"Potential use-after-free: '{var_name}' used after drop",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Ensure variable is not accessed after drop",
                        cwe_id="CWE-416",
                    )
                )

        # Box leak patterns
        box_pattern = r"Box::new\s*\([^)]+\)"
        self.find_pattern_matches(content, [box_pattern])

        # Check for Box::into_raw without corresponding from_raw function call
        if "into_raw" in content and not re.search(r"Box::from_raw\s*\(", content):
            into_raw_matches = self.find_pattern_matches(content, [r"Box::into_raw"])
            for _match_text, line, col in into_raw_matches:
                result.add_issue(
                    Issue(
                        type=IssueType.MEMORY_SAFETY,
                        severity=Severity.HIGH,
                        message="Potential memory leak: Box::into_raw without corresponding from_raw",
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        recommendation="Ensure Box::from_raw is called to reclaim memory",
                    )
                )

    def _analyze_ownership_issues(self, content: str, result: AnalyzerResult) -> None:
        """Analyze ownership and borrowing issues."""
        # Multiple mutable borrows
        mut_borrow_pattern = r"&\s*mut\s+(\w+)"
        mut_borrows = {}

        for match in re.finditer(mut_borrow_pattern, content):
            var_name = match.group(1)
            line_num = content[: match.start()].count("\n") + 1

            if var_name in mut_borrows:
                result.add_issue(
                    Issue(
                        type=IssueType.MEMORY_SAFETY,
                        severity=Severity.HIGH,
                        message=f"Potential multiple mutable borrows of '{var_name}'",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Ensure only one mutable borrow exists at a time",
                    )
                )
            else:
                mut_borrows[var_name] = line_num

        # Clone abuse
        clone_pattern = r"\.clone\(\)"
        clone_matches = self.find_pattern_matches(content, [clone_pattern])

        if len(clone_matches) > 5:  # Arbitrary threshold
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message=f"Excessive use of clone() ({len(clone_matches)} occurrences)",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Consider using references or Rc/Arc for shared ownership",
                    impact="Unnecessary memory allocations and copies",
                )
            )

    def _analyze_concurrency_safety(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze concurrency and thread safety issues."""
        # Static mutable variables
        static_mut_pattern = r"static\s+mut\s+\w+"
        matches = self.find_pattern_matches(content, [static_mut_pattern])

        for match_text, line, col in matches:
            result.add_issue(
                SecurityIssue(
                    type=IssueType.SECURITY,
                    severity=Severity.HIGH,
                    message="Static mutable variable is not thread-safe",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Use thread-safe alternatives like Mutex or atomic types",
                    cwe_id="CWE-362",
                )
            )

        # Rc in multi-threaded context
        if "thread::spawn" in content and ("Rc<" in content or "Rc::new" in content):
            result.add_issue(
                Issue(
                    type=IssueType.CONCURRENCY,
                    severity=Severity.HIGH,
                    message="Rc is not thread-safe, use Arc for multi-threaded contexts",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Replace Rc with Arc for thread-safe reference counting",
                )
            )

        # Missing Send/Sync bounds
        thread_spawn_pattern = r"thread::spawn\s*\(\s*(?:move\s+)?\|\s*\|"
        if re.search(thread_spawn_pattern, content):
            # Check if custom types implement Send/Sync
            struct_names = [
                node.value for node in ast.children if node.type == "struct"
            ]
            for struct_name in struct_names:
                if not re.search(rf"impl\s+Send\s+for\s+{struct_name}", content):
                    result.add_issue(
                        Issue(
                            type=IssueType.CONCURRENCY,
                            severity=Severity.LOW,
                            message=f"Consider implementing Send/Sync for '{struct_name}' if used across threads",
                            file_path=result.file_path,
                            line_number=1,
                            column_number=1,
                            recommendation="Explicitly implement Send/Sync or document thread-safety guarantees",
                        )
                    )

    def _analyze_crypto_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze cryptographic usage."""
        # Weak random number generation
        weak_rng_patterns = [
            (r"rand::random\s*\(\)", "Using rand::random for cryptographic purposes"),
            (r"thread_rng\s*\(\)", "thread_rng may not be cryptographically secure"),
        ]

        for pattern, message in weak_rng_patterns:
            if "crypto" in content or "key" in content or "token" in content:
                matches = self.find_pattern_matches(content, [pattern])
                for match_text, line, col in matches:
                    result.add_issue(
                        SecurityIssue(
                            type=IssueType.SECURITY,
                            severity=Severity.HIGH,
                            message=message,
                            file_path=result.file_path,
                            line_number=line,
                            column_number=col,
                            code_snippet=match_text,
                            recommendation="Use OsRng or crypto-grade RNG for security-sensitive operations",
                            cwe_id="CWE-338",
                        )
                    )

        # Hardcoded secrets
        secret_patterns = [
            (
                r'const\s+\w*(?:KEY|SECRET|TOKEN)\w*\s*:\s*&str\s*=\s*"[^"]+"',
                "Hardcoded secret",
            ),
            (
                r'let\s+\w*(?:password|key|secret)\w*\s*=\s*"[^"]+"',
                "Hardcoded credential",
            ),
        ]

        for pattern, message in secret_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line, col in matches:
                result.add_issue(
                    SecurityIssue(
                        type=IssueType.SECURITY,
                        severity=Severity.CRITICAL,
                        message=message,
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        code_snippet=match_text[:50] + "...",
                        recommendation="Use environment variables or secure key management",
                        cwe_id="CWE-798",
                    )
                )

    def _analyze_input_validation(self, content: str, result: AnalyzerResult) -> None:
        """Analyze input validation patterns."""
        # Unchecked conversions
        unsafe_conversions = [
            (r"as\s+usize(?!\s*\)?\s*<)", "Unchecked numeric conversion to usize"),
            (r"unwrap\s*\(\s*\)", "Using unwrap() can cause panics"),
            (r'expect\s*\(\s*"[^"]*"\s*\)', "Using expect() can cause panics"),
        ]

        for pattern, message in unsafe_conversions:
            matches = self.find_pattern_matches(content, [pattern])

            # Be more lenient with unwrap in tests
            if "unwrap" in pattern and "#[test]" in content:
                continue

            for match_text, line, col in matches:
                severity = Severity.LOW if "expect" in pattern else Severity.MEDIUM

                result.add_issue(
                    Issue(
                        type=IssueType.ERROR_HANDLING,
                        severity=severity,
                        message=message,
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use proper error handling with ? operator or match",
                    )
                )

    def analyze_performance(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Rust-specific performance issues."""
        content = Path(result.file_path).read_text()

        # Allocation patterns
        self._analyze_allocations(content, result)

        # Iterator usage
        self._analyze_iterator_usage(content, result)

        # String operations
        self._analyze_string_operations(content, result)

        # Collection usage
        self._analyze_collection_usage(content, result)

        # Async patterns
        self._analyze_async_patterns(content, ast, result)

    def _analyze_allocations(self, content: str, result: AnalyzerResult) -> None:
        """Analyze memory allocation patterns."""
        # Unnecessary allocations
        unnecessary_alloc_patterns = [
            (r"\.to_string\(\)\.as_str\(\)", "Unnecessary String allocation"),
            (
                r"\.collect::<Vec<_>>\(\)\.len\(\)",
                "Use .count() instead of collect().len()",
            ),
            (
                r'String::from\s*\(\s*"[^"]*"\s*\)',
                "Use &str literal instead of String::from for constants",
            ),
        ]

        for pattern, message in unnecessary_alloc_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line, col in matches:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message=message,
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Avoid unnecessary allocations",
                        impact="Extra heap allocation",
                    )
                )

        # Vec push in loop without capacity
        vec_push_loop = r"let\s+mut\s+(\w+)\s*=\s*Vec::new\(\);.*?for.*?\.push\("
        if re.search(vec_push_loop, content, re.DOTALL):
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message="Vec::push in loop without pre-allocated capacity causes multiple reallocations",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Use Vec::with_capacity() when size is known",
                    impact="Multiple reallocations during growth",
                )
            )

    def _analyze_iterator_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze iterator usage patterns."""
        # Inefficient iterator chains
        inefficient_patterns = [
            (
                r"\.filter\([^)]+\)\.count\(\)\s*==\s*0",
                "Use .any() or .all() instead of .filter().count() == 0",
            ),
            (r"for\s+_\s+in\s+0\.\.\w+", "Use (0..n).for_each() or repeat patterns"),
        ]

        for pattern, message in inefficient_patterns:
            matches = self.find_pattern_matches(content, [pattern])
            for match_text, line, col in matches:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message=message,
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use more efficient iterator methods",
                        impact="Unnecessary iterations or allocations",
                    )
                )

        # Check for multiline collect()->into_iter pattern
        if ".collect::<Vec<_>>" in content and ".into_iter()" in content:
            # Simple heuristic - if both patterns exist in the same function-like scope
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.LOW,
                    message="Unnecessary collect and into_iter",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    code_snippet="collect::<Vec<_>>().into_iter()",
                    recommendation="Use more efficient iterator methods",
                    impact="Unnecessary iterations or allocations",
                )
            )

    def _analyze_string_operations(self, content: str, result: AnalyzerResult) -> None:
        """Analyze string operation performance."""
        # String concatenation in loops
        string_push_loop = r"for.*?{[^}]*\.push_str\("
        matches = self.find_pattern_matches(content, [string_push_loop])

        for match_text, line, col in matches:
            result.add_issue(
                PerformanceIssue(
                    type=IssueType.PERFORMANCE,
                    severity=Severity.MEDIUM,
                    message="String concatenation in loop",
                    file_path=result.file_path,
                    line_number=line,
                    column_number=col,
                    code_snippet=match_text,
                    recommendation="Pre-allocate String capacity or use fmt::Write",
                    impact="Multiple reallocations",
                )
            )

    def _analyze_collection_usage(self, content: str, result: AnalyzerResult) -> None:
        """Analyze collection usage patterns."""
        # HashMap without capacity
        hashmap_pattern = r"HashMap::new\(\)"
        matches = self.find_pattern_matches(content, [hashmap_pattern])

        for match_text, line, col in matches:
            # Check if it's in a loop or function that processes many items
            context_start = max(0, line - 5)
            context = "\n".join(content.split("\n")[context_start : line + 5])

            if "for" in context or "iter" in context:
                result.add_issue(
                    PerformanceIssue(
                        type=IssueType.PERFORMANCE,
                        severity=Severity.LOW,
                        message="HashMap created without capacity hint",
                        file_path=result.file_path,
                        line_number=line,
                        column_number=col,
                        code_snippet=match_text,
                        recommendation="Use HashMap::with_capacity() when size is estimable",
                        impact="Potential rehashing during growth",
                    )
                )

    def _analyze_async_patterns(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze async/await patterns."""
        # Blocking operations in async context
        async_fns = [
            node
            for node in ast.children
            if node.type == "function" and node.metadata.get("is_async")
        ]

        if async_fns:
            blocking_patterns = [
                (r"std::thread::sleep", "Blocking sleep in async context"),
                (r"(?:std::fs::)?read_to_string\(", "Blocking I/O in async context"),
                (r"\.write_all\(\)", "Blocking I/O in async context"),
            ]

            for pattern, message in blocking_patterns:
                matches = self.find_pattern_matches(content, [pattern])
                for match_text, line, col in matches:
                    result.add_issue(
                        PerformanceIssue(
                            type=IssueType.PERFORMANCE,
                            severity=Severity.HIGH,
                            message=message,
                            file_path=result.file_path,
                            line_number=line,
                            column_number=col,
                            code_snippet=match_text,
                            recommendation="Use async equivalents (e.g., tokio::time::sleep)",
                            impact="Blocks entire thread in async runtime",
                        )
                    )

    def analyze_best_practices(self, ast: ASTNode, result: AnalyzerResult) -> None:
        """Analyze Rust best practices."""
        content = Path(result.file_path).read_text()

        # Error handling
        self._analyze_error_handling(content, result)

        # Code style
        self._analyze_code_style(content, result)

        # Documentation
        self._analyze_documentation(content, ast, result)

        # Testing
        self._analyze_testing(content, result)

    def _analyze_error_handling(self, content: str, result: AnalyzerResult) -> None:
        """Analyze error handling patterns."""
        # Custom error types
        if "Result<" in content and "enum" not in content:
            has_custom_error = bool(re.search(r"enum\s+\w*Error", content))
            if not has_custom_error and content.count("Result<") > 5:
                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.MEDIUM,
                        message="Consider defining custom error types",
                        file_path=result.file_path,
                        line_number=1,
                        column_number=1,
                        recommendation="Define custom error enums for better error handling",
                    )
                )

        # Panic in library code
        if "#[test]" not in content and "fn main" not in content:
            panic_patterns = [
                (r"panic!\s*\(", "Avoid panic! in library code"),
                (r"unimplemented!\s*\(", "Unimplemented code in production"),
                (r"todo!\s*\(", "TODO markers in production code"),
            ]

            for pattern, message in panic_patterns:
                matches = self.find_pattern_matches(content, [pattern])
                for match_text, line, col in matches:
                    result.add_issue(
                        Issue(
                            type=IssueType.ERROR_HANDLING,
                            severity=Severity.HIGH,
                            message=message,
                            file_path=result.file_path,
                            line_number=line,
                            column_number=col,
                            code_snippet=match_text,
                            recommendation="Return Result instead of panicking",
                        )
                    )

    def _analyze_code_style(self, content: str, result: AnalyzerResult) -> None:
        """Analyze Rust code style."""
        # Naming conventions
        const_pattern = r"const\s+([a-z_]+):"
        matches = re.finditer(const_pattern, content)

        for match in matches:
            const_name = match.group(1)
            if not const_name.isupper():
                line_num = content[: match.start()].count("\n") + 1

                result.add_issue(
                    Issue(
                        type=IssueType.BEST_PRACTICE,
                        severity=Severity.LOW,
                        message=f"Constant '{const_name}' should be SCREAMING_SNAKE_CASE",
                        file_path=result.file_path,
                        line_number=line_num,
                        column_number=1,
                        recommendation="Use SCREAMING_SNAKE_CASE for constants",
                    )
                )

        # Module organization
        if "mod.rs" in result.file_path and len(content.strip()) > 1000:
            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.LOW,
                    message="Large mod.rs file - consider splitting into submodules",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Keep mod.rs minimal, move implementations to separate files",
                )
            )

    def _analyze_documentation(
        self, content: str, ast: ASTNode, result: AnalyzerResult
    ) -> None:
        """Analyze documentation practices."""
        # Public items without docs
        pub_fns = [
            node
            for node in ast.children
            if node.type == "function" and node.metadata.get("is_pub")
        ]

        for fn_node in pub_fns:
            line_num = fn_node.metadata["line"]
            # Check if there's a doc comment before the function
            lines = content.split("\n")
            if line_num > 1:
                prev_line = lines[line_num - 2].strip()
                if not prev_line.startswith("///") and not prev_line.startswith("//!"):
                    result.add_issue(
                        Issue(
                            type=IssueType.BEST_PRACTICE,
                            severity=Severity.LOW,
                            message=f"Public function '{fn_node.value}' lacks documentation",
                            file_path=result.file_path,
                            line_number=line_num,
                            column_number=1,
                            recommendation="Add /// documentation comments for public items",
                        )
                    )

    def _analyze_testing(self, content: str, result: AnalyzerResult) -> None:
        """Analyze testing patterns."""
        # Check for tests
        has_tests = "#[test]" in content or "#[cfg(test)]" in content

        # Count functions vs tests
        fn_count = content.count("fn ")
        test_count = content.count("#[test]")

        if fn_count > 5 and test_count == 0 and not has_tests:
            result.add_issue(
                Issue(
                    type=IssueType.BEST_PRACTICE,
                    severity=Severity.MEDIUM,
                    message="No tests found in file with multiple functions",
                    file_path=result.file_path,
                    line_number=1,
                    column_number=1,
                    recommendation="Add unit tests for your functions",
                )
            )

    def _calculate_complexity(self, ast: ASTNode) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1

        # Count decision points
        decision_keywords = ["if", "else", "match", "for", "while", "&&", "||", "?"]
        content = Path(ast.value).read_text() if ast.value else ""

        for keyword in decision_keywords:
            complexity += content.count(keyword)

        # Count match arms
        complexity += content.count("=>")

        return complexity

    def _extract_dependencies(self, ast: ASTNode) -> list[str]:
        """Extract external dependencies from use statements."""
        dependencies = []

        for node in ast.children:
            if node.type == "use":
                # Extract crate name from use statement
                parts = node.value.split("::")
                if parts[0] not in ["std", "core", "alloc", "self", "super", "crate"]:
                    dependencies.append(parts[0])

        return list(set(dependencies))  # Remove duplicates
