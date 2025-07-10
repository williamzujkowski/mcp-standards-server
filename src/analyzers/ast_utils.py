"""AST parsing utilities for different languages."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Optional


class ASTNode:
    """Generic AST node representation."""

    def __init__(
        self, node_type: str, value: Any = None, children: list["ASTNode"] | None = None
    ):
        self.type = node_type
        self.value = value
        self.children = children or []
        self.parent: ASTNode | None = None
        self.metadata: dict[str, Any] = {}

        # Set parent references
        for child in self.children:
            child.parent = self

    def add_child(self, child: "ASTNode") -> None:
        """Add a child node."""
        child.parent = self
        self.children.append(child)

    def find_all(self, node_type: str) -> list["ASTNode"]:
        """Find all nodes of a specific type."""
        results = []
        if self.type == node_type:
            results.append(self)
        for child in self.children:
            results.extend(child.find_all(node_type))
        return results

    def find_parent(self, node_type: str) -> Optional["ASTNode"]:
        """Find the first parent of a specific type."""
        current = self.parent
        while current:
            if current.type == node_type:
                return current
            current = current.parent
        return None

    def get_source_location(self) -> tuple[int, int]:
        """Get line and column number."""
        return self.metadata.get("line", 0), self.metadata.get("column", 0)


class ASTParser(ABC):
    """Base AST parser interface."""

    @abstractmethod
    def parse(self, source: str) -> ASTNode:
        """Parse source code into AST."""
        pass

    @abstractmethod
    def get_imports(self, ast: ASTNode) -> list[str]:
        """Extract import statements."""
        pass

    @abstractmethod
    def get_functions(self, ast: ASTNode) -> list[ASTNode]:
        """Extract function definitions."""
        pass

    @abstractmethod
    def get_classes(self, ast: ASTNode) -> list[ASTNode]:
        """Extract class definitions."""
        pass


class PatternMatcher:
    """Pattern matching utilities for AST analysis."""

    def __init__(self) -> None:
        self.patterns: dict[str, Callable[[ASTNode], bool]] = {}

    def register_pattern(
        self, name: str, pattern_func: Callable[[ASTNode], bool]
    ) -> None:
        """Register a pattern matching function."""
        self.patterns[name] = pattern_func

    def match(self, node: ASTNode, pattern_name: str) -> bool:
        """Check if node matches a pattern."""
        if pattern_name in self.patterns:
            return self.patterns[pattern_name](node)
        return False

    def find_matches(self, ast: ASTNode, pattern_name: str) -> list[ASTNode]:
        """Find all nodes matching a pattern."""
        results: list[ASTNode] = []

        def traverse(node: ASTNode) -> None:
            if self.match(node, pattern_name):
                results.append(node)
            for child in node.children:
                traverse(child)

        traverse(ast)
        return results


class SecurityPatternDetector:
    """Detect security patterns in AST."""

    def __init__(self) -> None:
        self.detectors: dict[str, Callable[[ASTNode], list[ASTNode]]] = {
            "sql_injection": self._detect_sql_injection,
            "xss": self._detect_xss,
            "path_traversal": self._detect_path_traversal,
            "command_injection": self._detect_command_injection,
            "hardcoded_secrets": self._detect_hardcoded_secrets,
            "unsafe_deserialization": self._detect_unsafe_deserialization,
            "weak_crypto": self._detect_weak_crypto,
            "race_condition": self._detect_race_condition,
        }

    def detect_all(self, ast: ASTNode) -> dict[str, list[ASTNode]]:
        """Run all security detectors."""
        results: dict[str, list[ASTNode]] = {}
        for name, detector in self.detectors.items():
            matches = detector(ast)
            if matches:
                results[name] = matches
        return results

    def _detect_sql_injection(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential SQL injection vulnerabilities."""
        vulnerable_nodes = []

        # Look for string concatenation in SQL queries
        sql_keywords = {"SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE"}

        for node in ast.find_all("string_concat"):
            if any(keyword in str(node.value).upper() for keyword in sql_keywords):
                vulnerable_nodes.append(node)

        # Look for format strings in SQL contexts
        for node in ast.find_all("format_string"):
            if any(keyword in str(node.value).upper() for keyword in sql_keywords):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_xss(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential XSS vulnerabilities."""
        vulnerable_nodes = []

        # Dangerous DOM operations
        dangerous_ops = ["innerHTML", "outerHTML", "document.write", "eval"]

        for node in ast.find_all("assignment"):
            if any(op in str(node.value) for op in dangerous_ops):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_path_traversal(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential path traversal vulnerabilities."""
        vulnerable_nodes = []

        # File operations with user input
        file_ops = ["open", "read", "write", "readFile", "writeFile"]

        for node in ast.find_all("function_call"):
            if any(op in str(node.value) for op in file_ops):
                # Check if arguments contain user input
                for arg in node.children:
                    if arg.type == "variable" and "user" in str(arg.value).lower():
                        vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_command_injection(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential command injection vulnerabilities."""
        vulnerable_nodes = []

        # Dangerous functions
        dangerous_funcs = ["exec", "system", "popen", "subprocess", "shell_exec"]

        for node in ast.find_all("function_call"):
            if any(func in str(node.value) for func in dangerous_funcs):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_hardcoded_secrets(self, ast: ASTNode) -> list[ASTNode]:
        """Detect hardcoded secrets and credentials."""
        vulnerable_nodes = []

        # Secret-related variable names
        secret_names = [
            "password",
            "passwd",
            "pwd",
            "secret",
            "key",
            "token",
            "api_key",
            "apikey",
            "auth",
            "credential",
            "private_key",
        ]

        for node in ast.find_all("assignment"):
            var_name = str(node.value).lower()
            if any(secret in var_name for secret in secret_names):
                # Check if assigned value is a string literal
                if node.children and node.children[0].type == "string_literal":
                    vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_unsafe_deserialization(self, ast: ASTNode) -> list[ASTNode]:
        """Detect unsafe deserialization."""
        vulnerable_nodes = []

        # Dangerous deserialization functions
        unsafe_funcs = ["pickle.loads", "yaml.load", "eval", "exec"]

        for node in ast.find_all("function_call"):
            if any(func in str(node.value) for func in unsafe_funcs):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_weak_crypto(self, ast: ASTNode) -> list[ASTNode]:
        """Detect weak cryptographic algorithms."""
        vulnerable_nodes = []

        # Weak algorithms
        weak_algos = ["MD5", "SHA1", "DES", "RC4"]

        for node in ast.find_all("function_call"):
            if any(algo in str(node.value).upper() for algo in weak_algos):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_race_condition(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential race conditions."""
        vulnerable_nodes = []

        # Look for check-then-act patterns
        for node in ast.find_all("if_statement"):
            # Check if condition involves file/resource existence
            if "exists" in str(node.value) or "isfile" in str(node.value):
                # Check if body contains file operations
                for child in node.children:
                    if child.type == "function_call" and any(
                        op in str(child.value) for op in ["open", "create", "write"]
                    ):
                        vulnerable_nodes.append(node)

        return vulnerable_nodes


class PerformancePatternDetector:
    """Detect performance anti-patterns in AST."""

    def __init__(self) -> None:
        self.detectors: dict[str, Callable[[ASTNode], list[ASTNode]]] = {
            "n_plus_one": self._detect_n_plus_one,
            "inefficient_loop": self._detect_inefficient_loop,
            "memory_leak": self._detect_memory_leak,
            "blocking_io": self._detect_blocking_io,
            "excessive_allocation": self._detect_excessive_allocation,
        }

    def detect_all(self, ast: ASTNode) -> dict[str, list[ASTNode]]:
        """Run all performance detectors."""
        results = {}
        for name, detector in self.detectors.items():
            matches = detector(ast)
            if matches:
                results[name] = matches
        return results

    def _detect_n_plus_one(self, ast: ASTNode) -> list[ASTNode]:
        """Detect N+1 query patterns."""
        vulnerable_nodes = []

        # Look for loops containing database queries
        for loop_node in ast.find_all("loop"):
            for child in loop_node.children:
                if child.type == "function_call" and any(
                    db_op in str(child.value)
                    for db_op in ["query", "find", "select", "get"]
                ):
                    vulnerable_nodes.append(loop_node)

        return vulnerable_nodes

    def _detect_inefficient_loop(self, ast: ASTNode) -> list[ASTNode]:
        """Detect inefficient loop patterns."""
        vulnerable_nodes = []

        # Look for nested loops with high complexity
        for loop_node in ast.find_all("loop"):
            nested_loops = loop_node.find_all("loop")
            if len(nested_loops) > 1:  # Nested loop detected
                vulnerable_nodes.append(loop_node)

        return vulnerable_nodes

    def _detect_memory_leak(self, ast: ASTNode) -> list[ASTNode]:
        """Detect potential memory leaks."""
        vulnerable_nodes = []

        # Look for resource allocation without cleanup
        resource_allocs = ["open", "connect", "allocate", "new", "malloc"]

        for node in ast.find_all("function_call"):
            if any(alloc in str(node.value) for alloc in resource_allocs):
                # Check if there's a corresponding cleanup in the same scope
                scope = node.find_parent("function") or node.find_parent("method")
                if scope:
                    cleanup_found = False
                    for child in scope.find_all("function_call"):
                        if any(
                            cleanup in str(child.value)
                            for cleanup in ["close", "disconnect", "free", "delete"]
                        ):
                            cleanup_found = True
                            break
                    if not cleanup_found:
                        vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_blocking_io(self, ast: ASTNode) -> list[ASTNode]:
        """Detect blocking I/O operations."""
        vulnerable_nodes = []

        # Blocking I/O operations
        blocking_ops = ["readFileSync", "writeFileSync", "sleep", "time.sleep"]

        for node in ast.find_all("function_call"):
            if any(op in str(node.value) for op in blocking_ops):
                vulnerable_nodes.append(node)

        return vulnerable_nodes

    def _detect_excessive_allocation(self, ast: ASTNode) -> list[ASTNode]:
        """Detect excessive memory allocation."""
        vulnerable_nodes = []

        # Look for large allocations in loops
        for loop_node in ast.find_all("loop"):
            for child in loop_node.children:
                if child.type == "assignment" and any(
                    alloc in str(child.value) for alloc in ["new", "malloc", "allocate"]
                ):
                    vulnerable_nodes.append(child)

        return vulnerable_nodes
