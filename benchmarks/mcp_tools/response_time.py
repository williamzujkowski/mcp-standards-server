"""Response time benchmarks for MCP tools."""

import asyncio
import json
from typing import Any
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



from src.mcp_server import MCPStandardsServer

from benchmarks.framework import BaseBenchmark


class MCPResponseTimeBenchmark(BaseBenchmark):
    """Benchmark response times for all MCP tools."""

    def __init__(self, iterations: int = 100):
        super().__init__("MCP Response Time", iterations)
        self.server: MCPStandardsServer = None
        self.test_scenarios = self._create_test_scenarios()

    def _create_test_scenarios(self) -> list[dict[str, Any]]:
        """Create test scenarios for each MCP tool."""
        return [
            # get_applicable_standards
            {
                "tool": "get_applicable_standards",
                "args": {
                    "context": {
                        "project_type": "web_app",
                        "language": "javascript",
                        "framework": "react",
                        "environment": "production"
                    },
                    "include_resolution_details": True
                }
            },
            # search_standards
            {
                "tool": "search_standards",
                "args": {
                    "query": "security best practices authentication",
                    "limit": 10,
                    "min_relevance": 0.5
                }
            },
            # get_standard_details
            {
                "tool": "get_standard_details",
                "args": {
                    "standard_id": "react-18-patterns"
                }
            },
            # list_available_standards
            {
                "tool": "list_available_standards",
                "args": {
                    "category": "frontend",
                    "limit": 50
                }
            },
            # validate_against_standard
            {
                "tool": "validate_against_standard",
                "args": {
                    "code": """
                    import React from 'react';

                    class MyComponent extends React.Component {
                        render() {
                            return <div>Hello World</div>;
                        }
                    }
                    """,
                    "standard": "react-18-patterns",
                    "language": "javascript"
                }
            },
            # suggest_improvements
            {
                "tool": "suggest_improvements",
                "args": {
                    "code": """
                    var x = 10;
                    function getData() {
                        return fetch('/api/data').then(res => res.json());
                    }
                    """,
                    "context": {
                        "language": "javascript",
                        "framework": "react"
                    }
                }
            },
            # get_optimized_standard
            {
                "tool": "get_optimized_standard",
                "args": {
                    "standard_id": "react-18-patterns",
                    "format_type": "condensed",
                    "token_budget": 2000
                }
            },
            # estimate_token_usage
            {
                "tool": "estimate_token_usage",
                "args": {
                    "standard_ids": ["react-18-patterns", "typescript-5-guidelines"],
                    "format_types": ["full", "condensed", "summary"]
                }
            },
            # get_sync_status
            {
                "tool": "get_sync_status",
                "args": {}
            }
        ]

    async def setup(self):
        """Setup MCP server."""
        config = {
            "search": {"enabled": False},  # Disable search for speed
            "token_model": "gpt-4",
            "default_token_budget": 8000
        }
        self.server = MCPStandardsServer(config)

        # Ensure some standards exist
        await self._ensure_test_standards()

    async def _ensure_test_standards(self):
        """Ensure test standards exist in cache."""
        # Create minimal test standards
        test_standards = [
            {
                "id": "react-18-patterns",
                "name": "React 18 Best Practices",
                "category": "frontend",
                "tags": ["react", "javascript", "frontend"],
                "content": {
                    "overview": "Best practices for React 18",
                    "guidelines": ["Use functional components", "Prefer hooks"],
                    "examples": ["const MyComponent = () => { return <div>Hello</div>; }"]
                }
            },
            {
                "id": "typescript-5-guidelines",
                "name": "TypeScript 5 Guidelines",
                "category": "frontend",
                "tags": ["typescript", "javascript", "types"],
                "content": {
                    "overview": "Guidelines for TypeScript 5",
                    "guidelines": ["Use strict mode", "Avoid any type"],
                    "examples": ["interface User { name: string; age: number; }"]
                }
            }
        ]

        cache_dir = self.server.synchronizer.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        for standard in test_standards:
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run a single iteration testing all tools."""
        results = {}

        for scenario in self.test_scenarios:
            tool_name = scenario["tool"]
            args = scenario["args"]

            # Get the actual method
            method_name = f"_{tool_name}"
            method = getattr(self.server, method_name, None)

            if method:
                try:
                    # Time the tool execution
                    start = asyncio.get_event_loop().time()
                    await method(**args)
                    end = asyncio.get_event_loop().time()

                    elapsed = end - start
                    results[tool_name] = elapsed

                except Exception as e:
                    results[f"{tool_name}_error"] = str(e)

        return results

    async def teardown(self):
        """Cleanup after benchmark."""
        # No specific cleanup needed
        pass


class MCPToolSpecificBenchmark(BaseBenchmark):
    """Benchmark a specific MCP tool with various input sizes."""

    def __init__(self, tool_name: str, iterations: int = 50):
        super().__init__(f"MCP {tool_name} Benchmark", iterations)
        self.tool_name = tool_name
        self.server: MCPStandardsServer = None
        self.test_cases = []

    async def setup(self):
        """Setup MCP server and test cases."""
        self.server = MCPStandardsServer({
            "search": {"enabled": True},
            "token_model": "gpt-4"
        })

        # Generate tool-specific test cases
        self.test_cases = self._generate_test_cases()

    def _generate_test_cases(self) -> list[dict[str, Any]]:
        """Generate test cases based on tool name."""
        if self.tool_name == "search_standards":
            return [
                {"query": "security", "limit": 5},
                {"query": "authentication oauth jwt", "limit": 10},
                {"query": "react hooks performance optimization", "limit": 20},
                {"query": "microservices architecture patterns distributed systems", "limit": 50}
            ]
        elif self.tool_name == "get_applicable_standards":
            return [
                {"context": {"language": "python"}},
                {"context": {"language": "javascript", "framework": "react"}},
                {"context": {"language": "java", "framework": "spring", "project_type": "microservice"}},
                {"context": {
                    "language": "typescript",
                    "framework": "angular",
                    "project_type": "enterprise",
                    "team_size": "large",
                    "deployment": "kubernetes"
                }}
            ]
        elif self.tool_name == "validate_against_standard":
            # Different sizes of code
            return [
                {"code": "x = 1", "standard": "python-pep8", "language": "python"},
                {"code": "def hello():\n    return 'world'\n" * 10, "standard": "python-pep8", "language": "python"},
                {"code": "class MyClass:\n    def method(self):\n        pass\n" * 50, "standard": "python-pep8", "language": "python"},
                {"code": "# Large file\n" + "def func():\n    pass\n" * 200, "standard": "python-pep8", "language": "python"}
            ]
        else:
            return [{"default": True}]

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run benchmark for different input sizes."""
        results = {}

        method_name = f"_{self.tool_name}"
        method = getattr(self.server, method_name, None)

        if not method:
            return {"error": f"Tool {self.tool_name} not found"}

        for i, test_case in enumerate(self.test_cases):
            try:
                start = asyncio.get_event_loop().time()
                await method(**test_case)
                end = asyncio.get_event_loop().time()

                results[f"case_{i}_time"] = end - start
                results[f"case_{i}_size"] = len(str(test_case))

            except Exception as e:
                results[f"case_{i}_error"] = str(e)

        return results

    async def teardown(self):
        """Cleanup."""
        pass
