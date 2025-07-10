"""Throughput testing for MCP tools under concurrent load."""

import asyncio
import random
import time
from typing import Any
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark


class MCPThroughputBenchmark(BaseBenchmark):
    """Benchmark throughput under concurrent load."""

    def __init__(
        self,
        concurrent_clients: int = 10,
        duration_seconds: int = 30,
        iterations: int = 1
    ):
        super().__init__(
            f"MCP Throughput (clients={concurrent_clients})",
            iterations
        )
        self.concurrent_clients = concurrent_clients
        self.duration_seconds = duration_seconds
        self.server: MCPStandardsServer = None
        self.request_count = 0
        self.error_count = 0
        self.latencies: list[float] = []

    async def setup(self):
        """Setup MCP server."""
        self.server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })

        # Reset counters
        self.request_count = 0
        self.error_count = 0
        self.latencies = []

        # Ensure test data exists
        await self._setup_test_data()

    async def _setup_test_data(self):
        """Setup test standards."""
        test_standards = []

        # Generate multiple test standards
        for i in range(20):
            standard = {
                "id": f"test-standard-{i}",
                "name": f"Test Standard {i}",
                "category": random.choice(["frontend", "backend", "security", "testing"]),
                "tags": [f"tag{j}" for j in range(random.randint(2, 5))],
                "content": {
                    "overview": f"Overview for standard {i}",
                    "guidelines": [f"Guideline {j}" for j in range(10)],
                    "examples": [f"Example {j}" for j in range(5)]
                }
            }
            test_standards.append(standard)

        # Save to cache
        cache_dir = self.server.synchronizer.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        for standard in test_standards:
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                import json
                json.dump(standard, f)

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run concurrent load test."""
        # Create client tasks
        client_tasks = []

        for i in range(self.concurrent_clients):
            task = asyncio.create_task(
                self._client_worker(i, self.duration_seconds)
            )
            client_tasks.append(task)

        # Wait for all clients to complete
        await asyncio.gather(*client_tasks)

        # Calculate throughput
        throughput = self.request_count / self.duration_seconds

        # Calculate latency statistics
        from ..framework.stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "throughput_rps": throughput,
            "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
            "latency_mean": stats.mean(self.latencies) if self.latencies else 0,
            "latency_p50": stats.percentiles(self.latencies, [50])[50] if self.latencies else 0,
            "latency_p95": stats.percentiles(self.latencies, [95])[95] if self.latencies else 0,
            "latency_p99": stats.percentiles(self.latencies, [99])[99] if self.latencies else 0,
        }

    async def _client_worker(self, client_id: int, duration: int):
        """Simulate a client making requests."""
        end_time = time.time() + duration

        # Mix of different request types
        request_types = [
            self._make_search_request,
            self._make_get_standard_request,
            self._make_list_standards_request,
            self._make_applicable_standards_request,
            self._make_token_optimization_request,
        ]

        while time.time() < end_time:
            # Pick a random request type
            request_func = random.choice(request_types)

            try:
                start = time.perf_counter()
                await request_func()
                end = time.perf_counter()

                latency = end - start
                self.latencies.append(latency)
                self.request_count += 1

            except Exception:
                self.error_count += 1

            # Small random delay between requests
            await asyncio.sleep(random.uniform(0.01, 0.1))

    async def _make_search_request(self):
        """Make a search request."""
        queries = [
            "security",
            "performance",
            "react hooks",
            "authentication",
            "testing strategies"
        ]
        await self.server._search_standards(
            query=random.choice(queries),
            limit=random.randint(5, 20)
        )

    async def _make_get_standard_request(self):
        """Make a get standard request."""
        standard_id = f"test-standard-{random.randint(0, 19)}"
        await self.server._get_standard_details(standard_id)

    async def _make_list_standards_request(self):
        """Make a list standards request."""
        categories = [None, "frontend", "backend", "security", "testing"]
        await self.server._list_available_standards(
            category=random.choice(categories),
            limit=random.randint(10, 50)
        )

    async def _make_applicable_standards_request(self):
        """Make an applicable standards request."""
        contexts = [
            {"language": "python"},
            {"language": "javascript", "framework": "react"},
            {"language": "java", "framework": "spring", "project_type": "api"},
            {"language": "go", "project_type": "microservice"}
        ]
        await self.server._get_applicable_standards(
            context=random.choice(contexts)
        )

    async def _make_token_optimization_request(self):
        """Make a token optimization request."""
        standard_id = f"test-standard-{random.randint(0, 19)}"
        formats = ["full", "condensed", "summary", "reference"]

        await self.server._get_optimized_standard(
            standard_id=standard_id,
            format_type=random.choice(formats),
            token_budget=random.randint(1000, 5000)
        )

    async def teardown(self):
        """Cleanup."""
        pass


class MCPScalabilityBenchmark(BaseBenchmark):
    """Test scalability with increasing load."""

    def __init__(
        self,
        max_clients: int = 100,
        step_size: int = 10,
        duration_per_step: int = 10
    ):
        super().__init__("MCP Scalability Test", 1)
        self.max_clients = max_clients
        self.step_size = step_size
        self.duration_per_step = duration_per_step
        self.results_by_load: dict[int, dict[str, Any]] = {}

    async def setup(self):
        """Setup for scalability test."""
        # Reuse throughput benchmark setup
        self.throughput_bench = MCPThroughputBenchmark(1, 1)
        await self.throughput_bench.setup()
        self.server = self.throughput_bench.server

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run scalability test with increasing load."""
        results = {}

        for num_clients in range(self.step_size, self.max_clients + 1, self.step_size):
            print(f"\nTesting with {num_clients} concurrent clients...")

            # Run throughput test with this load
            bench = MCPThroughputBenchmark(
                concurrent_clients=num_clients,
                duration_seconds=self.duration_per_step
            )
            bench.server = self.server  # Reuse server

            # Reset counters
            bench.request_count = 0
            bench.error_count = 0
            bench.latencies = []

            # Run test
            step_result = await bench.run_single_iteration()

            self.results_by_load[num_clients] = step_result

            # Store in main results
            results[f"clients_{num_clients}_throughput"] = step_result["throughput_rps"]
            results[f"clients_{num_clients}_latency_p95"] = step_result["latency_p95"]
            results[f"clients_{num_clients}_error_rate"] = step_result["error_rate"]

            # Check if we're hitting limits
            if step_result["error_rate"] > 0.1:  # 10% error rate
                print(f"High error rate detected at {num_clients} clients")
                break

        # Analyze scalability
        results["scalability_analysis"] = self._analyze_scalability()

        return results

    def _analyze_scalability(self) -> dict[str, Any]:
        """Analyze scalability characteristics."""
        if len(self.results_by_load) < 2:
            return {"error": "Not enough data points"}

        loads = sorted(self.results_by_load.keys())
        throughputs = [self.results_by_load[load]["throughput_rps"] for load in loads]
        latencies = [self.results_by_load[load]["latency_p95"] for load in loads]

        # Find optimal load (highest throughput)
        max_throughput = max(throughputs)
        optimal_load = loads[throughputs.index(max_throughput)]

        # Find saturation point (throughput stops increasing)
        saturation_load = None
        for i in range(1, len(throughputs)):
            if throughputs[i] < throughputs[i-1] * 0.95:  # 5% drop
                saturation_load = loads[i-1]
                break

        # Calculate efficiency
        efficiencies = []
        for i, load in enumerate(loads):
            if load > 0:
                efficiency = throughputs[i] / load
                efficiencies.append(efficiency)

        return {
            "optimal_load": optimal_load,
            "max_throughput": max_throughput,
            "saturation_load": saturation_load or loads[-1],
            "efficiency_trend": "decreasing" if efficiencies[-1] < efficiencies[0] * 0.8 else "stable",
            "latency_increase": latencies[-1] / latencies[0] if latencies[0] > 0 else 0,
        }

    async def teardown(self):
        """Cleanup."""
        pass
