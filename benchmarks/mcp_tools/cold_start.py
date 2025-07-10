"""Cold start vs warm start performance benchmarks."""

import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark


class MCPColdStartBenchmark(BaseBenchmark):
    """Benchmark cold start vs warm start performance."""

    def __init__(self, iterations: int = 20):
        super().__init__("MCP Cold/Warm Start", iterations)
        self.cold_start_times: list[float] = []
        self.warm_start_times: list[float] = []
        self.first_request_times: list[float] = []
        self.subsequent_request_times: list[float] = []

    async def setup(self):
        """Setup test environment."""
        # Ensure we have test data
        await self._setup_test_data()

    async def _setup_test_data(self):
        """Create test standards."""
        import json
        from pathlib import Path

        data_dir = Path(os.environ.get("MCP_STANDARDS_DATA_DIR", "data"))
        cache_dir = data_dir / "standards" / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create test standards
        for i in range(5):
            standard = {
                "id": f"cold-start-test-{i}",
                "name": f"Cold Start Test {i}",
                "category": "test",
                "tags": ["test", "benchmark"],
                "content": {
                    "overview": f"Test standard {i} for cold start benchmarking",
                    "guidelines": [f"Guideline {j}" for j in range(10)],
                    "examples": [f"Example {j}" for j in range(5)],
                },
            }

            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, "w") as f:
                json.dump(standard, f)

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run cold and warm start tests."""
        # Test 1: Cold start - server initialization
        cold_start_time = await self._measure_cold_start()
        self.cold_start_times.append(cold_start_time)

        # Test 2: Warm start - server already initialized
        warm_start_time = await self._measure_warm_start()
        self.warm_start_times.append(warm_start_time)

        # Test 3: First request after cold start
        first_request_time = await self._measure_first_request()
        self.first_request_times.append(first_request_time)

        # Test 4: Subsequent requests (warmed up)
        subsequent_times = await self._measure_subsequent_requests(5)
        self.subsequent_request_times.extend(subsequent_times)

        return {
            "cold_start": cold_start_time,
            "warm_start": warm_start_time,
            "first_request": first_request_time,
            "avg_subsequent": (
                sum(subsequent_times) / len(subsequent_times) if subsequent_times else 0
            ),
        }

    async def _measure_cold_start(self) -> float:
        """Measure cold start time."""
        # Force garbage collection
        gc.collect()

        # Clear any module caches
        import sys

        # Remove cached modules
        modules_to_clear = [
            "src.mcp_server",
            "src.core.standards.rule_engine",
            "src.core.standards.sync",
            "src.core.standards.token_optimizer",
        ]

        for module in modules_to_clear:
            if module in sys.modules:
                del sys.modules[module]

        # Measure cold start
        start = time.perf_counter()

        # Import and initialize server
        from src.mcp_server import MCPStandardsServer

        server = MCPStandardsServer(
            {"search": {"enabled": False}, "token_model": "gpt-4"}
        )

        # Make first request to fully initialize
        await server._get_sync_status()

        end = time.perf_counter()

        # Cleanup
        del server
        gc.collect()

        return end - start

    async def _measure_warm_start(self) -> float:
        """Measure warm start time (modules already loaded)."""
        start = time.perf_counter()

        # Server initialization with modules already in memory
        server = MCPStandardsServer(
            {"search": {"enabled": False}, "token_model": "gpt-4"}
        )

        # Make first request
        await server._get_sync_status()

        end = time.perf_counter()

        # Keep server for next test
        self._warm_server = server

        return end - start

    async def _measure_first_request(self) -> float:
        """Measure first real request after cold start."""
        # Create new server
        server = MCPStandardsServer(
            {"search": {"enabled": False}, "token_model": "gpt-4"}
        )

        # Measure first meaningful request
        start = time.perf_counter()

        await server._get_applicable_standards(
            {"language": "javascript", "framework": "react", "project_type": "web_app"}
        )

        end = time.perf_counter()

        # Keep server for subsequent requests
        self._request_server = server

        return end - start

    async def _measure_subsequent_requests(self, count: int = 5) -> list[float]:
        """Measure subsequent requests on warmed-up server."""
        if not hasattr(self, "_request_server"):
            # Create server if needed
            self._request_server = MCPStandardsServer(
                {"search": {"enabled": False}, "token_model": "gpt-4"}
            )

        times = []

        # Different request types
        requests = [
            lambda: self._request_server._get_standard_details("cold-start-test-0"),
            lambda: self._request_server._list_available_standards(limit=10),
            lambda: self._request_server._get_applicable_standards(
                {"language": "python"}
            ),
            lambda: self._request_server._get_sync_status(),
            lambda: self._request_server._estimate_token_usage(["cold-start-test-1"]),
        ]

        for i in range(count):
            request = requests[i % len(requests)]

            start = time.perf_counter()
            await request()
            end = time.perf_counter()

            times.append(end - start)

        return times

    async def teardown(self):
        """Cleanup and analyze results."""
        # Cleanup servers
        for attr in ["_warm_server", "_request_server"]:
            if hasattr(self, attr):
                delattr(self, attr)

        gc.collect()

        # Analyze cold start penalty
        self.cold_start_analysis = self._analyze_cold_start_penalty()

    def _analyze_cold_start_penalty(self) -> dict[str, Any]:
        """Analyze cold start performance impact."""
        from ..framework.stats import StatisticalAnalyzer

        stats = StatisticalAnalyzer()

        analysis = {}

        if self.cold_start_times and self.warm_start_times:
            # Calculate penalties
            cold_mean = stats.mean(self.cold_start_times)
            warm_mean = stats.mean(self.warm_start_times)

            analysis["initialization"] = {
                "cold_start_mean": cold_mean,
                "warm_start_mean": warm_mean,
                "penalty_seconds": cold_mean - warm_mean,
                "penalty_factor": cold_mean / warm_mean if warm_mean > 0 else 0,
            }

        if self.first_request_times and self.subsequent_request_times:
            # Request penalties
            first_mean = stats.mean(self.first_request_times)
            subsequent_mean = stats.mean(self.subsequent_request_times)

            analysis["requests"] = {
                "first_request_mean": first_mean,
                "subsequent_mean": subsequent_mean,
                "penalty_seconds": first_mean - subsequent_mean,
                "penalty_factor": (
                    first_mean / subsequent_mean if subsequent_mean > 0 else 0
                ),
            }

            # Performance after warmup
            analysis["warmup_effect"] = {
                "p50_improvement": stats.percentiles(
                    self.subsequent_request_times, [50]
                )[50],
                "p95_improvement": stats.percentiles(
                    self.subsequent_request_times, [95]
                )[95],
                "consistency": (
                    1 - (stats.std_dev(self.subsequent_request_times) / subsequent_mean)
                    if subsequent_mean > 0
                    else 0
                ),
            }

        # Recommendations
        recommendations = []

        if analysis.get("initialization", {}).get("penalty_factor", 0) > 2:
            recommendations.append(
                "High cold start penalty - consider keeping server warm"
            )

        if analysis.get("requests", {}).get("penalty_factor", 0) > 1.5:
            recommendations.append(
                "First request penalty significant - implement request warming"
            )

        analysis["recommendations"] = recommendations

        return analysis


class MCPCacheBenchmark(BaseBenchmark):
    """Benchmark cache effectiveness."""

    def __init__(self, iterations: int = 50):
        super().__init__("MCP Cache Performance", iterations)
        self.server: MCPStandardsServer = None
        self.cache_hits: list[float] = []
        self.cache_misses: list[float] = []

    async def setup(self):
        """Setup server with cache."""
        self.server = MCPStandardsServer(
            {"search": {"enabled": False}, "token_model": "gpt-4"}
        )

        # Create test data
        await self._create_cache_test_data()

    async def _create_cache_test_data(self):
        """Create test standards for cache testing."""
        import json

        cache_dir = self.server.synchronizer.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create standards that will be cached
        for i in range(10):
            standard = {
                "id": f"cache-test-{i}",
                "name": f"Cache Test {i}",
                "category": "test",
                "content": {
                    "overview": "x" * 1000,  # 1KB of content
                    "guidelines": ["y" * 100 for _ in range(10)],
                },
            }

            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, "w") as f:
                json.dump(standard, f)

    async def run_single_iteration(self) -> dict[str, Any]:
        """Test cache performance."""
        # Cold cache - first access
        cold_times = []
        for i in range(5):
            start = time.perf_counter()
            await self.server._get_standard_details(f"cache-test-{i}")
            end = time.perf_counter()
            cold_times.append(end - start)

        # Warm cache - repeated access
        warm_times = []
        for i in range(5):
            start = time.perf_counter()
            await self.server._get_standard_details(f"cache-test-{i}")
            end = time.perf_counter()
            warm_times.append(end - start)

        # Cache miss - non-existent items
        miss_times = []
        for i in range(5):
            start = time.perf_counter()
            try:
                await self.server._get_standard_details(f"non-existent-{i}")
            except Exception:
                pass
            end = time.perf_counter()
            miss_times.append(end - start)

        return {
            "cold_cache_avg": sum(cold_times) / len(cold_times),
            "warm_cache_avg": sum(warm_times) / len(warm_times),
            "cache_miss_avg": sum(miss_times) / len(miss_times),
            "speedup_factor": (
                sum(cold_times) / sum(warm_times) if sum(warm_times) > 0 else 0
            ),
        }

    async def teardown(self):
        """Analyze cache effectiveness."""
        pass
