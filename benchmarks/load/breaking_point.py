"""Breaking point analysis for MCP server."""

import asyncio
import random
import sys
import time
from pathlib import Path
from typing import Any

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark


class BreakingPointBenchmark(BaseBenchmark):
    """Find the breaking point of the MCP server."""

    def __init__(
        self,
        initial_load: int = 10,
        load_increment: int = 10,
        failure_threshold: float = 0.5,  # 50% failure rate
        timeout_threshold: float = 5.0,  # 5 second timeout
        step_duration: int = 30  # seconds per step
    ):
        super().__init__("Breaking Point Analysis", 1)
        self.initial_load = initial_load
        self.load_increment = load_increment
        self.failure_threshold = failure_threshold
        self.timeout_threshold = timeout_threshold
        self.step_duration = step_duration
        self.breaking_point: int | None = None
        self.load_history: list[dict[str, Any]] = []

    async def setup(self):
        """Setup breaking point test."""
        self.server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })
        self.breaking_point = None
        self.load_history.clear()

    async def run_single_iteration(self) -> dict[str, Any]:
        """Find the breaking point through incremental load."""
        current_load = self.initial_load

        print("\nSearching for breaking point...")
        print(f"Failure threshold: {self.failure_threshold*100}%")
        print(f"Timeout threshold: {self.timeout_threshold}s")

        while not self.breaking_point:
            print(f"\nTesting with {current_load} concurrent requests...")

            # Test at current load
            metrics = await self._test_load_level(current_load)
            self.load_history.append({
                "load": current_load,
                "metrics": metrics
            })

            # Check if we've hit breaking point
            if self._is_breaking_point(metrics):
                self.breaking_point = current_load
                print(f"\n🔥 Breaking point found at {current_load} concurrent requests!")
                break

            # Check if system is still healthy
            if metrics["failure_rate"] < 0.01 and metrics["p95_latency"] < 1.0:
                # System is handling load well, increase more aggressively
                current_load += self.load_increment * 2
            else:
                # System showing stress, increase cautiously
                current_load += self.load_increment

            # Safety limit
            if current_load > 1000:
                print("Safety limit reached without finding breaking point")
                break

        # Analyze the progression
        analysis = self._analyze_breaking_point()

        return {
            "breaking_point": self.breaking_point or "Not found",
            "final_load_tested": current_load,
            "progression": self.load_history,
            "analysis": analysis
        }

    async def _test_load_level(self, concurrent_requests: int) -> dict[str, Any]:
        """Test server at specific load level."""
        results = {
            "successful": 0,
            "failed": 0,
            "timeouts": 0,
            "response_times": []
        }

        # Create test tasks
        tasks = []
        for _ in range(concurrent_requests):
            task = asyncio.create_task(self._make_request())
            tasks.append(task)

        # Execute all tasks
        time.time()

        # Wait for tasks with timeout
        done, pending = await asyncio.wait(
            tasks,
            timeout=self.step_duration
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        # Collect results
        for task in done:
            try:
                success, response_time = task.result()
                if success:
                    results["successful"] += 1
                    results["response_times"].append(response_time)
                else:
                    if response_time >= self.timeout_threshold:
                        results["timeouts"] += 1
                    else:
                        results["failed"] += 1
            except Exception:
                results["failed"] += 1

        # Calculate metrics
        total_requests = results["successful"] + results["failed"] + results["timeouts"]

        metrics = {
            "total_requests": total_requests,
            "successful": results["successful"],
            "failed": results["failed"],
            "timeouts": results["timeouts"],
            "failure_rate": (results["failed"] + results["timeouts"]) / total_requests if total_requests > 0 else 0,
            "timeout_rate": results["timeouts"] / total_requests if total_requests > 0 else 0,
        }

        # Calculate latency percentiles
        if results["response_times"]:
            sorted_times = sorted(results["response_times"])
            metrics["mean_latency"] = sum(sorted_times) / len(sorted_times)
            metrics["p50_latency"] = sorted_times[int(len(sorted_times) * 0.5)]
            metrics["p95_latency"] = sorted_times[int(len(sorted_times) * 0.95)]
            metrics["p99_latency"] = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            metrics["mean_latency"] = 0
            metrics["p50_latency"] = 0
            metrics["p95_latency"] = 0
            metrics["p99_latency"] = 0

        return metrics

    async def _make_request(self) -> tuple[bool, float]:
        """Make a single request and return success status and response time."""
        start = time.perf_counter()

        try:
            # Randomly select operation
            operations = [
                lambda: self.server._get_standard_details("test-standard-0"),
                lambda: self.server._list_available_standards(limit=20),
                lambda: self.server._get_applicable_standards({"language": "python"}),
                lambda: self.server._get_sync_status()
            ]

            operation = random.choice(operations)

            # Execute with timeout
            await asyncio.wait_for(
                operation(),
                timeout=self.timeout_threshold
            )

            response_time = time.perf_counter() - start
            return True, response_time

        except asyncio.TimeoutError:
            return False, self.timeout_threshold
        except Exception:
            response_time = time.perf_counter() - start
            return False, response_time

    def _is_breaking_point(self, metrics: dict[str, Any]) -> bool:
        """Check if current metrics indicate breaking point."""
        # Check failure rate
        if metrics["failure_rate"] >= self.failure_threshold:
            return True

        # Check timeout rate
        if metrics["timeout_rate"] >= 0.3:  # 30% timeouts
            return True

        # Check extreme latency
        if metrics["p95_latency"] >= self.timeout_threshold * 0.8:
            return True

        return False

    def _analyze_breaking_point(self) -> dict[str, Any]:
        """Analyze the path to breaking point."""
        if not self.load_history:
            return {}

        analysis = {
            "healthy_load": None,
            "degradation_start": None,
            "failure_acceleration": None
        }

        # Find last healthy load level
        for entry in self.load_history:
            metrics = entry["metrics"]
            if metrics["failure_rate"] < 0.01 and metrics["p95_latency"] < 1.0:
                analysis["healthy_load"] = entry["load"]

        # Find where degradation started
        for i, entry in enumerate(self.load_history):
            metrics = entry["metrics"]
            if metrics["failure_rate"] > 0.05 or metrics["p95_latency"] > 2.0:
                analysis["degradation_start"] = entry["load"]

                # Calculate acceleration
                if i > 0:
                    prev_metrics = self.load_history[i-1]["metrics"]
                    load_increase = entry["load"] - self.load_history[i-1]["load"]
                    failure_increase = metrics["failure_rate"] - prev_metrics["failure_rate"]
                    analysis["failure_acceleration"] = failure_increase / load_increase if load_increase > 0 else 0
                break

        return analysis

    async def teardown(self):
        """Generate breaking point report."""
        self._generate_breaking_point_report()

    def _generate_breaking_point_report(self):
        """Generate detailed breaking point analysis report."""
        print("\n" + "="*60)
        print("BREAKING POINT ANALYSIS REPORT")
        print("="*60)

        if self.breaking_point:
            print(f"\nBreaking Point: {self.breaking_point} concurrent requests")
        else:
            print("\nBreaking point not found within test limits")

        if self.load_history:
            print("\nLoad Progression:")
            print(f"{'Load':<10} {'Success':<10} {'Fail Rate':<12} {'P95 Latency':<12}")
            print("-" * 50)

            for entry in self.load_history[-10:]:  # Last 10 entries
                metrics = entry["metrics"]
                print(
                    f"{entry['load']:<10} "
                    f"{metrics['successful']:<10} "
                    f"{metrics['failure_rate']*100:<12.1f}% "
                    f"{metrics['p95_latency']*1000:<12.1f}ms"
                )
