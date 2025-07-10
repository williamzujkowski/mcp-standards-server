"""Scalability testing for MCP server."""

from typing import Any

from ..framework import BaseBenchmark
from .stress_test import StressTestBenchmark


class ScalabilityTestBenchmark(BaseBenchmark):
    """Test server scalability with increasing load."""

    def __init__(
        self, max_users: int = 200, step_size: int = 20, duration_per_step: int = 60
    ):
        super().__init__("Scalability Test", 1)
        self.max_users = max_users
        self.step_size = step_size
        self.duration_per_step = duration_per_step
        self.results_by_load: dict[int, dict[str, Any]] = {}

    async def setup(self):
        """Setup scalability test."""
        self.results_by_load.clear()

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run scalability test with increasing load."""
        print(f"\nRunning scalability test up to {self.max_users} users")

        for num_users in range(self.step_size, self.max_users + 1, self.step_size):
            print(f"\nTesting with {num_users} users...")

            # Run stress test at this load level
            stress_test = StressTestBenchmark(
                user_count=num_users,
                spawn_rate=5.0,
                test_duration=self.duration_per_step,
                scenario="mixed",
            )

            result = await stress_test.run()

            # Extract key metrics
            self.results_by_load[num_users] = {
                "throughput": result.custom_metrics.get("requests_per_second", 0),
                "mean_response_time": result.mean_time,
                "p95_response_time": result.percentiles.get(95, 0),
                "error_rate": result.custom_metrics.get("failure_rate", 0),
                "cpu_usage": result.custom_metrics.get("peak_cpu", 0),
                "memory_usage": result.peak_memory_mb,
            }

            # Check if we've hit saturation
            if self._check_saturation(num_users):
                print(f"Saturation detected at {num_users} users")
                break

        # Analyze scalability
        analysis = self._analyze_scalability()

        return {
            "max_users_tested": max(self.results_by_load.keys()),
            "optimal_load": analysis["optimal_load"],
            "saturation_point": analysis["saturation_point"],
            "max_throughput": analysis["max_throughput"],
            "scalability_factor": analysis["scalability_factor"],
            "detailed_results": self.results_by_load,
        }

    def _check_saturation(self, current_users: int) -> bool:
        """Check if server has reached saturation."""
        if current_users <= self.step_size:
            return False

        current = self.results_by_load[current_users]

        # Check error rate
        if current["error_rate"] > 0.1:  # 10% errors
            return True

        # Check response time degradation
        if current["p95_response_time"] > 2.0:  # 2 seconds
            return True

        # Check throughput plateau
        if current_users >= self.step_size * 3:
            throughputs = [
                self.results_by_load[u]["throughput"]
                for u in sorted(self.results_by_load.keys())[-3:]
            ]
            # If throughput isn't increasing
            if max(throughputs) - min(throughputs) < max(throughputs) * 0.05:
                return True

        return False

    def _analyze_scalability(self) -> dict[str, Any]:
        """Analyze scalability characteristics."""
        users = sorted(self.results_by_load.keys())

        # Find optimal load (best throughput)
        optimal_load = max(users, key=lambda u: self.results_by_load[u]["throughput"])

        # Find saturation point
        saturation_point = optimal_load
        for u in users:
            if self.results_by_load[u]["error_rate"] > 0.05:
                saturation_point = u
                break

        # Calculate scalability factor
        if len(users) >= 2:
            first_throughput = self.results_by_load[users[0]]["throughput"]
            last_throughput = self.results_by_load[users[-1]]["throughput"]
            scalability_factor = (
                last_throughput / first_throughput if first_throughput > 0 else 0
            )
        else:
            scalability_factor = 1.0

        return {
            "optimal_load": optimal_load,
            "saturation_point": saturation_point,
            "max_throughput": self.results_by_load[optimal_load]["throughput"],
            "scalability_factor": scalability_factor,
        }

    async def teardown(self):
        """Generate scalability report."""
        self._generate_scalability_report()

    def _generate_scalability_report(self):
        """Generate scalability analysis report."""
        print("\n" + "=" * 60)
        print("SCALABILITY ANALYSIS REPORT")
        print("=" * 60)

        if not self.results_by_load:
            print("No results to report")
            return

        analysis = self._analyze_scalability()

        print(f"\nOptimal Load: {analysis['optimal_load']} users")
        print(f"Saturation Point: {analysis['saturation_point']} users")
        print(f"Max Throughput: {analysis['max_throughput']:.1f} req/s")
        print(f"Scalability Factor: {analysis['scalability_factor']:.2f}x")

        print("\nLoad Progression:")
        print(f"{'Users':<10} {'Throughput':<12} {'Response':<12} {'Errors':<10}")
        print("-" * 50)

        for users in sorted(self.results_by_load.keys()):
            metrics = self.results_by_load[users]
            print(
                f"{users:<10} "
                f"{metrics['throughput']:<12.1f} "
                f"{metrics['mean_response_time']*1000:<12.1f}ms "
                f"{metrics['error_rate']*100:<10.1f}%"
            )
