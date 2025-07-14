#!/usr/bin/env python3
"""
MCP Standards Server Performance Benchmarking Suite

This script provides comprehensive performance testing for all MCP operations
including response time measurement, load testing, and resource monitoring.
"""

import asyncio
import json
import statistics
import time
import tracemalloc
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import aiohttp
import matplotlib.pyplot as plt
import psutil


@dataclass
class BenchmarkResult:
    """Store results from a single benchmark run"""

    operation: str
    scenario: str
    duration: float
    success: bool
    error: str = None
    memory_used: float = 0
    cpu_percent: float = 0
    timestamp: str = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class BenchmarkSummary:
    """Summary statistics for a benchmark scenario"""

    operation: str
    scenario: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    min_duration: float
    max_duration: float
    avg_duration: float
    median_duration: float
    p95_duration: float
    p99_duration: float
    avg_memory_mb: float
    avg_cpu_percent: float
    requests_per_second: float
    error_rate: float


class MCPBenchmarkClient:
    """Client for benchmarking MCP operations"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def list_available_standards(self, filters: dict = None) -> dict:
        """Benchmark list_available_standards operation"""
        params = filters or {}
        async with self.session.get(
            f"{self.base_url}/mcp/list_available_standards", params=params
        ) as response:
            return await response.json()

    async def get_applicable_standards(self, context: dict) -> dict:
        """Benchmark get_applicable_standards operation"""
        async with self.session.post(
            f"{self.base_url}/mcp/get_applicable_standards", json=context
        ) as response:
            return await response.json()

    async def search_standards(self, query: str, options: dict = None) -> dict:
        """Benchmark search_standards operation"""
        data = {"query": query}
        if options:
            data.update(options)
        async with self.session.post(
            f"{self.base_url}/mcp/search_standards", json=data
        ) as response:
            return await response.json()

    async def get_standard(self, standard_id: str, format: str = "full") -> dict:
        """Benchmark get_standard operation"""
        async with self.session.get(
            f"{self.base_url}/mcp/get_standard/{standard_id}", params={"format": format}
        ) as response:
            return await response.json()

    async def get_optimized_standard(self, standard_id: str, token_limit: int) -> dict:
        """Benchmark get_optimized_standard operation"""
        async with self.session.post(
            f"{self.base_url}/mcp/get_optimized_standard",
            json={"standard_id": standard_id, "token_limit": token_limit},
        ) as response:
            return await response.json()

    async def validate_against_standard(self, code_path: str, standard_id: str) -> dict:
        """Benchmark validate_against_standard operation"""
        async with self.session.post(
            f"{self.base_url}/mcp/validate_against_standard",
            json={"code_path": code_path, "standard_id": standard_id},
        ) as response:
            return await response.json()

    async def get_compliance_mapping(self, standard_id: str = None) -> dict:
        """Benchmark get_compliance_mapping operation"""
        params = {"standard_id": standard_id} if standard_id else {}
        async with self.session.get(
            f"{self.base_url}/mcp/get_compliance_mapping", params=params
        ) as response:
            return await response.json()


class PerformanceBenchmark:
    """Main benchmark orchestrator"""

    def __init__(self, output_dir: Path = None):
        self.output_dir = output_dir or Path("./evaluation/results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    async def measure_operation(
        self, operation: Callable, operation_name: str, scenario: str
    ) -> BenchmarkResult:
        """Measure a single operation's performance"""
        # Start resource monitoring
        process = psutil.Process()
        tracemalloc.start()
        cpu_before = process.cpu_percent()

        # Execute operation
        start_time = time.perf_counter()
        success = True
        error = None

        try:
            await operation()
        except Exception as e:
            success = False
            error = str(e)

        # Calculate metrics
        duration = time.perf_counter() - start_time
        current, peak = tracemalloc.get_traced_memory()
        memory_mb = peak / 1024 / 1024
        cpu_percent = process.cpu_percent() - cpu_before
        tracemalloc.stop()

        return BenchmarkResult(
            operation=operation_name,
            scenario=scenario,
            duration=duration,
            success=success,
            error=error,
            memory_used=memory_mb,
            cpu_percent=cpu_percent,
        )

    async def run_scenario(
        self,
        client: MCPBenchmarkClient,
        operation_name: str,
        operation: Callable,
        scenario: str,
        iterations: int = 100,
    ) -> list[BenchmarkResult]:
        """Run a benchmark scenario with multiple iterations"""
        scenario_results = []

        for i in range(iterations):
            result = await self.measure_operation(operation, operation_name, scenario)
            scenario_results.append(result)
            self.results.append(result)

            # Small delay to prevent overwhelming the server
            if i % 10 == 0:
                await asyncio.sleep(0.1)

        return scenario_results

    async def run_concurrent_scenario(
        self,
        client: MCPBenchmarkClient,
        operation_name: str,
        operation: Callable,
        scenario: str,
        concurrent_users: int,
        requests_per_user: int,
    ) -> list[BenchmarkResult]:
        """Run concurrent user scenario"""
        tasks = []

        for _ in range(concurrent_users):
            task = self.run_scenario(
                client,
                operation_name,
                operation,
                f"{scenario}_{concurrent_users}_users",
                requests_per_user,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        return [r for sublist in results for r in sublist]

    def summarize_results(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics for benchmark results"""
        successful_results = [r for r in results if r.success]
        durations = [r.duration for r in successful_results]

        if not durations:
            return None

        total_time = sum(durations)

        return BenchmarkSummary(
            operation=results[0].operation,
            scenario=results[0].scenario,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(results) - len(successful_results),
            min_duration=min(durations),
            max_duration=max(durations),
            avg_duration=statistics.mean(durations),
            median_duration=statistics.median(durations),
            p95_duration=(
                statistics.quantiles(durations, n=20)[18]
                if len(durations) > 20
                else max(durations)
            ),
            p99_duration=(
                statistics.quantiles(durations, n=100)[98]
                if len(durations) > 100
                else max(durations)
            ),
            avg_memory_mb=statistics.mean([r.memory_used for r in successful_results]),
            avg_cpu_percent=statistics.mean(
                [r.cpu_percent for r in successful_results]
            ),
            requests_per_second=(
                len(successful_results) / total_time if total_time > 0 else 0
            ),
            error_rate=(len(results) - len(successful_results)) / len(results),
        )

    async def run_all_benchmarks(self):
        """Run complete benchmark suite"""
        async with MCPBenchmarkClient() as client:
            # Define test scenarios
            scenarios = [
                # list_available_standards scenarios
                {
                    "name": "list_all_standards",
                    "operation": lambda: client.list_available_standards(),
                    "operation_name": "list_available_standards",
                },
                {
                    "name": "list_filtered_standards",
                    "operation": lambda: client.list_available_standards(
                        {"category": "security"}
                    ),
                    "operation_name": "list_available_standards",
                },
                # get_applicable_standards scenarios
                {
                    "name": "get_applicable_web_app",
                    "operation": lambda: client.get_applicable_standards(
                        {
                            "project_type": "web_application",
                            "framework": "react",
                            "requirements": ["security", "accessibility"],
                        }
                    ),
                    "operation_name": "get_applicable_standards",
                },
                {
                    "name": "get_applicable_complex",
                    "operation": lambda: client.get_applicable_standards(
                        {
                            "project_type": "microservice",
                            "languages": ["python", "go", "typescript"],
                            "requirements": ["security", "performance", "compliance"],
                            "frameworks": ["fastapi", "gin", "express"],
                        }
                    ),
                    "operation_name": "get_applicable_standards",
                },
                # search_standards scenarios
                {
                    "name": "search_simple",
                    "operation": lambda: client.search_standards("security"),
                    "operation_name": "search_standards",
                },
                {
                    "name": "search_complex",
                    "operation": lambda: client.search_standards(
                        "security AND authentication NOT oauth", {"fuzzy": True}
                    ),
                    "operation_name": "search_standards",
                },
                # get_standard scenarios
                {
                    "name": "get_standard_full",
                    "operation": lambda: client.get_standard(
                        "security-review-audit-process", "full"
                    ),
                    "operation_name": "get_standard",
                },
                {
                    "name": "get_standard_condensed",
                    "operation": lambda: client.get_standard(
                        "security-review-audit-process", "condensed"
                    ),
                    "operation_name": "get_standard",
                },
                # get_optimized_standard scenarios
                {
                    "name": "optimize_4k_tokens",
                    "operation": lambda: client.get_optimized_standard(
                        "security-review-audit-process", 4000
                    ),
                    "operation_name": "get_optimized_standard",
                },
                {
                    "name": "optimize_16k_tokens",
                    "operation": lambda: client.get_optimized_standard(
                        "security-review-audit-process", 16000
                    ),
                    "operation_name": "get_optimized_standard",
                },
                # get_compliance_mapping scenarios
                {
                    "name": "get_all_mappings",
                    "operation": lambda: client.get_compliance_mapping(),
                    "operation_name": "get_compliance_mapping",
                },
                {
                    "name": "get_standard_mapping",
                    "operation": lambda: client.get_compliance_mapping(
                        "security-review-audit-process"
                    ),
                    "operation_name": "get_compliance_mapping",
                },
            ]

            # Run baseline performance tests
            print("Running baseline performance tests...")
            baseline_results = {}
            for scenario in scenarios:
                print(f"  Testing {scenario['name']}...")
                results = await self.run_scenario(
                    client,
                    scenario["operation_name"],
                    scenario["operation"],
                    f"baseline_{scenario['name']}",
                    iterations=100,
                )
                baseline_results[scenario["name"]] = self.summarize_results(results)

            # Run concurrent user tests
            print("\nRunning concurrent user tests...")
            concurrent_tests = [
                (10, 10),  # 10 users, 10 requests each
                (50, 5),  # 50 users, 5 requests each
                (100, 3),  # 100 users, 3 requests each
            ]

            concurrent_results = {}
            for users, requests in concurrent_tests:
                print(f"  Testing with {users} concurrent users...")
                for scenario in scenarios[:3]:  # Test subset for concurrent
                    results = await self.run_concurrent_scenario(
                        client,
                        scenario["operation_name"],
                        scenario["operation"],
                        scenario["name"],
                        users,
                        requests,
                    )
                    key = f"{scenario['name']}_{users}_users"
                    concurrent_results[key] = self.summarize_results(results)

            # Run spike test
            print("\nRunning spike test...")
            spike_results = await self.run_spike_test(client)

            # Generate reports
            self.generate_reports(baseline_results, concurrent_results, spike_results)

    async def run_spike_test(self, client: MCPBenchmarkClient) -> list[BenchmarkResult]:
        """Simulate traffic spike from 0 to 500 users"""
        spike_results = []

        # Gradually increase load
        for users in [1, 10, 50, 100, 200, 500]:
            print(f"  Spike test: {users} users...")
            tasks = []

            for _ in range(users):

                def operation():
                    return client.list_available_standards()

                task = self.measure_operation(
                    operation, "list_available_standards", f"spike_{users}_users"
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            spike_results.extend(results)

            # Brief pause between waves
            await asyncio.sleep(2)

        return spike_results

    def generate_reports(
        self,
        baseline_results: dict[str, BenchmarkSummary],
        concurrent_results: dict[str, BenchmarkSummary],
        spike_results: list[BenchmarkResult],
    ):
        """Generate comprehensive benchmark reports"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save raw results
        raw_results = {
            "timestamp": timestamp,
            "baseline": {k: asdict(v) for k, v in baseline_results.items() if v},
            "concurrent": {k: asdict(v) for k, v in concurrent_results.items() if v},
            "spike": [asdict(r) for r in spike_results],
        }

        with open(self.output_dir / f"benchmark_results_{timestamp}.json", "w") as f:
            json.dump(raw_results, f, indent=2)

        # Generate summary report
        self.generate_summary_report(
            baseline_results, concurrent_results, spike_results, timestamp
        )

        # Generate visualizations
        self.generate_visualizations(
            baseline_results, concurrent_results, spike_results, timestamp
        )

    def generate_summary_report(
        self,
        baseline_results: dict[str, BenchmarkSummary],
        concurrent_results: dict[str, BenchmarkSummary],
        spike_results: list[BenchmarkResult],
        timestamp: str,
    ):
        """Generate markdown summary report"""
        report = f"""# MCP Performance Benchmark Report

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents comprehensive performance testing results for the MCP Standards Server.

## Baseline Performance Results

| Operation | Scenario | Avg Response (ms) | P95 (ms) | P99 (ms) | RPS | Error Rate |
|-----------|----------|-------------------|----------|----------|-----|------------|
"""

        for name, summary in baseline_results.items():
            if summary:
                report += f"| {summary.operation} | {name} | {summary.avg_duration*1000:.2f} | "
                report += f"{summary.p95_duration*1000:.2f} | {summary.p99_duration*1000:.2f} | "
                report += (
                    f"{summary.requests_per_second:.2f} | {summary.error_rate:.2%} |\n"
                )

        report += """
## Concurrent User Performance

| Scenario | Users | Avg Response (ms) | P95 (ms) | RPS | Error Rate |
|----------|-------|-------------------|----------|-----|------------|
"""

        for name, summary in concurrent_results.items():
            if summary:
                users = name.split("_")[-2]
                report += f"| {summary.operation} | {users} | {summary.avg_duration*1000:.2f} | "
                report += f"{summary.p95_duration*1000:.2f} | {summary.requests_per_second:.2f} | "
                report += f"{summary.error_rate:.2%} |\n"

        # Analyze spike test results
        spike_summary = self.analyze_spike_results(spike_results)
        report += f"""
## Spike Test Results

- Maximum successful concurrent users: {spike_summary['max_successful_users']}
- Performance degradation point: {spike_summary['degradation_point']} users
- Maximum response time: {spike_summary['max_response_time']:.2f}ms
- Error rate at peak: {spike_summary['peak_error_rate']:.2%}

## Performance vs. Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| list_available_standards | <100ms | {self.get_baseline_avg('list_all_standards', baseline_results):.2f}ms | {self.check_target(100, self.get_baseline_avg('list_all_standards', baseline_results))} |
| get_applicable_standards | <200ms | {self.get_baseline_avg('get_applicable_web_app', baseline_results):.2f}ms | {self.check_target(200, self.get_baseline_avg('get_applicable_web_app', baseline_results))} |
| search_standards | <150ms | {self.get_baseline_avg('search_simple', baseline_results):.2f}ms | {self.check_target(150, self.get_baseline_avg('search_simple', baseline_results))} |
| get_standard | <50ms | {self.get_baseline_avg('get_standard_full', baseline_results):.2f}ms | {self.check_target(50, self.get_baseline_avg('get_standard_full', baseline_results))} |

## Recommendations

1. **Performance Optimizations**:
   - Consider caching frequently accessed standards
   - Implement connection pooling for database queries
   - Add CDN for static standard content

2. **Scalability Improvements**:
   - Implement horizontal scaling for high concurrent loads
   - Add rate limiting to prevent resource exhaustion
   - Consider async processing for validation operations

3. **Monitoring Requirements**:
   - Set up alerts for response times exceeding P95 thresholds
   - Monitor error rates and set automatic scaling triggers
   - Track cache hit rates and optimize cache strategy
"""

        with open(self.output_dir / f"benchmark_report_{timestamp}.md", "w") as f:
            f.write(report)

    def generate_visualizations(
        self,
        baseline_results: dict[str, BenchmarkSummary],
        concurrent_results: dict[str, BenchmarkSummary],
        spike_results: list[BenchmarkResult],
        timestamp: str,
    ):
        """Generate performance visualization charts"""
        # Response time comparison chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("MCP Performance Benchmark Results", fontsize=16)

        # Baseline response times
        ax1 = axes[0, 0]
        operations = []
        avg_times = []
        p95_times = []

        for name, summary in baseline_results.items():
            if summary:
                operations.append(name)
                avg_times.append(summary.avg_duration * 1000)
                p95_times.append(summary.p95_duration * 1000)

        x = range(len(operations))
        ax1.bar([i - 0.2 for i in x], avg_times, 0.4, label="Average", alpha=0.8)
        ax1.bar([i + 0.2 for i in x], p95_times, 0.4, label="P95", alpha=0.8)
        ax1.set_xticks(x)
        ax1.set_xticklabels(operations, rotation=45, ha="right")
        ax1.set_ylabel("Response Time (ms)")
        ax1.set_title("Baseline Response Times")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Concurrent user impact
        ax2 = axes[0, 1]
        user_counts = [10, 50, 100]
        operation_types = [
            "list_all_standards",
            "get_applicable_web_app",
            "search_simple",
        ]

        for op in operation_types:
            response_times = []
            for users in user_counts:
                key = f"{op}_{users}_users"
                if key in concurrent_results and concurrent_results[key]:
                    response_times.append(concurrent_results[key].avg_duration * 1000)
                else:
                    response_times.append(0)
            ax2.plot(user_counts, response_times, marker="o", label=op)

        ax2.set_xlabel("Concurrent Users")
        ax2.set_ylabel("Avg Response Time (ms)")
        ax2.set_title("Response Time vs Concurrent Users")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Spike test results
        ax3 = axes[1, 0]
        spike_data = {}
        for result in spike_results:
            users = int(result.scenario.split("_")[1])
            if users not in spike_data:
                spike_data[users] = {"times": [], "errors": 0, "total": 0}
            spike_data[users]["total"] += 1
            if result.success:
                spike_data[users]["times"].append(result.duration * 1000)
            else:
                spike_data[users]["errors"] += 1

        users = sorted(spike_data.keys())
        avg_times = []
        error_rates = []

        for u in users:
            times = spike_data[u]["times"]
            avg_times.append(statistics.mean(times) if times else 0)
            error_rates.append(spike_data[u]["errors"] / spike_data[u]["total"] * 100)

        ax3_twin = ax3.twinx()
        line1 = ax3.plot(users, avg_times, "b-", marker="o", label="Avg Response Time")
        line2 = ax3_twin.plot(
            users, error_rates, "r--", marker="s", label="Error Rate %"
        )

        ax3.set_xlabel("Number of Users")
        ax3.set_ylabel("Avg Response Time (ms)", color="b")
        ax3_twin.set_ylabel("Error Rate (%)", color="r")
        ax3.set_title("Spike Test Results")

        # Combine legends
        lines = line1 + line2
        labels = [line.get_label() for line in lines]
        ax3.legend(lines, labels, loc="upper left")
        ax3.grid(True, alpha=0.3)

        # Resource utilization
        ax4 = axes[1, 1]
        operations = []
        memory_usage = []
        cpu_usage = []

        for name, summary in baseline_results.items():
            if summary:
                operations.append(name)
                memory_usage.append(summary.avg_memory_mb)
                cpu_usage.append(summary.avg_cpu_percent)

        x = range(len(operations))
        ax4_twin = ax4.twinx()

        bar1 = ax4.bar(
            [i - 0.2 for i in x],
            memory_usage,
            0.4,
            label="Memory (MB)",
            alpha=0.8,
            color="green",
        )
        bar2 = ax4_twin.bar(
            [i + 0.2 for i in x],
            cpu_usage,
            0.4,
            label="CPU (%)",
            alpha=0.8,
            color="orange",
        )

        ax4.set_xticks(x)
        ax4.set_xticklabels(operations, rotation=45, ha="right")
        ax4.set_ylabel("Memory Usage (MB)", color="green")
        ax4_twin.set_ylabel("CPU Usage (%)", color="orange")
        ax4.set_title("Resource Utilization")

        # Combine legends
        ax4.legend([bar1, bar2], ["Memory (MB)", "CPU (%)"], loc="upper left")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"benchmark_charts_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def get_baseline_avg(
        self, scenario: str, results: dict[str, BenchmarkSummary]
    ) -> float:
        """Get average response time for a scenario in milliseconds"""
        if scenario in results and results[scenario]:
            return results[scenario].avg_duration * 1000
        return 0

    def check_target(self, target: float, actual: float) -> str:
        """Check if performance meets target"""
        if actual == 0:
            return "❓ No Data"
        elif actual <= target:
            return "✅ Pass"
        elif actual <= target * 1.2:
            return "⚠️ Warning"
        else:
            return "❌ Fail"

    def analyze_spike_results(self, spike_results: list[BenchmarkResult]) -> dict:
        """Analyze spike test results"""
        analysis = {
            "max_successful_users": 0,
            "degradation_point": 0,
            "max_response_time": 0,
            "peak_error_rate": 0,
        }

        by_users = {}
        for result in spike_results:
            users = int(result.scenario.split("_")[1])
            if users not in by_users:
                by_users[users] = {"success": 0, "total": 0, "times": []}

            by_users[users]["total"] += 1
            if result.success:
                by_users[users]["success"] += 1
                by_users[users]["times"].append(result.duration * 1000)
                analysis["max_response_time"] = max(
                    analysis["max_response_time"], result.duration * 1000
                )

        # Find degradation point
        prev_avg = 0
        for users in sorted(by_users.keys()):
            error_rate = 1 - (by_users[users]["success"] / by_users[users]["total"])

            if error_rate < 0.01:  # Less than 1% errors
                analysis["max_successful_users"] = users

            if by_users[users]["times"]:
                avg_time = statistics.mean(by_users[users]["times"])
                if prev_avg > 0 and avg_time > prev_avg * 1.5:  # 50% increase
                    analysis["degradation_point"] = users
                    break
                prev_avg = avg_time

        # Peak error rate
        if by_users:
            max_users = max(by_users.keys())
            analysis["peak_error_rate"] = 1 - (
                by_users[max_users]["success"] / by_users[max_users]["total"]
            )

        return analysis


async def main():
    """Run the complete benchmark suite"""
    print("🚀 Starting MCP Performance Benchmark Suite")
    print("=" * 60)

    benchmark = PerformanceBenchmark()

    try:
        await benchmark.run_all_benchmarks()
        print("\n✅ Benchmark completed successfully!")
        print(f"Results saved to: {benchmark.output_dir}")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
