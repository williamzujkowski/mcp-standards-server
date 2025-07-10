#!/usr/bin/env python3
"""Comprehensive performance benchmarks for MCP Standards Server."""

import asyncio
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import psutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.cache.redis_client import RedisCache  # noqa: E402
from src.core.standards.rule_engine import RuleEngine  # noqa: E402
from src.core.standards.token_optimizer import TokenOptimizer  # noqa: E402
from src.mcp_server import MCPStandardsServer  # noqa: E402


class PerformanceMetrics:
    """Container for performance metrics."""

    def __init__(self, name: str):
        self.name = name
        self.response_times: list[float] = []
        self.memory_usage: list[float] = []
        self.cpu_usage: list[float] = []
        self.errors: list[str] = []
        self.start_time = None
        self.end_time = None

    def add_measurement(self, response_time: float, memory_mb: float, cpu_percent: float):
        """Add a measurement."""
        self.response_times.append(response_time)
        self.memory_usage.append(memory_mb)
        self.cpu_usage.append(cpu_percent)

    def add_error(self, error: str):
        """Add an error."""
        self.errors.append(error)

    def get_summary(self) -> dict[str, Any]:
        """Get summary statistics."""
        if not self.response_times:
            return {"error": "No measurements collected"}

        return {
            "name": self.name,
            "measurements": len(self.response_times),
            "errors": len(self.errors),
            "response_time": {
                "mean": mean(self.response_times),
                "median": median(self.response_times),
                "min": min(self.response_times),
                "max": max(self.response_times),
                "stdev": stdev(self.response_times) if len(self.response_times) > 1 else 0,
                "p95": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0,
                "p99": sorted(self.response_times)[int(len(self.response_times) * 0.99)] if self.response_times else 0,
            },
            "memory_mb": {
                "mean": mean(self.memory_usage),
                "max": max(self.memory_usage),
                "min": min(self.memory_usage),
            },
            "cpu_percent": {
                "mean": mean(self.cpu_usage),
                "max": max(self.cpu_usage),
            },
            "throughput": len(self.response_times) / (self.end_time - self.start_time) if self.start_time and self.end_time else 0,
        }


class ComprehensiveBenchmark:
    """Comprehensive benchmark suite for MCP Standards Server."""

    def __init__(self):
        self.server = None
        self.rule_engine = None
        self.token_optimizer = None
        self.metrics: dict[str, PerformanceMetrics] = {}
        self.process = psutil.Process()

    async def setup(self):
        """Setup components for benchmarking."""
        print("Setting up benchmark environment...")

        # Setup MCP Server
        config = {
            "search": {"enabled": False},
            "token_model": "gpt-4",
            "default_token_budget": 8000
        }
        self.server = MCPStandardsServer(config)

        # Setup other components
        self.rule_engine = RuleEngine()
        self.token_optimizer = TokenOptimizer()

        print("✓ Setup complete")

    async def benchmark_mcp_tools(self, iterations: int = 50):
        """Benchmark MCP tool response times."""
        print("\n=== MCP Tool Response Times ===")

        tools = [
            ("get_sync_status", {}),
            ("list_available_standards", {"limit": 10}),
            ("get_standard_details", {"standard_id": "CODING_STANDARDS"}),
            ("search_standards", {"query": "security", "limit": 5}),
            ("get_applicable_standards", {
                "context": {"language": "python", "framework": "fastapi"}
            }),
            ("estimate_token_usage", {
                "standard_ids": ["CODING_STANDARDS"],
                "format_types": ["full", "condensed"]
            }),
        ]

        for tool_name, args in tools:
            metrics = PerformanceMetrics(f"mcp_tool_{tool_name}")
            metrics.start_time = time.time()

            method = getattr(self.server, f"_{tool_name}", None)
            if not method:
                print(f"✗ Tool {tool_name} not found")
                continue

            # Warmup
            for _ in range(5):
                try:
                    await method(**args)
                except Exception:
                    pass

            # Benchmark
            print(f"\nBenchmarking {tool_name}...")
            for i in range(iterations):
                try:
                    # Get resource usage before
                    memory_before = self.process.memory_info().rss / 1024 / 1024
                    # cpu_before = self.process.cpu_percent(interval=None)  # Not used

                    # Measure response time
                    start = time.perf_counter()
                    await method(**args)
                    end = time.perf_counter()

                    # Get resource usage after
                    memory_after = self.process.memory_info().rss / 1024 / 1024
                    cpu_after = self.process.cpu_percent(interval=None)

                    metrics.add_measurement(
                        response_time=end - start,
                        memory_mb=memory_after - memory_before,
                        cpu_percent=cpu_after
                    )

                    if i == 0:
                        print(f"  First run: {end - start:.4f}s")

                except Exception as e:
                    metrics.add_error(str(e))

            metrics.end_time = time.time()
            self.metrics[tool_name] = metrics

            summary = metrics.get_summary()
            print(f"  ✓ Completed {summary['measurements']} iterations")
            print(f"  Mean time: {summary['response_time']['mean']:.4f}s")
            print(f"  P95 time: {summary['response_time']['p95']:.4f}s")

    async def benchmark_cache_performance(self, iterations: int = 100):
        """Benchmark cache performance."""
        print("\n=== Cache Performance ===")

        # Test Redis connection if available
        try:
            redis_cache = RedisCache()
            metrics = PerformanceMetrics("redis_cache")
            metrics.start_time = time.time()

            test_key = "benchmark_test"
            test_value = {"test": "data" * 100}

            print("\nBenchmarking Redis cache...")
            for i in range(iterations):
                try:
                    # Measure async SET operation
                    start = time.perf_counter()
                    await redis_cache.aset(test_key, test_value)
                    set_time = time.perf_counter() - start

                    # Measure async GET operation
                    start = time.perf_counter()
                    _ = await redis_cache.aget(test_key)
                    get_time = time.perf_counter() - start

                    metrics.add_measurement(
                        response_time=(set_time + get_time) / 2,
                        memory_mb=0,
                        cpu_percent=0
                    )

                    if i == 0:
                        print(f"  First SET: {set_time:.4f}s")
                        print(f"  First GET: {get_time:.4f}s")

                except Exception as e:
                    metrics.add_error(str(e))

            metrics.end_time = time.time()
            self.metrics["redis_cache"] = metrics

            summary = metrics.get_summary()
            print(f"  ✓ Completed {summary['measurements']} iterations")
            print(f"  Mean time: {summary['response_time']['mean']:.4f}s")
        except Exception as e:
            print(f"  ✗ Redis not available: {e}")

        # Test in-memory cache
        metrics = PerformanceMetrics("memory_cache")
        metrics.start_time = time.time()

        cache = {}
        test_data = {"test": "data" * 100}

        print("\nBenchmarking in-memory cache...")
        for i in range(iterations * 10):  # More iterations for in-memory
            try:
                # Measure write
                start = time.perf_counter()
                cache[f"key_{i}"] = test_data
                write_time = time.perf_counter() - start

                # Measure read
                start = time.perf_counter()
                _ = cache.get(f"key_{i}")
                read_time = time.perf_counter() - start

                metrics.add_measurement(
                    response_time=(write_time + read_time) / 2,
                    memory_mb=sys.getsizeof(cache) / 1024 / 1024,
                    cpu_percent=0
                )

            except Exception as e:
                metrics.add_error(str(e))

        metrics.end_time = time.time()
        self.metrics["memory_cache"] = metrics

        summary = metrics.get_summary()
        print(f"  ✓ Completed {summary['measurements']} iterations")
        print(f"  Mean time: {summary['response_time']['mean']:.6f}s")

    async def benchmark_rule_engine(self, iterations: int = 100):
        """Benchmark rule engine performance."""
        print("\n=== Rule Engine Performance ===")

        metrics = PerformanceMetrics("rule_engine")
        metrics.start_time = time.time()

        test_contexts = [
            {"language": "python"},
            {"language": "javascript", "framework": "react"},
            {"language": "go", "project_type": "microservice"},
            {"language": "java", "framework": "spring", "team_size": "large"},
            {
                "language": "typescript",
                "framework": "angular",
                "project_type": "enterprise",
                "deployment": "kubernetes",
                "team_size": "large",
                "compliance": ["pci", "gdpr"]
            }
        ]

        print("\nBenchmarking rule engine...")
        for i in range(iterations):
            for context in test_contexts:
                try:
                    start = time.perf_counter()
                    matches = self.rule_engine.evaluate(context)
                    end = time.perf_counter()

                    metrics.add_measurement(
                        response_time=end - start,
                        memory_mb=0,
                        cpu_percent=0
                    )

                    if i == 0 and context == test_contexts[0]:
                        print(f"  First run: {end - start:.4f}s ({len(matches)} matches)")

                except Exception as e:
                    metrics.add_error(str(e))

        metrics.end_time = time.time()
        self.metrics["rule_engine"] = metrics

        summary = metrics.get_summary()
        print(f"  ✓ Completed {summary['measurements']} evaluations")
        print(f"  Mean time: {summary['response_time']['mean']:.4f}s")
        print(f"  P95 time: {summary['response_time']['p95']:.4f}s")

    async def benchmark_memory_usage(self):
        """Benchmark memory usage patterns."""
        print("\n=== Memory Usage Analysis ===")

        # Enable tracemalloc
        tracemalloc.start()

        # Take initial snapshot
        snapshot1 = tracemalloc.take_snapshot()
        initial_memory = self.process.memory_info().rss / 1024 / 1024

        print(f"Initial memory: {initial_memory:.2f} MB")

        # Perform operations
        print("\nPerforming memory-intensive operations...")

        # Load many standards
        for _ in range(10):
            await self.server._list_available_standards(limit=100)

        # Create large contexts
        large_contexts = []
        for i in range(100):
            large_contexts.append({
                "language": "python",
                "data": "x" * 1000,
                "index": i
            })
            self.rule_engine.evaluate(large_contexts[-1])

        # Take second snapshot
        snapshot2 = tracemalloc.take_snapshot()
        final_memory = self.process.memory_info().rss / 1024 / 1024

        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory growth: {final_memory - initial_memory:.2f} MB")

        # Analyze top memory allocations
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        print("\nTop 5 memory allocations:")
        for stat in top_stats[:5]:
            print(f"  {stat}")

        tracemalloc.stop()

        # Store memory metrics
        metrics = PerformanceMetrics("memory_usage")
        metrics.add_measurement(
            response_time=0,
            memory_mb=final_memory - initial_memory,
            cpu_percent=0
        )
        self.metrics["memory_usage"] = metrics

    async def benchmark_throughput(self, duration_seconds: int = 30):
        """Benchmark throughput under concurrent load."""
        print(f"\n=== Throughput Test ({duration_seconds}s) ===")

        metrics = PerformanceMetrics("throughput")
        request_count = 0
        error_count = 0

        async def worker():
            nonlocal request_count, error_count
            while time.time() < end_time:
                try:
                    start = time.perf_counter()
                    await self.server._list_available_standards(limit=10)
                    end = time.perf_counter()

                    metrics.add_measurement(
                        response_time=end - start,
                        memory_mb=0,
                        cpu_percent=0
                    )
                    request_count += 1
                except Exception as e:
                    error_count += 1
                    metrics.add_error(str(e))

        print("\nRunning throughput test with 10 concurrent workers...")
        start_time = time.time()
        end_time = start_time + duration_seconds

        metrics.start_time = start_time

        # Run concurrent workers
        workers = [asyncio.create_task(worker()) for _ in range(10)]
        await asyncio.gather(*workers)

        metrics.end_time = time.time()
        self.metrics["throughput"] = metrics

        actual_duration = metrics.end_time - metrics.start_time
        rps = request_count / actual_duration

        print(f"  ✓ Completed {request_count} requests in {actual_duration:.1f}s")
        print(f"  Throughput: {rps:.1f} requests/second")
        print(f"  Error rate: {(error_count / (request_count + error_count)) * 100:.1f}%")

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive benchmark report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
                "python_version": sys.version,
                "platform": sys.platform,
            },
            "benchmarks": {},
            "summary": {
                "total_benchmarks": len(self.metrics),
                "total_errors": sum(len(m.errors) for m in self.metrics.values()),
            }
        }

        for name, metrics in self.metrics.items():
            report["benchmarks"][name] = metrics.get_summary()

        return report

    async def run(self):
        """Run all benchmarks."""
        print("MCP Standards Server - Comprehensive Performance Benchmarks")
        print("=" * 60)

        await self.setup()

        # Run all benchmarks
        await self.benchmark_mcp_tools(iterations=50)
        await self.benchmark_cache_performance(iterations=100)
        await self.benchmark_rule_engine(iterations=50)
        await self.benchmark_memory_usage()
        await self.benchmark_throughput(duration_seconds=10)

        # Generate and save report
        report = self.generate_report()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(project_root) / "benchmark_results" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "comprehensive_results.json", "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)

        for name, metrics in report["benchmarks"].items():
            if "response_time" in metrics:
                print(f"\n{name}:")
                print(f"  Mean response time: {metrics['response_time']['mean']:.4f}s")
                print(f"  P95 response time: {metrics['response_time'].get('p95', 0):.4f}s")
                if "throughput" in metrics:
                    print(f"  Throughput: {metrics['throughput']:.1f} ops/s")

        print(f"\n✓ Results saved to: {output_dir}")

        # Create/update baseline
        baseline_dir = Path(project_root) / "benchmark_results" / "baseline"
        baseline_file = baseline_dir / "comprehensive_baseline.json"

        if not baseline_file.exists():
            baseline_dir.mkdir(parents=True, exist_ok=True)
            with open(baseline_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"✓ Created baseline: {baseline_file}")
        else:
            # Compare with baseline
            with open(baseline_file) as f:
                baseline = json.load(f)

            print("\n" + "=" * 60)
            print("PERFORMANCE COMPARISON WITH BASELINE")
            print("=" * 60)

            for name, metrics in report["benchmarks"].items():
                if name in baseline["benchmarks"] and "response_time" in metrics:
                    baseline_mean = baseline["benchmarks"][name]["response_time"]["mean"]
                    current_mean = metrics["response_time"]["mean"]
                    diff = ((current_mean - baseline_mean) / baseline_mean) * 100

                    status = "✓" if diff < 10 else "⚠" if diff < 20 else "✗"
                    print(f"\n{name}:")
                    print(f"  Baseline: {baseline_mean:.4f}s")
                    print(f"  Current:  {current_mean:.4f}s")
                    print(f"  {status} Change: {diff:+.1f}%")


async def main():
    """Main entry point."""
    benchmark = ComprehensiveBenchmark()
    await benchmark.run()


if __name__ == "__main__":
    asyncio.run(main())
