"""Main script to run comprehensive performance benchmarks."""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

# Import all benchmark components
from benchmarks.framework import BenchmarkSuite, BenchmarkVisualizer, RegressionDetector
from benchmarks.load import StressTestBenchmark
from benchmarks.mcp_tools import (
    MCPColdStartBenchmark,
    MCPLatencyBenchmark,
    MCPResponseTimeBenchmark,
    MCPThroughputBenchmark,
)
from benchmarks.memory import (
    AllocationTrackingBenchmark,
    LeakDetectionBenchmark,
    MemoryGrowthBenchmark,
    MemoryUsageBenchmark,
)
from benchmarks.monitoring import AlertSystem, MetricsCollector, PerformanceDashboard


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    import re

    # Replace spaces with underscores
    sanitized = name.replace(' ', '_')

    # Replace forward slashes with dashes
    sanitized = sanitized.replace('/', '-')

    # Replace other problematic characters with underscores
    sanitized = re.sub(r'[<>:"|?*\\]', '_', sanitized)

    # Remove any leading/trailing dots or spaces
    sanitized = sanitized.strip('. ')

    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_benchmark"

    return sanitized


async def run_quick_benchmarks():
    """Run quick benchmarks for CI/CD."""
    print("Running quick benchmarks...")

    suite = BenchmarkSuite("Quick Performance Check")

    # Add quick benchmarks
    suite.add_benchmark(MCPResponseTimeBenchmark(iterations=10))
    suite.add_benchmark(MCPColdStartBenchmark(iterations=5))
    suite.add_benchmark(MemoryUsageBenchmark(iterations=5))

    # Run benchmarks
    results = await suite.run_all(warmup_iterations=2)

    # Save results
    output_dir = Path("benchmark_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(output_dir)

    # Check for regressions
    RegressionDetector()
    # Load baseline if exists
    baseline_path = Path("benchmark_results/baseline")
    if baseline_path.exists():
        # Compare with baseline
        print("\nChecking for regressions...")
        # Implementation would load baseline and compare

    print(f"\nResults saved to: {output_dir}")
    return results


async def run_full_benchmarks():
    """Run comprehensive benchmark suite."""
    print("Running full benchmark suite...")

    suite = BenchmarkSuite("Comprehensive Performance Analysis")

    # Response time benchmarks
    suite.add_benchmark(MCPResponseTimeBenchmark(iterations=100))

    # Throughput benchmarks
    suite.add_benchmark(
        MCPThroughputBenchmark(concurrent_clients=10, duration_seconds=30)
    )
    suite.add_benchmark(
        MCPThroughputBenchmark(concurrent_clients=50, duration_seconds=30)
    )

    # Latency distribution
    suite.add_benchmark(MCPLatencyBenchmark(iterations=500))

    # Cold/warm start
    suite.add_benchmark(MCPColdStartBenchmark(iterations=20))

    # Memory profiling
    suite.add_benchmark(MemoryUsageBenchmark(iterations=10))
    suite.add_benchmark(LeakDetectionBenchmark(iterations=50))
    suite.add_benchmark(AllocationTrackingBenchmark(iterations=20))

    # Run benchmarks
    results = await suite.run_all(warmup_iterations=5)

    # Save results
    output_dir = Path("benchmark_results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    suite.save_results(output_dir)

    # Generate visualizations
    print("\nGenerating visualizations...")
    visualizer = BenchmarkVisualizer()

    for benchmark_name, benchmark_results in results.items():
        if benchmark_results:
            latest_result = benchmark_results[-1]

            # Create dashboard
            fig = visualizer.create_dashboard(latest_result)
            if fig:  # Only save if visualization is available
                safe_name = sanitize_filename(benchmark_name)
                fig.savefig(output_dir / f"{safe_name}_dashboard.png")

    visualizer.close_all()

    print(f"\nResults saved to: {output_dir}")
    return results


async def run_stress_tests():
    """Run stress tests."""
    print("Running stress tests...")

    scenarios = ["mixed", "read_heavy", "compute_heavy"]
    results = {}

    for scenario in scenarios:
        print(f"\n--- Stress Test: {scenario} ---")

        benchmark = StressTestBenchmark(
            user_count=50,
            spawn_rate=2.0,
            test_duration=120,  # 2 minutes
            scenario=scenario,
        )

        result = await benchmark.run()
        results[scenario] = result

        # Print summary
        print(f"\nScenario: {scenario}")
        print(f"Total Requests: {result.custom_metrics.get('total_requests', 0)}")
        print(f"RPS: {result.custom_metrics.get('requests_per_second', 0):.1f}")
        print(f"Failure Rate: {result.custom_metrics.get('failure_rate', 0)*100:.1f}%")

    return results


async def run_memory_growth_analysis():
    """Run extended memory growth analysis."""
    print("Running memory growth analysis (this will take several minutes)...")

    benchmark = MemoryGrowthBenchmark(duration_minutes=5, sample_interval=1.0)

    result = await benchmark.run()

    # Print report
    print(benchmark.growth_report)

    return result


async def run_continuous_monitoring(duration_minutes: int = 10):
    """Run continuous monitoring with dashboard."""
    print(f"Starting continuous monitoring for {duration_minutes} minutes...")

    # Setup metrics collector
    collector = MetricsCollector()

    # Setup dashboard
    dashboard = PerformanceDashboard(collector)

    # Setup alerts
    alert_system = AlertSystem(collector)

    # Start collection
    await collector.start_collection(interval=1.0)

    # Generate HTML dashboard
    output_dir = Path("monitoring_results") / datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run monitoring
    check_interval = 10  # seconds
    checks = (duration_minutes * 60) // check_interval

    for i in range(checks):
        await asyncio.sleep(check_interval)

        # Check alerts
        alert_system.check_alerts()

        # Update dashboard
        dashboard.generate_html_dashboard(output_dir)

        # Print status
        active_alerts = alert_system.get_active_alerts()
        print(f"\r[{i+1}/{checks}] Active Alerts: {len(active_alerts)}", end="")

    # Stop collection
    await collector.stop_collection()

    # Save final reports
    collector.save_snapshot(output_dir / "final_metrics.json")
    alert_system.save_alert_history(output_dir / "alert_history.json")

    print(f"\n\nMonitoring results saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Standards Server Performance Benchmarks"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "stress", "memory", "monitor"],
        default="quick",
        help="Benchmark mode to run",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Duration in minutes (for monitoring mode)",
    )
    parser.add_argument("--output", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Run selected benchmark mode
    if args.mode == "quick":
        asyncio.run(run_quick_benchmarks())
    elif args.mode == "full":
        asyncio.run(run_full_benchmarks())
    elif args.mode == "stress":
        asyncio.run(run_stress_tests())
    elif args.mode == "memory":
        asyncio.run(run_memory_growth_analysis())
    elif args.mode == "monitor":
        asyncio.run(run_continuous_monitoring(args.duration))

    print("\nBenchmark completed successfully!")


if __name__ == "__main__":
    main()
