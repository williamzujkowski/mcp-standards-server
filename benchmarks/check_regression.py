#!/usr/bin/env python3
"""Check for performance regressions against baseline."""

import json
import sys
from pathlib import Path

# Regression thresholds
THRESHOLDS = {
    "response_time": 0.10,  # 10% increase is warning
    "response_time_critical": 0.20,  # 20% increase is critical
    "memory": 0.15,  # 15% increase is warning
    "memory_critical": 0.25,  # 25% increase is critical
    "throughput": -0.10,  # 10% decrease is warning
    "throughput_critical": -0.20,  # 20% decrease is critical
}


def load_results(filepath: Path) -> dict:
    """Load benchmark results from file."""
    with open(filepath) as f:
        return json.load(f)


def compare_metrics(baseline: dict, current: dict) -> list[tuple[str, str, float, str]]:
    """Compare current metrics against baseline."""
    regressions = []

    for benchmark_name, current_metrics in current.get("benchmarks", {}).items():
        if benchmark_name not in baseline.get("benchmarks", {}):
            continue

        baseline_metrics = baseline["benchmarks"][benchmark_name]

        # Compare response times
        if "response_time" in current_metrics and "response_time" in baseline_metrics:
            baseline_mean = baseline_metrics["response_time"]["mean"]
            current_mean = current_metrics["response_time"]["mean"]

            if baseline_mean > 0:
                change = (current_mean - baseline_mean) / baseline_mean

                if change > THRESHOLDS["response_time_critical"]:
                    regressions.append(
                        (benchmark_name, "response_time", change, "CRITICAL")
                    )
                elif change > THRESHOLDS["response_time"]:
                    regressions.append(
                        (benchmark_name, "response_time", change, "WARNING")
                    )

        # Compare throughput
        if "throughput" in current_metrics and "throughput" in baseline_metrics:
            baseline_throughput = baseline_metrics["throughput"]
            current_throughput = current_metrics["throughput"]

            if baseline_throughput > 0:
                change = (
                    current_throughput - baseline_throughput
                ) / baseline_throughput

                if change < THRESHOLDS["throughput_critical"]:
                    regressions.append(
                        (benchmark_name, "throughput", change, "CRITICAL")
                    )
                elif change < THRESHOLDS["throughput"]:
                    regressions.append(
                        (benchmark_name, "throughput", change, "WARNING")
                    )

        # Compare memory usage
        if "memory_mb" in current_metrics and "memory_mb" in baseline_metrics:
            baseline_memory = baseline_metrics["memory_mb"]["max"]
            current_memory = current_metrics["memory_mb"]["max"]

            if baseline_memory > 0:
                change = (current_memory - baseline_memory) / baseline_memory

                if change > THRESHOLDS["memory_critical"]:
                    regressions.append((benchmark_name, "memory", change, "CRITICAL"))
                elif change > THRESHOLDS["memory"]:
                    regressions.append((benchmark_name, "memory", change, "WARNING"))

    return regressions


def print_report(
    baseline: dict, current: dict, regressions: list[tuple[str, str, float, str]]
):
    """Print regression report."""
    print("Performance Regression Analysis")
    print("=" * 60)
    print(f"Baseline: {baseline.get('timestamp', 'Unknown')}")
    print(f"Current:  {current.get('timestamp', 'Unknown')}")
    print()

    if not regressions:
        print("✅ No performance regressions detected!")
        return

    # Group by severity
    criticals = [r for r in regressions if r[3] == "CRITICAL"]
    warnings = [r for r in regressions if r[3] == "WARNING"]

    if criticals:
        print(f"❌ CRITICAL Regressions ({len(criticals)}):")
        for benchmark, metric, change, _ in criticals:
            print(f"  - {benchmark}.{metric}: {change:+.1%}")

    if warnings:
        print(f"\n⚠️  WARNING Regressions ({len(warnings)}):")
        for benchmark, metric, change, _ in warnings:
            print(f"  - {benchmark}.{metric}: {change:+.1%}")

    print("\nDetailed Comparison:")
    print("-" * 60)

    for benchmark_name in sorted(current.get("benchmarks", {}).keys()):
        if benchmark_name not in baseline.get("benchmarks", {}):
            continue

        baseline_metrics = baseline["benchmarks"][benchmark_name]
        current_metrics = current["benchmarks"][benchmark_name]

        print(f"\n{benchmark_name}:")

        if "response_time" in current_metrics and "response_time" in baseline_metrics:
            baseline_mean = baseline_metrics["response_time"]["mean"]
            current_mean = current_metrics["response_time"]["mean"]
            change = (
                ((current_mean - baseline_mean) / baseline_mean * 100)
                if baseline_mean > 0
                else 0
            )

            status = "✓" if abs(change) < 10 else "⚠" if abs(change) < 20 else "✗"
            print(
                f"  Response Time: {baseline_mean:.4f}s → {current_mean:.4f}s ({change:+.1f}%) {status}"
            )

        if "throughput" in current_metrics and "throughput" in baseline_metrics:
            baseline_tps = baseline_metrics["throughput"]
            current_tps = current_metrics["throughput"]
            change = (
                ((current_tps - baseline_tps) / baseline_tps * 100)
                if baseline_tps > 0
                else 0
            )

            status = "✓" if change > -10 else "⚠" if change > -20 else "✗"
            print(
                f"  Throughput: {baseline_tps:.1f} → {current_tps:.1f} ops/s ({change:+.1f}%) {status}"
            )


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument(
        "--baseline",
        type=str,
        default="benchmark_results/baseline/comprehensive_baseline.json",
        help="Path to baseline results",
    )
    parser.add_argument(
        "--current", type=str, help="Path to current results (defaults to latest)"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with non-zero code if regressions found",
    )

    args = parser.parse_args()

    # Find project root
    project_root = Path(__file__).parent.parent

    # Load baseline
    baseline_path = project_root / args.baseline
    if not baseline_path.exists():
        print(f"❌ Baseline not found: {baseline_path}")
        sys.exit(1)

    baseline = load_results(baseline_path)

    # Find current results
    if args.current:
        current_path = Path(args.current)
    else:
        # Find latest results
        results_dir = project_root / "benchmark_results"
        dirs = [d for d in results_dir.iterdir() if d.is_dir() and d.name != "baseline"]
        if not dirs:
            print("❌ No benchmark results found")
            sys.exit(1)

        latest_dir = sorted(dirs)[-1]
        current_path = latest_dir / "comprehensive_results.json"

    if not current_path.exists():
        print(f"❌ Current results not found: {current_path}")
        sys.exit(1)

    current = load_results(current_path)

    # Compare
    regressions = compare_metrics(baseline, current)

    # Print report
    print_report(baseline, current, regressions)

    # Exit code
    if args.fail_on_regression and regressions:
        criticals = [r for r in regressions if r[3] == "CRITICAL"]
        if criticals:
            sys.exit(2)  # Critical regression
        else:
            sys.exit(1)  # Warning regression

    sys.exit(0)


if __name__ == "__main__":
    main()
