#!/usr/bin/env python3
"""
Example of running E2E tests programmatically.

This demonstrates how to run tests from Python code and analyze results.
"""

import json
import subprocess
import sys
from pathlib import Path


def run_e2e_tests(test_type="all", coverage=True, performance=False):
    """Run E2E tests and return results."""

    cmd = ["pytest", "tests/e2e/"]

    # Add test selection
    if test_type == "performance":
        cmd.extend(["-m", "performance"])
    elif test_type == "functional":
        cmd.extend(["-m", "not performance and not benchmark"])

    # Add coverage
    if coverage:
        cmd.extend([
            "--cov=src",
            "--cov-report=json",
            "--cov-report=term"
        ])

    # Add JSON output for parsing
    cmd.extend(["--json-report", "--json-report-file=test_results.json"])

    # Run tests
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse results
    test_results = None
    if Path("test_results.json").exists():
        with open("test_results.json") as f:
            test_results = json.load(f)

    coverage_data = None
    if coverage and Path("coverage.json").exists():
        with open("coverage.json") as f:
            coverage_data = json.load(f)

    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "test_results": test_results,
        "coverage": coverage_data
    }


def analyze_results(results):
    """Analyze test results and print summary."""

    print("\n" + "="*60)
    print("E2E TEST RESULTS SUMMARY")
    print("="*60)

    if results["exit_code"] == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")

    if results["test_results"]:
        summary = results["test_results"]["summary"]
        print(f"\nTests run: {summary['total']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"Duration: {summary['duration']:.2f}s")

        # Show failed tests
        if summary["failed"] > 0:
            print("\nFailed tests:")
            for test in results["test_results"]["tests"]:
                if test["outcome"] == "failed":
                    print(f"  - {test['nodeid']}")
                    if "message" in test:
                        print(f"    {test['message']}")

    if results["coverage"]:
        coverage = results["coverage"]["totals"]
        print("\nCode Coverage:")
        print(f"  Lines: {coverage['percent_covered']:.1f}%")
        print(f"  Statements: {coverage['num_statements']}")
        print(f"  Missing: {coverage['missing_lines']}")


def run_performance_benchmarks():
    """Run performance benchmarks and check for regressions."""

    print("\nRunning performance benchmarks...")

    cmd = [
        "pytest",
        "tests/e2e/test_performance.py",
        "-m", "benchmark",
        "--benchmark-json=benchmark_results.json",
        "--benchmark-only"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0 and Path("benchmark_results.json").exists():
        with open("benchmark_results.json") as f:
            benchmarks = json.load(f)

        print("\nBenchmark Results:")
        for bench in benchmarks.get("benchmarks", []):
            print(f"  {bench['name']}:")
            print(f"    Mean: {bench['stats']['mean']*1000:.2f}ms")
            print(f"    Min: {bench['stats']['min']*1000:.2f}ms")
            print(f"    Max: {bench['stats']['max']*1000:.2f}ms")

        # Check for regressions
        if Path("benchmark_baseline.json").exists():
            check_cmd = [
                "python",
                "scripts/detect_performance_regression.py",
                "benchmark_results.json",
                "benchmark_baseline.json",
                "--threshold", "10.0"
            ]

            regression_result = subprocess.run(
                check_cmd,
                capture_output=True,
                text=True
            )

            print("\nRegression Check:")
            print(regression_result.stdout)


def main():
    """Main entry point."""

    print("MCP Standards Server - E2E Test Runner Example")
    print("=" * 60)

    # Run functional tests
    print("\n1. Running functional E2E tests...")
    functional_results = run_e2e_tests(test_type="functional", coverage=True)
    analyze_results(functional_results)

    # Run performance tests
    print("\n2. Running performance tests...")
    performance_results = run_e2e_tests(test_type="performance", coverage=False)

    if performance_results["exit_code"] == 0:
        run_performance_benchmarks()

    # Overall result
    overall_success = (
        functional_results["exit_code"] == 0 and
        performance_results["exit_code"] == 0
    )

    print("\n" + "="*60)
    if overall_success:
        print("✅ All E2E tests completed successfully!")
        return 0
    else:
        print("❌ Some E2E tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
