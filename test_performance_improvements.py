#!/usr/bin/env python3
"""Script to test performance improvements."""

import subprocess
import sys
import time


def run_tests(test_path, description):
    """Run tests and measure time."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print('='*60)

    start_time = time.time()

    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-v",
        "--durations=10",
        "--no-cov",  # Disable coverage for performance testing
        "-x",  # Stop on first failure
        "--tb=short"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    elapsed = time.time() - start_time

    print(f"\nElapsed time: {elapsed:.2f} seconds")

    if result.returncode != 0:
        print("\nTest failures detected:")
        print(result.stdout[-2000:])  # Last 2000 chars
        print("\nErrors:")
        print(result.stderr[-1000:])  # Last 1000 chars
    else:
        print("\nAll tests passed!")
        # Extract test count from output
        for line in result.stdout.split('\n'):
            if " passed" in line and "failed" not in line:
                print(f"Summary: {line.strip()}")
                break

    return elapsed, result.returncode == 0

def main():
    """Run various test suites and compare performance."""
    test_suites = [
        ("tests/unit/core/cache", "Unit tests - Cache"),
        ("tests/unit/analyzers", "Unit tests - Analyzers"),
        ("tests/unit/core/standards", "Unit tests - Standards"),
        ("tests/integration", "Integration tests"),
        ("tests/e2e/test_mcp_server.py", "E2E - MCP Server"),
        ("tests/performance", "Performance tests"),
    ]

    total_time = 0
    all_passed = True
    results = []

    print("Running test performance analysis...")

    for test_path, description in test_suites:
        elapsed, passed = run_tests(test_path, description)
        total_time += elapsed
        all_passed = all_passed and passed
        results.append((description, elapsed, passed))

    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print('='*60)

    for desc, elapsed, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{desc:<40} {elapsed:>8.2f}s  {status}")

    print(f"\n{'Total time:':<40} {total_time:>8.2f}s")
    print(f"Overall status: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
