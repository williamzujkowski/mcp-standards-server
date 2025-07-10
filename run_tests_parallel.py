#!/usr/bin/env python3
"""Run tests with parallel execution for improved performance."""

import os
import subprocess
import sys
import time


def check_xdist_available():
    """Check if pytest-xdist is available."""
    try:
        import pytest_xdist  # noqa: F401
        return True
    except ImportError:
        return False

def get_cpu_count():
    """Get optimal number of workers."""
    cpu_count = os.cpu_count() or 4
    # Use 1 less than total CPUs to leave room for system
    return max(1, cpu_count - 1)

def run_tests_parallel(test_path="tests", workers=None):
    """Run tests in parallel."""
    if workers is None:
        workers = get_cpu_count()

    print(f"Running tests with {workers} parallel workers...")

    cmd = [
        sys.executable, "-m", "pytest",
        test_path,
        "-n", str(workers),  # Number of parallel workers
        "--dist", "loadgroup",  # Group tests by xdist_group mark
        "--durations=20",
        "--no-cov",  # Disable coverage for speed
        "-v"
    ]

    if not check_xdist_available():
        print("WARNING: pytest-xdist not available. Running tests serially.")
        cmd.remove("-n")
        cmd.remove(str(workers))
        cmd.remove("--dist")
        cmd.remove("loadgroup")

    start_time = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start_time

    print(f"\nTotal time: {elapsed:.2f} seconds")
    return result.returncode

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run tests in parallel")
    parser.add_argument("path", nargs="?", default="tests", help="Test path")
    parser.add_argument("-n", "--workers", type=int, help="Number of workers")
    parser.add_argument("--install-xdist", action="store_true",
                       help="Install pytest-xdist if not available")

    args = parser.parse_args()

    if args.install_xdist and not check_xdist_available():
        print("Installing pytest-xdist...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest-xdist"])

    return run_tests_parallel(args.path, args.workers)

if __name__ == "__main__":
    sys.exit(main())
