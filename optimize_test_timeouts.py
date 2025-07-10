#!/usr/bin/env python3
"""Add timeout markers to slow tests."""

import os
import re

# Tests known to be slow or problematic
SLOW_TEST_PATTERNS = [
    (r"test_semantic_search", 10),  # 10 second timeout for semantic search
    (r"test_embedding", 10),
    (r"test_performance", 30),  # 30 seconds for performance tests
    (r"test_load", 30),
    (r"test_concurrent", 20),
    (r"test_memory", 20),
    (r"test_cache_warmup", 15),
    (r"test_sync", 15),
]


def add_timeout_marker(file_path, test_name, timeout):
    """Add timeout marker to a test function."""
    with open(file_path) as f:
        content = f.read()

    # Find the test function
    pattern = rf"(\n    )(def {test_name}\()"

    # Check if already has timeout marker
    if f"@pytest.mark.timeout({timeout})" in content:
        return False

    # Add timeout marker
    replacement = rf"\1@pytest.mark.timeout({timeout})\n\1\2"
    new_content = re.sub(pattern, replacement, content)

    if new_content != content:
        with open(file_path, "w") as f:
            f.write(new_content)
        return True
    return False


def process_test_file(file_path):
    """Process a test file to add timeout markers."""
    if not file_path.endswith(".py"):
        return

    with open(file_path) as f:
        content = f.read()

    # Find all test functions
    test_functions = re.findall(r"def (test_\w+)\(", content)

    modified = False
    for test_func in test_functions:
        for pattern, timeout in SLOW_TEST_PATTERNS:
            if re.search(pattern, test_func):
                if add_timeout_marker(file_path, test_func, timeout):
                    print(f"Added timeout({timeout}) to {test_func} in {file_path}")
                    modified = True
                break

    return modified


def main():
    """Main entry point."""
    test_dirs = ["tests/unit", "tests/integration", "tests/e2e", "tests/performance"]

    total_modified = 0
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            continue

        for root, _dirs, files in os.walk(test_dir):
            for file in files:
                if file.startswith("test_") and file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    if process_test_file(file_path):
                        total_modified += 1

    print(f"\nModified {total_modified} test files with timeout markers.")


if __name__ == "__main__":
    main()
