#!/usr/bin/env python3
"""Create test data for E2E tests."""

import json
from pathlib import Path

# Import the setup function
from test_data_setup import setup_test_data


def main():
    """Create test data in multiple locations for E2E tests."""

    # Create test data in cross-platform temporary directory
    import tempfile

    tmp_dir = Path(tempfile.gettempdir()) / "test_standards_data"
    print(f"Creating test data in {tmp_dir}")
    setup_test_data(tmp_dir)

    # Also create test data in project's data directory
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print(f"Creating test data in {data_dir}")
    setup_test_data(data_dir)

    # Verify created files
    cache_dir = tmp_dir / "standards" / "cache"
    if cache_dir.exists():
        files = list(cache_dir.glob("*.json"))
        print(f"\nCreated {len(files)} standard files:")
        for file_path in files:
            print(f"  - {file_path.name}")

    rules_file = tmp_dir / "standards" / "meta" / "enhanced-selection-rules.json"
    if rules_file.exists():
        print(f"\nCreated rules file: {rules_file}")
        with open(rules_file) as rule_file:
            rules_data = json.load(rule_file)
            print(f"  - Contains {len(rules_data['rules'])} rules")

    sync_file = tmp_dir / "standards" / "sync_config.yaml"
    if sync_file.exists():
        print(f"\nCreated sync config: {sync_file}")


if __name__ == "__main__":
    main()
