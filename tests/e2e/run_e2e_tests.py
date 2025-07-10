#!/usr/bin/env python
"""Helper script to run E2E tests with proper coverage setup."""

import os
import subprocess
import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set environment variables
env = os.environ.copy()
env["PYTHONPATH"] = str(project_root)
env["COVERAGE_PROCESS_START"] = str(project_root / ".coveragerc")

# Run coverage with pytest
cmd = [
    sys.executable,
    "-m",
    "coverage",
    "run",
    "--parallel-mode",
    "-m",
    "pytest",
    "tests/e2e/",
    "--timeout=300",
    "-v",
]

print(f"Running: {' '.join(cmd)}")
print(f"PYTHONPATH: {env['PYTHONPATH']}")
print(f"COVERAGE_PROCESS_START: {env['COVERAGE_PROCESS_START']}")

result = subprocess.run(cmd, env=env, cwd=project_root)
sys.exit(result.returncode)
