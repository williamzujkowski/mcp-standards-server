"""E2E test configuration."""
import pytest
import subprocess
import os


def pytest_sessionfinish(session, exitstatus):
    """Combine coverage data after all tests."""
    if os.environ.get("COVERAGE_PROCESS_START"):
        # Coverage is running in subprocess mode
        try:
            subprocess.run(["coverage", "combine"], check=False)
        except FileNotFoundError:
            pass


@pytest.fixture(scope="session", autouse=True)
def setup_coverage():
    """Setup coverage for subprocesses."""
    # Set environment variable to enable subprocess coverage
    os.environ["COVERAGE_PROCESS_START"] = ".coveragerc"
    yield
    # Cleanup is handled in pytest_sessionfinish