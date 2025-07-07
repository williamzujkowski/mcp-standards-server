"""
Global pytest configuration and fixtures for all tests.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Generator

import pytest
import pytest_asyncio

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Configure asyncio for tests
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for tests."""
    if sys.platform == "win32":
        # Windows requires special handling
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    return asyncio.get_event_loop_policy()


@pytest.fixture(scope="session")
def event_loop(event_loop_policy):
    """Create event loop for session."""
    loop = event_loop_policy.new_event_loop()
    yield loop
    loop.close()


# Environment setup
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    os.environ["MCP_TEST_MODE"] = "true"
    os.environ["MCP_LOG_LEVEL"] = "DEBUG"
    os.environ["MCP_DISABLE_TELEMETRY"] = "true"
    
    yield
    
    # Cleanup
    os.environ.pop("MCP_TEST_MODE", None)
    os.environ.pop("MCP_LOG_LEVEL", None)
    os.environ.pop("MCP_DISABLE_TELEMETRY", None)


# Test data directories
@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to test data directory."""
    return PROJECT_ROOT / "tests" / "data"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Path to fixtures directory."""
    return PROJECT_ROOT / "tests" / "fixtures"


# Benchmark fixtures
@pytest.fixture
def benchmark_data():
    """Provide data for benchmark tests."""
    return {
        "small_context": {
            "project_type": "api",
            "language": "python"
        },
        "medium_context": {
            "project_type": "web_application",
            "framework": "react",
            "language": "javascript",
            "requirements": ["accessibility", "performance"]
        },
        "large_context": {
            "project_type": "microservice",
            "framework": "spring-boot",
            "language": "java",
            "requirements": ["security", "scalability", "monitoring"],
            "deployment": "kubernetes",
            "database": ["postgresql", "redis"],
            "messaging": "kafka",
            "team_size": "large"
        }
    }


# Skip markers for conditional tests
@pytest.fixture(autouse=True)
def skip_on_ci(request):
    """Skip tests marked with 'skip_on_ci' when running in CI."""
    if request.node.get_closest_marker("skip_on_ci"):
        if os.environ.get("CI"):
            pytest.skip("Skipping test in CI environment")


@pytest.fixture(autouse=True)
def skip_without_mcp(request):
    """Skip tests that require MCP when it's not available."""
    if request.node.get_closest_marker("mcp"):
        try:
            import mcp
        except ImportError:
            pytest.skip("MCP not installed")


# Performance tracking
@pytest.fixture
def track_performance(request):
    """Track test performance metrics."""
    import time
    import psutil
    
    process = psutil.Process()
    
    # Before test
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    yield
    
    # After test
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    duration = end_time - start_time
    memory_delta = end_memory - start_memory
    
    # Add metrics to test report
    if hasattr(request.node, "user_properties"):
        request.node.user_properties.append(("duration", duration))
        request.node.user_properties.append(("memory_delta_mb", memory_delta))


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear any test caches
    if hasattr(asyncio, "_test_cache"):
        asyncio._test_cache.clear()


# Custom assertions
class CustomAssertions:
    """Custom assertion helpers for tests."""
    
    @staticmethod
    def assert_performance(duration: float, max_duration: float, operation: str = "Operation"):
        """Assert that an operation completed within time limit."""
        assert duration <= max_duration, (
            f"{operation} took {duration:.3f}s, "
            f"exceeding limit of {max_duration:.3f}s"
        )
        
    @staticmethod
    def assert_memory_usage(memory_mb: float, max_memory_mb: float):
        """Assert that memory usage is within limits."""
        assert memory_mb <= max_memory_mb, (
            f"Memory usage {memory_mb:.2f}MB "
            f"exceeds limit of {max_memory_mb:.2f}MB"
        )


@pytest.fixture
def custom_assertions():
    """Provide custom assertion helpers."""
    return CustomAssertions()


# Test doubles and mocks
@pytest.fixture
def mock_github_api(monkeypatch):
    """Mock GitHub API calls."""
    class MockGitHubAPI:
        def __init__(self):
            self.calls = []
            
        def get_file(self, path):
            self.calls.append(("get_file", path))
            return {"content": "mocked content", "sha": "abc123"}
            
        def list_files(self, path):
            self.calls.append(("list_files", path))
            return ["file1.md", "file2.yaml"]
            
    mock_api = MockGitHubAPI()
    monkeypatch.setattr("src.core.standards.sync.github_api", mock_api)
    return mock_api


# Pytest plugins configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "skip_on_ci: skip test when running in CI"
    )
    config.addinivalue_line(
        "markers", "requires_redis: test requires Redis server"
    )
    config.addinivalue_line(
        "markers", "requires_docker: test requires Docker"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        # Add markers based on path
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
            
        # Add markers based on test name
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "benchmark" in item.name:
            item.add_marker(pytest.mark.benchmark)