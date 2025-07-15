"""
Global pytest configuration and fixtures for all tests.
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

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
        "small_context": {"project_type": "api", "language": "python"},
        "medium_context": {
            "project_type": "web_application",
            "framework": "react",
            "language": "javascript",
            "requirements": ["accessibility", "performance"],
        },
        "large_context": {
            "project_type": "microservice",
            "framework": "spring-boot",
            "language": "java",
            "requirements": ["security", "scalability", "monitoring"],
            "deployment": "kubernetes",
            "database": ["postgresql", "redis"],
            "messaging": "kafka",
            "team_size": "large",
        },
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
        import importlib.util

        if importlib.util.find_spec("mcp") is None:
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
@pytest.fixture(autouse=True, scope="function")
def cleanup_after_test(request):
    """Cleanup after each test - only when needed."""
    yield

    # Only force GC for memory-intensive tests
    if request.node.get_closest_marker("memory_intensive"):
        import gc

        gc.collect()

    # Clear any test caches
    if hasattr(asyncio, "_test_cache"):
        asyncio._test_cache.clear()


# Custom assertions
class CustomAssertions:
    """Custom assertion helpers for tests."""

    @staticmethod
    def assert_performance(
        duration: float, max_duration: float, operation: str = "Operation"
    ):
        """Assert that an operation completed within time limit."""
        assert duration <= max_duration, (
            f"{operation} took {duration:.3f}s, "
            f"exceeding limit of {max_duration:.3f}s"
        )

    @staticmethod
    def assert_memory_usage(memory_mb: float, max_memory_mb: float):
        """Assert that memory usage is within limits."""
        assert memory_mb <= max_memory_mb, (
            f"Memory usage {memory_mb:.2f}MB " f"exceeds limit of {max_memory_mb:.2f}MB"
        )


@pytest.fixture
def custom_assertions():
    """Provide custom assertion helpers."""
    return CustomAssertions()


# Apply ML mocks for tests that need them
@pytest.fixture
def use_ml_mocks(mock_ml_dependencies):
    """Ensure ML dependencies are mocked for tests that need them."""
    # Just by depending on mock_ml_dependencies, we ensure it's initialized
    pass


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


@pytest.fixture(scope="session", autouse=True)
def mock_ml_dependencies():
    """Mock ML dependencies for tests - session scoped for performance."""
    import sys
    from types import ModuleType
    from typing import cast
    from unittest.mock import Mock

    from tests.mocks import MockSentenceTransformer
    from tests.mocks.semantic_search_mocks import (
        MockCosineSimilarity,
        MockFuzz,
        MockNearestNeighbors,
        MockProcess,
        MockRedisClient,
    )

    # Store original modules for cleanup
    original_modules = {}

    # Mock sentence-transformers EARLY and COMPLETELY
    class MockSentenceTransformersModule:
        SentenceTransformer = MockSentenceTransformer
        __version__ = "2.0.0"  # Mock version

    # Mock sklearn components
    class MockPairwiseModule:
        cosine_similarity = MockCosineSimilarity.cosine_similarity

    class MockMetricsModule:
        pairwise = MockPairwiseModule()

    class MockNeighborsModule:
        NearestNeighbors = MockNearestNeighbors

    class MockPCA:
        """Mock PCA for sklearn.decomposition."""

        def __init__(self, n_components=None, **kwargs):
            self.n_components = n_components

        def fit(self, X):
            return self

        def transform(self, X):
            # Return data with reduced dimensions if specified
            if self.n_components and hasattr(X, "shape"):
                return X[:, : self.n_components]
            return X

        def fit_transform(self, X):
            return self.fit(X).transform(X)

    class MockDecompositionModule:
        PCA = MockPCA

    class MockSklearnModule:
        neighbors = MockNeighborsModule()
        metrics = MockMetricsModule()
        decomposition = MockDecompositionModule()

    # Mock NLTK download function - this is the main issue with timeouts
    def mock_nltk_download(*args, **kwargs):
        return True  # Always succeed, don't download anything

    # Mock fuzzywuzzy
    class MockFuzzyWuzzyModule:
        fuzz = MockFuzz()
        process = MockProcess()

    # Mock huggingface_hub to prevent downloads
    import tempfile

    class MockHuggingFaceHub:
        def snapshot_download(*args, **kwargs):
            return str(Path(tempfile.gettempdir()) / "mock_model")

        def hf_hub_download(*args, **kwargs):
            return str(Path(tempfile.gettempdir()) / "mock_file")

        def try_to_load_from_cache(*args, **kwargs):
            return None

        def cached_download(*args, **kwargs):
            return str(Path(tempfile.gettempdir()) / "mock_file")

        def get_hf_file_metadata(*args, **kwargs):
            return {"etag": "mock_etag"}

    # Mock torch to prevent CUDA checks and model loading
    class MockTorch:
        cuda = type("cuda", (), {"is_available": lambda: False})()

        @staticmethod
        def device(x):
            return x

        @staticmethod
        def no_grad():
            return type(
                "no_grad",
                (),
                {"__enter__": lambda self: None, "__exit__": lambda self, *args: None},
            )()

        class FloatTensor:
            pass

        @staticmethod
        def tensor(data):
            import numpy as np

            return np.array(data)

        @staticmethod
        def load(*args, **kwargs):
            return {"model": "mock"}

    # Mock transformers completely to prevent any model loading
    class MockTransformers:
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return type(
                    "MockTokenizer",
                    (),
                    {
                        "encode": lambda self, text: [1, 2, 3],
                        "decode": lambda self, tokens: "mock text",
                        "tokenize": lambda self, text: ["mock", "tokens"],
                    },
                )()

        class AutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                import numpy as np

                return type(
                    "MockModel",
                    (),
                    {
                        "eval": lambda self: None,
                        "encode": lambda self, *args, **kwargs: np.random.randn(1, 384),
                    },
                )()

        class AutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return type("MockConfig", (), {"hidden_size": 384})()

    # Aggressively set HuggingFace offline environment variables
    import os

    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    # Use cross-platform temp directory for cache locations
    temp_dir = tempfile.gettempdir()
    os.environ["TORCH_HOME"] = str(Path(temp_dir) / "torch_cache")
    os.environ["HF_HOME"] = str(Path(temp_dir) / "hf_cache")
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(Path(temp_dir) / "st_cache")

    # Store and replace modules BEFORE any imports happen
    modules_to_mock = {
        "sentence_transformers": MockSentenceTransformersModule(),
        "sklearn": MockSklearnModule(),
        "sklearn.neighbors": MockNeighborsModule(),
        "sklearn.metrics": MockMetricsModule(),
        "sklearn.metrics.pairwise": MockPairwiseModule(),
        "sklearn.decomposition": MockDecompositionModule(),
        "fuzzywuzzy": MockFuzzyWuzzyModule(),
        "huggingface_hub": MockHuggingFaceHub(),
        "torch": MockTorch(),
        "transformers": MockTransformers(),
    }

    for module_name, mock_module in modules_to_mock.items():
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
        sys.modules[module_name] = cast(ModuleType, mock_module)

    # Create mock redis module
    redis_module = type(sys)("redis")
    redis_module.Redis = MockRedisClient
    redis_module.StrictRedis = MockRedisClient  # Some code uses StrictRedis
    if "redis" in sys.modules:
        original_modules["redis"] = sys.modules["redis"]
    sys.modules["redis"] = redis_module

    # Patch already imported modules if they exist
    try:
        import sentence_transformers

        sentence_transformers.SentenceTransformer = MockSentenceTransformer
    except ImportError:
        pass

    try:
        import redis

        redis.Redis = MockRedisClient
        redis.StrictRedis = MockRedisClient
    except ImportError:
        pass

    # Patch NLTK download function if NLTK is available
    try:
        import nltk

        # Store original for restoration
        original_modules["nltk.download"] = getattr(nltk, "download", None)
        # Replace with mock
        nltk.download = mock_nltk_download

        # Mock nltk.corpus.stopwords to prevent LookupError
        from tests.mocks.semantic_search_mocks import MockStopwords

        # Create a mock corpus module with proper stopwords attribute
        class MockCorpus:
            class stopwords:
                @staticmethod
                def words(language="english"):
                    return MockStopwords.words(language)

        # Patch nltk.corpus if it exists
        if hasattr(nltk, "corpus"):
            original_modules["nltk.corpus"] = nltk.corpus
            nltk.corpus = MockCorpus()

    except ImportError:
        # If NLTK isn't installed, create minimal mock only if needed
        pass

    # Mock requests to prevent any HTTP calls
    class MockResponse:
        def __init__(self, status_code=200):
            self.status_code = status_code
            self.text = "mocked response"
            self.content = b"mocked response"

        def json(self):
            return {"mocked": True}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise Exception(f"HTTP {self.status_code}")

    def mock_requests_get(*args, **kwargs):
        return MockResponse()

    def mock_requests_post(*args, **kwargs):
        return MockResponse()

    # Mock requests module
    class MockRequestsModule:
        get = staticmethod(mock_requests_get)
        post = staticmethod(mock_requests_post)
        Session = Mock

    if "requests" in sys.modules:
        original_modules["requests"] = sys.modules["requests"]
    sys.modules["requests"] = cast(ModuleType, MockRequestsModule())

    yield

    # Restore original modules if needed (optional for session scope)
    # for module_name, original_module in original_modules.items():
    #     sys.modules[module_name] = original_module


# Pytest plugins configuration
def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line("markers", "skip_on_ci: skip test when running in CI")
    config.addinivalue_line("markers", "requires_redis: test requires Redis server")
    config.addinivalue_line("markers", "requires_docker: test requires Docker")


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
