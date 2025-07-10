"""Mock implementations for testing."""

# Import semantic search mocks
from .semantic_search_mocks import (
    MockCosineSimilarity,
    MockFuzz,
    MockNearestNeighbors,
    MockNLTKComponents,
    MockProcess,
    MockRedisClient,
    MockSentenceTransformer,
    TestDataGenerator,
    patch_ml_dependencies,
)

__all__ = [
    "MockSentenceTransformer",
    "MockRedisClient",
    "MockNLTKComponents",
    "MockFuzz",
    "MockProcess",
    "MockCosineSimilarity",
    "MockNearestNeighbors",
    "TestDataGenerator",
    "patch_ml_dependencies",
]
