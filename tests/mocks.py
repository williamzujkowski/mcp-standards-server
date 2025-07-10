"""Mock implementations for heavy dependencies during testing."""

import numpy as np


class MockSentenceTransformer:
    """Mock sentence transformer for testing without installing the full package."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.embedding_dim = 384  # Standard dimension for all-MiniLM-L6-v2

    def encode(self, texts: str | list[str], **kwargs) -> np.ndarray:
        """Generate mock embeddings."""
        if isinstance(texts, str):
            texts = [texts]

        # Generate deterministic mock embeddings based on text hash
        embeddings = []
        for text in texts:
            # Create a deterministic embedding based on text content
            text_hash = hash(text) % 1000
            embedding = np.random.RandomState(text_hash).randn(self.embedding_dim)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)

        return np.array(embeddings)

    @property
    def max_seq_length(self) -> int:
        """Return mock max sequence length."""
        return 512


class MockNearestNeighbors:
    """Mock scikit-learn NearestNeighbors for testing."""

    def __init__(self, n_neighbors=5, metric="cosine", algorithm="brute"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self._fit_X = None

    def fit(self, X: np.ndarray):
        """Store the training data."""
        self._fit_X = X
        return self

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors."""
        if self._fit_X is None:
            raise ValueError("Must fit before calling kneighbors")

        n_neighbors = n_neighbors or self.n_neighbors
        n_samples = X.shape[0]
        n_indexed = self._fit_X.shape[0]

        # Generate mock distances and indices
        distances = np.random.rand(n_samples, min(n_neighbors, n_indexed))
        indices = np.random.randint(
            0, n_indexed, size=(n_samples, min(n_neighbors, n_indexed))
        )

        # Sort by distance
        for i in range(n_samples):
            sort_idx = np.argsort(distances[i])
            distances[i] = distances[i][sort_idx]
            indices[i] = indices[i][sort_idx]

        return distances, indices
