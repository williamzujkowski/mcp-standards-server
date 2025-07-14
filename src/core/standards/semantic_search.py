"""
Enhanced semantic search implementation for MCP standards.

This module provides advanced search capabilities including:
- Query preprocessing with synonyms and stemming
- Embedding generation with caching
- Query expansion techniques
- Re-ranking based on relevance scores
- Boolean operator support
- Fuzzy matching for typos
- Search analytics and performance tracking
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Import SentenceTransformer with fallback for test environments
from typing import TYPE_CHECKING, Any, cast

import nltk
import numpy as np
import redis
from fuzzywuzzy import fuzz, process
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    SentenceTransformer = None


# Runtime import logic
def _get_sentence_transformer_class() -> Any:
    """Get SentenceTransformer class at runtime."""
    import os

    if (
        os.getenv("MCP_TEST_MODE") == "true"
        or os.getenv("CI") is not None
        or os.getenv("PYTEST_CURRENT_TEST") is not None
    ):
        # Use mock in test environments
        try:
            import sys

            if "tests.mocks.semantic_search_mocks" in sys.modules:
                mock_module = sys.modules["tests.mocks.semantic_search_mocks"]
            else:
                import importlib

                mock_module = importlib.import_module(
                    "tests.mocks.semantic_search_mocks"
                )
            print("INFO: Using MockSentenceTransformer due to test mode environment")
            return mock_module.MockSentenceTransformer
        except ImportError:
            pass

    # Use real SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer as ST

        return ST
    except ImportError:
        return None


# Get the actual class to use
_SentenceTransformerCls = _get_sentence_transformer_class()


# Initialize NLTK data - only download if not already present
def _initialize_nltk_data() -> None:
    """Initialize NLTK data with proper error handling and timeouts."""
    import signal
    from contextlib import contextmanager

    # Skip downloads in test mode
    if os.environ.get("MCP_TEST_MODE") == "true":
        return

    @contextmanager
    def timeout(seconds: int) -> Any:
        """Context manager for timing out operations."""

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("NLTK download timed out")

        # Set up signal alarm (Unix only)
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        else:
            # On Windows or if SIGALRM not available, just yield
            yield

    required_data = [
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
    ]

    for data_name, data_path in required_data:
        try:
            # Check if data already exists
            try:
                nltk.data.find(data_path)
                continue  # Already downloaded
            except LookupError:
                pass  # Need to download

            # Download with timeout
            with timeout(30):  # 30 second timeout per download
                nltk.download(data_name, quiet=True)
        except Exception as e:
            # Log but don't fail - NLTK will use fallback tokenization
            logging.debug(f"Failed to download NLTK data '{data_name}': {e}")


# Only initialize NLTK data if not in test mode
if os.environ.get("MCP_TEST_MODE") != "true":
    try:
        _initialize_nltk_data()
    except Exception:
        pass  # nosec B110

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result with metadata."""

    id: str
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    highlights: list[str] = field(default_factory=list)
    explanation: str | None = None


@dataclass
class SearchQuery:
    """Represents a parsed search query."""

    original: str
    preprocessed: str
    tokens: list[str]
    stems: list[str]
    expanded_terms: list[str] = field(default_factory=list)
    boolean_operators: dict[str, list[str]] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchAnalytics:
    """Tracks search analytics and metrics."""

    query_count: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    popular_queries: Counter = field(default_factory=Counter)
    failed_queries: list[tuple[str, str]] = field(default_factory=list)
    average_results_per_query: float = 0.0
    click_through_data: dict[str, list[str]] = field(default_factory=dict)


class QueryPreprocessor:
    """Handles query preprocessing including synonyms, stemming, and expansion."""

    def __init__(self) -> None:
        self.stemmer = PorterStemmer()
        self.stopwords = set(nltk.corpus.stopwords.words("english"))

        # Domain-specific synonyms for standards/MCP context
        self.synonyms = {
            "api": ["interface", "endpoint", "service"],
            "test": ["testing", "validation", "verification", "check"],
            "security": ["secure", "safety", "protection", "auth", "authentication"],
            "standard": ["convention", "guideline", "best practice", "rule"],
            "web": ["website", "webapp", "frontend", "browser"],
            "react": ["reactjs", "react.js"],
            "vue": ["vuejs", "vue.js"],
            "angular": ["angularjs", "angular.js"],
            "mcp": ["model context protocol", "context protocol"],
            "llm": ["language model", "ai model", "gpt", "claude"],
            "performance": ["speed", "optimization", "efficiency", "fast"],
            "accessibility": ["a11y", "wcag", "aria"],
            "database": ["db", "data store", "storage"],
            "deploy": ["deployment", "release", "publish"],
            "config": ["configuration", "settings", "setup"],
        }

        # Build reverse synonym mapping
        self.reverse_synonyms: dict[str, list[str]] = {}
        for key, values in self.synonyms.items():
            for value in values:
                if value not in self.reverse_synonyms:
                    self.reverse_synonyms[value] = []
                self.reverse_synonyms[value].append(key)

    def preprocess(self, query: str) -> SearchQuery:
        """Preprocess a query with all enhancement techniques."""
        # Parse boolean operators
        boolean_ops = self._extract_boolean_operators(query)
        clean_query = self._remove_boolean_operators(query)

        # Tokenize and clean
        tokens = word_tokenize(clean_query.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in self.stopwords]

        # Generate stems
        stems = [self.stemmer.stem(token) for token in tokens]

        # Expand query with synonyms
        expanded_terms = self._expand_with_synonyms(tokens)

        # Create search query object
        search_query = SearchQuery(
            original=query,
            preprocessed=clean_query.lower(),
            tokens=tokens,
            stems=stems,
            expanded_terms=expanded_terms,
            boolean_operators=boolean_ops,
        )

        return search_query

    def _extract_boolean_operators(self, query: str) -> dict[str, list[str]]:
        """Extract AND, OR, NOT operators from query."""
        operators: dict[str, list[Any]] = {"AND": [], "OR": [], "NOT": []}

        # Match patterns like "term1 AND term2"
        and_pattern = r"(\w+)\s+AND\s+(\w+)"
        or_pattern = r"(\w+)\s+OR\s+(\w+)"
        not_pattern = r"NOT\s+(\w+)"

        for match in re.finditer(and_pattern, query):
            operators["AND"].append((match.group(1).lower(), match.group(2).lower()))

        for match in re.finditer(or_pattern, query):
            operators["OR"].append((match.group(1).lower(), match.group(2).lower()))

        for match in re.finditer(not_pattern, query):
            operators["NOT"].append(match.group(1).lower())

        return operators

    def _remove_boolean_operators(self, query: str) -> str:
        """Remove boolean operators from query."""
        # Remove boolean operators but keep the terms
        query = re.sub(r"\s+AND\s+", " ", query)
        query = re.sub(r"\s+OR\s+", " ", query)
        query = re.sub(r"\s+NOT\s+", " ", query)
        return query

    def _expand_with_synonyms(self, tokens: list[str]) -> list[str]:
        """Expand tokens with synonyms."""
        expanded = set()

        for token in tokens:
            # Add original token
            expanded.add(token)

            # Add direct synonyms
            if token in self.synonyms:
                expanded.update(self.synonyms[token])

            # Check if token is a synonym of something else
            if token in self.reverse_synonyms:
                expanded.update(self.reverse_synonyms[token])

        return list(expanded)


class EmbeddingCache:
    """Manages embedding generation with caching."""

    def __init__(
        self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Path | None = None
    ):
        # Check if we're in test/CI mode to avoid downloading models
        import os

        is_test_mode = (
            os.getenv("MCP_TEST_MODE") == "true"
            or os.getenv("CI") is not None
            or os.getenv("PYTEST_CURRENT_TEST") is not None
        )

        if is_test_mode:
            # Use mock in test mode to prevent HuggingFace downloads
            try:
                import importlib
                import sys

                if "tests.mocks.semantic_search_mocks" in sys.modules:
                    mock_module = sys.modules["tests.mocks.semantic_search_mocks"]
                else:
                    mock_module = importlib.import_module(
                        "tests.mocks.semantic_search_mocks"
                    )

                MockSentenceTransformer = mock_module.MockSentenceTransformer
                self.model = MockSentenceTransformer(model_name)
                logger.info(
                    f"Using MockSentenceTransformer for model {model_name} in test mode"
                )
            except ImportError:
                # Fallback mock if the test mocks aren't available
                logger.warning("Test mocks not available, creating minimal mock")
                self.model = self._create_minimal_mock(model_name)
        else:
            # Production mode - use real SentenceTransformer with retry logic
            self.model = self._create_sentence_transformer_with_retry(model_name)
        if cache_dir is None:
            self.cache_dir = Path.home() / ".mcp_search_cache"
        else:
            self.cache_dir = (
                Path(cache_dir) if isinstance(cache_dir, str) else cache_dir
            )
        self.cache_dir.mkdir(exist_ok=True)

        # In-memory cache with TTL
        self.memory_cache: dict[str, tuple[datetime, np.ndarray]] = {}
        self.cache_ttl = timedelta(hours=24)

        # Redis cache for distributed systems (optional)
        self.redis_client = None
        try:
            temp_client = redis.Redis(
                host="localhost", port=6379, decode_responses=False
            )
            temp_client.ping()
            self.redis_client = temp_client
        except Exception:
            logger.info("Redis not available, using file-based cache only")

    def _create_sentence_transformer_with_retry(
        self, model_name: str, max_retries: int = 3
    ) -> Any:
        """Create SentenceTransformer with retry logic and exponential backoff."""
        import time

        # Check if SentenceTransformer is available
        if SentenceTransformer is None:
            logger.warning(
                f"SentenceTransformer not available, using minimal mock for {model_name}"
            )
            return self._create_minimal_mock(model_name)

        for attempt in range(max_retries):
            try:
                # Set offline environment variables if they're not already set
                import os

                if os.getenv("HF_DATASETS_OFFLINE") != "1":
                    os.environ["HF_DATASETS_OFFLINE"] = (
                        "0"  # Allow downloads in production
                    )

                logger.info(
                    f"Attempting to load SentenceTransformer model {model_name} (attempt {attempt + 1}/{max_retries})"
                )
                return _SentenceTransformerCls(model_name)

            except Exception as e:
                if "429" in str(e) or "rate limit" in str(e).lower():
                    # Rate limiting detected - apply exponential backoff
                    wait_time = (2**attempt) * 1.0  # 1s, 2s, 4s
                    logger.warning(
                        f"Rate limit hit loading {model_name}, retrying in {wait_time}s: {e}"
                    )
                    if attempt < max_retries - 1:  # Don't wait on the last attempt
                        time.sleep(wait_time)
                    continue
                else:
                    # Check if we're in test environment - if so, be more lenient and retry
                    import os

                    is_test_env = (
                        os.getenv("PYTEST_CURRENT_TEST") is not None
                        or "test" in str(e).lower()
                        or "mock" in str(e).lower()
                    )

                    if is_test_env and attempt < max_retries - 1:
                        logger.warning(
                            f"Test environment error loading {model_name}, retrying: {e}"
                        )
                        continue
                    else:
                        # Non-rate-limit error in production - re-raise immediately
                        logger.error(f"Non-rate-limit error loading {model_name}: {e}")
                        raise

        # If all retries failed, fall back to minimal mock to keep tests running
        logger.error(
            f"Failed to load {model_name} after {max_retries} attempts, falling back to minimal mock"
        )
        return self._create_minimal_mock(model_name)

    def _create_minimal_mock(self, model_name: str) -> Any:
        """Create a minimal mock SentenceTransformer for fallback scenarios."""
        import numpy as np

        class MinimalMockSentenceTransformer:
            def __init__(self, model_name: str) -> None:
                self.model_name = model_name
                self.embedding_dim = 384  # Standard dimension for most models

            def encode(
                self, texts: Any, convert_to_numpy: bool = True, **kwargs: Any
            ) -> Any:
                if isinstance(texts, str):
                    texts = [texts]
                # Return deterministic but realistic embeddings
                embeddings = []
                for text in texts:
                    # Use hash of text for deterministic embeddings (not security-related)
                    import hashlib

                    hash_val = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                    np.random.seed(hash_val % (2**31))  # Ensure positive seed
                    embedding = np.random.randn(self.embedding_dim).astype(np.float32)
                    embedding = cast(Any, embedding) / (
                        np.linalg.norm(embedding) + 1e-9
                    )  # Normalize
                    embeddings.append(embedding)
                result = np.array(embeddings)
                return result[0] if len(texts) == 1 else result

            def get_sentence_embedding_dimension(self) -> int:
                return self.embedding_dim

        return MinimalMockSentenceTransformer(model_name)

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        """Serialize numpy array to bytes safely."""
        # Convert to base64-encoded string for safe storage
        serialized_str = (
            base64.b64encode(embedding.tobytes()).decode("ascii")
            + f'|{embedding.dtype}|{"x".join(map(str, embedding.shape))}'
        )
        return serialized_str.encode("utf-8")

    def _deserialize_embedding(self, data: str | bytes) -> Any:
        """Deserialize bytes to numpy array safely."""
        if isinstance(data, bytes):
            data = data.decode("ascii")

        # Parse the serialized format
        parts = data.split("|")
        if len(parts) != 3:
            raise ValueError("Invalid embedding format")

        array_bytes = base64.b64decode(parts[0])
        dtype = np.dtype(parts[1])
        shape = tuple(map(int, parts[2].split("x")))

        # Reconstruct the array
        return np.frombuffer(array_bytes, dtype=dtype).reshape(shape)

    def get_embedding(self, text: str, use_cache: bool = True) -> Any:
        """Get embedding for text with caching."""
        # Generate cache key
        cache_key = hashlib.sha256(text.encode()).hexdigest()

        if use_cache:
            # Check memory cache
            if cache_key in self.memory_cache:
                cached_time, embedding = self.memory_cache[cache_key]
                if datetime.now() - cached_time < self.cache_ttl:
                    return embedding

            # Check Redis cache
            if self.redis_client:
                try:
                    cached = self.redis_client.get(f"emb:{cache_key}")
                    if cached:
                        return self._deserialize_embedding(cached)
                except Exception:
                    pass  # nosec B110

            # Check file cache
            cache_file = self.cache_dir / f"{cache_key}.npy"
            if cache_file.exists():
                try:
                    return cast(np.ndarray, np.load(cache_file))
                except Exception:
                    pass  # nosec B110

        # Generate embedding
        embedding = self.model.encode(text, convert_to_numpy=True)

        # Cache the result
        if use_cache:
            self._cache_embedding(cache_key, embedding)

        return embedding

    def get_embeddings_batch(self, texts: list[str], use_cache: bool = True) -> Any:
        """Get embeddings for multiple texts with batching."""
        if not use_cache:
            return self.model.encode(texts, convert_to_numpy=True)

        # Separate cached and uncached texts
        embeddings: list[np.ndarray | None] = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cache_key = hashlib.sha256(text.encode()).hexdigest()

            # Try to get from cache
            cached_emb = self._get_cached_embedding(cache_key)
            if cached_emb is not None:
                embeddings[i] = cached_emb
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)

            # Cache and assign results
            for idx, text, emb in zip(
                uncached_indices, uncached_texts, new_embeddings, strict=False
            ):
                cache_key = hashlib.sha256(text.encode()).hexdigest()
                self._cache_embedding(cache_key, emb)
                embeddings[idx] = emb

        # Filter out None values and convert to numpy array
        valid_embeddings: list[np.ndarray] = [
            emb for emb in embeddings if emb is not None
        ]
        if valid_embeddings:
            return np.vstack(valid_embeddings)
        else:
            # Return empty array with proper shape
            try:
                dim = self.model.get_sentence_embedding_dimension()
                if dim is None:
                    dim = 384  # Default dimension
                return np.empty((0, dim), dtype=np.float32)
            except Exception:
                # Default dimension if model not properly initialized
                return np.empty((0, 384), dtype=np.float32)

    def _get_cached_embedding(self, cache_key: str) -> Any:
        """Try to get embedding from various cache layers."""
        # Memory cache
        if cache_key in self.memory_cache:
            cached_time, embedding = self.memory_cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return embedding

        # Redis cache
        if self.redis_client:
            try:
                cached = self.redis_client.get(f"emb:{cache_key}")
                if cached:
                    return self._deserialize_embedding(cached)
            except Exception:
                pass  # nosec B110

        # File cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        if cache_file.exists():
            try:
                return cast(np.ndarray, np.load(cache_file))
            except Exception:
                pass  # nosec B110

        return None

    def _cache_embedding(self, cache_key: str, embedding: np.ndarray) -> None:
        """Cache embedding in multiple layers."""
        # Memory cache
        self.memory_cache[cache_key] = (datetime.now(), embedding)

        # Redis cache
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"emb:{cache_key}",
                    int(self.cache_ttl.total_seconds()),
                    self._serialize_embedding(embedding),
                )
            except Exception:
                pass  # nosec B110

        # File cache
        cache_file = self.cache_dir / f"{cache_key}.npy"
        try:
            np.save(cache_file, embedding)
        except Exception:
            pass  # nosec B110

    def clear_cache(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()

        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter("emb:*"):
                    self.redis_client.delete(key)
            except Exception:
                pass  # nosec B110

        # Clear file cache
        for pattern in ["*.npy", "*.pkl"]:
            for cache_file in self.cache_dir.glob(pattern):
                try:
                    cache_file.unlink()
                except Exception:
                    pass  # nosec B110


class FuzzyMatcher:
    """Handles fuzzy matching for typo tolerance."""

    def __init__(self, threshold: int = 80) -> None:
        self.threshold = threshold
        self.known_terms: set[str] = set()

    def add_known_terms(self, terms: list[str]) -> None:
        """Add terms to the known terms set."""
        self.known_terms.update(terms)

    def find_matches(
        self, query: str, candidates: list[str] | None = None
    ) -> list[tuple[str, int]]:
        """Find fuzzy matches for a query."""
        if candidates is None:
            candidates = list(self.known_terms)

        # Use token set ratio for better matching of partial terms
        matches = process.extract(
            query, candidates, scorer=fuzz.token_set_ratio, limit=5
        )

        # Filter by threshold
        return [(match, score) for match, score in matches if score >= self.threshold]

    def correct_query(self, query: str) -> tuple[str, list[str]]:
        """Attempt to correct typos in query."""
        tokens = query.lower().split()
        corrections = []
        corrected_tokens = []

        for token in tokens:
            matches = self.find_matches(token)
            if matches and matches[0][1] < 100:  # Not exact match
                best_match = matches[0][0]
                corrections.append(f"{token} -> {best_match}")
                corrected_tokens.append(best_match)
            else:
                corrected_tokens.append(token)

        corrected_query = " ".join(corrected_tokens)
        return corrected_query, corrections


class SemanticSearch:
    """Main semantic search engine with all enhanced features."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_analytics: bool = True,
        cache_dir: Path | str | None = None,
    ):
        self.preprocessor = QueryPreprocessor()
        # Convert string to Path if needed
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)
        self.embedding_cache = EmbeddingCache(embedding_model, cache_dir)
        self.fuzzy_matcher = FuzzyMatcher()
        self.analytics = SearchAnalytics() if enable_analytics else None

        # Document store (in production, this would be a vector database)
        self.documents: dict[str, str] = {}
        self.document_embeddings: dict[str, np.ndarray] = {}
        self.document_metadata: dict[str, dict[str, Any]] = {}

        # Query result cache
        self.result_cache: dict[str, tuple[datetime, list[SearchResult]]] = {}
        self.result_cache_ttl = timedelta(minutes=30)

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

    def index_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Index a document for searching."""
        # Store document
        self.documents[doc_id] = content
        self.document_metadata[doc_id] = metadata or {}

        # Generate and cache embedding
        embedding = self.embedding_cache.get_embedding(content)
        self.document_embeddings[doc_id] = embedding

        # Update fuzzy matcher with document terms
        tokens = word_tokenize(content.lower())
        self.fuzzy_matcher.add_known_terms(tokens)

    def index_documents_batch(
        self, documents: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Index multiple documents efficiently."""
        # Extract components
        doc_ids = [doc[0] for doc in documents]
        contents = [doc[1] for doc in documents]
        metadatas = [doc[2] if len(doc) > 2 else {} for doc in documents]

        # Store documents
        for doc_id, content, metadata in zip(
            doc_ids, contents, metadatas, strict=False
        ):
            self.documents[doc_id] = content
            self.document_metadata[doc_id] = metadata

        # Generate embeddings in batch
        embeddings = self.embedding_cache.get_embeddings_batch(contents)

        # Store embeddings
        for doc_id, embedding in zip(doc_ids, embeddings, strict=False):
            self.document_embeddings[doc_id] = embedding

        # Update fuzzy matcher
        all_tokens = []
        for content in contents:
            tokens = word_tokenize(content.lower())
            all_tokens.extend(tokens)
        self.fuzzy_matcher.add_known_terms(list(set(all_tokens)))

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        rerank: bool = True,
        use_fuzzy: bool = True,
        use_cache: bool = True,
    ) -> list[SearchResult]:
        """Perform semantic search with all enhancements."""
        start_time = time.time()

        # Track query attempt in analytics (regardless of success/failure)
        if self.analytics:
            self.analytics.query_count += 1
            self.analytics.popular_queries[query] += 1

        # Check result cache
        cache_key = self._get_result_cache_key(query, top_k, filters, rerank, use_fuzzy)
        if use_cache and cache_key in self.result_cache:
            cached_time, cached_results = self.result_cache[cache_key]
            if datetime.now() - cached_time < self.result_cache_ttl:
                if self.analytics:
                    self.analytics.cache_hits += 1
                return cached_results

        if self.analytics:
            self.analytics.cache_misses += 1

        try:
            # Preprocess query
            search_query = self.preprocessor.preprocess(query)

            # Apply fuzzy matching if enabled
            if use_fuzzy:
                corrected_query, corrections = self.fuzzy_matcher.correct_query(
                    search_query.preprocessed
                )
                if corrections:
                    logger.info(f"Applied corrections: {corrections}")
                    # Re-preprocess with corrected query
                    search_query = self.preprocessor.preprocess(corrected_query)

            # Generate query embedding - use original query for primary embedding
            # Only include top relevant synonyms to avoid dilution
            primary_query = " ".join(search_query.tokens)
            query_embedding = self.embedding_cache.get_embedding(primary_query)

            # Generate secondary embedding with limited expansion (top 3 synonyms only)
            limited_expansion = (
                search_query.expanded_terms[:3] if search_query.expanded_terms else []
            )
            if limited_expansion:
                expanded_query = " ".join(search_query.tokens + limited_expansion)
                expanded_embedding = self.embedding_cache.get_embedding(expanded_query)
                # Blend embeddings with more weight on original
                query_embedding = 0.8 * query_embedding + 0.2 * expanded_embedding

            # Calculate similarities with hybrid approach
            similarities = self._calculate_similarities(
                query_embedding, search_query, filters
            )

            # Get top results - get more than needed to account for filtering
            top_k_extended = min(top_k * 2, len(similarities))
            top_indices = np.argsort(similarities)[-top_k_extended:][::-1]

            # Create initial results
            results = []
            score_threshold = 0.05  # Lower threshold for better recall
            prev_score = None

            for idx in top_indices:
                doc_id = list(self.documents.keys())[idx]
                score = float(similarities[idx])

                # Only apply score drop detection for focused queries (not general searches)
                if (
                    prev_score is not None
                    and len(results) >= 3
                    and len(search_query.tokens) >= 4
                ):
                    score_drop = (
                        (prev_score - score) / prev_score if prev_score > 0 else 1.0
                    )
                    # If score drops by more than 50%, consider stopping
                    if score_drop > 0.5 and score < 0.3:
                        break

                if score > score_threshold:
                    result = SearchResult(
                        id=doc_id,
                        content=self.documents[doc_id],
                        score=score,
                        metadata=self.document_metadata.get(doc_id, {}),
                    )
                    results.append(result)
                    prev_score = score

                    # Stop when we have enough results
                    if len(results) >= top_k:
                        break

            # Apply re-ranking if enabled
            if rerank and len(results) > 1:
                results = self._rerank_results(results, search_query)

            # Generate highlights
            for result in results:
                result.highlights = self._generate_highlights(
                    result.content, search_query.tokens
                )

            # Cache results
            if use_cache:
                self.result_cache[cache_key] = (datetime.now(), results)

            # Update analytics
            if self.analytics:
                elapsed = time.time() - start_time
                self.analytics.total_latency += elapsed
                self.analytics.average_results_per_query = (
                    self.analytics.average_results_per_query
                    * (self.analytics.query_count - 1)
                    + len(results)
                ) / self.analytics.query_count

            return results

        except Exception as e:
            logger.error(f"Search error: {e}")
            if self.analytics:
                self.analytics.failed_queries.append((query, str(e)))
            return []

    def _calculate_similarities(
        self,
        query_embedding: np.ndarray,
        search_query: SearchQuery,
        filters: dict[str, Any] | None,
    ) -> np.ndarray:
        """Calculate similarities with boolean operators and filters."""
        # Get all document embeddings as matrix
        doc_ids = list(self.documents.keys())
        doc_embeddings = np.vstack(
            [self.document_embeddings[doc_id] for doc_id in doc_ids]
        )

        # Calculate semantic similarities
        semantic_similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        # Calculate keyword matching scores with improved logic
        keyword_scores = np.zeros(len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            content_lower = self.documents[doc_id].lower()
            doc_tokens = set(word_tokenize(content_lower))
            metadata = self.document_metadata.get(doc_id, {})

            # Score based on query token matches
            matches = 0.0
            exact_matches = 0

            for token in search_query.tokens:
                # Check content for exact word match
                if token in doc_tokens:
                    matches += 1.0
                    exact_matches += 1
                # Check for substring match (less weight)
                elif token in content_lower:
                    matches += 0.5

                # Also check metadata (important for categorization)
                metadata_match = False
                for _field, value in metadata.items():
                    if isinstance(value, str) and token in value.lower():
                        matches += 0.5
                        metadata_match = True
                        break
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and token in item.lower():
                                matches += 0.5
                                metadata_match = True
                                break
                        if metadata_match:
                            break

            # Bonus for documents that contain multiple query terms close together
            if len(search_query.tokens) > 1:
                # Check if all tokens appear within a window
                words = content_lower.split()
                for window_start in range(len(words) - len(search_query.tokens) + 1):
                    window = words[window_start : window_start + 20]  # 20-word window
                    if all(token in window for token in search_query.tokens):
                        matches += 0.5  # Proximity bonus
                        break

            # Normalize by number of query tokens
            if search_query.tokens:
                keyword_scores[i] = min(matches / len(search_query.tokens), 1.0)

        # Identify critical terms (usually nouns or key concepts)
        # For compound queries like "security best practices", "security" is critical
        critical_terms = []
        if len(search_query.tokens) >= 2:
            # Consider first term as critical if it's not a common word
            common_words = {
                "best",
                "good",
                "new",
                "top",
                "all",
                "how",
                "what",
                "when",
                "where",
                "why",
            }
            for token in search_query.tokens:
                if token not in common_words and len(token) > 3:
                    critical_terms.append(token)
                    break  # Just take the first critical term

        # Check if documents contain critical terms
        has_critical_term = np.zeros(len(doc_ids), dtype=bool)
        for i, doc_id in enumerate(doc_ids):
            content_lower = self.documents[doc_id].lower()
            doc_tokens = set(word_tokenize(content_lower))
            metadata = self.document_metadata.get(doc_id, {})

            # Check if any critical term is present
            for term in critical_terms:
                if term in doc_tokens or term in content_lower:
                    has_critical_term[i] = True
                    break
                # Also check metadata
                for _field, value in metadata.items():
                    if isinstance(value, str) and term in value.lower():
                        has_critical_term[i] = True
                        break
                    elif isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and term in item.lower():
                                has_critical_term[i] = True
                                break

        # Combine semantic and keyword scores with critical term requirement
        min_keyword_threshold = 0.2  # Lowered threshold for better recall

        combined_similarities = np.zeros(len(doc_ids))
        for i in range(len(doc_ids)):
            # Apply critical term penalty only for certain types of queries
            # For single-word queries or very short queries, don't apply critical term filtering
            apply_critical_filter = (
                len(search_query.tokens) >= 3 and len(critical_terms) > 0
            )

            # Moderate penalty if missing critical terms (not too harsh)
            critical_multiplier = 1.0
            if apply_critical_filter and not has_critical_term[i]:
                critical_multiplier = (
                    0.5  # Reduced from 0.3 to 0.5 for better semantic matching
                )

            if keyword_scores[i] == 0:
                # No keyword matches - moderate penalty for better semantic matching
                combined_similarities[i] = (
                    semantic_similarities[i] * 0.3 * critical_multiplier
                )
            elif keyword_scores[i] < min_keyword_threshold:
                # Poor keyword matches - mild penalty
                combined_similarities[i] = (
                    semantic_similarities[i] * 0.5 * critical_multiplier
                )
            else:
                # Good keyword matches - balanced weighting
                # Balance semantic and keyword signals
                keyword_weight = 0.5 if keyword_scores[i] > 0.5 else 0.4
                semantic_weight = 1.0 - keyword_weight
                combined_similarities[i] = (
                    semantic_weight * semantic_similarities[i]
                    + keyword_weight * keyword_scores[i]
                ) * critical_multiplier

        similarities = combined_similarities

        # Apply boolean operators
        if search_query.boolean_operators:
            similarities = self._apply_boolean_operators(
                similarities, doc_ids, search_query.boolean_operators
            )

        # Apply filters
        if filters:
            similarities = self._apply_filters(similarities, doc_ids, filters)

        return similarities

    def _apply_boolean_operators(
        self, similarities: np.ndarray, doc_ids: list[str], operators: dict[str, list]
    ) -> np.ndarray:
        """Apply AND, OR, NOT boolean operators."""
        modified_similarities = similarities.copy()

        for i, doc_id in enumerate(doc_ids):
            content = self.documents[doc_id].lower()
            tokens = set(word_tokenize(content))

            # Apply NOT operators - exclude documents containing the term
            for not_term in operators.get("NOT", []):
                # For NOT, only check whole word matches to avoid false positives
                # (e.g., "java" should not match "javascript")
                if not_term in tokens:
                    modified_similarities[i] = 0

            # Apply AND operators - both terms must be present
            for term1, term2 in operators.get("AND", []):
                # Check if both terms exist as whole words or substrings
                term1_present = term1 in tokens or term1 in content
                term2_present = term2 in tokens or term2 in content

                if not (term1_present and term2_present):
                    modified_similarities[i] = 0  # Exclude documents without both terms

            # Apply OR operators - at least one term must be present
            for term1, term2 in operators.get("OR", []):
                term1_present = term1 in tokens or term1 in content
                term2_present = term2 in tokens or term2 in content

                if term1_present or term2_present:
                    # Boost based on how many terms are present
                    boost = 1.5 if (term1_present and term2_present) else 1.25
                    modified_similarities[i] *= boost
                else:
                    # Neither term present - penalize
                    modified_similarities[i] *= 0.3

        return cast(np.ndarray, modified_similarities)

    def _apply_filters(
        self, similarities: np.ndarray, doc_ids: list[str], filters: dict[str, Any]
    ) -> np.ndarray:
        """Apply metadata filters."""
        modified_similarities = similarities.copy()

        for i, doc_id in enumerate(doc_ids):
            metadata = self.document_metadata.get(doc_id, {})

            # Check each filter
            for key, value in filters.items():
                if key not in metadata:
                    modified_similarities[i] = 0
                elif isinstance(value, list):
                    # Filter value is a list - check if metadata value is in list
                    if metadata[key] not in value:
                        modified_similarities[i] = 0
                else:
                    # Direct comparison
                    if metadata[key] != value:
                        modified_similarities[i] = 0

        return cast(np.ndarray, modified_similarities)

    def _rerank_results(
        self, results: list[SearchResult], search_query: SearchQuery
    ) -> list[SearchResult]:
        """Re-rank results based on additional factors."""
        # Calculate additional scoring factors
        for result in results:
            content_lower = result.content.lower()
            initial_score = result.score  # Preserve initial score

            # Exact match boost - check if exact query appears
            exact_match_score = (
                1.0 if search_query.original.lower() in content_lower else 0.0
            )

            # Term frequency boost with position weighting
            term_freq_score = self._calculate_term_frequency_score(
                result.content, search_query.tokens
            )

            # Phrase proximity score - bonus for terms appearing close together
            proximity_score = self._calculate_proximity_score(
                content_lower, search_query.tokens
            )

            # Title/beginning boost - check if terms appear early in document
            position_score = self._calculate_position_score(
                content_lower, search_query.tokens
            )

            # Recency boost (if timestamp in metadata)
            recency_score = self._calculate_recency_score(result.metadata)

            # Metadata match boost
            metadata_score = self._calculate_metadata_score(
                result.metadata, search_query.tokens
            )

            # Combine scores with better weighting
            new_score = (
                initial_score * 0.4  # Semantic similarity (reduced from 0.7)
                + exact_match_score * 0.2  # Exact match bonus
                + term_freq_score * 0.15  # Term frequency
                + proximity_score * 0.1  # Term proximity
                + position_score * 0.05  # Position importance
                + metadata_score * 0.05  # Metadata relevance
                + recency_score * 0.05  # Recency
            )

            # Cap the new score - allow reasonable boost for documents with good matches
            # This prevents documents with very low initial relevance from jumping too high
            max_boost = 2.0  # Increased from 1.5 to allow more flexibility
            result.score = min(new_score, initial_score * max_boost)

            # Add detailed explanation
            result.explanation = (
                f"Semantic: {result.score:.3f}, "
                f"Exact: {exact_match_score:.3f}, "
                f"TermFreq: {term_freq_score:.3f}, "
                f"Proximity: {proximity_score:.3f}, "
                f"Position: {position_score:.3f}"
            )

        # Re-sort by new scores
        return sorted(results, key=lambda x: x.score, reverse=True)

    def _calculate_term_frequency_score(self, content: str, terms: list[str]) -> float:
        """Calculate term frequency score with improved weighting."""
        content_lower = content.lower()
        content_tokens = word_tokenize(content_lower)

        # Count exact word matches and substring matches separately
        exact_matches = 0
        substring_matches = 0

        for term in terms:
            # Count exact word matches
            exact_matches += content_tokens.count(term)
            # Count substring matches (excluding exact matches)
            substring_matches += content_lower.count(term) - content_tokens.count(term)

        # Weight exact matches more than substring matches
        weighted_occurrences = exact_matches + (substring_matches * 0.5)

        # Normalize by content length with diminishing returns
        doc_length = len(content_tokens)
        if doc_length == 0:
            return 0.0

        # Use log normalization to prevent very long documents from dominating
        normalized_score = weighted_occurrences / (1 + np.log(doc_length))

        # Cap at 1.0 but allow high frequency to show
        return float(min(normalized_score, 1.0))

    def _calculate_recency_score(self, metadata: dict[str, Any]) -> float:
        """Calculate recency score based on timestamp."""
        if "timestamp" not in metadata:
            return 0.5  # Neutral score if no timestamp

        try:
            timestamp = datetime.fromisoformat(metadata["timestamp"])
            days_old = (datetime.now() - timestamp).days

            # Exponential decay - newer documents get higher scores
            return float(np.exp(-days_old / 30))  # Half-life of 30 days
        except Exception:
            return 0.5

    def _calculate_proximity_score(self, content: str, terms: list[str]) -> float:
        """Calculate score based on proximity of query terms to each other."""
        if len(terms) < 2:
            return 0.5  # Neutral score for single term queries

        # Find positions of all terms
        term_positions = defaultdict(list)
        words = content.split()

        for i, word in enumerate(words):
            for term in terms:
                if term in word:
                    term_positions[term].append(i)

        # If not all terms are present, return low score
        if len(term_positions) < len(terms):
            return 0.0

        # Calculate minimum distance between different terms
        min_distances = []
        term_list = list(term_positions.keys())

        for i in range(len(term_list)):
            for j in range(i + 1, len(term_list)):
                term1_positions = term_positions[term_list[i]]
                term2_positions = term_positions[term_list[j]]

                # Find minimum distance between any occurrence of term1 and term2
                min_dist = float("inf")
                for pos1 in term1_positions:
                    for pos2 in term2_positions:
                        min_dist = min(min_dist, abs(pos1 - pos2))

                min_distances.append(min_dist)

        if not min_distances:
            return 0.5

        # Convert distances to score (closer = higher score)
        avg_min_distance = np.mean(min_distances)

        # Use exponential decay - terms within 5 words get high score
        proximity_score = np.exp(-avg_min_distance / 5)

        return float(min(proximity_score, 1.0))

    def _calculate_position_score(self, content: str, terms: list[str]) -> float:
        """Calculate score based on where terms appear in document."""
        words = content.split()
        doc_length = len(words)

        if doc_length == 0:
            return 0.0

        # Find earliest position of any query term
        earliest_position = doc_length

        for i, word in enumerate(words):
            for term in terms:
                if term in word:
                    earliest_position = min(earliest_position, i)
                    break

        # Convert position to score (earlier = higher score)
        # First 10% of document gets high score
        position_ratio = earliest_position / doc_length

        if position_ratio <= 0.1:
            return 1.0
        elif position_ratio <= 0.3:
            return 0.8
        elif position_ratio <= 0.5:
            return 0.6
        else:
            return 0.4

    def _calculate_metadata_score(
        self, metadata: dict[str, Any], terms: list[str]
    ) -> float:
        """Calculate score based on term matches in metadata."""
        if not metadata:
            return 0.0

        matches = 0.0
        total_fields = 0.0

        # Check each metadata field for term matches
        important_fields = ["title", "category", "tags", "description", "summary"]

        for field_name, value in metadata.items():
            if isinstance(value, str):
                value_lower = value.lower()
                field_importance = 2.0 if field_name in important_fields else 1.0

                for term in terms:
                    if term in value_lower:
                        matches += field_importance

                total_fields += field_importance
            elif isinstance(value, list):
                # Handle list fields like tags
                field_importance = 2.0 if field in important_fields else 1.0

                for item in value:
                    if isinstance(item, str):
                        item_lower = item.lower()
                        for term in terms:
                            if term in item_lower:
                                matches += field_importance
                                break

                total_fields += field_importance

        if total_fields == 0:
            return 0.0

        return min(matches / total_fields, 1.0)

    def _generate_highlights(self, content: str, terms: list[str]) -> list[str]:
        """Generate highlighted snippets."""
        highlights = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(term in sentence_lower for term in terms):
                # Highlight matching terms
                highlighted = sentence
                for term in terms:
                    # Case-insensitive replacement
                    pattern = re.compile(re.escape(term), re.IGNORECASE)
                    highlighted = pattern.sub(f"**{term}**", highlighted)

                highlights.append(highlighted.strip())

                if len(highlights) >= 3:  # Limit to 3 highlights
                    break

        return highlights

    def _get_result_cache_key(
        self,
        query: str,
        top_k: int,
        filters: dict | None,
        rerank: bool = True,
        use_fuzzy: bool = True,
    ) -> str:
        """Generate cache key for results."""
        filter_str = json.dumps(filters, sort_keys=True) if filters else ""
        return hashlib.sha256(
            f"{query}:{top_k}:{filter_str}:{rerank}:{use_fuzzy}".encode()
        ).hexdigest()

    def get_analytics_report(self) -> dict[str, Any]:
        """Generate analytics report."""
        if not self.analytics:
            return {"error": "Analytics not enabled"}

        avg_latency = (
            self.analytics.total_latency / self.analytics.query_count
            if self.analytics.query_count > 0
            else 0
        )

        cache_hit_rate = (
            self.analytics.cache_hits
            / (self.analytics.cache_hits + self.analytics.cache_misses)
            if (self.analytics.cache_hits + self.analytics.cache_misses) > 0
            else 0
        )

        return {
            "total_queries": self.analytics.query_count,
            "average_latency_ms": avg_latency * 1000,
            "cache_hit_rate": cache_hit_rate,
            "average_results_per_query": self.analytics.average_results_per_query,
            "top_queries": self.analytics.popular_queries.most_common(10),
            "failed_queries_count": len(self.analytics.failed_queries),
            "recent_failures": self.analytics.failed_queries[-5:],
        }

    def track_click(self, query: str, result_id: str) -> None:
        """Track click-through data for analytics."""
        if self.analytics:
            if query not in self.analytics.click_through_data:
                self.analytics.click_through_data[query] = []
            self.analytics.click_through_data[query].append(result_id)

    def close(self) -> None:
        """Clean up resources."""
        self.executor.shutdown()


# Async wrapper for non-blocking search
class AsyncSemanticSearch:
    """Async wrapper for SemanticSearch."""

    def __init__(self, semantic_search: SemanticSearch) -> None:
        self.search_engine = semantic_search
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self) -> None:
        """Run the async event loop in a separate thread."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def search_async(self, query: str, **kwargs: Any) -> list[SearchResult]:
        """Perform search asynchronously."""
        import asyncio
        import functools

        # Use partial to properly handle keyword arguments
        search_func = functools.partial(self.search_engine.search, query, **kwargs)

        # Use current event loop instead of self.loop for better compatibility
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, search_func)
        return await future

    async def index_documents_batch_async(
        self, documents: list[tuple[str, str, dict[str, Any]]]
    ) -> None:
        """Index documents asynchronously."""
        import asyncio

        # Use current event loop instead of self.loop for better compatibility
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(
            None, self.search_engine.index_documents_batch, documents
        )
        await future

    def close(self) -> None:
        """Clean up async resources."""
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join()
        self.search_engine.close()


# Convenience function for creating search engine
def create_search_engine(
    embedding_model: str = "all-MiniLM-L6-v2",
    enable_analytics: bool = True,
    cache_dir: Path | None = None,
    async_mode: bool = False,
) -> SemanticSearch | AsyncSemanticSearch:
    """Create a semantic search engine instance."""
    search_engine = SemanticSearch(
        embedding_model=embedding_model,
        enable_analytics=enable_analytics,
        cache_dir=cache_dir,
    )

    if async_mode:
        return AsyncSemanticSearch(search_engine)

    return search_engine
