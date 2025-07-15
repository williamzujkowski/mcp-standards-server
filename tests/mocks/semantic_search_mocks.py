"""
Comprehensive mock implementations for semantic search ML components.

This module provides deterministic mocks for ML dependencies to enable
reliable and fast testing without requiring actual model downloads.
"""

import hashlib
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, cast

import numpy as np


class MockSentenceTransformer:
    """
    Deterministic mock for sentence-transformers models.

    Generates consistent embeddings based on text content while simulating
    realistic model behavior including latency and resource usage.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.embedding_dim = self._get_embedding_dim(model_name)
        self.max_seq_length = 512

        # Simulate model loading time
        self._loading_time = 0.1  # 100ms
        self._encode_base_time = 0.01  # 10ms per text

        # Track resource usage
        self._encode_count = 0
        self._total_tokens_processed = 0

        # For testing failure scenarios
        self._should_fail = False
        self._failure_message = ""

    def get_sentence_embedding_dimension(self) -> int:
        """Get sentence embedding dimension - matches SentenceTransformer API."""
        return self.embedding_dim

    def _get_embedding_dim(self, model_name: str) -> int:
        """Get embedding dimension based on model name."""
        dims = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "all-MiniLM-L12-v2": 384,
            "multi-qa-MiniLM-L6-cos-v1": 384,
        }
        return dims.get(model_name, 384)

    def encode(
        self,
        texts: str | list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Generate deterministic embeddings with realistic behavior."""
        if self._should_fail:
            raise RuntimeError(self._failure_message)

        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        # Simulate encoding time
        time.sleep(self._encode_base_time * len(texts))

        # Generate embeddings
        embeddings = []
        for text in texts:
            # Create deterministic embedding from text
            embedding = self._text_to_embedding(text)

            if normalize_embeddings:
                # L2 normalize
                embedding = embedding / (np.linalg.norm(embedding) + 1e-9)

            embeddings.append(embedding)

            # Update stats
            self._encode_count += 1
            self._total_tokens_processed += len(text.split())

        result = np.array(embeddings)

        # Return single embedding if input was single text
        if single_text:
            return cast(np.ndarray, result[0])
        return result

    def _text_to_embedding(self, text: str) -> np.ndarray:
        """Generate deterministic embedding from text."""
        # Check for strong negative sentiment first
        text_lower = text.lower()
        has_strong_negative = any(
            neg_word in text_lower
            for neg_word in [
                "remove",
                "disable",
                "without",
                "issues",
                "problems",
                "slow",
                "degrade",
            ]
        )

        # For negative sentiment, modify the text to reduce positive keyword influence
        if has_strong_negative:
            # Replace positive keywords with negative equivalents for embedding generation
            modified_text = text_lower
            modified_text = modified_text.replace("performance", "failure")
            modified_text = modified_text.replace("optimization", "degradation")
            modified_text = modified_text.replace("improve", "worsen")
            modified_text = modified_text.replace("speed", "slowness")
            modified_text = modified_text.replace("faster", "slower")
            modified_text = modified_text.replace("auth", "no_auth")
            modified_text = modified_text.replace("login", "logout")
            text_for_embedding = modified_text
        else:
            text_for_embedding = text.lower()

        # Use SHA256 for deterministic hashing
        text_bytes = text_for_embedding.encode("utf-8")
        hash_hex = hashlib.sha256(text_bytes).hexdigest()

        # Convert hash to seed - modify seed for negative sentiment to create very different embeddings
        base_seed = int(hash_hex[:8], 16)
        if has_strong_negative:
            # Flip the seed to create dramatically different embeddings for negative sentiment
            seed = base_seed ^ 0xFFFFFFFF  # XOR with all 1s to flip bits
        else:
            seed = base_seed

        rng = np.random.RandomState(seed)

        # Generate embedding with semantic properties
        base_embedding = rng.randn(self.embedding_dim)

        # Add semantic features based on text content (use original text for feature extraction)
        semantic_features = self._extract_semantic_features(text)

        # Blend base with semantic features
        for i, feature in enumerate(semantic_features):
            if i < self.embedding_dim:
                weight = 0.5
                if has_strong_negative:
                    # For negative sentiment text, invert the positive features
                    feature_idx = i % len(semantic_features) if semantic_features else 0
                    if feature_idx in [2, 4]:  # Auth positive and DB positive indices
                        weight = (
                            -2.0
                        )  # Strongly invert positive sentiment for negative queries
                    elif feature_idx in [3, 5]:  # Auth negative and DB negative indices
                        weight = 3.0  # Strongly amplify negative sentiment
                base_embedding[i] += feature * weight

        return base_embedding

    def _extract_semantic_features(self, text: str) -> list[float]:
        """Extract simple semantic features for testing."""
        features = []
        text_lower = text.lower()

        # Word-based features
        features.append(len(text) / 100.0)  # Length feature
        features.append(text_lower.count(" ") / 10.0)  # Word count

        # Enhanced keyword features with semantic groups and sentiment
        auth_positive = [
            "authentication",
            "login",
            "auth",
            "implement",
            "setup",
            "enable",
            "secure",
        ]
        auth_negative = ["remove", "disable", "without", "public", "unsecured"]
        database_positive = [
            "optimize",
            "improve",
            "faster",
            "performance",
            "speed",
            "efficiency",
        ]
        database_negative = ["slow", "degrade", "issues", "problems", "bottleneck"]
        security_keywords = ["security", "secure", "protection", "vulnerability"]
        tech_keywords = ["test", "api", "react", "python", "standard", "mcp"]

        # Calculate positive and negative sentiment scores
        auth_pos_score = sum(1.0 for keyword in auth_positive if keyword in text_lower)
        auth_neg_score = sum(1.0 for keyword in auth_negative if keyword in text_lower)
        db_pos_score = sum(
            1.0 for keyword in database_positive if keyword in text_lower
        )
        db_neg_score = sum(
            1.0 for keyword in database_negative if keyword in text_lower
        )

        # Add semantic features with sentiment differentiation
        features.append(auth_pos_score / len(auth_positive))  # Auth positive sentiment
        features.append(auth_neg_score / len(auth_negative))  # Auth negative sentiment
        features.append(db_pos_score / len(database_positive))  # DB positive sentiment
        features.append(db_neg_score / len(database_negative))  # DB negative sentiment

        # Add overall sentiment polarity
        features.append(max(0, auth_pos_score - auth_neg_score))  # Auth polarity
        features.append(max(0, db_pos_score - db_neg_score))  # DB polarity

        # Traditional keyword groups
        for keyword_group in [security_keywords, tech_keywords]:
            group_score = sum(1.0 for keyword in keyword_group if keyword in text_lower)
            features.append(group_score / len(keyword_group))

        # Individual keyword features for important discriminators
        important_keywords = [
            "implement",
            "remove",
            "setup",
            "disable",
            "optimize",
            "slow",
            "improve",
            "degrade",
        ]
        for keyword in important_keywords:
            features.append(1.0 if keyword in text_lower else 0.0)

        return features

    def tokenize(self, texts: list[str]) -> dict[str, Any]:
        """Mock tokenization."""
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }

    def set_failure_mode(self, should_fail: bool, message: str = "Mock failure"):
        """Set failure mode for testing error handling."""
        self._should_fail = should_fail
        self._failure_message = message

    def get_stats(self) -> dict[str, Any]:
        """Get usage statistics."""
        return {
            "encode_count": self._encode_count,
            "total_tokens": self._total_tokens_processed,
            "model_name": self.model_name,
            "device": self.device,
        }


class MockRedisClient:
    """
    In-memory mock Redis client with TTL support.

    Simulates Redis behavior including expiration, pub/sub, and pipelines.
    """

    def __init__(self, host="localhost", port=6379, decode_responses=True, **kwargs):
        self.host = host
        self.port = port
        self.decode_responses = decode_responses
        self._store = {}
        self._expiry = {}
        self._lock = threading.Lock()

        # Connection state
        self._connected = True
        self._pipeline_mode = False
        self._pipeline_commands = []

        # Pub/sub
        self._subscribers = defaultdict(list)

    def ping(self) -> bool:
        """Check connection."""
        if not self._connected:
            raise ConnectionError("Connection refused")
        return True

    def get(self, key: str) -> str | bytes | None:
        """Get value by key."""
        self._check_connection()
        self._cleanup_expired()

        with self._lock:
            value = self._store.get(key)

        if value is not None and self.decode_responses and isinstance(value, bytes):
            return cast(str, value.decode("utf-8"))
        return value

    def set(self, key: str, value: str | bytes, ex: int | None = None) -> bool:
        """Set key-value pair with optional expiration."""
        self._check_connection()

        if self._pipeline_mode:
            self._pipeline_commands.append(("set", key, value, ex))
            return True

        with self._lock:
            if isinstance(value, str) and not self.decode_responses:
                value = value.encode("utf-8")

            self._store[key] = value

            if ex:
                self._expiry[key] = datetime.now() + timedelta(seconds=ex)

        return True

    def setex(self, key: str, seconds: int, value: str | bytes) -> bool:
        """Set with expiration."""
        return self.set(key, value, ex=seconds)

    def delete(self, *keys: str) -> int:
        """Delete keys."""
        self._check_connection()
        deleted = 0

        with self._lock:
            for key in keys:
                if key in self._store:
                    del self._store[key]
                    self._expiry.pop(key, None)
                    deleted += 1

        return deleted

    def exists(self, *keys: str) -> int:
        """Check if keys exist."""
        self._check_connection()
        self._cleanup_expired()

        count = 0
        with self._lock:
            for key in keys:
                if key in self._store:
                    count += 1
        return count

    def expire(self, key: str, seconds: int) -> bool:
        """Set expiration on existing key."""
        self._check_connection()

        with self._lock:
            if key in self._store:
                self._expiry[key] = datetime.now() + timedelta(seconds=seconds)
                return True
        return False

    def ttl(self, key: str) -> int:
        """Get time to live for key."""
        self._check_connection()

        with self._lock:
            if key not in self._store:
                return -2  # Key doesn't exist

            if key not in self._expiry:
                return -1  # No expiration

            ttl = int((self._expiry[key] - datetime.now()).total_seconds())
            return max(0, ttl)

    def scan_iter(self, match: str | None = None, count: int = 100):
        """Scan keys matching pattern."""
        self._check_connection()
        self._cleanup_expired()

        import re

        pattern = None
        if match:
            # Convert Redis pattern to regex
            regex_pattern = match.replace("*", ".*").replace("?", ".")
            pattern = re.compile(f"^{regex_pattern}$")

        with self._lock:
            keys = list(self._store.keys())

        for key in keys:
            if pattern is None or pattern.match(key):
                yield key

    def pipeline(self, transaction: bool = True):
        """Create pipeline."""
        return MockRedisPipeline(self)

    def _cleanup_expired(self):
        """Remove expired keys."""
        now = datetime.now()
        expired_keys = []

        with self._lock:
            for key, expiry_time in self._expiry.items():
                if expiry_time <= now:
                    expired_keys.append(key)

            for key in expired_keys:
                self._store.pop(key, None)
                self._expiry.pop(key, None)

    def _check_connection(self):
        """Check if connected."""
        if not self._connected:
            raise ConnectionError("Connection lost")

    def close(self):
        """Close connection."""
        self._connected = False

    def flushdb(self):
        """Clear all data."""
        with self._lock:
            self._store.clear()
            self._expiry.clear()


class MockRedisPipeline:
    """Mock Redis pipeline."""

    def __init__(self, client: MockRedisClient):
        self.client = client
        self.commands: list[tuple[str, ...]] = []

    def set(self, key: str, value: str | bytes, ex: int | None = None):
        """Queue set command."""
        self.commands.append(("set", key, value, ex))
        return self

    def get(self, key: str):
        """Queue get command."""
        self.commands.append(("get", key))
        return self

    def delete(self, key: str):
        """Queue delete command."""
        self.commands.append(("delete", key))
        return self

    def execute(self) -> list[Any]:
        """Execute all commands."""
        results = []

        for cmd in self.commands:
            if cmd[0] == "set":
                _, key, value, ex = cmd
                result: Any = self.client.set(key, value, ex=ex)
                results.append(result)
            elif cmd[0] == "get":
                _, key = cmd
                result = self.client.get(key)
                results.append(result)
            elif cmd[0] == "delete":
                _, key = cmd
                result = self.client.delete(key)
                results.append(result)

        self.commands.clear()
        return results


class MockNLTKComponents:
    """Mock NLTK components without requiring downloads."""

    @staticmethod
    def word_tokenize(text: str) -> list[str]:
        """Simple word tokenization."""
        # Basic tokenization - split on whitespace and punctuation
        import re

        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    @staticmethod
    def sent_tokenize(text: str) -> list[str]:
        """Simple sentence tokenization."""
        # Split on common sentence endings
        import re

        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]


class MockPorterStemmer:
    """Simple mock stemmer."""

    def stem(self, word: str) -> str:
        """Basic stemming rules."""
        word = word.lower()

        # Simple suffix removal
        suffixes = ["ing", "ed", "es", "s", "er", "est", "ly"]
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[: -len(suffix)]

        return word


class MockStopwords:
    """Mock stopwords without NLTK download."""

    @staticmethod
    def words(language: str = "english") -> list[str]:
        """Return common stopwords."""
        if language == "english":
            return [
                "a",
                "an",
                "and",
                "are",
                "as",
                "at",
                "be",
                "by",
                "for",
                "from",
                "has",
                "he",
                "in",
                "is",
                "it",
                "its",
                "of",
                "on",
                "that",
                "the",
                "to",
                "was",
                "will",
                "with",
                "the",
                "this",
                "but",
                "they",
                "have",
                "had",
                "what",
                "when",
                "where",
                "who",
                "which",
                "why",
                "how",
            ]
        return []


class MockFuzz:
    """Mock fuzzywuzzy fuzz module."""

    @staticmethod
    def ratio(s1: str, s2: str) -> int:
        """Calculate similarity ratio."""
        if s1 == s2:
            return 100

        # Simple character overlap
        s1_lower = s1.lower()
        s2_lower = s2.lower()

        if s1_lower == s2_lower:
            return 95

        # Count common characters
        common = sum(1 for c in s1_lower if c in s2_lower)
        total = max(len(s1_lower), len(s2_lower))

        return int((common / total) * 100) if total > 0 else 0

    @staticmethod
    def token_set_ratio(s1: str, s2: str) -> int:
        """Token set ratio for better partial matching."""
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())

        if not tokens1 or not tokens2:
            return 0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return int((intersection / union) * 100) if union > 0 else 0


class MockProcess:
    """Mock fuzzywuzzy process module."""

    @staticmethod
    def extract(
        query: str, choices: list[str], scorer=None, limit: int = 5
    ) -> list[tuple[str, int]]:
        """Extract best matches."""
        if scorer is None:
            scorer = MockFuzz.ratio

        scores = []
        for choice in choices:
            score = scorer(query, choice)
            scores.append((choice, score))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:limit]


class MockCosineSimilarity:
    """Mock sklearn cosine similarity."""

    @staticmethod
    def cosine_similarity(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
        """Calculate cosine similarity."""
        if Y is None:
            Y = X

        # Normalize rows
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        Y_norm = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)

        # Compute similarity
        similarity = np.dot(X_norm, Y_norm.T)

        return cast(np.ndarray, similarity)


class MockNearestNeighbors:
    """Mock sklearn NearestNeighbors."""

    def __init__(self, n_neighbors=5, metric="cosine", algorithm="auto"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.algorithm = algorithm
        self.X_train = None
        self.fitted = False

    def fit(self, X: np.ndarray):
        """Fit the model with training data."""
        self.X_train = X.copy()
        self.fitted = True
        return self

    def kneighbors(
        self,
        X: np.ndarray,
        n_neighbors: int | None = None,
        return_distance: bool = True,
    ):
        """Find k nearest neighbors."""
        if not self.fitted:
            raise ValueError("Model not fitted yet")

        if n_neighbors is None:
            n_neighbors = self.n_neighbors

        # Calculate distances using cosine similarity
        if self.metric == "cosine":
            similarities = MockCosineSimilarity.cosine_similarity(X, self.X_train)
            distances = 1 - similarities  # Convert similarity to distance
        else:
            # Default to euclidean distance
            if self.X_train is not None:
                distances = np.array(
                    [[np.linalg.norm(x - y) for y in self.X_train] for x in X]
                )
            else:
                raise ValueError("X_train is not set")

        # Find k nearest neighbors
        neighbor_indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        neighbor_distances = np.array(
            [distances[i, neighbor_indices[i]] for i in range(len(X))]
        )

        if return_distance:
            return neighbor_distances, neighbor_indices
        else:
            return neighbor_indices


class TestDataGenerator:
    """Generate realistic test data for semantic search tests."""

    @staticmethod
    def generate_standards_corpus(
        num_documents: int = 100,
    ) -> list[tuple[str, str, dict[str, Any]]]:
        """Generate a corpus of standards documents."""
        frameworks = ["React", "Angular", "Vue", "Django", "FastAPI", "Spring"]
        categories = [
            "security",
            "testing",
            "performance",
            "accessibility",
            "api",
            "frontend",
            "authentication",
            "authorization",
            "database",
            "deployment",
        ]
        languages = ["Python", "JavaScript", "Java", "Go", "TypeScript", "Rust"]

        # Specific authentication and security content templates
        auth_templates = [
            {
                "title": "User Authentication Implementation",
                "content": """
                # User Authentication Implementation Standards

                This document outlines authentication standards for secure user login
                and authentication system setup in web applications.

                ## Authentication Best Practices
                - Implement secure user login workflows
                - Use strong password policies
                - Enable multi-factor authentication
                - Secure session management
                - Authentication token handling

                ## User Login Implementation
                - Implement proper authentication flows
                - Secure user credential validation
                - Login session security
                - Authentication system architecture

                Keywords: authentication, user login, auth, security, login implementation, implement, setup, enable
                """,
                "category": "authentication",
            },
            {
                "title": "Public Access Configuration",
                "content": """
                # Public Access Configuration Standards

                Guidelines for configuring public access systems without authentication
                requirements for open content and public resources.

                ## Public Access Best Practices
                - Remove unnecessary authentication barriers
                - Disable authentication for public content
                - Configure open access policies
                - Public resource management
                - Open content delivery

                ## Public System Configuration
                - Public access without authentication requirements
                - Open system architecture
                - Unrestricted content access
                - Public resource optimization

                Keywords: public, access, open, remove, disable, without, unrestricted
                """,
                "category": "public_access",
            },
            {
                "title": "Disable User Login Procedures",
                "content": """
                # Disable User Login Procedures

                Documentation for procedures to disable user login functionality
                and remove user authentication from systems when needed.

                ## Disable Login Best Practices
                - Disable user login safely
                - Remove user authentication flows
                - Deactivate login systems
                - Turn off user access controls
                - Disable authentication mechanisms

                ## User Login Removal
                - Procedures to disable user login
                - Steps to remove login functionality
                - Deactivate user authentication
                - Turn off login systems

                Keywords: disable, user, login, remove, deactivate, turn off, authentication
                """,
                "category": "disable_auth",
            },
            {
                "title": "Database Performance Optimization",
                "content": """
                # Database Performance Optimization Standards

                Guidelines for optimizing database performance and improving database speed
                through various database optimization techniques.

                ## Performance Best Practices
                - Optimize database queries
                - Implement proper indexing strategies
                - Database connection pooling
                - Query performance tuning
                - Make database faster through caching

                ## Database Speed Improvements
                - Improve database speed through optimization
                - Database optimization techniques and patterns
                - Performance monitoring and profiling

                Keywords: database, performance, optimization, speed, improve database, optimize, faster
                """,
                "category": "performance",
            },
            {
                "title": "Database Performance Issues",
                "content": """
                # Database Performance Issues and Troubleshooting

                Documentation of common database performance issues and problems
                that cause slow database operations and degraded performance.

                ## Performance Problems
                - Slow database query execution
                - Database performance bottlenecks
                - Query performance issues
                - Database speed problems
                - Performance degradation patterns

                ## Troubleshooting Slow Operations
                - Identify slow database operations
                - Diagnose performance issues
                - Database slowdown causes
                - Performance problem analysis

                Keywords: database, slow, issues, problems, degrade, bottleneck, troubleshooting
                """,
                "category": "troubleshooting",
            },
        ]

        documents = []
        for i in range(num_documents):
            framework = frameworks[i % len(frameworks)]
            category = categories[i % len(categories)]
            language = languages[i % len(languages)]

            doc_id = f"std-{i:04d}"

            # Use specific templates for different categories
            if category == "authentication":
                template = auth_templates[0]  # Use auth template
                content = template["content"]
                category = template["category"]
            elif (
                category == "security" and (i % 15) < 3
            ):  # Some security docs about public access
                template = auth_templates[1]  # Use public access template
                content = template["content"]
                category = template["category"]
            elif (
                category == "security" and (i % 15) >= 3 and (i % 15) < 6
            ):  # Some security docs about disabling auth
                template = auth_templates[2]  # Use disable login template
                content = template["content"]
                category = template["category"]
            elif (
                category == "performance" and (i % 20) < 8
            ):  # Some performance docs about database optimization
                template = auth_templates[3]  # Use database performance template
                content = template["content"]
                category = template["category"]
            elif (
                category == "performance" and (i % 20) >= 8 and (i % 20) < 12
            ):  # Some performance docs about issues
                template = auth_templates[4]  # Use database issues template
                content = template["content"]
                category = template["category"]
            else:
                # Use original generic template
                content = f"""
                # {framework} {category.title()} Standards

                This document outlines {category} standards for {framework} applications
                using {language} programming language.

                ## Best Practices
                - Follow {category} guidelines
                - Implement proper {framework} patterns
                - Use {language} idioms

                ## Requirements
                - All {framework} components must follow {category} standards
                - {language} code must be properly tested
                - Documentation must be complete

                Keywords: {framework}, {category}, {language}, standards, mcp, guidelines
                """

            metadata = {
                "framework": framework.lower(),
                "category": category,
                "language": language.lower(),
                "version": f"{(i % 3) + 1}.0",
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "index": i,
            }

            documents.append((doc_id, content.strip(), metadata))

        return documents

    @staticmethod
    def generate_test_queries() -> list[dict[str, Any]]:
        """Generate test queries with expected results."""
        return [
            {
                "query": "React security best practices",
                "expected_keywords": ["react", "security"],
                "expected_categories": ["security"],
                "min_results": 1,
            },
            {
                "query": "Python API testing",
                "expected_keywords": ["python", "api", "testing"],
                "expected_languages": ["python"],
                "min_results": 1,
            },
            {
                "query": "frontend performance optimization",
                "expected_keywords": ["frontend", "performance"],
                "expected_categories": ["performance", "frontend"],
                "min_results": 1,
            },
            {
                "query": "testing NOT angular",
                "expected_keywords": ["testing"],
                "excluded_keywords": ["angular"],
                "min_results": 1,
            },
            {
                "query": "security AND java",
                "expected_keywords": ["security", "java"],
                "required_all": True,
                "min_results": 1,
            },
        ]


class MockAsyncio:
    """Mock asyncio operations for deterministic testing."""

    @staticmethod
    def create_deterministic_event_loop():
        """Create a deterministic event loop for testing."""
        import asyncio

        class DeterministicEventLoop(asyncio.BaseEventLoop):
            def __init__(self):
                super().__init__()
                self._time = 0.0
                self._scheduled = []

            def time(self):
                return self._time

            def advance_time(self, seconds: float):
                self._time += seconds

        return DeterministicEventLoop()


# Helper functions for easy mocking
def patch_ml_dependencies():
    """Decorator to patch ML dependencies for a test."""
    import asyncio
    import functools
    from unittest.mock import patch

    def decorator(func):
        # Check if the function is async
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            @patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer)
            @patch("redis.Redis", MockRedisClient)
            @patch("nltk.stem.PorterStemmer", MockPorterStemmer)
            @patch("nltk.tokenize.word_tokenize", MockNLTKComponents.word_tokenize)
            @patch("nltk.corpus.stopwords", MockStopwords)
            @patch("fuzzywuzzy.fuzz", MockFuzz)
            @patch("fuzzywuzzy.process", MockProcess)
            @patch(
                "sklearn.metrics.pairwise.cosine_similarity",
                MockCosineSimilarity.cosine_similarity,
            )
            @patch("sklearn.neighbors.NearestNeighbors", MockNearestNeighbors)
            async def async_wrapped(*args, **kwargs):
                return await func(*args, **kwargs)

            return async_wrapped
        else:

            @functools.wraps(func)
            @patch("sentence_transformers.SentenceTransformer", MockSentenceTransformer)
            @patch("redis.Redis", MockRedisClient)
            @patch("nltk.stem.PorterStemmer", MockPorterStemmer)
            @patch("nltk.tokenize.word_tokenize", MockNLTKComponents.word_tokenize)
            @patch("nltk.corpus.stopwords", MockStopwords)
            @patch("fuzzywuzzy.fuzz", MockFuzz)
            @patch("fuzzywuzzy.process", MockProcess)
            @patch(
                "sklearn.metrics.pairwise.cosine_similarity",
                MockCosineSimilarity.cosine_similarity,
            )
            @patch("sklearn.neighbors.NearestNeighbors", MockNearestNeighbors)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

    return decorator
