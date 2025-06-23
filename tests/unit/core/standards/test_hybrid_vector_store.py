"""
Unit tests for Hybrid Vector Store implementation.

@nist-controls: SA-11, CA-7
@evidence: Comprehensive testing of three-tier vector store
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.core.standards.hybrid_vector_store import (
    FAISSHotCache,
    HybridConfig,
    HybridVectorStore,
    RedisQueryCache,
    SearchResult,
    TierMetrics,
)

# Check if FAISS is available
try:
    import faiss  # noqa: F401
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

skipif_no_faiss = pytest.mark.skipif(not FAISS_AVAILABLE, reason="FAISS not installed")


class TestSearchResult:
    """Test SearchResult dataclass"""

    def test_search_result_creation(self):
        """Test creating a search result"""
        result = SearchResult(
            id="test_001",
            content="Test content",
            score=0.95,
            metadata={"type": "test"},
            source_tier="faiss"
        )

        assert result.id == "test_001"
        assert result.content == "Test content"
        assert result.score == 0.95
        assert result.metadata["type"] == "test"
        assert result.source_tier == "faiss"


class TestHybridConfig:
    """Test HybridConfig configuration"""

    def test_default_config(self):
        """Test default configuration values"""
        config = HybridConfig()

        assert config.hot_cache_size == 1000
        assert config.faiss_dimension == 384
        assert config.faiss_index_type == "Flat"
        assert config.chroma_path == ".chroma_db"
        assert config.chroma_collection == "standards"
        assert config.redis_ttl == 3600
        assert config.redis_prefix == "mcp:search:"
        assert config.enable_monitoring is True
        assert config.access_threshold == 10

    def test_custom_config(self):
        """Test custom configuration values"""
        config = HybridConfig(
            hot_cache_size=500,
            redis_ttl=7200,
            access_threshold=5
        )

        assert config.hot_cache_size == 500
        assert config.redis_ttl == 7200
        assert config.access_threshold == 5


class TestTierMetrics:
    """Test TierMetrics tracking"""

    def test_metrics_initialization(self):
        """Test metrics initialization"""
        metrics = TierMetrics()

        assert metrics.hits == 0
        assert metrics.misses == 0
        assert len(metrics.latencies) == 0
        assert metrics.last_reset > 0

    def test_record_hit(self):
        """Test recording a cache hit"""
        metrics = TierMetrics()
        metrics.record_hit(0.5)

        assert metrics.hits == 1
        assert metrics.misses == 0
        assert len(metrics.latencies) == 1
        assert metrics.latencies[0] == 0.5

    def test_record_miss(self):
        """Test recording a cache miss"""
        metrics = TierMetrics()
        metrics.record_miss()

        assert metrics.hits == 0
        assert metrics.misses == 1
        assert len(metrics.latencies) == 0

    def test_get_stats(self):
        """Test getting statistics"""
        metrics = TierMetrics()
        metrics.record_hit(0.1)
        metrics.record_hit(0.2)
        metrics.record_miss()

        stats = metrics.get_stats()

        assert stats["total_requests"] == 3
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.666, rel=0.01)
        assert stats["avg_latency_ms"] == pytest.approx(150, rel=0.01)  # 0.15 * 1000
        assert stats["p95_latency_ms"] == pytest.approx(195, rel=0.1)  # p95 with 2 points

    def test_latency_buffer_limit(self):
        """Test that latency buffer doesn't grow indefinitely"""
        metrics = TierMetrics()

        # Add more than 10000 latencies
        for _ in range(15000):
            metrics.record_hit(0.1)

        # Should only keep last 10000
        assert len(metrics.latencies) == 10000


@skipif_no_faiss
class TestFAISSHotCache:
    """Test FAISS hot cache tier"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return HybridConfig(hot_cache_size=10, faiss_dimension=384)

    @pytest.fixture
    def faiss_cache(self, config):
        """Create FAISS cache instance"""
        return FAISSHotCache(config)

    def test_initialization(self, faiss_cache):
        """Test FAISS cache initialization"""
        assert faiss_cache.config.hot_cache_size == 10
        assert faiss_cache.config.faiss_dimension == 384
        # _index might be None if FAISS is not installed
        assert hasattr(faiss_cache, '_index')
        assert len(faiss_cache._id_map) == 0
        assert len(faiss_cache._lru_cache) == 0

    @pytest.mark.asyncio
    async def test_add_document(self, faiss_cache):
        """Test adding a document to FAISS cache"""
        embedding = np.random.rand(384).astype(np.float32)

        success = await faiss_cache.add(
            id="doc_001",
            embedding=embedding,
            content="Test document",
            metadata={"type": "test"}
        )

        assert success is True
        assert "doc_001" in faiss_cache._id_map
        assert faiss_cache._lru_cache["doc_001"] == ("Test document", {"type": "test"})

    @pytest.mark.asyncio
    async def test_search(self, faiss_cache):
        """Test searching in FAISS cache"""
        # Add some documents
        for i in range(5):
            embedding = np.random.rand(384).astype(np.float32)
            await faiss_cache.add(
                id=f"doc_{i:03d}",
                embedding=embedding,
                content=f"Document {i}",
                metadata={"index": i}
            )

        # Search
        query_embedding = np.random.rand(384).astype(np.float32)
        results = await faiss_cache.search(query_embedding, k=3)

        assert len(results) == 3
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source_tier == "faiss" for r in results)

    @pytest.mark.asyncio
    async def test_lru_eviction(self, faiss_cache):
        """Test LRU eviction when cache is full"""
        # Fill cache beyond capacity
        for i in range(15):
            embedding = np.random.rand(384).astype(np.float32)
            await faiss_cache.add(
                id=f"doc_{i:03d}",
                embedding=embedding,
                content=f"Document {i}",
                metadata={"index": i}
            )

        # Should only have 10 documents (max capacity)
        assert len(faiss_cache._id_map) == 10
        assert len(faiss_cache._lru_cache) == 10

        # First 5 documents should have been evicted
        for i in range(5):
            assert f"doc_{i:03d}" not in faiss_cache._id_map

    @pytest.mark.asyncio
    async def test_remove_document(self, faiss_cache):
        """Test removing a document"""
        embedding = np.random.rand(384).astype(np.float32)
        await faiss_cache.add(
            id="doc_001",
            embedding=embedding,
            content="Test document",
            metadata={}
        )

        success = await faiss_cache.remove("doc_001")
        assert success is True
        assert "doc_001" not in faiss_cache._id_map
        assert "doc_001" not in faiss_cache._lru_cache

    @pytest.mark.asyncio
    async def test_get_stats(self, faiss_cache):
        """Test getting cache statistics"""
        # Add some documents
        for i in range(3):
            embedding = np.random.rand(384).astype(np.float32)
            await faiss_cache.add(
                id=f"doc_{i:03d}",
                embedding=embedding,
                content=f"Document {i}",
                metadata={}
            )

        stats = await faiss_cache.get_stats()

        assert stats["size"] == 3
        assert stats["capacity"] == 10
        assert stats["utilization"] == 0.3
        assert "metrics" in stats


class TestRedisQueryCache:
    """Test Redis query cache tier"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return HybridConfig(redis_ttl=300, redis_prefix="test:")

    @pytest.fixture
    def redis_cache(self, config):
        """Create Redis cache instance with mocked Redis"""
        with patch('src.core.standards.hybrid_vector_store.get_redis_client') as mock_get_redis:
            mock_redis = MagicMock()
            mock_get_redis.return_value = mock_redis
            cache = RedisQueryCache(config)
            cache._redis = mock_redis
            return cache

    def test_initialization(self, redis_cache):
        """Test Redis cache initialization"""
        assert redis_cache.config.redis_ttl == 300
        assert redis_cache.config.redis_prefix == "test:"
        assert redis_cache._redis is not None

    @pytest.mark.asyncio
    async def test_cache_results(self, redis_cache):
        """Test caching query results"""
        query_embedding = np.random.rand(384).astype(np.float32)
        results = [
            SearchResult(
                id="doc_001",
                content="Test 1",
                score=0.95,
                metadata={},
                source_tier="chromadb"
            ),
            SearchResult(
                id="doc_002",
                content="Test 2",
                score=0.90,
                metadata={},
                source_tier="chromadb"
            )
        ]

        success = await redis_cache.cache_results(
            query_embedding=query_embedding,
            results=results,
            k=5,
            filters=None
        )

        assert success is True
        assert redis_cache._redis.setex.called

    @pytest.mark.asyncio
    async def test_search_cache_hit(self, redis_cache):
        """Test searching with cache hit"""
        # Mock cached data
        cached_data = [
            {
                'id': 'doc_001',
                'score': 0.95,
                'content': 'Cached content',
                'metadata': {'cached': True}
            }
        ]
        redis_cache._redis.get.return_value = json.dumps(cached_data).encode()

        query_embedding = np.random.rand(384).astype(np.float32)
        results = await redis_cache.search(query_embedding, k=5)

        assert len(results) == 1
        assert results[0].id == "doc_001"
        assert results[0].source_tier == "redis"
        assert results[0].metadata['cached'] is True

    @pytest.mark.asyncio
    async def test_search_cache_miss(self, redis_cache):
        """Test searching with cache miss"""
        redis_cache._redis.get.return_value = None

        query_embedding = np.random.rand(384).astype(np.float32)
        results = await redis_cache.search(query_embedding, k=5)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_get_stats(self, redis_cache):
        """Test getting cache statistics"""
        stats = await redis_cache.get_stats()

        assert stats["redis_connected"] is True
        # Metrics are included directly in stats
        assert "hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats


class TestHybridVectorStore:
    """Test main hybrid vector store"""

    @pytest.fixture
    def config(self):
        """Create test config"""
        return HybridConfig(
            hot_cache_size=10,
            faiss_dimension=384,
            redis_ttl=300
        )

    @pytest.fixture
    def hybrid_store(self, config):
        """Create hybrid store instance with mocked dependencies"""
        with patch('src.core.standards.hybrid_vector_store.get_redis_client') as mock_get_redis:
            mock_redis = MagicMock()
            mock_get_redis.return_value = mock_redis

            # Mock ChromaDB availability
            with patch('src.core.standards.hybrid_vector_store.CHROMADB_AVAILABLE', True):
                with patch('src.core.standards.hybrid_vector_store.ChromaDBTier') as MockChromaDB:
                    mock_chroma = AsyncMock()
                    MockChromaDB.return_value = mock_chroma

                    store = HybridVectorStore(config)
                    store.redis_tier._redis = mock_redis
                    store.chroma_tier = mock_chroma

                    return store

    def test_initialization(self, hybrid_store):
        """Test hybrid store initialization"""
        assert hybrid_store.config.hot_cache_size == 10
        assert hybrid_store.redis_tier is not None
        assert hybrid_store.faiss_tier is not None
        assert hybrid_store.chroma_tier is not None

    @pytest.mark.asyncio
    async def test_add_document(self, hybrid_store):
        """Test adding a document to hybrid store"""
        embedding = np.random.rand(384).astype(np.float32)

        # Mock ChromaDB add
        hybrid_store.chroma_tier.add.return_value = True

        success = await hybrid_store.add(
            id="doc_001",
            content="Test document",
            embedding=embedding,
            metadata={"type": "test"}
        )

        assert success is True
        assert hybrid_store.chroma_tier.add.called

    @pytest.mark.asyncio
    async def test_search_redis_hit(self, hybrid_store):
        """Test search with Redis cache hit"""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock Redis hit
        redis_result = SearchResult(
            id="doc_001",
            content="Cached content",
            score=0.95,
            metadata={},
            source_tier="redis"
        )

        hybrid_store.redis_tier.search = AsyncMock(return_value=[redis_result])

        results = await hybrid_store.search(
            query="test query",
            query_embedding=query_embedding,
            k=5
        )

        assert len(results) == 1
        assert results[0].source_tier == "redis"

    @pytest.mark.asyncio
    async def test_search_fallback_to_faiss(self, hybrid_store):
        """Test search fallback from Redis to FAISS"""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock Redis miss
        hybrid_store.redis_tier.search = AsyncMock(return_value=[])

        # Mock FAISS hit
        faiss_results = [
            SearchResult(
                id="doc_001",
                content="FAISS content",
                score=0.90,
                metadata={},
                source_tier="faiss"
            )
        ]
        hybrid_store.faiss_tier.search = AsyncMock(return_value=faiss_results)

        results = await hybrid_store.search(
            query="test query",
            query_embedding=query_embedding,
            k=5
        )

        assert len(results) == 1
        assert results[0].source_tier == "faiss"

    @pytest.mark.asyncio
    async def test_search_fallback_to_chromadb(self, hybrid_store):
        """Test search fallback to ChromaDB"""
        query_embedding = np.random.rand(384).astype(np.float32)

        # Mock Redis and FAISS miss
        hybrid_store.redis_tier.search = AsyncMock(return_value=[])
        hybrid_store.faiss_tier.search = AsyncMock(return_value=[])

        # Mock ChromaDB hit
        chroma_results = [
            SearchResult(
                id="doc_001",
                content="ChromaDB content",
                score=0.85,
                metadata={},
                source_tier="chromadb"
            )
        ]
        hybrid_store.chroma_tier.search = AsyncMock(return_value=chroma_results)

        results = await hybrid_store.search(
            query="test query",
            query_embedding=query_embedding,
            k=5
        )

        assert len(results) == 1
        assert results[0].source_tier == "chromadb"

    @pytest.mark.asyncio
    async def test_remove_from_all_tiers(self, hybrid_store):
        """Test removing document from all tiers"""
        # Mock remove methods
        hybrid_store.redis_tier.remove = AsyncMock(return_value=True)
        hybrid_store.faiss_tier.remove = AsyncMock(return_value=True)
        hybrid_store.chroma_tier.remove = AsyncMock(return_value=True)

        success = await hybrid_store.remove("doc_001")

        assert success is True
        assert hybrid_store.faiss_tier.remove.called
        assert hybrid_store.chroma_tier.remove.called

    @pytest.mark.asyncio
    async def test_optimize(self, hybrid_store):
        """Test optimization process"""
        # Mock access counts
        hybrid_store.faiss_tier.get_access_counts = MagicMock(
            return_value={"doc_001": 15, "doc_002": 5}
        )

        # Mock ChromaDB get_by_ids
        hybrid_store.chroma_tier.get_by_ids = AsyncMock(return_value=[])

        await hybrid_store.optimize()

        # Should have analyzed access patterns
        assert hybrid_store.faiss_tier.get_access_counts.called

    @pytest.mark.asyncio
    async def test_get_stats(self, hybrid_store):
        """Test getting statistics from all tiers"""
        # Mock tier stats
        hybrid_store.redis_tier.get_stats = AsyncMock(
            return_value={"redis_connected": True}
        )
        hybrid_store.faiss_tier.get_stats = AsyncMock(
            return_value={"size": 5, "capacity": 10}
        )
        hybrid_store.chroma_tier.get_stats = AsyncMock(
            return_value={"total_documents": 100}
        )

        stats = await hybrid_store.get_stats()

        assert "tiers" in stats
        assert "redis" in stats["tiers"]
        assert "faiss" in stats["tiers"]
        assert "chromadb" in stats["tiers"]
        assert "total_searches" in stats
        assert "tier_hits" in stats
        assert "overall_hit_rate" in stats
