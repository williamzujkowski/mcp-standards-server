"""
Unit tests for Tiered Storage Strategy implementation.

@nist-controls: SA-11, CA-7
@evidence: Comprehensive testing of tiered storage placement logic
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.standards.tiered_storage_strategy import (
    AccessPattern,
    DocumentMetadata,
    PlacementDecision,
    StorageTier,
    TieredStorageStrategy,
)


class TestAccessPattern:
    """Test AccessPattern class"""

    def test_access_pattern_initialization(self):
        """Test creating access pattern"""
        pattern = AccessPattern(document_id="doc_001")

        assert pattern.document_id == "doc_001"
        assert pattern.access_count == 0
        assert pattern.last_accessed > 0
        assert len(pattern.access_times) == 0
        assert len(pattern.query_contexts) == 0

    def test_record_access(self):
        """Test recording document access"""
        pattern = AccessPattern(document_id="doc_001")
        initial_time = pattern.last_accessed

        # Wait a tiny bit to ensure time difference
        time.sleep(0.01)

        pattern.record_access("search_query")

        assert pattern.access_count == 1
        assert pattern.last_accessed > initial_time
        assert len(pattern.access_times) == 1
        assert pattern.query_contexts == ["search_query"]

    def test_record_access_limits(self):
        """Test access recording limits"""
        pattern = AccessPattern(document_id="doc_001")

        # Record more than limit
        for i in range(150):
            pattern.record_access(f"query_{i}")

        assert pattern.access_count == 150
        assert len(pattern.access_times) == 100  # Limited to 100
        assert len(pattern.query_contexts) == 20  # Limited to 20

        # Check that we kept the most recent ones
        assert pattern.query_contexts[-1] == "query_149"

    def test_get_access_frequency(self):
        """Test calculating access frequency"""
        pattern = AccessPattern(document_id="doc_001")

        # Record 10 accesses
        for _ in range(10):
            pattern.record_access()

        # Default window is 1 hour (3600 seconds)
        frequency = pattern.get_access_frequency()
        assert frequency == pytest.approx(10.0, rel=0.1)  # 10 per hour

        # Test with different window
        frequency_per_minute = pattern.get_access_frequency(window_seconds=60)
        assert frequency_per_minute == pytest.approx(600.0, rel=0.1)  # 10 per minute = 600 per hour

    def test_get_access_frequency_with_old_accesses(self):
        """Test frequency calculation excludes old accesses"""
        pattern = AccessPattern(document_id="doc_001")

        # Add some old access times (2 hours ago)
        old_time = time.time() - 7200
        pattern.access_times = [old_time] * 5
        pattern.access_count = 5

        # Add recent accesses
        for _ in range(3):
            pattern.record_access()

        # Only recent accesses should count
        frequency = pattern.get_access_frequency(window_seconds=3600)
        assert frequency == pytest.approx(3.0, rel=0.1)

    def test_get_recency_score(self):
        """Test calculating recency score"""
        pattern = AccessPattern(document_id="doc_001")

        # Fresh access
        pattern.record_access()
        score = pattern.get_recency_score()
        assert score > 0.99  # Very recent, close to 1

        # Simulate 1 hour old
        pattern.last_accessed = time.time() - 3600
        score = pattern.get_recency_score()
        assert score == pytest.approx(0.5, rel=0.01)  # Half-life is 1 hour

        # Simulate 2 hours old
        pattern.last_accessed = time.time() - 7200
        score = pattern.get_recency_score()
        assert score == pytest.approx(0.25, rel=0.01)


class TestDocumentMetadata:
    """Test DocumentMetadata dataclass"""

    def test_document_metadata_creation(self):
        """Test creating document metadata"""
        metadata = DocumentMetadata(
            id="doc_001",
            size_bytes=1024,
            token_count=100,
            language="python",
            framework="django",
            control_families=["AC", "AU"],
            priority="high",
            document_type="micro_standard",
            version="2.0"
        )

        assert metadata.id == "doc_001"
        assert metadata.size_bytes == 1024
        assert metadata.token_count == 100
        assert metadata.language == "python"
        assert metadata.framework == "django"
        assert metadata.control_families == ["AC", "AU"]
        assert metadata.priority == "high"
        assert metadata.document_type == "micro_standard"
        assert metadata.version == "2.0"

    def test_document_metadata_defaults(self):
        """Test default values"""
        metadata = DocumentMetadata(
            id="doc_002",
            size_bytes=512,
            token_count=50
        )

        assert metadata.language is None
        assert metadata.framework is None
        # control_families has a default_factory, so it should be an empty list
        assert hasattr(metadata, 'control_families')
        assert metadata.priority == "normal"
        assert metadata.document_type == "standard"
        assert metadata.version == "latest"


class TestPlacementDecision:
    """Test PlacementDecision model"""

    def test_placement_decision_creation(self):
        """Test creating placement decision"""
        decision = PlacementDecision(
            document_id="doc_001",
            recommended_tiers=[StorageTier.FAISS_HOT, StorageTier.CHROMADB],
            reasoning="High access frequency",
            score=0.85,
            metadata={"access_count": 100}
        )

        assert decision.document_id == "doc_001"
        assert len(decision.recommended_tiers) == 2
        assert StorageTier.FAISS_HOT in decision.recommended_tiers
        assert decision.reasoning == "High access frequency"
        assert decision.score == 0.85
        assert decision.metadata["access_count"] == 100


class TestTieredStorageStrategy:
    """Test TieredStorageStrategy class"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        return MagicMock()

    @pytest.fixture
    def strategy(self, mock_redis):
        """Create storage strategy with mocked Redis"""
        with patch('src.core.standards.tiered_storage_strategy.get_redis_client', return_value=mock_redis):
            strategy = TieredStorageStrategy(
                hot_cache_size=100,
                access_threshold=5,
                recency_weight=0.3,
                frequency_weight=0.5,
                priority_weight=0.2
            )
            strategy._redis = mock_redis
            return strategy

    def test_initialization(self, mock_redis):
        """Test strategy initialization"""
        with patch('src.core.standards.tiered_storage_strategy.get_redis_client', return_value=mock_redis):
            strategy = TieredStorageStrategy(
                hot_cache_size=50,
                access_threshold=10
            )

            assert strategy.hot_cache_size == 50
            assert strategy.access_threshold == 10
            assert strategy.recency_weight == 0.3
            assert strategy.frequency_weight == 0.5
            assert strategy.priority_weight == 0.2
            assert len(strategy._access_patterns) == 0
            assert len(strategy._faiss_contents) == 0

    def test_record_access_new_document(self, strategy):
        """Test recording access for new document"""
        strategy.record_access("doc_001", "search_standards")

        assert "doc_001" in strategy._access_patterns
        pattern = strategy._access_patterns["doc_001"]
        assert pattern.access_count == 1
        assert pattern.query_contexts == ["search_standards"]

    def test_record_access_existing_document(self, strategy):
        """Test recording access for existing document"""
        strategy.record_access("doc_001", "first_query")
        strategy.record_access("doc_001", "second_query")

        pattern = strategy._access_patterns["doc_001"]
        assert pattern.access_count == 2
        assert len(pattern.query_contexts) == 2

    def test_get_placement_decision_new_document(self, strategy):
        """Test placement decision for document with no access"""
        metadata = DocumentMetadata(
            id="new_doc",
            size_bytes=1024,
            token_count=100
        )

        decision = strategy.get_placement_decision(metadata)

        assert decision.document_id == "new_doc"
        assert StorageTier.CHROMADB in decision.recommended_tiers
        assert StorageTier.FAISS_HOT not in decision.recommended_tiers
        assert decision.score == 0.0
        assert "ChromaDB for persistent storage" in decision.reasoning

    def test_get_placement_decision_high_access(self, strategy):
        """Test placement decision for frequently accessed document"""
        # Create document with high access
        doc_id = "popular_doc"
        for _ in range(10):
            strategy.record_access(doc_id, "search")

        metadata = DocumentMetadata(
            id=doc_id,
            size_bytes=1024,
            token_count=100,
            priority="normal"
        )

        decision = strategy.get_placement_decision(metadata)

        assert StorageTier.FAISS_HOT in decision.recommended_tiers
        assert StorageTier.CHROMADB in decision.recommended_tiers
        assert decision.score > 0.5
        assert "FAISS hot cache" in decision.reasoning

    def test_get_placement_decision_critical_priority(self, strategy):
        """Test placement decision for critical priority document"""
        metadata = DocumentMetadata(
            id="critical_doc",
            size_bytes=1024,
            token_count=100,
            priority="critical"
        )

        decision = strategy.get_placement_decision(metadata)

        assert StorageTier.FAISS_HOT in decision.recommended_tiers
        assert "Critical priority requires hot cache" in decision.reasoning

    def test_get_placement_decision_micro_standard(self, strategy):
        """Test placement decision for micro standard"""
        metadata = DocumentMetadata(
            id="micro_001",
            size_bytes=512,
            token_count=500,
            document_type="micro_standard"
        )

        decision = strategy.get_placement_decision(metadata)

        assert StorageTier.FAISS_HOT in decision.recommended_tiers
        assert "Micro-standard prioritized for hot cache" in decision.reasoning

    def test_get_eviction_candidates_no_access(self, strategy):
        """Test eviction candidates with no access patterns"""
        # Add documents to FAISS contents
        strategy._faiss_contents = {"doc_1", "doc_2", "doc_3"}

        candidates = strategy.get_eviction_candidates(count=2)

        assert len(candidates) == 2
        # All should have score 0 (no access pattern)
        assert all(score == 0.0 for _, score in candidates)

    def test_get_eviction_candidates_with_access(self, strategy):
        """Test eviction candidates with varying access patterns"""
        # Set up documents with different access patterns
        docs = {
            "old_doc": 1,    # 1 access, old
            "recent_doc": 5,  # 5 accesses, recent
            "popular_doc": 20 # 20 accesses, recent
        }

        for doc_id, access_count in docs.items():
            for _ in range(access_count):
                strategy.record_access(doc_id)
            strategy._faiss_contents.add(doc_id)

        # Make old_doc actually old
        strategy._access_patterns["old_doc"].last_accessed = time.time() - 7200

        candidates = strategy.get_eviction_candidates(count=3)

        # old_doc should be first (lowest score)
        assert candidates[0][0] == "old_doc"
        assert candidates[0][1] < candidates[1][1]
        assert candidates[1][1] < candidates[2][1]

    def test_get_promotion_candidates(self, strategy):
        """Test getting promotion candidates"""
        # Create documents with high access not in FAISS
        high_access_docs = ["hot_1", "hot_2", "hot_3"]
        for doc_id in high_access_docs:
            for _ in range(10):
                strategy.record_access(doc_id)

        # Create documents with low access
        low_access_docs = ["cold_1", "cold_2"]
        for doc_id in low_access_docs:
            strategy.record_access(doc_id)

        candidates = strategy.get_promotion_candidates(count=5)

        # Should get decisions for high-access docs
        candidate_ids = [c.document_id for c in candidates]
        assert all(doc_id in candidate_ids for doc_id in high_access_docs)

        # High access docs should have higher scores
        assert all(c.score > 0.5 for c in candidates if c.document_id in high_access_docs)

    def test_optimize_tier_placement_cache_full(self, strategy):
        """Test optimization when cache is full"""
        # Fill cache beyond 90%
        for i in range(95):
            doc_id = f"doc_{i}"
            strategy._faiss_contents.add(doc_id)
            # Give some minimal access
            strategy.record_access(doc_id)

        # Add some cold documents
        for i in range(10):
            doc_id = f"cold_{i}"
            strategy._faiss_contents.add(doc_id)
            # No access pattern

        # Add some hot candidates not in cache
        for i in range(5):
            doc_id = f"hot_candidate_{i}"
            for _ in range(20):
                strategy.record_access(doc_id)

        results = strategy.optimize_tier_placement()

        assert len(results["evictions"]) > 0
        assert len(results["promotions"]) > 0
        assert results["cache_utilization_before"] > 0.9
        assert results["cache_utilization_after"] <= 1.0

    def test_optimize_tier_placement_cache_available(self, strategy):
        """Test optimization with available cache space"""
        # Partially fill cache
        for i in range(50):
            doc_id = f"existing_{i}"
            strategy._faiss_contents.add(doc_id)
            strategy.record_access(doc_id)

        # Add hot candidates
        for i in range(10):
            doc_id = f"promote_{i}"
            for _ in range(15):
                strategy.record_access(doc_id)

        results = strategy.optimize_tier_placement()

        assert len(results["evictions"]) == 0  # No need to evict
        assert len(results["promotions"]) > 0
        assert results["cache_utilization_after"] > results["cache_utilization_before"]

    def test_get_tier_stats(self, strategy):
        """Test getting tier statistics"""
        # Set up some data
        for i in range(20):
            doc_id = f"doc_{i}"
            # Record at least one access for doc_0
            if i == 0:
                strategy.record_access(doc_id)
            else:
                for _ in range(i):
                    strategy.record_access(doc_id)
            if i < 10:
                strategy._faiss_contents.add(doc_id)

        stats = strategy.get_tier_stats()

        assert stats["total_documents_tracked"] == 20
        assert stats["hot_cache_items"] == 10
        assert stats["hot_cache_utilization"] == 0.1  # 10/100
        assert stats["access_threshold"] == 5
        assert "access_count_mean" in stats
        assert "access_count_median" in stats
        assert "access_count_p95" in stats
        assert stats["documents_above_threshold"] == 15  # docs 5-19 have >= 5 accesses

    def test_calculate_placement_score(self, strategy):
        """Test placement score calculation"""
        # Create access pattern
        pattern = AccessPattern(document_id="test_doc")
        for _ in range(10):
            pattern.record_access()

        # Test normal priority
        metadata = DocumentMetadata(
            id="test_doc",
            size_bytes=1024,
            token_count=100,
            priority="normal",
            document_type="standard"
        )

        score = strategy._calculate_placement_score(metadata, pattern)
        assert 0 <= score <= 1

        # Test critical priority (should be higher)
        metadata_critical = DocumentMetadata(
            id="test_doc",
            size_bytes=1024,
            token_count=100,
            priority="critical",
            document_type="standard"
        )

        score_critical = strategy._calculate_placement_score(metadata_critical, pattern)
        assert score_critical > score

        # Test micro_standard type (should have bonus)
        metadata_micro = DocumentMetadata(
            id="test_doc",
            size_bytes=512,
            token_count=500,
            priority="normal",
            document_type="micro_standard"
        )

        score_micro = strategy._calculate_placement_score(metadata_micro, pattern)
        assert score_micro > score

    def test_should_be_in_hot_cache(self, strategy):
        """Test hot cache inclusion logic"""
        # No access pattern - never in cache
        normal_meta = DocumentMetadata(
            id="normal",
            size_bytes=1024,
            token_count=100,
            priority="normal"
        )
        assert strategy._should_be_in_hot_cache(normal_meta, None, 0.0) is False

        # Critical document with access pattern - always in cache
        critical_pattern = AccessPattern(document_id="critical")
        critical_pattern.record_access()
        critical_meta = DocumentMetadata(
            id="critical",
            size_bytes=1024,
            token_count=100,
            priority="critical"
        )
        assert strategy._should_be_in_hot_cache(critical_meta, critical_pattern, 0.0) is True

        # Micro standard with access
        micro_pattern = AccessPattern(document_id="micro")
        micro_pattern.record_access()
        micro_meta = DocumentMetadata(
            id="micro",
            size_bytes=512,
            token_count=500,
            document_type="micro_standard"
        )
        assert strategy._should_be_in_hot_cache(micro_meta, micro_pattern, 0.3) is True

        # Normal document below threshold
        low_pattern = AccessPattern(document_id="low")
        for _ in range(3):
            low_pattern.record_access()
        normal_meta = DocumentMetadata(
            id="low",
            size_bytes=1024,
            token_count=100
        )
        assert strategy._should_be_in_hot_cache(normal_meta, low_pattern, 0.4) is False

        # High frequency document
        high_freq_pattern = AccessPattern(document_id="high_freq")
        # Simulate 6 accesses in recent time
        current_time = time.time()
        high_freq_pattern.access_times = [current_time - i * 300 for i in range(6)]  # Every 5 min
        high_freq_pattern.access_count = 6
        assert strategy._should_be_in_hot_cache(normal_meta, high_freq_pattern, 0.4) is True

    def test_load_access_patterns(self, strategy, mock_redis):
        """Test loading access patterns from Redis"""
        # Mock Redis data
        patterns_data = {
            "doc_1": {
                "document_id": "doc_1",
                "access_count": 10,
                "last_accessed": time.time(),
                "access_times": [time.time()],
                "query_contexts": ["search"]
            },
            "doc_2": {
                "document_id": "doc_2",
                "access_count": 5,
                "last_accessed": time.time(),
                "access_times": [],
                "query_contexts": []
            }
        }

        mock_redis.get.return_value = json.dumps(patterns_data)

        # Clear and reload
        strategy._access_patterns.clear()
        strategy._load_access_patterns()

        assert len(strategy._access_patterns) == 2
        assert strategy._access_patterns["doc_1"].access_count == 10
        assert strategy._access_patterns["doc_2"].access_count == 5

    def test_save_access_patterns(self, strategy, mock_redis):
        """Test saving access patterns to Redis"""
        # Add some patterns
        strategy.record_access("doc_1", "query_1")
        strategy.record_access("doc_2", "query_2")

        strategy._save_access_patterns()

        # Verify Redis setex was called
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][0] == "mcp:access_patterns"
        assert call_args[0][1] == 86400  # 24 hour TTL

        # Verify data format
        saved_data = json.loads(call_args[0][2])
        assert "doc_1" in saved_data
        assert "doc_2" in saved_data
        assert saved_data["doc_1"]["access_count"] == 1

    def test_clear_stats(self, strategy, mock_redis):
        """Test clearing all statistics"""
        # Add some data
        strategy.record_access("doc_1")
        strategy._faiss_contents.add("doc_1")

        strategy.clear_stats()

        assert len(strategy._access_patterns) == 0
        assert len(strategy._faiss_contents) == 0
        mock_redis.delete.assert_called_once_with("mcp:access_patterns")

    def test_redis_not_available(self):
        """Test behavior when Redis is not available"""
        with patch('src.core.standards.tiered_storage_strategy.get_redis_client', return_value=None):
            strategy = TieredStorageStrategy()

            # Should still work without Redis
            strategy.record_access("doc_1")
            assert "doc_1" in strategy._access_patterns

            # Save/load should not crash
            strategy._save_access_patterns()
            strategy._load_access_patterns()

    def test_periodic_save_on_record_access(self, strategy, mock_redis):
        """Test that access patterns are saved periodically"""
        # Record 99 accesses - should not trigger save
        for i in range(99):
            strategy.record_access(f"doc_{i}")

        mock_redis.setex.assert_not_called()

        # 100th access should trigger save
        strategy.record_access("doc_99")
        mock_redis.setex.assert_called()


class TestIntegration:
    """Integration tests for tiered storage strategy"""

    def test_full_optimization_cycle(self):
        """Test complete optimization cycle"""
        with patch('src.core.standards.tiered_storage_strategy.get_redis_client', return_value=MagicMock()):
            strategy = TieredStorageStrategy(
                hot_cache_size=10,
                access_threshold=3
            )

            # Simulate document access patterns
            # High access documents
            for i in range(5):
                doc_id = f"high_{i}"
                for _ in range(10):
                    strategy.record_access(doc_id, "frequent_search")

            # Medium access documents
            for i in range(10):
                doc_id = f"medium_{i}"
                for _ in range(4):
                    strategy.record_access(doc_id, "occasional_search")

            # Low access documents
            for i in range(20):
                doc_id = f"low_{i}"
                strategy.record_access(doc_id, "rare_search")

            # Fill cache with some medium access docs
            for i in range(8):
                strategy._faiss_contents.add(f"medium_{i}")

            # Run optimization
            results = strategy.optimize_tier_placement()

            # High access docs should be promoted
            promoted_ids = [p["document_id"] for p in results["promotions"]]
            assert any("high_" in doc_id for doc_id in promoted_ids)

            # Get final stats
            stats = strategy.get_tier_stats()
            assert stats["hot_cache_items"] <= 10
            assert stats["documents_above_threshold"] >= 15  # high + medium docs

    def test_micro_standards_prioritization(self):
        """Test that micro standards are prioritized correctly"""
        with patch('src.core.standards.tiered_storage_strategy.get_redis_client', return_value=MagicMock()):
            strategy = TieredStorageStrategy(hot_cache_size=5)

            # Create mix of document types
            docs = [
                ("micro_1", "micro_standard", "normal", 1),
                ("micro_2", "micro_standard", "high", 0),
                ("standard_1", "standard", "normal", 10),
                ("standard_2", "standard", "critical", 0),
                ("template_1", "template", "normal", 5)
            ]

            decisions = []
            for doc_id, doc_type, priority, access_count in docs:
                # Record accesses
                for _ in range(access_count):
                    strategy.record_access(doc_id)

                # Get placement decision
                metadata = DocumentMetadata(
                    id=doc_id,
                    size_bytes=1024,
                    token_count=100,
                    document_type=doc_type,
                    priority=priority
                )
                decision = strategy.get_placement_decision(metadata)
                decisions.append((doc_id, decision))

            # Check micro standards are recommended for hot cache
            micro_decisions = [d for doc_id, d in decisions if "micro_" in doc_id]
            assert all(StorageTier.FAISS_HOT in d.recommended_tiers for d in micro_decisions)

            # Critical document should be in hot cache
            critical_decision = next(d for doc_id, d in decisions if doc_id == "standard_2")
            assert StorageTier.FAISS_HOT in critical_decision.recommended_tiers
