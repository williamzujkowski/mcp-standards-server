"""
Comprehensive tests for enhanced semantic search functionality.
"""

import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from src.core.standards.semantic_search import (
    AsyncSemanticSearch,
    EmbeddingCache,
    FuzzyMatcher,
    QueryPreprocessor,
    SearchQuery,
    SearchResult,
    SemanticSearch,
    create_search_engine,
)


class TestQueryPreprocessor:
    """Test query preprocessing functionality."""

    def setup_method(self):
        """Set up test instance."""
        self.preprocessor = QueryPreprocessor()

    def test_basic_preprocessing(self):
        """Test basic query preprocessing."""
        query = "React API testing standards"
        result = self.preprocessor.preprocess(query)

        assert isinstance(result, SearchQuery)
        assert result.original == query
        assert result.preprocessed == "react api testing standards"
        assert "react" in result.tokens
        assert "api" in result.tokens
        assert "testing" in result.tokens
        assert "standards" in result.tokens

    def test_stemming(self):
        """Test word stemming."""
        query = "testing tested tests"
        result = self.preprocessor.preprocess(query)

        # All should stem to 'test'
        assert len(set(result.stems)) == 1
        assert result.stems[0] == "test"

    def test_synonym_expansion(self):
        """Test synonym expansion."""
        query = "web security"
        result = self.preprocessor.preprocess(query)

        # Should expand 'web' and 'security'
        assert "website" in result.expanded_terms or "webapp" in result.expanded_terms
        assert "secure" in result.expanded_terms or "auth" in result.expanded_terms

    def test_boolean_operators(self):
        """Test boolean operator extraction."""
        query = "react AND testing NOT angular OR vue"
        result = self.preprocessor.preprocess(query)

        assert len(result.boolean_operators["AND"]) > 0
        assert len(result.boolean_operators["NOT"]) > 0
        assert len(result.boolean_operators["OR"]) > 0
        assert "angular" in result.boolean_operators["NOT"]

    def test_stopword_removal(self):
        """Test stopword removal."""
        query = "the testing of the react components"
        result = self.preprocessor.preprocess(query)

        assert "the" not in result.tokens
        assert "of" not in result.tokens
        assert "testing" in result.tokens
        assert "react" in result.tokens


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def setup_method(self):
        """Set up test instance with temporary cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EmbeddingCache(cache_dir=Path(self.temp_dir))

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir)

    def test_embedding_generation(self):
        """Test basic embedding generation."""
        text = "Test document for embedding"
        embedding = self.cache.get_embedding(text)

        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] > 0  # Has dimensions

    def test_embedding_cache_hit(self):
        """Test cache hit for same text."""
        text = "Test document for caching"

        # First call - cache miss
        start = time.time()
        embedding1 = self.cache.get_embedding(text)
        first_time = time.time() - start

        # Second call - cache hit
        start = time.time()
        embedding2 = self.cache.get_embedding(text)
        second_time = time.time() - start

        # Cache hit should be faster
        assert second_time < first_time / 2
        assert np.array_equal(embedding1, embedding2)

    def test_batch_embedding(self):
        """Test batch embedding generation."""
        texts = ["Document 1", "Document 2", "Document 3"]

        embeddings = self.cache.get_embeddings_batch(texts)

        assert embeddings.shape[0] == 3
        assert isinstance(embeddings, np.ndarray)

    def test_cache_persistence(self):
        """Test cache persistence across instances."""
        text = "Persistent cache test"

        # Generate and cache
        embedding1 = self.cache.get_embedding(text)

        # Create new cache instance with same directory
        new_cache = EmbeddingCache(cache_dir=Path(self.temp_dir))
        embedding2 = new_cache.get_embedding(text)

        assert np.array_equal(embedding1, embedding2)

    def test_cache_clear(self):
        """Test cache clearing."""
        text = "Clear cache test"

        # Generate and cache
        self.cache.get_embedding(text)

        # Clear cache
        self.cache.clear_cache()

        # Check that cache is empty
        cache_files = list(Path(self.temp_dir).glob("*.pkl"))
        assert len(cache_files) == 0


class TestFuzzyMatcher:
    """Test fuzzy matching functionality."""

    def setup_method(self):
        """Set up test instance."""
        self.matcher = FuzzyMatcher(threshold=80)
        self.matcher.add_known_terms(["react", "angular", "vue", "testing", "security"])

    def test_exact_match(self):
        """Test exact matching."""
        matches = self.matcher.find_matches("react")

        assert len(matches) > 0
        assert matches[0][0] == "react"
        assert matches[0][1] == 100

    def test_fuzzy_match(self):
        """Test fuzzy matching with typos."""
        matches = self.matcher.find_matches("reakt")  # Typo in 'react'

        assert len(matches) > 0
        assert matches[0][0] == "react"
        assert matches[0][1] >= 80

    def test_query_correction(self):
        """Test query correction."""
        query = "reakt and anglar testing"  # Typos
        corrected, corrections = self.matcher.correct_query(query)

        assert "react" in corrected
        assert "angular" in corrected
        assert len(corrections) >= 2

    def test_threshold_filtering(self):
        """Test threshold-based filtering."""
        # Very different word shouldn't match
        matches = self.matcher.find_matches("python")

        # Should have no high-scoring matches with the known terms
        high_scores = [m for m in matches if m[1] >= 80]
        assert len(high_scores) == 0 or high_scores[0][1] < 90


class TestSemanticSearch:
    """Test main semantic search functionality."""

    def setup_method(self):
        """Set up test instance."""
        self.temp_dir = tempfile.mkdtemp()
        self.search = SemanticSearch(
            cache_dir=Path(self.temp_dir), enable_analytics=True
        )

        # Index test documents
        self.test_docs = [
            (
                "doc1",
                "React component testing best practices",
                {"type": "testing", "framework": "react"},
            ),
            (
                "doc2",
                "Angular security standards and guidelines",
                {"type": "security", "framework": "angular"},
            ),
            (
                "doc3",
                "Vue.js performance optimization techniques",
                {"type": "performance", "framework": "vue"},
            ),
            (
                "doc4",
                "API design patterns for REST services",
                {"type": "api", "framework": "generic"},
            ),
            (
                "doc5",
                "Web accessibility WCAG compliance",
                {"type": "accessibility", "framework": "generic"},
            ),
        ]

        self.search.index_documents_batch(self.test_docs)

    def teardown_method(self):
        """Clean up."""
        self.search.close()
        shutil.rmtree(self.temp_dir)

    def test_basic_search(self):
        """Test basic semantic search."""
        results = self.search.search("React testing", top_k=3)

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0

        # Should find React testing document as top result
        assert "React" in results[0].content

    def test_synonym_search(self):
        """Test search with synonyms."""
        results = self.search.search("website security", top_k=3)

        # Should find security-related documents even with 'website' instead of 'web'
        assert len(results) > 0
        security_found = any("security" in r.content.lower() for r in results)
        assert security_found

    def test_fuzzy_search(self):
        """Test fuzzy search with typos."""
        results = self.search.search("Reakt testng", top_k=3, use_fuzzy=True)

        # Should still find React testing document
        assert len(results) > 0
        react_found = any("React" in r.content for r in results)
        assert react_found

    def test_boolean_operators(self):
        """Test boolean operator support."""
        # AND operator
        results = self.search.search("security AND angular", top_k=3)
        assert len(results) > 0
        # Top result should contain both terms
        top_content = results[0].content.lower()
        assert "security" in top_content and "angular" in top_content

        # NOT operator
        results = self.search.search("testing NOT angular", top_k=3)
        # Results should not contain Angular
        for result in results:
            assert "angular" not in result.content.lower()

    def test_metadata_filters(self):
        """Test metadata filtering."""
        filters = {"type": "testing"}
        results = self.search.search("best practices", top_k=5, filters=filters)

        # All results should be testing type
        for result in results:
            assert result.metadata.get("type") == "testing"

    def test_reranking(self):
        """Test result reranking."""
        results = self.search.search("testing", top_k=5, rerank=True)

        # Check that results have explanations
        assert all(r.explanation is not None for r in results)

        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_highlighting(self):
        """Test search result highlighting."""
        results = self.search.search("React component", top_k=3)

        # Should have highlights
        assert len(results[0].highlights) > 0

        # Highlights should contain search terms
        highlight_text = " ".join(results[0].highlights).lower()
        assert "react" in highlight_text or "component" in highlight_text

    def test_result_caching(self):
        """Test result caching."""
        query = "caching test query"

        # First search
        start = time.time()
        results1 = self.search.search(query, top_k=3)
        first_time = time.time() - start

        # Second search (should hit cache)
        start = time.time()
        results2 = self.search.search(query, top_k=3)
        second_time = time.time() - start

        # Cache hit should be faster
        assert second_time < first_time / 2

        # Results should be identical
        assert len(results1) == len(results2)
        assert all(r1.id == r2.id for r1, r2 in zip(results1, results2, strict=False))

    def test_analytics_tracking(self):
        """Test analytics tracking."""
        # Perform several searches
        self.search.search("test query 1")
        self.search.search("test query 2")
        self.search.search("test query 1")  # Repeat

        # Get analytics report
        report = self.search.get_analytics_report()

        assert report["total_queries"] == 3
        assert report["average_latency_ms"] > 0
        assert len(report["top_queries"]) > 0
        assert report["top_queries"][0][0] == "test query 1"  # Most popular
        assert report["top_queries"][0][1] == 2  # Count

    def test_click_tracking(self):
        """Test click-through tracking."""
        query = "test query"
        results = self.search.search(query)

        # Track clicks
        if results:
            self.search.track_click(query, results[0].id)
            self.search.track_click(query, results[0].id)

        # Check analytics
        if self.search.analytics is not None:
            assert query in self.search.analytics.click_through_data
            assert len(self.search.analytics.click_through_data[query]) == 2
        else:
            pytest.skip("Analytics not enabled")

    def test_batch_indexing_performance(self):
        """Test batch indexing performance."""
        # Generate many documents
        large_batch = [
            (f"doc_{i}", f"Test document {i} with content", {"index": i})
            for i in range(100)
        ]

        start = time.time()
        self.search.index_documents_batch(large_batch)
        batch_time = time.time() - start

        # Should complete reasonably fast (less than 10 seconds for 100 docs)
        assert batch_time < 10.0

        # All documents should be indexed
        assert len(self.search.documents) >= 100


class TestAsyncSemanticSearch:
    """Test async semantic search functionality."""

    @pytest.mark.asyncio
    async def test_async_search(self):
        """Test async search operation."""
        search = create_search_engine(async_mode=True)

        # Index some documents
        docs: list[tuple[str, str, dict[str, str]]] = [
            ("async1", "Async programming in Python", {}),
            ("async2", "JavaScript async await patterns", {}),
        ]
        if isinstance(search, AsyncSemanticSearch):
            await search.index_documents_batch_async(docs)

            # Perform async search
            results = await search.search_async("async programming")

            assert len(results) > 0
            assert "async" in results[0].content.lower()

            search.close()
        else:
            raise TypeError("Expected AsyncSemanticSearch instance for async test")

    @pytest.mark.asyncio
    async def test_concurrent_searches(self):
        """Test concurrent async searches."""
        search = create_search_engine(async_mode=True)

        # Index documents
        docs: list[tuple[str, str, dict[str, str]]] = [
            ("doc1", "Python programming", {}),
            ("doc2", "JavaScript development", {}),
            ("doc3", "Java enterprise", {}),
        ]
        if isinstance(search, AsyncSemanticSearch):
            await search.index_documents_batch_async(docs)

            # Perform multiple concurrent searches
            import asyncio

            queries = ["Python", "JavaScript", "Java"]
            tasks = [search.search_async(q) for q in queries]
            results = await asyncio.gather(*tasks)

            assert len(results) == 3
            assert all(len(r) > 0 for r in results)

            search.close()
        else:
            raise TypeError("Expected AsyncSemanticSearch instance for async test")


class TestSearchIntegration:
    """Integration tests for the complete search system."""

    def test_end_to_end_search_workflow(self):
        """Test complete search workflow."""
        # Create search engine
        search = create_search_engine(enable_analytics=True)

        # Index standards documents
        standards_docs = [
            (
                "std-001",
                """
            # React Component Standards

            ## Best Practices
            - Use functional components with hooks
            - Implement proper error boundaries
            - Follow accessibility guidelines

            ## Testing Requirements
            - Unit tests for all components
            - Integration tests for complex flows
            - Snapshot tests for UI consistency
            """,
                {"category": "frontend", "framework": "react", "version": "18"},
            ),
            (
                "std-002",
                """
            # API Security Standards

            ## Authentication
            - Use OAuth 2.0 or JWT tokens
            - Implement rate limiting
            - Validate all inputs

            ## HTTPS Requirements
            - TLS 1.3 minimum
            - Strong cipher suites only
            """,
                {"category": "security", "type": "api", "version": "2.0"},
            ),
            (
                "std-003",
                """
            # Python Testing Standards

            ## Framework
            - Use pytest for unit testing
            - Coverage minimum 80%
            - Mock external dependencies

            ## Best Practices
            - Test-driven development
            - Continuous integration
            """,
                {"category": "testing", "language": "python", "version": "3.0"},
            ),
        ]

        search.index_documents_batch(standards_docs)

        # Test various search scenarios

        # 1. Basic keyword search
        results = search.search("React testing")
        assert len(results) > 0
        assert "std-001" in [r.id for r in results[:2]]  # React standards should be top

        # 2. Fuzzy search with typos
        results = search.search("pythn testng", use_fuzzy=True)
        assert len(results) > 0
        assert any("python" in r.content.lower() for r in results)

        # 3. Boolean operators
        results = search.search("security AND api NOT react")
        assert len(results) > 0
        assert all("react" not in r.content.lower() for r in results)
        assert any(
            "security" in r.content.lower() and "api" in r.content.lower()
            for r in results
        )

        # 4. Filtered search
        results = search.search("standards", filters={"category": "frontend"})
        assert all(r.metadata.get("category") == "frontend" for r in results)

        # 5. Check analytics
        report = search.get_analytics_report()
        assert report["total_queries"] >= 4
        assert report["cache_hit_rate"] >= 0  # Some queries might hit cache

        if isinstance(search, SemanticSearch):
            search.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    def test_performance_under_load(self):
        """Test search performance with many documents."""
        search = create_search_engine()

        # Generate 1000 documents
        docs = []
        for i in range(1000):
            doc_id = f"perf-{i}"
            content = f"""
            Document {i} about {['Python', 'JavaScript', 'Java', 'Go', 'Rust'][i % 5]} programming.
            Topics include {['testing', 'security', 'performance', 'design', 'deployment'][i % 5]}.
            Framework: {['React', 'Angular', 'Vue', 'Django', 'Spring'][i % 5]}.
            """
            metadata = {
                "index": i,
                "language": ["Python", "JavaScript", "Java", "Go", "Rust"][i % 5],
                "topic": ["testing", "security", "performance", "design", "deployment"][
                    i % 5
                ],
            }
            docs.append((doc_id, content, metadata))

        # Time indexing
        start = time.time()
        search.index_documents_batch(docs)
        index_time = time.time() - start
        print(f"Indexed 1000 documents in {index_time:.2f} seconds")

        # Time searching
        queries = [
            "Python testing",
            "JavaScript security",
            "performance optimization",
            "React components",
            "deployment strategies",
        ]

        search_times = []
        for query in queries:
            start = time.time()
            results = search.search(query, top_k=10)
            search_time = time.time() - start
            search_times.append(search_time)
            assert len(results) > 0

        avg_search_time = sum(search_times) / len(search_times)
        print(f"Average search time: {avg_search_time*1000:.2f} ms")

        # Performance assertions
        assert index_time < 60  # Should index 1000 docs in under a minute
        assert avg_search_time < 0.5  # Searches should complete in under 500ms

        search.close()


def test_create_search_engine():
    """Test search engine factory function."""
    # Create sync engine
    sync_engine = create_search_engine()
    assert isinstance(sync_engine, SemanticSearch)
    sync_engine.close()

    # Create async engine
    async_engine = create_search_engine(async_mode=True)
    assert isinstance(async_engine, AsyncSemanticSearch)
    async_engine.close()

    # Create with custom settings
    temp_dir = tempfile.mkdtemp()
    custom_engine = create_search_engine(
        embedding_model="all-MiniLM-L6-v2",
        enable_analytics=False,
        cache_dir=Path(temp_dir),
    )
    assert custom_engine.analytics is None
    custom_engine.close()
    shutil.rmtree(temp_dir)
