"""
Integration tests for semantic search with real-world scenarios.

Tests the semantic search system in integration with other components
like the standards management system and MCP server.
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, Mock, MagicMock
import json
import time

from tests.mocks.semantic_search_mocks import (
    patch_ml_dependencies,
    TestDataGenerator,
    MockSentenceTransformer,
    MockRedisClient
)

from src.core.standards.semantic_search import (
    SemanticSearch,
    create_search_engine
)
from src.core.standards.models import Standard, StandardMetadata
from src.core.standards.engine import StandardsEngine
from src.core.mcp.server import MCPServer
from src.core.mcp.handlers import StandardsHandler


class TestSemanticSearchStandardsIntegration:
    """Test semantic search integration with standards management."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    @patch_ml_dependencies()
    def standards_engine(self, temp_dir):
        """Create standards engine with semantic search."""
        # Mock the standards engine components
        with patch('src.core.standards.engine.ChromaDBTier'):
            with patch('src.core.standards.engine.RedisCache'):
                engine = StandardsEngine(
                    data_dir=temp_dir / "standards",
                    enable_semantic_search=True
                )
                yield engine
    
    @patch_ml_dependencies()
    def test_standards_indexing_integration(self, standards_engine, temp_dir):
        """Test indexing standards for semantic search."""
        # Create test standards
        standards = [
            Standard(
                id="api-security-001",
                title="API Security Best Practices",
                category="security",
                subcategory="api",
                description="Comprehensive API security guidelines",
                content="""
                # API Security Standards
                
                Implement OAuth 2.0 for authentication.
                Use HTTPS for all communications.
                Validate all input parameters.
                Implement rate limiting.
                """,
                tags=["api", "security", "oauth", "https"],
                metadata=StandardMetadata(
                    version="2.0",
                    last_updated=datetime.now().isoformat(),
                    authors=["Security Team"],
                    source="internal"
                )
            ),
            Standard(
                id="react-testing-001",
                title="React Component Testing",
                category="testing",
                subcategory="frontend",
                description="Testing standards for React components",
                content="""
                # React Testing Standards
                
                Use Jest and React Testing Library.
                Test component behavior, not implementation.
                Achieve 80% code coverage.
                Mock external dependencies.
                """,
                tags=["react", "testing", "jest", "frontend"],
                metadata=StandardMetadata(
                    version="1.5",
                    last_updated=datetime.now().isoformat(),
                    authors=["Frontend Team"],
                    source="internal"
                )
            )
        ]
        
        # Index standards
        for standard in standards:
            # Simulate engine indexing
            if hasattr(standards_engine, 'semantic_search') and standards_engine.semantic_search:
                doc_content = f"{standard.title}\n{standard.description}\n{standard.content}"
                standards_engine.semantic_search.index_document(
                    doc_id=standard.id,
                    content=doc_content,
                    metadata={
                        "category": standard.category,
                        "subcategory": standard.subcategory,
                        "tags": standard.tags,
                        "version": standard.metadata.version
                    }
                )
        
        # Search for standards
        if hasattr(standards_engine, 'semantic_search') and standards_engine.semantic_search:
            results = standards_engine.semantic_search.search(
                "API authentication security",
                top_k=5
            )
            
            # Should find API security standard
            assert len(results) > 0
            assert any("api-security-001" in r.id for r in results)
    
    @patch_ml_dependencies()
    def test_standards_query_with_filters(self, standards_engine):
        """Test semantic search with standards-specific filters."""
        # Create mock semantic search
        mock_search = SemanticSearch()
        
        # Index test documents
        docs = TestDataGenerator.generate_standards_corpus(50)
        mock_search.index_documents_batch(docs)
        
        # Attach to engine
        standards_engine.semantic_search = mock_search
        
        # Search with category filter
        results = mock_search.search(
            "testing best practices",
            filters={"category": "testing"},
            top_k=10
        )
        
        # Verify filtering
        assert all(r.metadata.get("category") == "testing" for r in results)
        
        # Search with multiple filters
        results = mock_search.search(
            "programming standards",
            filters={
                "category": "security",
                "language": ["python", "javascript"]
            },
            top_k=10
        )
        
        # Verify all filters applied
        for result in results:
            assert result.metadata.get("category") == "security"
            assert result.metadata.get("language") in ["python", "javascript"]
    
    @patch_ml_dependencies()
    def test_incremental_indexing(self, standards_engine):
        """Test incremental indexing of new standards."""
        mock_search = SemanticSearch()
        standards_engine.semantic_search = mock_search
        
        # Initial indexing
        initial_docs = [
            ("std-001", "Python coding standards", {"category": "coding"}),
            ("std-002", "API design guidelines", {"category": "api"})
        ]
        
        for doc_id, content, metadata in initial_docs:
            mock_search.index_document(doc_id, content, metadata)
        
        # Verify initial search
        results = mock_search.search("standards")
        initial_count = len(results)
        assert initial_count == 2
        
        # Add new standards incrementally
        new_docs = [
            ("std-003", "Security best practices", {"category": "security"}),
            ("std-004", "Testing methodologies", {"category": "testing"})
        ]
        
        for doc_id, content, metadata in new_docs:
            mock_search.index_document(doc_id, content, metadata)
        
        # Verify incremental indexing
        results = mock_search.search("standards")
        assert len(results) == initial_count + 2
        
        # Verify new documents are searchable
        security_results = mock_search.search("security")
        assert any("std-003" in r.id for r in security_results)


class TestSemanticSearchMCPIntegration:
    """Test semantic search integration with MCP server."""
    
    @pytest.fixture
    @patch_ml_dependencies()
    async def mcp_server(self):
        """Create MCP server with semantic search enabled."""
        with patch('src.core.mcp.server.StandardsEngine') as mock_engine:
            # Create mock semantic search
            mock_search = SemanticSearch()
            
            # Index test documents
            docs = TestDataGenerator.generate_standards_corpus(30)
            mock_search.index_documents_batch(docs)
            
            # Configure mock engine
            mock_engine.return_value.semantic_search = mock_search
            
            server = MCPServer(
                name="test-mcp-server",
                version="1.0.0"
            )
            
            # Attach handlers
            handler = StandardsHandler(mock_engine.return_value)
            server.add_handler(handler)
            
            yield server
    
    @pytest.mark.asyncio
    async def test_mcp_search_tool(self, mcp_server):
        """Test MCP search tool integration."""
        # Create mock request for semantic search
        search_request = {
            "method": "tools/call",
            "params": {
                "name": "search_standards",
                "arguments": {
                    "query": "React security best practices",
                    "top_k": 5,
                    "use_fuzzy": True
                }
            }
        }
        
        # Mock the tool execution
        with patch.object(mcp_server, 'handle_request') as mock_handle:
            # Simulate search results
            mock_results = [
                {
                    "id": "std-001",
                    "title": "React Security Guidelines",
                    "score": 0.95,
                    "highlights": ["React security best practices"]
                }
            ]
            mock_handle.return_value = {
                "results": mock_results,
                "total": 1
            }
            
            # Execute search
            response = await mock_handle(search_request)
            
            # Verify response
            assert "results" in response
            assert len(response["results"]) > 0
            assert response["results"][0]["score"] > 0
    
    @pytest.mark.asyncio
    async def test_mcp_filtered_search(self, mcp_server):
        """Test MCP search with filters."""
        search_request = {
            "method": "tools/call",
            "params": {
                "name": "search_standards",
                "arguments": {
                    "query": "testing standards",
                    "filters": {
                        "category": "testing",
                        "framework": ["react", "vue"]
                    },
                    "top_k": 10
                }
            }
        }
        
        with patch.object(mcp_server, 'handle_request') as mock_handle:
            mock_handle.return_value = {
                "results": [],
                "total": 0
            }
            
            response = await mock_handle(search_request)
            
            # Verify filter parameters were passed
            call_args = mock_handle.call_args[0][0]
            assert call_args["params"]["arguments"]["filters"] == {
                "category": "testing",
                "framework": ["react", "vue"]
            }
    
    @pytest.mark.asyncio
    async def test_mcp_search_analytics(self, mcp_server):
        """Test MCP search analytics tracking."""
        # Perform multiple searches
        queries = [
            "python security",
            "react testing",
            "api design",
            "python security"  # Duplicate
        ]
        
        with patch.object(mcp_server, 'handle_request') as mock_handle:
            mock_handle.return_value = {"results": [], "total": 0}
            
            for query in queries:
                request = {
                    "method": "tools/call",
                    "params": {
                        "name": "search_standards",
                        "arguments": {"query": query}
                    }
                }
                await mock_handle(request)
        
        # Request analytics
        analytics_request = {
            "method": "tools/call",
            "params": {
                "name": "get_search_analytics",
                "arguments": {}
            }
        }
        
        with patch.object(mcp_server, 'handle_request') as mock_handle:
            mock_handle.return_value = {
                "total_queries": 4,
                "unique_queries": 3,
                "top_queries": [("python security", 2)]
            }
            
            response = await mock_handle(analytics_request)
            
            assert response["total_queries"] == 4
            assert response["unique_queries"] == 3


class TestSemanticSearchCrossComponent:
    """Test semantic search across multiple components."""
    
    @patch_ml_dependencies()
    def test_search_with_redis_caching(self):
        """Test semantic search with Redis caching integration."""
        # Create search engine with Redis
        temp_dir = tempfile.mkdtemp()
        
        try:
            with patch('redis.Redis', MockRedisClient):
                engine = create_search_engine(
                    cache_dir=Path(temp_dir),
                    enable_analytics=True
                )
                
                # Index documents
                docs = TestDataGenerator.generate_standards_corpus(20)
                engine.index_documents_batch(docs)
                
                # First search - populate caches
                results1 = engine.search("python testing", top_k=5)
                
                # Verify Redis cache was used
                cache_key = engine._get_result_cache_key("python testing", 5, None)
                
                # Second search - should hit cache
                results2 = engine.search("python testing", top_k=5)
                
                # Results should be identical
                assert len(results1) == len(results2)
                for r1, r2 in zip(results1, results2):
                    assert r1.id == r2.id
                
                # Check analytics
                report = engine.get_analytics_report()
                assert report["cache_hit_rate"] > 0
                
                engine.close()
        finally:
            shutil.rmtree(temp_dir)
    
    @patch_ml_dependencies()
    def test_search_with_versioning(self):
        """Test semantic search with document versioning."""
        engine = create_search_engine()
        
        # Index versioned documents
        versions = ["1.0", "1.1", "2.0"]
        for version in versions:
            doc_id = f"api-standard-v{version}"
            content = f"API Standard Version {version} - Updated security guidelines"
            metadata = {
                "version": version,
                "category": "api",
                "timestamp": datetime.now().isoformat()
            }
            engine.index_document(doc_id, content, metadata)
        
        # Search for latest version
        results = engine.search("API security", top_k=10)
        
        # Should find all versions
        version_ids = [r.id for r in results]
        assert any("v2.0" in vid for vid in version_ids)
        assert any("v1.1" in vid for vid in version_ids)
        assert any("v1.0" in vid for vid in version_ids)
        
        # Filter by version
        results = engine.search(
            "API security",
            filters={"version": "2.0"},
            top_k=10
        )
        
        # Should only find v2.0
        assert all("v2.0" in r.id for r in results)
        
        engine.close()
    
    @pytest.mark.asyncio
    @patch_ml_dependencies()
    async def test_concurrent_component_access(self):
        """Test concurrent access from multiple components."""
        engine = create_search_engine(async_mode=True)
        
        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(50)
        await engine.index_documents_batch_async(docs)
        
        # Simulate concurrent access from different components
        async def standards_component():
            """Simulate standards engine queries."""
            for i in range(5):
                results = await engine.search_async(f"standard {i}", top_k=5)
                assert len(results) >= 0
        
        async def mcp_component():
            """Simulate MCP server queries."""
            for i in range(5):
                results = await engine.search_async(f"api {i}", top_k=5)
                assert len(results) >= 0
        
        async def analytics_component():
            """Simulate analytics queries."""
            for i in range(5):
                results = await engine.search_async(f"security {i}", top_k=5)
                assert len(results) >= 0
        
        # Run all components concurrently
        await asyncio.gather(
            standards_component(),
            mcp_component(),
            analytics_component()
        )
        
        # Verify no errors and get final analytics
        report = engine.search_engine.get_analytics_report()
        assert report["total_queries"] == 15
        assert report["failed_queries_count"] == 0
        
        engine.close()


class TestSemanticSearchErrorRecovery:
    """Test error recovery and resilience in integration scenarios."""
    
    @patch_ml_dependencies()
    def test_model_loading_failure_recovery(self):
        """Test recovery from model loading failures."""
        fail_count = 0
        
        class FailingThenSucceedingModel(MockSentenceTransformer):
            def __init__(self, *args, **kwargs):
                nonlocal fail_count
                if fail_count < 2:
                    fail_count += 1
                    raise Exception("Model loading failed")
                super().__init__(*args, **kwargs)
        
        with patch('sentence_transformers.SentenceTransformer', FailingThenSucceedingModel):
            # First two attempts should fail
            for i in range(2):
                with pytest.raises(Exception):
                    create_search_engine()
            
            # Third attempt should succeed
            engine = create_search_engine()
            assert engine is not None
            engine.close()
    
    @patch_ml_dependencies()
    def test_partial_index_failure_recovery(self):
        """Test recovery from partial indexing failures."""
        engine = create_search_engine()
        
        # Create documents where some will fail
        docs = []
        for i in range(10):
            if i == 5:
                # This will cause an error
                docs.append((f"doc-{i}", None, {}))  # None content
            else:
                docs.append((f"doc-{i}", f"Content {i}", {"index": i}))
        
        # Index with error handling
        success_count = 0
        for doc_id, content, metadata in docs:
            try:
                if content is None:
                    raise ValueError("Invalid content")
                engine.index_document(doc_id, content, metadata)
                success_count += 1
            except Exception:
                pass
        
        # Should have indexed 9 out of 10
        assert success_count == 9
        
        # Search should still work
        results = engine.search("Content")
        assert len(results) > 0
        assert all("doc-5" not in r.id for r in results)  # Failed doc not in results
        
        engine.close()
    
    @patch_ml_dependencies()
    def test_cache_corruption_recovery(self, temp_dir):
        """Test recovery from cache corruption."""
        engine = create_search_engine(cache_dir=temp_dir)
        
        # Index and search to populate cache
        engine.index_document("test-1", "Test content", {})
        results1 = engine.search("test")
        
        # Corrupt the cache file
        cache_files = list(temp_dir.glob("*.npy"))
        if cache_files:
            # Write invalid data
            with open(cache_files[0], 'wb') as f:
                f.write(b"CORRUPTED DATA")
        
        # Clear memory cache to force file read
        engine.embedding_cache.memory_cache.clear()
        
        # Should handle corruption gracefully
        results2 = engine.search("test")
        assert len(results2) > 0  # Should still return results
        
        engine.close()


class TestSemanticSearchPerformanceIntegration:
    """Integration tests focused on performance."""
    
    @patch_ml_dependencies()
    def test_large_scale_integration_performance(self):
        """Test performance with large-scale integration."""
        engine = create_search_engine()
        
        # Generate large corpus
        print("\nGenerating large corpus...")
        docs = TestDataGenerator.generate_standards_corpus(5000)
        
        # Benchmark batch indexing
        print("Indexing documents...")
        start = time.time()
        engine.index_documents_batch(docs)
        index_time = time.time() - start
        print(f"Indexed 5000 documents in {index_time:.2f}s")
        print(f"Rate: {5000/index_time:.2f} docs/second")
        
        # Benchmark various search patterns
        search_patterns = [
            ("Simple keyword", "security"),
            ("Multiple keywords", "python api testing"),
            ("With synonyms", "web application security"),
            ("Boolean AND", "testing AND python"),
            ("Boolean NOT", "security NOT java"),
            ("Fuzzy search", "pythn secruity"),
            ("Filtered search", "standards", {"category": "testing"}),
        ]
        
        print("\nBenchmarking search patterns:")
        for pattern_name, query, *args in search_patterns:
            filters = args[0] if args else None
            
            # Warm up cache
            engine.search(query, filters=filters)
            
            # Benchmark
            times = []
            for _ in range(5):
                start = time.time()
                results = engine.search(query, filters=filters, top_k=20)
                elapsed = time.time() - start
                times.append(elapsed)
            
            avg_time = sum(times) / len(times)
            print(f"{pattern_name}: {avg_time*1000:.2f}ms avg, {len(results)} results")
        
        # Check memory usage
        report = engine.get_analytics_report()
        print(f"\nTotal queries: {report['total_queries']}")
        print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")
        
        engine.close()