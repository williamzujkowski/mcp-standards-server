"""
Comprehensive tests for performance optimizations.

This module tests all the performance optimization components:
- Async semantic search
- Redis connection pooling
- Vector index caching
- Memory management
- Performance monitoring
- Database query optimization
- Async MCP server
"""

import asyncio
import time

import numpy as np
import pytest

from src.core.cache.redis_client import CacheConfig, RedisCache
from src.core.database.query_optimizer import DatabaseOptimizer, QueryCacheConfig
from src.core.mcp.async_server import AsyncMCPServer, ServerConfig
from src.core.performance.memory_manager import (
    MemoryConfig,
    MemoryEfficientDict,
    MemoryEfficientList,
)
from src.core.performance.memory_manager import MemoryManager as MemoryMgr
from src.core.performance.metrics import (
    PerformanceConfig,
    PerformanceMonitor,
)
from src.core.standards.async_semantic_search import (
    AsyncSearchConfig,
    AsyncSemanticSearch,
)
from src.core.standards.vector_index_cache import VectorIndexCache as VectorCache
from src.core.standards.vector_index_cache import VectorIndexConfig


class TestAsyncSemanticSearch:
    """Test async semantic search optimizations."""

    @pytest.fixture
    async def search_engine(self):
        """Create async search engine for testing."""
        config = AsyncSearchConfig(
            batch_size=2,
            max_batch_wait_time=0.01,
            enable_vector_cache=True,
            enable_cache_warming=False,  # Disable for testing
        )

        engine = AsyncSemanticSearch(config)
        await engine.initialize()
        yield engine
        await engine.close()

    @pytest.mark.asyncio
    async def test_async_initialization(self, search_engine):
        """Test async search engine initialization."""
        assert search_engine.initialized
        assert search_engine.batch_processor is not None
        assert search_engine.vector_cache is not None
        assert search_engine.memory_manager is not None

    @pytest.mark.asyncio
    async def test_document_indexing(self, search_engine):
        """Test document indexing performance."""
        # Index single document
        await search_engine.index_document(
            "doc1", "This is a test document about machine learning", {"category": "ai"}
        )

        assert "doc1" in search_engine.documents
        assert "doc1" in search_engine.document_embeddings
        assert search_engine.document_metadata["doc1"]["category"] == "ai"

    @pytest.mark.asyncio
    async def test_batch_document_indexing(self, search_engine):
        """Test batch document indexing performance."""
        documents = [
            ("doc1", "Machine learning basics", {"category": "ai"}),
            ("doc2", "Deep learning fundamentals", {"category": "ai"}),
            ("doc3", "Python programming guide", {"category": "programming"}),
        ]

        start_time = time.time()
        await search_engine.index_documents_batch(documents)
        indexing_time = time.time() - start_time

        assert len(search_engine.documents) == 3
        assert indexing_time < 1.0  # Should be fast

    @pytest.mark.asyncio
    async def test_search_performance(self, search_engine):
        """Test search performance."""
        # Index some documents
        documents = [
            (
                "doc1",
                "Machine learning and artificial intelligence",
                {"category": "ai"},
            ),
            ("doc2", "Python programming tutorials", {"category": "programming"}),
            ("doc3", "Data science fundamentals", {"category": "data"}),
        ]
        await search_engine.index_documents_batch(documents)

        # Perform search
        start_time = time.time()
        results = await search_engine.search("machine learning", top_k=5)
        search_time = time.time() - start_time

        assert len(results) > 0
        assert search_time < 1.0
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.asyncio
    async def test_batch_search(self, search_engine):
        """Test batch search performance."""
        # Index documents
        documents = [
            ("doc1", "Machine learning basics", {"category": "ai"}),
            ("doc2", "Python programming", {"category": "programming"}),
            ("doc3", "Data analysis", {"category": "data"}),
        ]
        await search_engine.index_documents_batch(documents)

        # Perform batch search
        queries = ["machine learning", "python", "data"]
        start_time = time.time()
        results = await search_engine.search_batch(queries)
        batch_time = time.time() - start_time

        assert len(results) == 3
        assert batch_time < 2.0
        assert all(isinstance(r, list) for r in results)

    @pytest.mark.asyncio
    async def test_performance_metrics(self, search_engine):
        """Test performance metrics collection."""
        # Index and search
        await search_engine.index_document("doc1", "test content")
        await search_engine.search("test")

        # Get metrics
        metrics = await search_engine.get_performance_metrics()

        assert "search_engine" in metrics
        assert "memory" in metrics
        assert "vector_cache" in metrics
        assert metrics["search_engine"]["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_health_check(self, search_engine):
        """Test health check functionality."""
        health = await search_engine.health_check()

        assert "status" in health
        assert "initialized" in health
        assert "components" in health
        assert health["initialized"] is True


class TestRedisConnectionPooling:
    """Test Redis connection pooling optimizations."""

    @pytest.fixture
    def redis_config(self):
        """Create Redis configuration for testing."""
        return CacheConfig(
            host="localhost",
            port=6379,
            connection_pool_size=5,
            max_connections=10,
            health_check_interval=1.0,
            enable_metrics=True,
        )

    @pytest.fixture
    def redis_cache(self, redis_config):
        """Create Redis cache for testing."""
        cache = RedisCache(redis_config)
        yield cache
        # Cleanup in case Redis is available
        try:
            cache.clear_l1_cache()
        except Exception:
            pass

    def test_redis_initialization(self, redis_cache):
        """Test Redis cache initialization."""
        assert redis_cache.config is not None
        assert redis_cache._l1_cache is not None
        assert redis_cache._circuit_breaker is not None

    def test_cache_operations(self, redis_cache):
        """Test basic cache operations."""
        # Test set and get
        key = "test_key"
        value = {"test": "data", "number": 42}

        # This will work with L1 cache even without Redis
        result = redis_cache.set(key, value)
        assert result is not None  # May be False if Redis unavailable

        retrieved = redis_cache.get(key)
        assert retrieved == value

    def test_batch_operations(self, redis_cache):
        """Test batch cache operations."""
        data = {"key1": "value1", "key2": {"nested": "data"}, "key3": [1, 2, 3]}

        # Test mset
        redis_cache.mset(data)

        # Test mget
        results = redis_cache.mget(list(data.keys()))

        assert len(results) == 3
        for key, expected_value in data.items():
            assert results[key] == expected_value

    def test_health_check(self, redis_cache):
        """Test Redis health check."""
        health = redis_cache.health_check()

        assert "status" in health
        assert "l1_cache_size" in health
        assert "circuit_breaker_state" in health
        assert "metrics" in health
        assert "connection_health" in health

    def test_metrics_collection(self, redis_cache):
        """Test metrics collection."""
        # Perform some operations
        redis_cache.set("test1", "value1")
        redis_cache.get("test1")
        redis_cache.get("nonexistent")

        metrics = redis_cache.get_metrics()

        assert "l1_hits" in metrics
        assert "l1_misses" in metrics
        assert "l1_hit_rate" in metrics
        assert metrics["l1_hits"] > 0

    def test_pipeline_operations(self, redis_cache):
        """Test pipeline operations."""
        operations = [
            {"method": "set", "args": ["key1", "value1"]},
            {"method": "set", "args": ["key2", "value2"]},
            {"method": "get", "args": ["key1"]},
            {"method": "get", "args": ["key2"]},
        ]

        results = redis_cache.execute_pipeline(operations)

        assert len(results) == 4
        assert results[2] == "value1"
        assert results[3] == "value2"


class TestVectorIndexCaching:
    """Test vector index caching optimizations."""

    @pytest.fixture
    def vector_cache_config(self):
        """Create vector cache configuration."""
        return VectorIndexConfig(
            memory_cache_size=100,
            memory_cache_ttl=300,
            enable_compression=True,
            warming_batch_size=10,
            warming_strategies=[],  # Disable warming strategies for testing
        )

    @pytest.fixture
    async def vector_cache(self, vector_cache_config):
        """Create vector cache for testing."""
        cache = VectorCache(vector_cache_config)
        await cache.start()
        yield cache
        await cache.stop()

    @pytest.mark.asyncio
    async def test_vector_index_building(self, vector_cache):
        """Test vector index building."""
        # Create test vectors
        vectors = np.random.rand(100, 384).astype(np.float32)

        # Build and cache index
        index = await vector_cache.build_and_cache_index(
            "test_index", vectors, index_type="Flat"
        )

        assert index is not None
        assert index.ntotal == 100
        assert index.d == 384

    @pytest.mark.asyncio
    async def test_index_caching(self, vector_cache):
        """Test index caching and retrieval."""
        # Create and cache index
        vectors = np.random.rand(50, 384).astype(np.float32)
        await vector_cache.build_and_cache_index("cached_index", vectors)

        # Retrieve from cache
        retrieved_index = await vector_cache.get_index("cached_index")

        assert retrieved_index is not None
        assert retrieved_index.ntotal == 50
        assert retrieved_index.d == 384

    @pytest.mark.asyncio
    async def test_search_performance(self, vector_cache):
        """Test search performance with caching."""
        # Create and cache index
        vectors = np.random.rand(1000, 384).astype(np.float32)
        await vector_cache.build_and_cache_index("search_index", vectors)

        # Perform search
        query_vectors = np.random.rand(5, 384).astype(np.float32)

        start_time = time.time()
        distances, indices = await vector_cache.search_index(
            "search_index", query_vectors, k=10
        )
        search_time = time.time() - start_time

        assert distances.shape == (5, 10)
        assert indices.shape == (5, 10)
        assert search_time < 1.0

    @pytest.mark.asyncio
    async def test_cache_metrics(self, vector_cache):
        """Test cache metrics collection."""
        # Perform operations
        vectors = np.random.rand(100, 384).astype(np.float32)
        await vector_cache.build_and_cache_index("metrics_index", vectors)
        await vector_cache.get_index("metrics_index")
        await vector_cache.get_index("nonexistent")

        # Get metrics
        metrics = vector_cache.get_metrics()

        assert "cache_metrics" in metrics
        assert "memory_cache_size" in metrics
        assert metrics["cache_metrics"]["cache_hits"] > 0

    @pytest.mark.asyncio
    async def test_invalidation(self, vector_cache):
        """Test cache invalidation."""
        # Create and cache index
        vectors = np.random.rand(50, 384).astype(np.float32)
        await vector_cache.build_and_cache_index("invalidate_index", vectors)

        # Verify it's cached
        index = await vector_cache.get_index("invalidate_index")
        assert index is not None

        # Invalidate
        await vector_cache.invalidate_index("invalidate_index")

        # Verify it's gone
        index = await vector_cache.get_index("invalidate_index")
        assert index is None


class TestMemoryManagement:
    """Test memory management optimizations."""

    @pytest.fixture
    def memory_config(self):
        """Create memory configuration for testing."""
        return MemoryConfig(
            max_memory_usage=512,  # 512MB
            monitoring_interval=0.1,
            enable_object_pooling=True,
            enable_gc_optimization=True,
        )

    @pytest.fixture
    async def memory_manager(self, memory_config):
        """Create memory manager for testing."""
        manager = MemoryMgr(memory_config)
        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_memory_tracking(self, memory_manager):
        """Test memory tracking functionality."""
        # Track some objects
        test_objects = [{"data": f"test_{i}"} for i in range(10)]

        for i, obj in enumerate(test_objects):
            memory_manager.track_object(obj, f"test_category_{i % 3}")

        # Get stats
        stats = memory_manager.get_memory_stats()

        assert "memory_stats" in stats
        assert "object_summary" in stats
        assert stats["object_summary"]["total_tracked"] == 10

    def test_efficient_dict(self):
        """Test memory-efficient dictionary."""
        efficient_dict = MemoryEfficientDict(initial_size=5)

        # Add items
        for i in range(10):
            efficient_dict[f"key_{i}"] = f"value_{i}"

        # Should have triggered eviction
        assert len(efficient_dict) <= 5

        # Get stats
        stats = efficient_dict.get_stats()
        assert stats["size"] <= 5
        assert stats["eviction_count"] > 0

    def test_efficient_list(self):
        """Test memory-efficient list."""
        efficient_list = MemoryEfficientList(max_size=5)

        # Add items
        for i in range(10):
            efficient_list.append(f"item_{i}")

        # Should have triggered eviction
        assert len(efficient_list) <= 5

        # Get stats
        stats = efficient_list.get_stats()
        assert stats["size"] <= 5
        assert stats["eviction_count"] > 0

    @pytest.mark.asyncio
    async def test_memory_context(self, memory_manager):
        """Test memory context manager."""
        with memory_manager.memory_context("test_operation"):
            # Simulate some work
            data = list(range(1000))
            del data

        # Context should complete without errors
        assert True

    @pytest.mark.asyncio
    async def test_object_pooling(self, memory_manager):
        """Test object pooling."""
        # Get objects from pool
        obj1 = memory_manager.get_object_from_pool("list")
        obj2 = memory_manager.get_object_from_pool("dict")

        assert obj1 is not None
        assert obj2 is not None
        assert isinstance(obj1, list)
        assert isinstance(obj2, dict)

        # Return to pool
        memory_manager.return_object_to_pool("list", obj1)
        memory_manager.return_object_to_pool("dict", obj2)

        # Get stats
        stats = memory_manager.get_memory_stats()
        assert "pool_stats" in stats


class TestPerformanceMonitoring:
    """Test performance monitoring system."""

    @pytest.fixture
    def perf_config(self):
        """Create performance monitoring configuration."""
        return PerformanceConfig(
            collection_interval=0.1,
            enable_system_metrics=True,
            enable_application_metrics=True,
            enable_prometheus=False,  # Disable for testing
        )

    @pytest.fixture
    async def performance_monitor(self, perf_config):
        """Create performance monitor for testing."""
        monitor = PerformanceMonitor(perf_config)
        await monitor.start()
        yield monitor
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_metric_recording(self, performance_monitor):
        """Test metric recording functionality."""
        # Record some metrics
        performance_monitor.record_metric("test_counter", 1.0)
        performance_monitor.record_metric("test_gauge", 42.0)
        performance_monitor.record_metric("test_histogram", 0.5)

        # Get metrics summary
        summary = performance_monitor.get_metrics_summary()

        assert "test_counter" in summary
        assert "test_gauge" in summary
        assert "test_histogram" in summary

    @pytest.mark.asyncio
    async def test_timing_context(self, performance_monitor):
        """Test timing context manager."""
        with performance_monitor.time_operation("test_operation"):
            time.sleep(0.01)  # Simulate work

        # Check if metric was recorded
        summary = performance_monitor.get_metrics_summary()
        assert "test_operation" in summary

    @pytest.mark.asyncio
    async def test_benchmarking(self, performance_monitor):
        """Test performance benchmarking."""

        def test_operation():
            time.sleep(0.001)  # Simulate work
            return "result"

        benchmark = performance_monitor.create_benchmark("test_bench", test_operation)
        results = benchmark.run(iterations=10)

        assert "name" in results
        assert "iterations" in results
        assert "success_rate" in results
        assert results["iterations"] == 10
        assert results["success_rate"] > 0

    @pytest.mark.asyncio
    async def test_dashboard_data(self, performance_monitor):
        """Test dashboard data generation."""
        # Record some metrics
        performance_monitor.record_metric("system_cpu_percent", 50.0)
        performance_monitor.record_metric("app_request_count", 100)

        # Get dashboard data
        dashboard = performance_monitor.get_dashboard_data()

        assert "timestamp" in dashboard
        assert "system_metrics" in dashboard
        assert "application_metrics" in dashboard
        assert "alerts" in dashboard


class TestDatabaseOptimization:
    """Test database query optimization."""

    @pytest.fixture
    def db_config(self):
        """Create database configuration for testing."""
        return QueryCacheConfig(
            enable_query_cache=True,
            enable_query_batching=True,
            batch_size=5,
            batch_timeout=0.01,
        )

    @pytest.fixture
    async def db_optimizer(self, db_config):
        """Create database optimizer for testing."""
        optimizer = DatabaseOptimizer(db_config)
        await optimizer.initialize()
        yield optimizer
        await optimizer.close()

    @pytest.mark.asyncio
    async def test_query_execution(self, db_optimizer):
        """Test query execution with caching."""
        query = "SELECT * FROM users WHERE age > 18"
        params = {"min_age": 18}

        # Execute query
        result = await db_optimizer.execute_query(query, params)

        assert result is not None
        assert "query" in result
        assert "mock_data" in result

    @pytest.mark.asyncio
    async def test_query_caching(self, db_optimizer):
        """Test query result caching."""
        query = "SELECT * FROM products WHERE price > 100"
        params = {"min_price": 100}

        # Execute query twice
        result1 = await db_optimizer.execute_query(query, params)
        result2 = await db_optimizer.execute_query(query, params)

        # Should get same result
        assert result1 == result2

        # Check cache metrics
        stats = db_optimizer.get_performance_stats()
        assert stats["cache_stats"]["hits"] > 0

    @pytest.mark.asyncio
    async def test_batch_execution(self, db_optimizer):
        """Test batch query execution."""
        queries: list[tuple[str, dict[str, str]]] = [
            ("SELECT * FROM users", {}),
            ("SELECT * FROM products", {}),
            ("SELECT * FROM orders", {}),
        ]

        results = await db_optimizer.execute_batch(queries)

        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_performance_stats(self, db_optimizer):
        """Test performance statistics collection."""
        # Execute some queries
        await db_optimizer.execute_query("SELECT * FROM test1")
        await db_optimizer.execute_query("SELECT * FROM test2")

        # Get stats
        stats = db_optimizer.get_performance_stats()

        assert "metrics" in stats
        assert "cache_stats" in stats
        assert stats["metrics"]["total_queries"] > 0

    @pytest.mark.asyncio
    async def test_health_check(self, db_optimizer):
        """Test database health check."""
        health = await db_optimizer.health_check()

        assert "status" in health
        assert "cache_status" in health
        assert "metrics" in health


class TestAsyncMCPServer:
    """Test async MCP server performance."""

    @pytest.fixture
    def server_config(self):
        """Create server configuration for testing."""
        return ServerConfig(
            host="127.0.0.1",
            port=8081,  # Use different port for testing
            max_connections=10,
            enable_request_batching=True,
            batch_size=5,
            batch_timeout=0.01,
            enable_metrics=True,
        )

    @pytest.fixture
    async def mcp_server(self, server_config):
        """Create MCP server for testing."""
        server = AsyncMCPServer(server_config)
        await server.start()
        yield server
        await server.stop()

    @pytest.mark.asyncio
    async def test_server_initialization(self, mcp_server):
        """Test server initialization."""
        assert mcp_server.running
        assert mcp_server.connection_manager is not None
        assert mcp_server.request_batcher is not None

    @pytest.mark.asyncio
    async def test_request_processing(self, mcp_server):
        """Test request processing."""
        request = {"method": "list_tools", "params": {}}

        result = await mcp_server._process_request(request, "test_connection")

        assert result is not None
        assert "tools" in result

    @pytest.mark.asyncio
    async def test_health_check(self, mcp_server):
        """Test server health check."""
        health = await mcp_server.get_health_status()

        assert "status" in health
        assert "connections" in health
        assert "requests" in health
        assert "memory" in health

    @pytest.mark.asyncio
    async def test_performance_stats(self, mcp_server):
        """Test performance statistics collection."""
        # Process some requests
        request = {"method": "list_tools", "params": {}}
        await mcp_server._process_request(request, "test_connection_1")
        await mcp_server._process_request(request, "test_connection_2")

        # Get stats
        stats = mcp_server.get_performance_stats()

        assert "metrics" in stats
        assert "connections" in stats
        assert stats["metrics"]["total_requests"] > 0

    @pytest.mark.asyncio
    async def test_connection_management(self, mcp_server):
        """Test connection management."""
        # Add connections
        await mcp_server.connection_manager.add_connection(
            "conn1", "127.0.0.1", "test-agent"
        )
        await mcp_server.connection_manager.add_connection(
            "conn2", "127.0.0.1", "test-agent"
        )

        # Check connections
        connections = mcp_server.get_active_connections()
        assert len(connections) == 2

        # Remove connection
        await mcp_server.connection_manager.remove_connection("conn1")

        connections = mcp_server.get_active_connections()
        assert len(connections) == 1


class TestIntegrationPerformance:
    """Integration tests for performance optimizations."""

    @pytest.mark.asyncio
    async def test_end_to_end_performance(self):
        """Test end-to-end performance with all optimizations."""
        # Create integrated system
        search_config = AsyncSearchConfig(batch_size=10, enable_cache_warming=False)
        cache_config = CacheConfig(connection_pool_size=5)
        server_config = ServerConfig(port=8082, enable_request_batching=True)

        # Initialize components
        search_engine = AsyncSemanticSearch(search_config)
        RedisCache(cache_config)
        mcp_server = AsyncMCPServer(server_config)

        try:
            # Start all components
            await search_engine.initialize()
            await mcp_server.start()

            # Test integrated functionality
            documents = [
                ("doc1", "Machine learning fundamentals", {"category": "ai"}),
                ("doc2", "Python programming guide", {"category": "programming"}),
                ("doc3", "Data science basics", {"category": "data"}),
            ]

            # Index documents
            start_time = time.time()
            await search_engine.index_documents_batch(documents)
            indexing_time = time.time() - start_time

            # Perform searches
            start_time = time.time()
            results = await search_engine.search_batch(
                ["machine learning", "python", "data science"]
            )
            search_time = time.time() - start_time

            # Verify performance
            assert indexing_time < 2.0
            assert search_time < 1.0
            assert len(results) == 3

            # Test server health
            health = await mcp_server.get_health_status()
            assert health["status"] in ["healthy", "degraded"]

        finally:
            # Cleanup
            await search_engine.close()
            await mcp_server.stop()

    @pytest.mark.asyncio
    async def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        search_config = AsyncSearchConfig(
            batch_size=20, max_concurrent_batches=4, enable_cache_warming=False
        )

        search_engine = AsyncSemanticSearch(search_config)
        await search_engine.initialize()

        try:
            # Index documents
            documents = [
                (f"doc_{i}", f"Document content {i}", {"id": i}) for i in range(100)
            ]
            await search_engine.index_documents_batch(documents)

            # Perform concurrent searches
            async def search_task(query_id):
                return await search_engine.search(f"content {query_id % 10}")

            # Run concurrent searches
            start_time = time.time()
            tasks = [search_task(i) for i in range(50)]
            results = await asyncio.gather(*tasks)
            concurrent_time = time.time() - start_time

            # Verify results
            assert len(results) == 50
            assert all(len(r) > 0 for r in results)
            assert concurrent_time < 5.0  # Should handle 50 concurrent searches quickly

            # Check performance metrics
            metrics = await search_engine.get_performance_metrics()
            assert metrics["search_engine"]["total_queries"] >= 50

        finally:
            await search_engine.close()

    @pytest.mark.asyncio
    async def test_memory_efficiency(self):
        """Test memory efficiency under load."""
        memory_config = MemoryConfig(
            max_memory_usage=256,  # 256MB limit
            monitoring_interval=0.1,
            enable_gc_optimization=True,
        )

        memory_manager = MemoryMgr(memory_config)
        await memory_manager.start()

        try:
            # Create memory-intensive operations
            efficient_dict = memory_manager.create_efficient_dict("test_dict", 1000)
            efficient_list = memory_manager.create_efficient_list("test_list", 1000)

            # Add lots of data
            for i in range(5000):
                efficient_dict[f"key_{i}"] = f"value_{i}" * 10
                efficient_list.append(f"item_{i}" * 10)

            # Check memory usage
            stats = memory_manager.get_memory_stats()

            # Should have triggered cleanup/eviction
            assert len(efficient_dict) <= 1000
            assert len(efficient_list) <= 1000
            assert (
                stats["memory_stats"]["current_usage_mb"] < 512
            )  # Should be reasonable

        finally:
            await memory_manager.stop()


@pytest.mark.asyncio
async def test_performance_regression():
    """Test for performance regressions."""
    # This test ensures that optimizations don't degrade performance

    # Baseline measurements
    baseline_times = {
        "search_time": 0.1,
        "indexing_time": 0.5,
        "cache_time": 0.001,
        "memory_cleanup": 0.05,
    }

    # Test search performance
    search_config = AsyncSearchConfig(enable_cache_warming=False)
    search_engine = AsyncSemanticSearch(search_config)
    await search_engine.initialize()

    try:
        # Index test documents
        documents = [(f"doc_{i}", f"Test content {i}", {"id": i}) for i in range(100)]

        start_time = time.time()
        await search_engine.index_documents_batch(documents)
        indexing_time = time.time() - start_time

        # Perform search
        start_time = time.time()
        results = await search_engine.search("test content")
        search_time = time.time() - start_time

        # Verify no regression
        assert indexing_time < baseline_times["indexing_time"] * 2  # Allow 2x slower
        assert search_time < baseline_times["search_time"] * 2
        assert len(results) > 0

    finally:
        await search_engine.close()


# Performance test configuration
pytestmark = pytest.mark.asyncio
