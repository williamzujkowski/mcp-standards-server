#!/usr/bin/env python3
"""
Performance test runner for MCP Standards Server optimizations.

This script runs performance tests and demonstrates the improvements
made to the system.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.cache.redis_client import CacheConfig, RedisCache
from src.core.database.query_optimizer import DatabaseOptimizer, QueryCacheConfig
from src.core.mcp.async_server import AsyncMCPServer, ServerConfig
from src.core.performance.memory_manager import MemoryConfig, MemoryManager
from src.core.performance.metrics import PerformanceConfig, PerformanceMonitor
from src.core.standards.async_semantic_search import AsyncSearchConfig, AsyncSemanticSearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_async_semantic_search():
    """Test async semantic search performance."""
    logger.info("Testing async semantic search performance...")

    config = AsyncSearchConfig(
        batch_size=32,
        max_concurrent_batches=4,
        enable_vector_cache=True,
        enable_cache_warming=False
    )

    search_engine = AsyncSemanticSearch(config)
    await search_engine.initialize()

    try:
        # Test data
        documents = [
            (f"doc_{i}", f"Document about {topic} with content {i}", {"topic": topic, "id": i})
            for i, topic in enumerate([
                "machine learning", "artificial intelligence", "data science",
                "python programming", "web development", "database design",
                "cloud computing", "cybersecurity", "mobile apps", "blockchain"
            ] * 10)  # 100 documents
        ]

        # Test indexing performance
        start_time = time.time()
        await search_engine.index_documents_batch(documents)
        indexing_time = time.time() - start_time

        # Test search performance
        queries = [
            "machine learning algorithms",
            "python web development",
            "database security",
            "cloud infrastructure",
            "mobile application development"
        ]

        start_time = time.time()
        await search_engine.search_batch(queries)
        search_time = time.time() - start_time

        # Test concurrent searches
        concurrent_queries = ["data science"] * 20
        start_time = time.time()
        await search_engine.search_batch(concurrent_queries)
        concurrent_time = time.time() - start_time

        # Get metrics
        metrics = await search_engine.get_performance_metrics()

        results_data = {
            "indexing_time": indexing_time,
            "search_time": search_time,
            "concurrent_time": concurrent_time,
            "documents_indexed": len(documents),
            "queries_processed": len(queries),
            "concurrent_queries": len(concurrent_queries),
            "metrics": metrics
        }

        logger.info(f"Async Search Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await search_engine.close()


async def test_redis_performance():
    """Test Redis connection pooling performance."""
    logger.info("Testing Redis connection pooling performance...")

    config = CacheConfig(
        connection_pool_size=20,
        max_connections=50,
        enable_pipeline=True,
        enable_compression=True
    )

    redis_cache = RedisCache(config)

    try:
        # Test individual operations
        start_time = time.time()
        for i in range(100):
            redis_cache.set(f"key_{i}", f"value_{i}")
        individual_time = time.time() - start_time

        # Test batch operations
        batch_data = {f"batch_key_{i}": f"batch_value_{i}" for i in range(100)}
        start_time = time.time()
        redis_cache.mset(batch_data)
        batch_time = time.time() - start_time

        # Test pipeline operations
        pipeline_ops = [
            {"method": "set", "args": [f"pipe_key_{i}", f"pipe_value_{i}"]}
            for i in range(100)
        ]
        start_time = time.time()
        redis_cache.execute_pipeline(pipeline_ops)
        pipeline_time = time.time() - start_time

        # Get metrics
        metrics = redis_cache.get_metrics()
        health = redis_cache.health_check()

        results_data = {
            "individual_operations_time": individual_time,
            "batch_operations_time": batch_time,
            "pipeline_operations_time": pipeline_time,
            "operations_count": 100,
            "metrics": metrics,
            "health": health
        }

        logger.info(f"Redis Performance Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        # Cleanup
        try:
            redis_cache.clear_l1_cache()
        except Exception:
            pass


async def test_memory_management():
    """Test memory management performance."""
    logger.info("Testing memory management performance...")

    config = MemoryConfig(
        max_memory_usage=512,
        monitoring_interval=0.1,
        enable_object_pooling=True,
        enable_gc_optimization=True
    )

    memory_manager = MemoryManager(config)
    await memory_manager.start()

    try:
        # Test efficient data structures
        efficient_dict = memory_manager.create_efficient_dict("test_dict", 1000)
        efficient_list = memory_manager.create_efficient_list("test_list", 1000)

        # Test memory usage
        start_time = time.time()
        for i in range(5000):
            efficient_dict[f"key_{i}"] = f"value_{i}" * 10
            efficient_list.append(f"item_{i}" * 10)
        creation_time = time.time() - start_time

        # Test object pooling
        start_time = time.time()
        for _ in range(1000):
            obj = memory_manager.get_object_from_pool("list")
            if obj is not None:
                memory_manager.return_object_to_pool("list", obj)
        pooling_time = time.time() - start_time

        # Force cleanup
        start_time = time.time()
        await memory_manager.force_cleanup()
        cleanup_time = time.time() - start_time

        # Get metrics
        stats = memory_manager.get_memory_stats()

        results_data = {
            "data_structure_creation_time": creation_time,
            "object_pooling_time": pooling_time,
            "cleanup_time": cleanup_time,
            "items_created": 5000,
            "pool_operations": 1000,
            "stats": stats
        }

        logger.info(f"Memory Management Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await memory_manager.stop()


async def test_performance_monitoring():
    """Test performance monitoring system."""
    logger.info("Testing performance monitoring system...")

    config = PerformanceConfig(
        collection_interval=0.1,
        enable_system_metrics=True,
        enable_prometheus=False
    )

    monitor = PerformanceMonitor(config)
    await monitor.start()

    try:
        # Record metrics
        start_time = time.time()
        for i in range(100):
            monitor.record_metric("test_counter", 1.0)
            monitor.record_metric("test_gauge", i)
            monitor.record_metric("test_histogram", 0.001 * i)
        recording_time = time.time() - start_time

        # Test timing context
        start_time = time.time()
        for _ in range(50):
            with monitor.time_operation("test_operation"):
                time.sleep(0.001)  # Simulate work
        timing_time = time.time() - start_time

        # Test benchmark
        def test_func():
            time.sleep(0.001)
            return "result"

        benchmark = monitor.create_benchmark("test_benchmark", test_func)
        start_time = time.time()
        benchmark_results = benchmark.run(iterations=20)
        benchmark_time = time.time() - start_time

        # Get metrics
        summary = monitor.get_metrics_summary()
        dashboard = monitor.get_dashboard_data()

        results_data = {
            "metric_recording_time": recording_time,
            "timing_operations_time": timing_time,
            "benchmark_time": benchmark_time,
            "metrics_recorded": 100,
            "timing_operations": 50,
            "benchmark_iterations": 20,
            "summary": summary,
            "dashboard": dashboard,
            "benchmark_results": benchmark_results
        }

        logger.info(f"Performance Monitoring Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await monitor.stop()


async def test_database_optimization():
    """Test database query optimization."""
    logger.info("Testing database query optimization...")

    config = QueryCacheConfig(
        enable_query_cache=True,
        enable_query_batching=True,
        batch_size=10,
        batch_timeout=0.01
    )

    optimizer = DatabaseOptimizer(config)
    await optimizer.initialize()

    try:
        # Test individual queries
        queries = [
            "SELECT * FROM users WHERE age > 18",
            "SELECT * FROM products WHERE price > 100",
            "SELECT * FROM orders WHERE status = 'pending'",
            "SELECT COUNT(*) FROM customers",
            "SELECT * FROM inventory WHERE stock > 0"
        ]

        start_time = time.time()
        for query in queries:
            await optimizer.execute_query(query, {"param": "value"})
        individual_time = time.time() - start_time

        # Test cached queries (should be faster)
        start_time = time.time()
        for query in queries:
            await optimizer.execute_query(query, {"param": "value"})
        cached_time = time.time() - start_time

        # Test batch queries
        batch_queries = [(query, {"param": "value"}) for query in queries]
        start_time = time.time()
        await optimizer.execute_batch(batch_queries)
        batch_time = time.time() - start_time

        # Get metrics
        stats = optimizer.get_performance_stats()
        health = await optimizer.health_check()

        results_data = {
            "individual_queries_time": individual_time,
            "cached_queries_time": cached_time,
            "batch_queries_time": batch_time,
            "query_count": len(queries),
            "stats": stats,
            "health": health
        }

        logger.info(f"Database Optimization Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await optimizer.close()


async def test_mcp_server_performance():
    """Test MCP server performance."""
    logger.info("Testing MCP server performance...")

    config = ServerConfig(
        port=8083,
        max_connections=100,
        enable_request_batching=True,
        batch_size=20,
        enable_metrics=True
    )

    server = AsyncMCPServer(config)
    await server.start()

    try:
        # Test request processing
        requests = [
            {"method": "list_tools", "params": {}},
            {"method": "call_tool", "params": {"name": "test_tool", "arguments": {}}},
        ]

        start_time = time.time()
        for i, request in enumerate(requests * 25):  # 50 requests
            await server._process_request(request, f"test_connection_{i}")
        processing_time = time.time() - start_time

        # Test concurrent requests
        async def process_request(request_id):
            return await server._process_request(
                {"method": "list_tools", "params": {}},
                f"concurrent_connection_{request_id}"
            )

        start_time = time.time()
        tasks = [process_request(i) for i in range(20)]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time

        # Get metrics
        health = await server.get_health_status()
        stats = server.get_performance_stats()

        results_data = {
            "request_processing_time": processing_time,
            "concurrent_requests_time": concurrent_time,
            "requests_processed": 50,
            "concurrent_requests": 20,
            "health": health,
            "stats": stats
        }

        logger.info(f"MCP Server Performance Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await server.stop()


async def run_integration_test():
    """Run comprehensive integration test."""
    logger.info("Running integration performance test...")

    # Test all components together
    start_time = time.time()

    # Initialize all components
    search_engine = AsyncSemanticSearch(AsyncSearchConfig(enable_cache_warming=False))
    memory_manager = MemoryManager(MemoryConfig(max_memory_usage=512))
    monitor = PerformanceMonitor(PerformanceConfig(enable_prometheus=False))

    await search_engine.initialize()
    await memory_manager.start()
    await monitor.start()

    try:
        # Simulate integrated workload

        # 1. Index documents
        documents = [
            (f"doc_{i}", f"Content about {topic} number {i}", {"topic": topic})
            for i, topic in enumerate(["ai", "ml", "data", "python", "web"] * 20)
        ]

        indexing_start = time.time()
        await search_engine.index_documents_batch(documents)
        indexing_time = time.time() - indexing_start

        # 2. Perform searches with monitoring
        search_start = time.time()
        for i in range(10):
            with monitor.time_operation("integrated_search"):
                await search_engine.search(f"content about topic {i % 5}")
        search_time = time.time() - search_start

        # 3. Memory operations
        memory_start = time.time()
        efficient_dict = memory_manager.create_efficient_dict("integrated_dict", 500)
        for i in range(1000):
            efficient_dict[f"key_{i}"] = f"value_{i}"
        memory_time = time.time() - memory_start

        # 4. Get comprehensive metrics
        search_metrics = await search_engine.get_performance_metrics()
        memory_stats = memory_manager.get_memory_stats()
        monitor_summary = monitor.get_metrics_summary()

        total_time = time.time() - start_time

        results_data = {
            "total_integration_time": total_time,
            "indexing_time": indexing_time,
            "search_time": search_time,
            "memory_operations_time": memory_time,
            "documents_indexed": len(documents),
            "searches_performed": 10,
            "memory_operations": 1000,
            "search_metrics": search_metrics,
            "memory_stats": memory_stats,
            "monitor_summary": monitor_summary
        }

        logger.info(f"Integration Test Results: {json.dumps(results_data, indent=2)}")

        return results_data

    finally:
        await search_engine.close()
        await memory_manager.stop()
        await monitor.stop()


async def main():
    """Run all performance tests."""
    logger.info("Starting comprehensive performance tests...")

    test_results = {}

    try:
        # Run individual component tests
        test_results["async_semantic_search"] = await test_async_semantic_search()
        test_results["redis_performance"] = await test_redis_performance()
        test_results["memory_management"] = await test_memory_management()
        test_results["performance_monitoring"] = await test_performance_monitoring()
        test_results["database_optimization"] = await test_database_optimization()
        test_results["mcp_server_performance"] = await test_mcp_server_performance()

        # Run integration test
        test_results["integration_test"] = await run_integration_test()

        # Generate summary
        logger.info("=" * 60)
        logger.info("PERFORMANCE TEST SUMMARY")
        logger.info("=" * 60)

        for test_name, results in test_results.items():
            logger.info(f"\n{test_name.upper()}:")

            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, int | float) and "time" in key:
                        logger.info(f"  {key}: {value:.4f}s")
                    elif isinstance(value, int | float) and "count" in key:
                        logger.info(f"  {key}: {value}")

        logger.info("\n" + "=" * 60)
        logger.info("All performance tests completed successfully!")
        logger.info("Key improvements demonstrated:")
        logger.info("- Async semantic search with batching")
        logger.info("- Redis connection pooling and pipelining")
        logger.info("- Memory-efficient data structures")
        logger.info("- Comprehensive performance monitoring")
        logger.info("- Database query optimization with caching")
        logger.info("- Async MCP server with request batching")

        return test_results

    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
