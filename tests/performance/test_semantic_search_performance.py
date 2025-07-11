"""
Performance tests for semantic search functionality.

Tests focused on performance characteristics including latency,
throughput, memory usage, and scalability.
"""

import asyncio
import gc
import json
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import psutil
import pytest

from src.core.standards.semantic_search import (
    AsyncSemanticSearch,
    SemanticSearch,
    create_search_engine,
)
from tests.mocks.semantic_search_mocks import (
    TestDataGenerator,
    patch_ml_dependencies,
)


class PerformanceMetrics:
    """Helper class for collecting performance metrics."""

    def __init__(self):
        self.latencies = []
        self.memory_usage = []
        self.cpu_usage = []
        self.timestamps = []
        self.process = psutil.Process()

    def record_operation(self, operation_time: float):
        """Record metrics for an operation."""
        self.latencies.append(operation_time)
        self.memory_usage.append(self.process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(self.process.cpu_percent())
        self.timestamps.append(datetime.now())

    def get_percentiles(self, percentiles=None):
        """Get latency percentiles."""
        if percentiles is None:
            percentiles = [50, 90, 95, 99]
        if not self.latencies:
            return {}

        return {
            f"p{p}": np.percentile(self.latencies, p) * 1000  # Convert to ms
            for p in percentiles
        }

    def get_summary(self):
        """Get performance summary."""
        if not self.latencies:
            return {}

        return {
            "total_operations": len(self.latencies),
            "avg_latency_ms": np.mean(self.latencies) * 1000,
            "min_latency_ms": np.min(self.latencies) * 1000,
            "max_latency_ms": np.max(self.latencies) * 1000,
            "percentiles": self.get_percentiles(),
            "avg_memory_mb": np.mean(self.memory_usage),
            "max_memory_mb": np.max(self.memory_usage),
            "avg_cpu_percent": np.mean(self.cpu_usage),
        }

    def plot_metrics(self, output_path: Path):
        """Plot performance metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Latency over time
        ax1.plot(self.timestamps, [latency * 1000 for latency in self.latencies])
        ax1.set_ylabel("Latency (ms)")
        ax1.set_title("Search Latency Over Time")
        ax1.grid(True)

        # Memory usage over time
        ax2.plot(self.timestamps, self.memory_usage, color="orange")
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title("Memory Usage Over Time")
        ax2.grid(True)

        # CPU usage over time
        ax3.plot(self.timestamps, self.cpu_usage, color="green")
        ax3.set_ylabel("CPU (%)")
        ax3.set_xlabel("Time")
        ax3.set_title("CPU Usage Over Time")
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


class TestSemanticSearchLatency:
    """Test search latency under various conditions."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    @patch_ml_dependencies()
    def search_engine(self, temp_dir):
        """Create search engine for testing."""
        engine = create_search_engine(cache_dir=temp_dir)
        yield engine
        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_search_latency_percentiles(self, search_engine):
        """Test search latency percentiles."""
        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(search_engine, SemanticSearch):
            search_engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Generate diverse queries
        queries = [
            "python security",
            "react testing best practices",
            "api design patterns",
            "database optimization techniques",
            "microservices architecture",
            "cloud deployment strategies",
            "security AND authentication",
            "testing NOT integration",
            "performance OR optimization",
        ] * 20  # Repeat for more samples

        metrics = PerformanceMetrics()

        # Perform searches and measure latency
        for query in queries:
            start = time.perf_counter()
            if isinstance(search_engine, SemanticSearch):
                search_engine.search(query, top_k=10)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")
            elapsed = time.perf_counter() - start
            metrics.record_operation(elapsed)

        # Analyze results
        summary = metrics.get_summary()
        percentiles = summary["percentiles"]

        print(f"\nSearch Latency Percentiles (n={len(queries)}):")
        print(f"P50: {percentiles['p50']:.2f}ms")
        print(f"P90: {percentiles['p90']:.2f}ms")
        print(f"P95: {percentiles['p95']:.2f}ms")
        print(f"P99: {percentiles['p99']:.2f}ms")

        # Performance assertions
        assert percentiles["p50"] < 50  # P50 under 50ms
        assert percentiles["p90"] < 100  # P90 under 100ms
        assert percentiles["p99"] < 200  # P99 under 200ms

    @patch_ml_dependencies()
    def test_cold_vs_warm_cache_latency(self, search_engine):
        """Test latency difference between cold and warm cache."""
        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(500)
        if isinstance(search_engine, SemanticSearch):
            search_engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        query = "python api security testing"

        # Cold cache search
        if isinstance(search_engine, SemanticSearch):
            search_engine.result_cache.clear()
            search_engine.embedding_cache.memory_cache.clear()

            cold_start = time.perf_counter()
            cold_results = search_engine.search(query, top_k=10)
            cold_latency = time.perf_counter() - cold_start

            # Warm cache search
            warm_start = time.perf_counter()
            warm_results = search_engine.search(query, top_k=10)
            warm_latency = time.perf_counter() - warm_start
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        print(f"\nCold cache latency: {cold_latency*1000:.2f}ms")
        print(f"Warm cache latency: {warm_latency*1000:.2f}ms")
        print(f"Speedup: {cold_latency/warm_latency:.2f}x")

        # Warm cache should be significantly faster
        assert warm_latency < cold_latency * 0.5
        assert len(cold_results) == len(warm_results)

    @patch_ml_dependencies()
    def test_latency_vs_corpus_size(self, temp_dir):
        """Test how latency scales with corpus size."""
        corpus_sizes = [100, 500, 1000, 5000, 10000]
        latencies = []

        for size in corpus_sizes:
            # Create fresh engine
            engine = create_search_engine(cache_dir=temp_dir)

            # Index documents
            docs = TestDataGenerator.generate_standards_corpus(size)
            if isinstance(engine, SemanticSearch):
                engine.index_documents_batch(docs)
            else:
                # AsyncSemanticSearch doesn't have sync index_documents_batch
                # This test expects sync behavior, so we shouldn't get here
                raise TypeError("Expected SemanticSearch instance for sync test")

            # Measure search latency (average of 10 searches)
            query = "python testing security"
            times = []

            for _ in range(10):
                start = time.perf_counter()
                if isinstance(engine, SemanticSearch):
                    engine.search(query, top_k=10)
                else:
                    raise TypeError("Expected SemanticSearch instance for sync test")
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            avg_latency = np.mean(times[1:])  # Skip first (cold cache)
            latencies.append(avg_latency * 1000)  # Convert to ms

            print(f"Corpus size: {size:5d}, Avg latency: {avg_latency*1000:.2f}ms")

            if isinstance(engine, SemanticSearch):
                engine.close()
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

        # Check scaling characteristics
        # Latency should not increase linearly with corpus size
        scaling_factor = latencies[-1] / latencies[0]
        size_factor = corpus_sizes[-1] / corpus_sizes[0]

        print(
            f"\nScaling factor: {scaling_factor:.2f}x for {size_factor}x size increase"
        )
        assert scaling_factor < size_factor * 0.5  # Sub-linear scaling


class TestSemanticSearchThroughput:
    """Test search throughput and concurrent performance."""

    @patch_ml_dependencies()
    def test_single_thread_throughput(self):
        """Test single-threaded search throughput."""
        engine = create_search_engine()

        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Generate queries
        queries = [f"query {i % 100}" for i in range(1000)]

        # Measure throughput
        start = time.time()
        for query in queries:
            if isinstance(engine, SemanticSearch):
                engine.search(query, top_k=5)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")
        elapsed = time.time() - start

        throughput = len(queries) / elapsed
        print(f"\nSingle-thread throughput: {throughput:.2f} queries/second")

        # Should handle at least 100 queries/second
        assert throughput > 100

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_concurrent_throughput(self):
        """Test concurrent search throughput."""
        engine = create_search_engine()

        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Test with different thread counts
        thread_counts = [1, 2, 4, 8, 16]
        results = {}

        for num_threads in thread_counts:
            queries_per_thread = 100
            total_queries = num_threads * queries_per_thread

            def worker(thread_id, queries_count=queries_per_thread):
                times = []
                for i in range(queries_count):
                    query = f"thread {thread_id} query {i}"
                    start = time.perf_counter()
                    if isinstance(engine, SemanticSearch):
                        engine.search(query, top_k=5)
                    else:
                        raise TypeError(
                            "Expected SemanticSearch instance for sync test"
                        )
                    elapsed = time.perf_counter() - start
                    times.append(elapsed)
                return times

            # Run concurrent searches
            start = time.time()
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker, i) for i in range(num_threads)]
                all_times = []
                for future in as_completed(futures):
                    all_times.extend(future.result())

            total_time = time.time() - start
            throughput = total_queries / total_time
            avg_latency = np.mean(all_times) * 1000

            results[num_threads] = {
                "throughput": throughput,
                "avg_latency_ms": avg_latency,
                "total_time": total_time,
            }

            print(
                f"Threads: {num_threads:2d}, Throughput: {throughput:.2f} q/s, "
                f"Avg latency: {avg_latency:.2f}ms"
            )

        # Throughput should scale with threads (not linearly due to contention)
        assert results[4]["throughput"] > results[1]["throughput"] * 2

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @pytest.mark.asyncio
    @patch_ml_dependencies()
    async def test_async_throughput(self):
        """Test async search throughput."""
        engine = create_search_engine(async_mode=True)

        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, AsyncSemanticSearch):
            await engine.index_documents_batch_async(docs)
        else:
            raise TypeError("Expected AsyncSemanticSearch instance for async test")

        # Generate queries
        num_queries = 1000
        queries = [f"async query {i}" for i in range(num_queries)]

        # Measure async throughput
        start = time.time()

        # Create tasks in batches to avoid overwhelming
        batch_size = 100
        for i in range(0, num_queries, batch_size):
            batch = queries[i : i + batch_size]
            if isinstance(engine, AsyncSemanticSearch):
                tasks = [engine.search_async(q, top_k=5) for q in batch]
                await asyncio.gather(*tasks)
            else:
                raise TypeError("Expected AsyncSemanticSearch instance for async test")

        elapsed = time.time() - start
        throughput = num_queries / elapsed

        print(f"\nAsync throughput: {throughput:.2f} queries/second")

        # Async should achieve high throughput
        assert throughput > 200

        if isinstance(engine, AsyncSemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected AsyncSemanticSearch instance for async test")


class TestSemanticSearchMemory:
    """Test memory usage and leak detection."""

    @patch_ml_dependencies()
    def test_memory_usage_during_indexing(self):
        """Test memory usage during large-scale indexing."""
        engine = create_search_engine()
        process = psutil.Process()

        # Get baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Index documents in batches
        batch_sizes = [100, 500, 1000]
        memory_usage = []

        for batch_size in batch_sizes:
            docs = TestDataGenerator.generate_standards_corpus(batch_size)

            # Measure memory before and after
            gc.collect()
            before = process.memory_info().rss / 1024 / 1024

            if isinstance(engine, SemanticSearch):
                engine.index_documents_batch(docs)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

            gc.collect()
            after = process.memory_info().rss / 1024 / 1024

            delta = after - before
            memory_usage.append(delta)

            print(f"Batch size: {batch_size}, Memory delta: {delta:.2f}MB")

        # Memory usage should be reasonable
        total_memory_used = process.memory_info().rss / 1024 / 1024 - baseline_memory
        print(f"\nTotal memory used: {total_memory_used:.2f}MB")

        # Should use less than 500MB for ~1600 documents
        assert total_memory_used < 500

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        engine = create_search_engine()
        process = psutil.Process()

        # Index initial documents
        docs = TestDataGenerator.generate_standards_corpus(500)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Get baseline after initial setup
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024

        # Perform many operations
        memory_samples = []

        for iteration in range(10):
            # Perform 100 searches
            for i in range(100):
                query = f"test query {i % 20}"
                if isinstance(engine, SemanticSearch):
                    results = engine.search(query, top_k=10)
                else:
                    raise TypeError("Expected SemanticSearch instance for sync test")

                # Simulate result processing
                for result in results:
                    _ = result.content
                    _ = result.highlights

            # Index new documents
            new_docs = TestDataGenerator.generate_standards_corpus(10)
            if isinstance(engine, SemanticSearch):
                engine.index_documents_batch(new_docs)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

            # Clear caches periodically
            if iteration % 3 == 0:
                engine.result_cache.clear()

            # Measure memory
            gc.collect()
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)

        # Check for memory leak
        memory_growth = memory_samples[-1] - baseline_memory
        avg_growth_per_iteration = memory_growth / len(memory_samples)

        print(
            f"\nMemory growth: {memory_growth:.2f}MB over {len(memory_samples)} iterations"
        )
        print(f"Average growth per iteration: {avg_growth_per_iteration:.2f}MB")

        # Memory growth should be minimal
        assert avg_growth_per_iteration < 5  # Less than 5MB per iteration

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_cache_memory_management(self, temp_dir):
        """Test memory management of caching layers."""
        # Create engine with custom cache settings
        engine = create_search_engine(cache_dir=temp_dir)

        # Configure cache limits
        if isinstance(engine, SemanticSearch):
            engine.result_cache_ttl = timedelta(seconds=30)
            engine.embedding_cache.cache_ttl = timedelta(minutes=5)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        process = psutil.Process()

        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Fill caches
        queries = [f"cache test {i}" for i in range(100)]
        for query in queries:
            if isinstance(engine, SemanticSearch):
                engine.search(query, top_k=10)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

        # Check memory with full caches
        gc.collect()
        full_cache_memory = process.memory_info().rss / 1024 / 1024

        # Clear caches
        if isinstance(engine, SemanticSearch):
            engine.result_cache.clear()
            engine.embedding_cache.clear_cache()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Check memory after clearing
        gc.collect()
        cleared_cache_memory = process.memory_info().rss / 1024 / 1024

        memory_freed = full_cache_memory - cleared_cache_memory
        print(f"\nMemory with full caches: {full_cache_memory:.2f}MB")
        print(f"Memory after clearing: {cleared_cache_memory:.2f}MB")
        print(f"Memory freed: {memory_freed:.2f}MB")

        # Should free significant memory
        assert memory_freed > 10  # At least 10MB freed

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")


class TestSemanticSearchScalability:
    """Test scalability with large datasets."""

    @patch_ml_dependencies()
    def test_large_corpus_indexing(self, temp_dir):
        """Test indexing performance with large corpus."""
        engine = create_search_engine(cache_dir=temp_dir)

        corpus_sizes = [1000, 5000, 10000]
        indexing_times = []

        for size in corpus_sizes:
            # Generate documents
            print(f"\nGenerating {size} documents...")
            docs = TestDataGenerator.generate_standards_corpus(size)

            # Time indexing
            start = time.time()
            if isinstance(engine, SemanticSearch):
                engine.index_documents_batch(docs)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")
            elapsed = time.time() - start

            indexing_times.append(elapsed)
            rate = size / elapsed

            print(f"Indexed {size} documents in {elapsed:.2f}s ({rate:.2f} docs/s)")

        # Check scaling
        # Time should not scale linearly (due to batching optimizations)
        time_ratio = indexing_times[-1] / indexing_times[0]
        size_ratio = corpus_sizes[-1] / corpus_sizes[0]

        print(f"\nTime scaling: {time_ratio:.2f}x for {size_ratio}x size increase")
        assert time_ratio < size_ratio * 0.8

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_search_quality_at_scale(self):
        """Test search quality doesn't degrade with scale."""
        # Test with different corpus sizes
        sizes = [100, 1000, 5000]
        quality_metrics = {}

        for size in sizes:
            engine = create_search_engine()

            # Create corpus with known relevant documents
            docs = []

            # Add specific documents we'll search for
            target_docs = [
                (
                    "python-security",
                    "Python security best practices for web applications",
                    {"category": "security", "language": "python"},
                ),
                (
                    "react-testing",
                    "React component testing with Jest and React Testing Library",
                    {"category": "testing", "framework": "react"},
                ),
                (
                    "api-design",
                    "RESTful API design patterns and guidelines",
                    {"category": "api", "type": "design"},
                ),
            ]

            for doc_id, content, metadata in target_docs:
                docs.append((doc_id, content, metadata))

            # Add noise documents
            noise_docs = TestDataGenerator.generate_standards_corpus(
                size - len(target_docs)
            )
            docs.extend(noise_docs)

            # Index all documents
            if isinstance(engine, SemanticSearch):
                engine.index_documents_batch(docs)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

            # Search for target documents
            searches = [
                ("Python security", "python-security"),
                ("React testing Jest", "react-testing"),
                ("API design patterns", "api-design"),
            ]

            hits_at_1 = 0
            hits_at_5 = 0

            for query, expected_id in searches:
                if isinstance(engine, SemanticSearch):
                    results = engine.search(query, top_k=5)
                else:
                    raise TypeError("Expected SemanticSearch instance for sync test")

                if results and results[0].id == expected_id:
                    hits_at_1 += 1

                if any(r.id == expected_id for r in results[:5]):
                    hits_at_5 += 1

            quality_metrics[size] = {
                "precision_at_1": hits_at_1 / len(searches),
                "precision_at_5": hits_at_5 / len(searches),
            }

            if isinstance(engine, SemanticSearch):
                engine.close()
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")

        # Print results
        print("\nSearch quality at different scales:")
        for size, metrics in quality_metrics.items():
            print(
                f"Size {size}: P@1={metrics['precision_at_1']:.2f}, "
                f"P@5={metrics['precision_at_5']:.2f}"
            )

        # Quality should remain high regardless of scale
        for _size, metrics in quality_metrics.items():
            assert metrics["precision_at_5"] >= 0.8  # At least 80% recall at 5


class TestSemanticSearchStressTest:
    """Stress tests for semantic search."""

    @patch_ml_dependencies()
    def test_sustained_high_load(self):
        """Test system under sustained high load."""
        engine = create_search_engine()

        # Index moderate corpus
        docs = TestDataGenerator.generate_standards_corpus(2000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Simulate high load for extended period
        duration_seconds = 30
        queries_per_second = 50

        metrics = PerformanceMetrics()
        errors = []

        def load_generator():
            """Generate continuous load."""
            start_time = time.time()
            query_count = 0

            while time.time() - start_time < duration_seconds:
                try:
                    query = f"load test query {query_count % 100}"

                    op_start = time.perf_counter()
                    if isinstance(engine, SemanticSearch):
                        engine.search(query, top_k=10)
                    else:
                        raise TypeError(
                            "Expected SemanticSearch instance for sync test"
                        )
                    op_elapsed = time.perf_counter() - op_start

                    metrics.record_operation(op_elapsed)
                    query_count += 1

                    # Pace the queries
                    sleep_time = max(0, (1.0 / queries_per_second) - op_elapsed)
                    time.sleep(sleep_time)

                except Exception as e:
                    errors.append(str(e))

        # Run load test
        print(
            f"\nRunning stress test for {duration_seconds}s at {queries_per_second} q/s..."
        )
        load_generator()

        # Analyze results
        summary = metrics.get_summary()

        print("\nStress test results:")
        print(f"Total queries: {summary['total_operations']}")
        print(f"Errors: {len(errors)}")
        print(f"Avg latency: {summary['avg_latency_ms']:.2f}ms")
        print(f"P99 latency: {summary['percentiles']['p99']:.2f}ms")
        print(f"Max memory: {summary['max_memory_mb']:.2f}MB")

        # System should remain stable
        assert len(errors) == 0
        assert summary["percentiles"]["p99"] < 500  # P99 under 500ms even under load

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

    @patch_ml_dependencies()
    def test_burst_traffic_handling(self):
        """Test handling of traffic bursts."""
        engine = create_search_engine()

        # Index documents
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Simulate burst traffic pattern
        burst_sizes = [10, 50, 100, 200]

        for burst_size in burst_sizes:
            print(f"\nTesting burst of {burst_size} concurrent requests...")

            # Prepare queries
            queries = [f"burst query {i}" for i in range(burst_size)]

            # Launch all queries concurrently
            start = time.time()
            with ThreadPoolExecutor(max_workers=burst_size) as executor:
                if isinstance(engine, SemanticSearch):
                    futures = [
                        executor.submit(engine.search, query, top_k=5)
                        for query in queries
                    ]
                else:
                    raise TypeError("Expected SemanticSearch instance for sync test")

                # Wait for all to complete
                results = []
                errors = []
                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=5.0)
                        results.append(result)
                    except Exception as e:
                        errors.append(str(e))

            elapsed = time.time() - start

            print(f"Completed in {elapsed:.2f}s")
            print(f"Success: {len(results)}, Errors: {len(errors)}")

            # All requests should succeed
            assert len(errors) == 0
            assert len(results) == burst_size

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")


def test_performance_regression_suite():
    """
    Comprehensive performance regression test suite.

    This test establishes performance baselines and can be used
    to detect performance regressions in CI/CD pipelines.
    """
    with patch_ml_dependencies():
        results: dict[str, Any] = {"timestamp": datetime.now().isoformat(), "tests": {}}

        # Test 1: Basic search latency
        engine = create_search_engine()
        docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        latencies = []
        for i in range(100):
            query = f"test query {i % 20}"
            start = time.perf_counter()
            if isinstance(engine, SemanticSearch):
                engine.search(query, top_k=10)
            else:
                raise TypeError("Expected SemanticSearch instance for sync test")
            latencies.append(time.perf_counter() - start)

        results["tests"]["basic_search"] = {
            "p50_ms": np.percentile(latencies, 50) * 1000,
            "p90_ms": np.percentile(latencies, 90) * 1000,
            "p99_ms": np.percentile(latencies, 99) * 1000,
        }

        # Test 2: Indexing throughput
        start = time.time()
        new_docs = TestDataGenerator.generate_standards_corpus(1000)
        if isinstance(engine, SemanticSearch):
            engine.index_documents_batch(new_docs)
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")
        indexing_time = time.time() - start

        results["tests"]["indexing"] = {
            "docs_per_second": 1000 / indexing_time,
            "total_time_seconds": indexing_time,
        }

        # Test 3: Memory efficiency
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024

        results["tests"]["memory"] = {
            "total_memory_mb": memory_mb,
            "memory_per_doc_kb": (memory_mb * 1024) / 2000,  # 2000 total docs
        }

        if isinstance(engine, SemanticSearch):
            engine.close()
        else:
            raise TypeError("Expected SemanticSearch instance for sync test")

        # Save results for comparison
        output_path = Path("performance_baseline.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nPerformance baseline saved to {output_path}")
        print(json.dumps(results, indent=2))

        # Assert baseline requirements
        assert results["tests"]["basic_search"]["p50_ms"] < 50
        assert results["tests"]["basic_search"]["p99_ms"] < 200
        assert results["tests"]["indexing"]["docs_per_second"] > 100
        assert results["tests"]["memory"]["memory_per_doc_kb"] < 100
