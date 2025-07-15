"""
Performance tests for MCP Standards Server.

Tests cover:
- Load testing for concurrent requests
- Memory usage monitoring
- Response time benchmarks
"""

import asyncio
import gc
import os
import statistics
import time
from typing import Any

import psutil
import pytest

from tests.e2e.conftest import MCPTestClient
from tests.e2e.fixtures import SAMPLE_CONTEXTS

# Mark entire module as serial to avoid parallel execution conflicts
pytestmark = pytest.mark.serial


class PerformanceMetrics:
    """Track and analyze performance metrics."""

    def __init__(self):
        self.response_times: list[float] = []
        self.memory_usage: list[float] = []
        self.cpu_usage: list[float] = []
        self.error_count: int = 0
        self.start_time: float = time.time()

    def record_response_time(self, duration: float):
        """Record a response time."""
        self.response_times.append(duration)

    def record_memory_usage(self):
        """Record current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)

    def record_cpu_usage(self):
        """Record current CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def get_summary(self) -> dict[str, Any]:
        """Get performance summary statistics."""
        total_duration = time.time() - self.start_time

        return {
            "total_duration": total_duration,
            "total_requests": len(self.response_times),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(len(self.response_times), 1),
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "mean": (
                    statistics.mean(self.response_times) if self.response_times else 0
                ),
                "median": (
                    statistics.median(self.response_times) if self.response_times else 0
                ),
                "p95": (
                    self._percentile(self.response_times, 95)
                    if self.response_times
                    else 0
                ),
                "p99": (
                    self._percentile(self.response_times, 99)
                    if self.response_times
                    else 0
                ),
            },
            "memory_usage": {
                "min": min(self.memory_usage) if self.memory_usage else 0,
                "max": max(self.memory_usage) if self.memory_usage else 0,
                "mean": statistics.mean(self.memory_usage) if self.memory_usage else 0,
            },
            "cpu_usage": {
                "min": min(self.cpu_usage) if self.cpu_usage else 0,
                "max": max(self.cpu_usage) if self.cpu_usage else 0,
                "mean": statistics.mean(self.cpu_usage) if self.cpu_usage else 0,
            },
            "throughput": (
                len(self.response_times) / total_duration if total_duration > 0 else 0
            ),
        }

    @staticmethod
    def _percentile(data: list[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestLoadPerformance:
    """Test server performance under load."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.timeout(120)  # Allow more time for load test
    async def test_concurrent_standard_requests(self, mcp_client):
        """Test performance with concurrent get_applicable_standards requests."""
        metrics = PerformanceMetrics()
        # Further reduce for CI stability
        num_requests = 25 if os.environ.get("CI") == "true" else 50
        concurrent_limit = 5 if os.environ.get("CI") == "true" else 10

        async def make_request(context: dict) -> float:
            """Make a single request and measure response time."""
            start = time.time()
            try:
                await mcp_client.call_tool(
                    "get_applicable_standards", {"context": context}
                )
                duration = time.time() - start
                metrics.record_response_time(duration)
                return duration
            except Exception as e:
                metrics.record_error()
                raise e

        # Create varied contexts
        contexts = []
        for i in range(num_requests):
            context_type = list(SAMPLE_CONTEXTS.keys())[i % len(SAMPLE_CONTEXTS)]
            context = SAMPLE_CONTEXTS[context_type].copy()
            context["request_id"] = f"perf_test_{i}"
            contexts.append(context)

        # Run requests with concurrency limit
        semaphore = asyncio.Semaphore(concurrent_limit)

        async def bounded_request(context):
            async with semaphore:
                return await make_request(context)

        # Execute all requests
        start_time = time.time()
        tasks = [bounded_request(ctx) for ctx in contexts]
        await asyncio.gather(*tasks, return_exceptions=True)
        time.time() - start_time

        # Analyze results
        summary = metrics.get_summary()

        # Performance assertions
        assert summary["error_rate"] < 0.01  # Less than 1% error rate
        assert summary["response_times"]["p95"] < 1.0  # 95% under 1 second
        assert summary["response_times"]["p99"] < 2.0  # 99% under 2 seconds
        assert summary["throughput"] > 10  # At least 10 requests per second

        print("\nLoad Test Results:")
        print(f"Total requests: {summary['total_requests']}")
        print(f"Total duration: {summary['total_duration']:.2f}s")
        print(f"Throughput: {summary['throughput']:.2f} req/s")
        print(
            f"Response times - p50: {summary['response_times']['median']:.3f}s, "
            f"p95: {summary['response_times']['p95']:.3f}s, "
            f"p99: {summary['response_times']['p99']:.3f}s"
        )

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_search_performance(self, mcp_client):
        """Test semantic search performance."""
        metrics = PerformanceMetrics()
        queries = [
            "React component performance optimization",
            "Python async programming patterns",
            "API security best practices",
            "Mobile app accessibility guidelines",
            "Microservice communication patterns",
            "Database query optimization",
            "Frontend testing strategies",
            "Cloud deployment best practices",
            "Error handling in distributed systems",
            "Authentication and authorization",
        ]

        # Run searches multiple times
        num_iterations = 5
        search_disabled = False

        for _iteration in range(num_iterations):
            for query in queries:
                start = time.time()
                try:
                    result = await mcp_client.call_tool(
                        "search_standards", {"query": query, "limit": 5}
                    )
                    duration = time.time() - start

                    # Check if search is disabled
                    if (
                        "error" in result
                        and "search" in result.get("error", "").lower()
                    ):
                        search_disabled = True
                        break

                    metrics.record_response_time(duration)
                except Exception:
                    metrics.record_error()

            if search_disabled:
                break

        # Skip performance assertions if search is disabled
        if search_disabled:
            pytest.skip("Search is disabled in test environment")

        summary = metrics.get_summary()

        # Search should be fast
        assert summary["response_times"]["mean"] < 0.5  # Average under 500ms
        assert summary["response_times"]["p95"] < 1.0  # 95% under 1 second

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_mixed_workload_performance(self, mcp_client):
        """Test performance with mixed workload types."""
        metrics = PerformanceMetrics()

        # Define different operation types
        async def get_standards(context):
            start = time.time()
            result = await mcp_client.call_tool(
                "get_applicable_standards", {"context": context}
            )
            metrics.record_response_time(time.time() - start)
            return result

        async def search_standards(query):
            start = time.time()
            try:
                result = await mcp_client.call_tool(
                    "search_standards", {"query": query, "limit": 3}
                )
                metrics.record_response_time(time.time() - start)
                return result
            except Exception as e:
                metrics.record_error()
                return {"error": str(e)}

        async def validate_code(code, standard):
            start = time.time()
            result = await mcp_client.call_tool(
                "validate_against_standard", {"code": code, "standard": standard}
            )
            metrics.record_response_time(time.time() - start)
            return result

        # Create mixed workload (reduced for CI)
        tasks = []
        ci_mode = os.environ.get("CI") == "true"

        # 50% get standards requests
        get_requests = 20 if ci_mode else 50
        for i in range(get_requests):
            context = SAMPLE_CONTEXTS["react_web_app"].copy()
            context["request_id"] = f"mixed_{i}"
            tasks.append(get_standards(context))

        # 30% search requests
        search_queries = [
            "performance optimization",
            "security best practices",
            "testing strategies",
        ]
        search_requests = 10 if ci_mode else 30
        for i in range(search_requests):
            query = search_queries[i % len(search_queries)]
            tasks.append(search_standards(query))

        # 20% validation requests
        sample_code = "const Component = () => <div>Hello</div>;"
        validation_requests = 8 if ci_mode else 20
        for _i in range(validation_requests):
            tasks.append(validate_code(sample_code, "react-18-patterns"))

        # Execute mixed workload
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        time.time() - start_time

        # Count errors
        error_count = sum(1 for r in results if isinstance(r, Exception))
        metrics.error_count = error_count

        summary = metrics.get_summary()

        # Mixed workload should maintain good performance
        assert summary["error_rate"] < 0.02  # Less than 2% errors
        assert summary["throughput"] > 20  # At least 20 req/s


class TestMemoryPerformance:
    """Test memory usage and leak detection."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(self, mcp_client):
        """Test memory usage remains stable under load."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Perform many operations
        for i in range(50):  # Reduced iterations for faster tests
            context: dict[str, Any] = SAMPLE_CONTEXTS["react_web_app"].copy()
            context["iteration"] = i

            await mcp_client.call_tool("get_applicable_standards", {"context": context})

            # Check memory periodically
            if i % 20 == 0:
                gc.collect()  # Force garbage collection
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory

                # Memory increase should be reasonable
                assert memory_increase < 100  # Less than 100MB increase

        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        print("\nMemory Usage:")
        print(f"Initial: {initial_memory:.2f} MB")
        print(f"Final: {final_memory:.2f} MB")
        print(f"Increase: {total_increase:.2f} MB")

        # Total memory increase should be minimal
        assert total_increase < 50  # Less than 50MB total increase

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cache_memory_management(self, mcp_client):
        """Test that caching doesn't cause memory issues."""
        process = psutil.Process()
        metrics = PerformanceMetrics()

        # Access many different standards
        standard_ids = [f"standard_{i}" for i in range(50)]

        # First pass - populate cache
        for std_id in standard_ids:
            try:
                await mcp_client.call_tool(
                    "get_standard_details", {"standard_id": std_id}
                )
            except Exception:
                pass  # Some standards may not exist

            metrics.record_memory_usage()

        # Check memory growth
        memory_growth = max(metrics.memory_usage) - min(metrics.memory_usage)
        assert memory_growth < 200  # Less than 200MB growth

        # Second pass - should use cache
        cache_memory_start = metrics.memory_usage[-1]

        for std_id in standard_ids:
            try:
                await mcp_client.call_tool(
                    "get_standard_details", {"standard_id": std_id}
                )
            except Exception:
                pass

        # Memory should not grow significantly when using cache
        cache_memory_end = process.memory_info().rss / 1024 / 1024
        cache_growth = cache_memory_end - cache_memory_start
        assert cache_growth < 10  # Less than 10MB growth when using cache


class TestResponseTimeBenchmarks:
    """Benchmark response times for different operations."""

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_standard_selection_benchmark(self, mcp_client, benchmark):
        """Benchmark get_applicable_standards operation."""
        context = SAMPLE_CONTEXTS["react_web_app"]

        async def operation():
            return await mcp_client.call_tool(
                "get_applicable_standards", {"context": context}
            )

        # Run benchmark
        result = await benchmark(operation)

        # Verify result is valid
        assert "standards" in result
        assert len(result["standards"]) > 0

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_search_benchmark(self, mcp_client, benchmark):
        """Benchmark search_standards operation."""
        query = "React performance optimization techniques"

        async def operation():
            return await mcp_client.call_tool(
                "search_standards", {"query": query, "limit": 5}
            )

        result = await benchmark(operation)

        # Handle case where search is disabled
        if "error" in result:
            assert (
                "search" in result["error"].lower()
                or "disabled" in result["error"].lower()
            )
        else:
            assert "results" in result
            assert isinstance(result["results"], list)

    @pytest.mark.asyncio
    @pytest.mark.benchmark
    async def test_validation_benchmark(self, mcp_client, benchmark):
        """Benchmark validate_against_standard operation."""
        code = """
        import React from 'react';
        const MyComponent = ({ data }) => {
            return <div>{data.map(item => <p key={item.id}>{item.name}</p>)}</div>;
        };
        export default MyComponent;
        """

        async def operation():
            return await mcp_client.call_tool(
                "validate_against_standard",
                {
                    "code": code,
                    "standard": "react-18-patterns",
                    "language": "javascript",
                },
            )

        result = await benchmark(operation)

        # Check for validation structure (it's at the top level now)
        assert "passed" in result
        assert "violations" in result
        assert "standard" in result
        assert isinstance(result["violations"], list)
        assert result["standard"] == "react-18-patterns"


class TestScalabilityLimits:
    """Test server scalability limits."""

    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.memory_intensive
    @pytest.mark.skipif(
        os.environ.get("CI") == "true",
        reason="Skipping memory-intensive test in CI to prevent timeouts",
    )
    @pytest.mark.timeout(180)  # Allow more time for connection test
    async def test_max_concurrent_connections(self, mcp_server):
        """Test maximum concurrent client connections."""
        max_clients = 50

        # Try to create many clients
        async def create_client(i):
            try:
                client = MCPTestClient(mcp_server)
                async with client as connected:
                    return connected
            except Exception as e:
                print(f"Failed to create client {i}: {e}")
                return None

        # Create clients concurrently
        tasks = [create_client(i) for i in range(max_clients)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful connections
        connected_count = sum(
            1 for r in results if r is not None and not isinstance(r, Exception)
        )

        # Should handle at least 20 concurrent clients
        assert connected_count >= 20

    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_large_response_handling(self, mcp_client):
        """Test handling of large responses."""
        # This test is limited by our test data
        # We only have 3 test standards, so adjust expectations
        result = await mcp_client.call_tool(
            "list_available_standards", {"limit": 1000}  # Request many standards
        )

        assert "standards" in result

        # Measure serialization time
        start = time.time()
        import json

        serialized = json.dumps(result)
        serialization_time = time.time() - start

        # Should handle responses efficiently
        assert serialization_time < 1.0  # Under 1 second
        # With our test data (3 standards), response will be smaller
        assert len(serialized) > 100  # Basic response size check
