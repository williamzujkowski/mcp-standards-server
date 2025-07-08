"""
Performance benchmarks for MCP Standards Server.

Measures performance of key operations to ensure they meet MCP standards.
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any
from contextlib import asynccontextmanager

import pytest
import pytest_benchmark
from mcp import ClientSession, StdioServerParameters

from tests.e2e.conftest import MCPTestClient, mcp_server, mcp_client
from tests.e2e.fixtures import SAMPLE_CONTEXTS


class BenchmarkMetrics:
    """Collect and analyze benchmark metrics."""
    
    def __init__(self):
        self.results = {
            "tool_latencies": {},
            "cache_performance": {},
            "auth_overhead": {},
            "concurrent_performance": {}
        }
        
    def record_tool_latency(self, tool: str, latency: float):
        """Record tool call latency."""
        if tool not in self.results["tool_latencies"]:
            self.results["tool_latencies"][tool] = []
        self.results["tool_latencies"][tool].append(latency)
        
    def record_cache_performance(self, operation: str, hit_rate: float, time_saved: float):
        """Record cache performance metrics."""
        self.results["cache_performance"][operation] = {
            "hit_rate": hit_rate,
            "time_saved_ms": time_saved * 1000
        }
        
    def record_auth_overhead(self, auth_type: str, overhead_ms: float):
        """Record authentication overhead."""
        self.results["auth_overhead"][auth_type] = overhead_ms
        
    def record_concurrent_performance(self, concurrent_level: int, avg_latency: float, throughput: float):
        """Record concurrent request performance."""
        self.results["concurrent_performance"][str(concurrent_level)] = {
            "avg_latency_ms": avg_latency * 1000,
            "throughput_rps": throughput
        }
        
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            "tool_latencies_summary": {},
            "cache_performance": self.results["cache_performance"],
            "auth_overhead": self.results["auth_overhead"],
            "concurrent_performance": self.results["concurrent_performance"]
        }
        
        # Summarize tool latencies
        for tool, latencies in self.results["tool_latencies"].items():
            if latencies:
                summary["tool_latencies_summary"][tool] = {
                    "avg_ms": statistics.mean(latencies) * 1000,
                    "p50_ms": statistics.median(latencies) * 1000,
                    "p95_ms": self._percentile(latencies, 95) * 1000,
                    "p99_ms": self._percentile(latencies, 99) * 1000,
                    "min_ms": min(latencies) * 1000,
                    "max_ms": max(latencies) * 1000
                }
                
        return summary
        
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]


@pytest.mark.benchmark(group="tool-latency")
class TestToolLatencyBenchmarks:
    """Benchmark tool call latencies."""
    
    @pytest.mark.asyncio
    async def test_get_applicable_standards_latency(self, mcp_client, benchmark):
        """Benchmark get_applicable_standards latency."""
        context = SAMPLE_CONTEXTS["react_web_app"]
        
        async def operation():
            return await mcp_client.call_tool(
                "get_applicable_standards",
                {"context": context}
            )
            
        # Run benchmark
        result = await benchmark.pedantic(
            operation,
            rounds=50,
            iterations=1,
            warmup_rounds=5
        )
        
        # Verify result and check against target
        assert "standards" in result
        assert benchmark.stats["mean"] < 0.050  # Target: <50ms
        
    @pytest.mark.asyncio
    async def test_validate_against_standard_latency(self, mcp_client, benchmark):
        """Benchmark validate_against_standard latency."""
        code = """
        import React from 'react';
        const MyComponent = ({data}) => <div>{data.map(item => <p key={item.id}>{item.name}</p>)}</div>;
        export default MyComponent;
        """
        
        async def operation():
            return await mcp_client.call_tool(
                "validate_against_standard",
                {
                    "code": code,
                    "standard": "react-18-patterns",
                    "language": "javascript"
                }
            )
            
        result = await benchmark.pedantic(
            operation,
            rounds=30,
            iterations=1,
            warmup_rounds=3
        )
        
        assert "violations" in result
        assert benchmark.stats["mean"] < 0.050  # Target: <50ms
        
    @pytest.mark.asyncio
    async def test_search_standards_latency(self, mcp_client, benchmark):
        """Benchmark search_standards latency."""
        async def operation():
            return await mcp_client.call_tool(
                "search_standards",
                {
                    "query": "React hooks best practices",
                    "limit": 5
                }
            )
            
        result = await benchmark.pedantic(
            operation,
            rounds=30,
            iterations=1,
            warmup_rounds=3
        )
        
        assert "results" in result
        assert benchmark.stats["mean"] < 0.100  # Target: <100ms for search
        
    @pytest.mark.asyncio
    async def test_get_standard_details_latency(self, mcp_client, benchmark):
        """Benchmark get_standard_details latency."""
        async def operation():
            return await mcp_client.call_tool(
                "get_standard_details",
                {"standard_id": "react-18-patterns"}
            )
            
        result = await benchmark.pedantic(
            operation,
            rounds=50,
            iterations=1,
            warmup_rounds=5
        )
        
        assert "id" in result
        assert benchmark.stats["mean"] < 0.050  # Target: <50ms


@pytest.mark.benchmark(group="cache-performance")
class TestCachePerformanceBenchmarks:
    """Benchmark cache performance."""
    
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, mcp_client, benchmark):
        """Benchmark cache hit vs miss performance."""
        standard_id = "react-18-patterns"
        
        # Prime the cache
        await mcp_client.call_tool(
            "get_standard_details",
            {"standard_id": standard_id}
        )
        
        # Measure cache hit performance
        async def cache_hit_operation():
            return await mcp_client.call_tool(
                "get_standard_details",
                {"standard_id": standard_id}
            )
            
        # Measure performance
        cache_hit_result = await benchmark.pedantic(
            cache_hit_operation,
            rounds=100,
            iterations=1
        )
        
        cache_hit_time = benchmark.stats["mean"]
        
        # Clear cache and measure miss performance
        await mcp_client.call_tool("cache_clear_all", {})
        
        async def cache_miss_operation():
            return await mcp_client.call_tool(
                "get_standard_details",
                {"standard_id": standard_id}
            )
            
        # Reset benchmark for new measurement
        benchmark.reset()
        cache_miss_result = await benchmark.pedantic(
            cache_miss_operation,
            rounds=10,
            iterations=1
        )
        
        cache_miss_time = benchmark.stats["mean"]
        
        # Cache should provide significant speedup
        speedup = cache_miss_time / cache_hit_time
        assert speedup > 2.0  # At least 2x faster with cache
        
        # Record metrics
        metrics = BenchmarkMetrics()
        metrics.record_cache_performance(
            "get_standard_details",
            hit_rate=0.9,  # Assumed hit rate
            time_saved=(cache_miss_time - cache_hit_time)
        )


@pytest.mark.benchmark(group="concurrent-performance")
class TestConcurrentPerformanceBenchmarks:
    """Benchmark concurrent request handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests_throughput(self, mcp_server):
        """Benchmark throughput with concurrent requests."""
        metrics = BenchmarkMetrics()
        
        async def make_concurrent_requests(num_concurrent: int):
            """Make concurrent requests and measure performance."""
            clients = []
            
            # Create multiple clients
            for _ in range(num_concurrent):
                client = MCPTestClient(mcp_server)
                clients.append(client)
                
            # Prepare tasks
            tasks = []
            context = SAMPLE_CONTEXTS["react_web_app"]
            
            async def single_request(client):
                async with client.connect() as connected:
                    start = time.time()
                    result = await connected.call_tool(
                        "get_applicable_standards",
                        {"context": context}
                    )
                    latency = time.time() - start
                    return latency, result
                    
            # Run concurrent requests
            start_time = time.time()
            
            for client in clients:
                task = single_request(client)
                tasks.append(task)
                
            results = await asyncio.gather(*tasks)
            
            total_time = time.time() - start_time
            
            # Calculate metrics
            latencies = [r[0] for r in results]
            avg_latency = statistics.mean(latencies)
            throughput = num_concurrent / total_time
            
            metrics.record_concurrent_performance(
                num_concurrent,
                avg_latency,
                throughput
            )
            
            return avg_latency, throughput
            
        # Test different concurrency levels
        for concurrent in [1, 5, 10, 20]:
            avg_latency, throughput = await make_concurrent_requests(concurrent)
            
            # Performance targets
            assert avg_latency < 0.200  # <200ms average latency
            assert throughput > concurrent * 2  # At least 2 req/s per connection
            
        # Get summary
        summary = metrics.get_summary()
        print(f"\nConcurrent Performance Summary:")
        print(json.dumps(summary["concurrent_performance"], indent=2))


@pytest.mark.benchmark(group="auth-overhead")
class TestAuthenticationOverheadBenchmarks:
    """Benchmark authentication overhead."""
    
    @pytest.mark.asyncio
    async def test_jwt_auth_overhead(self, mcp_server, benchmark):
        """Benchmark JWT authentication overhead."""
        # Create clients with and without auth
        client_no_auth = MCPTestClient(mcp_server)
        client_with_auth = MCPTestClient(mcp_server)
        
        # Set auth header for authenticated client
        async def operation_no_auth():
            async with client_no_auth.connect() as client:
                return await client.call_tool(
                    "list_available_standards",
                    {"limit": 10}
                )
                
        async def operation_with_auth():
            async with client_with_auth.connect() as client:
                # Simulate JWT token in header
                client.session._write_transport._process.env["MCP_AUTH_TOKEN"] = "fake.jwt.token"
                return await client.call_tool(
                    "list_available_standards",
                    {"limit": 10}
                )
                
        # Benchmark without auth
        result_no_auth = await benchmark.pedantic(
            operation_no_auth,
            rounds=30,
            iterations=1
        )
        time_no_auth = benchmark.stats["mean"]
        
        # Reset and benchmark with auth
        benchmark.reset()
        result_with_auth = await benchmark.pedantic(
            operation_with_auth,
            rounds=30,
            iterations=1
        )
        time_with_auth = benchmark.stats["mean"]
        
        # Calculate overhead
        overhead_ms = (time_with_auth - time_no_auth) * 1000
        
        # Auth overhead should be minimal
        assert overhead_ms < 5.0  # Less than 5ms overhead
        
        metrics = BenchmarkMetrics()
        metrics.record_auth_overhead("jwt", overhead_ms)


@pytest.mark.benchmark(group="memory-usage")
class TestMemoryUsageBenchmarks:
    """Benchmark memory usage patterns."""
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, mcp_client):
        """Test memory usage remains stable under load."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Perform many operations
        operations_count = 1000
        context = SAMPLE_CONTEXTS["react_web_app"]
        
        for i in range(operations_count):
            if i % 100 == 0:
                gc.collect()  # Force garbage collection periodically
                
            result = await mcp_client.call_tool(
                "get_applicable_standards",
                {"context": context}
            )
            
            # Verify result
            assert "standards" in result
            
        # Final memory check
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"\nMemory Usage:")
        print(f"Initial: {initial_memory:.2f} MB")
        print(f"Final: {final_memory:.2f} MB")
        print(f"Increase: {memory_increase:.2f} MB")
        print(f"Per operation: {memory_increase / operations_count * 1000:.2f} KB")
        
        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB for 1000 operations
        

def run_all_benchmarks():
    """Run all benchmarks and generate report."""
    import subprocess
    
    # Run pytest with benchmark plugin
    result = subprocess.run([
        "pytest",
        __file__,
        "-v",
        "--benchmark-only",
        "--benchmark-json=benchmark_results.json",
        "--benchmark-columns=min,max,mean,stddev,median,iqr,outliers,rounds,iterations",
        "--benchmark-group-by=group"
    ], capture_output=True, text=True)
    
    print(result.stdout)
    
    # Load and analyze results
    try:
        with open("benchmark_results.json", "r") as f:
            results = json.load(f)
            
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)
        
        # Check against MCP performance targets
        targets_met = True
        
        for benchmark in results["benchmarks"]:
            name = benchmark["name"]
            mean = benchmark["stats"]["mean"]
            
            print(f"\n{name}:")
            print(f"  Mean: {mean*1000:.2f}ms")
            print(f"  Min: {benchmark['stats']['min']*1000:.2f}ms")
            print(f"  Max: {benchmark['stats']['max']*1000:.2f}ms")
            
            # Check targets
            if "tool_latency" in benchmark["group"]:
                if mean > 0.050:  # 50ms target
                    print(f"  ❌ FAILED: Exceeds 50ms target")
                    targets_met = False
                else:
                    print(f"  ✅ PASSED: Meets 50ms target")
                    
        print("\n" + "="*80)
        print(f"Overall: {'✅ ALL TARGETS MET' if targets_met else '❌ SOME TARGETS MISSED'}")
        print("="*80)
        
    except FileNotFoundError:
        print("No benchmark results found. Run benchmarks first.")
        

if __name__ == "__main__":
    run_all_benchmarks()