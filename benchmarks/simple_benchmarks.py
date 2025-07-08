"""
Simple performance benchmarks for MCP Standards Server.

Can be run standalone without pytest infrastructure.
"""

import asyncio
import json
import time
import statistics
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.standards.micro_standards import MicroStandardsEngine
from src.core.standards.semantic_search import SemanticSearchEngine
from src.core.validation import InputValidator, GetApplicableStandardsInput
from src.core.auth import AuthManager
from src.core.cache.mcp_cache import MCPCache, CacheConfig


class PerformanceBenchmark:
    """Run performance benchmarks for core components."""
    
    def __init__(self):
        self.results = {}
        self.engine = MicroStandardsEngine()
        self.validator = InputValidator()
        self.auth_manager = AuthManager()
        self.cache = MCPCache(CacheConfig())
        
    async def benchmark_standards_engine(self):
        """Benchmark standards engine operations."""
        print("\n📊 Benchmarking Standards Engine...")
        
        # Test context
        context = {
            "project_type": "web_application",
            "framework": "react",
            "language": "javascript",
            "requirements": ["accessibility", "performance"]
        }
        
        # Warm up
        for _ in range(5):
            await self.engine.get_applicable_standards(context)
            
        # Benchmark
        times = []
        for _ in range(100):
            start = time.time()
            result = await self.engine.get_applicable_standards(context)
            elapsed = time.time() - start
            times.append(elapsed)
            
        self.results["standards_engine"] = {
            "operation": "get_applicable_standards",
            "iterations": 100,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000,
            "p95_ms": self._percentile(times, 95) * 1000,
            "p99_ms": self._percentile(times, 99) * 1000
        }
        
        print(f"  ✓ Mean: {self.results['standards_engine']['mean_ms']:.2f}ms")
        print(f"  ✓ P95: {self.results['standards_engine']['p95_ms']:.2f}ms")
        print(f"  ✓ Target: <50ms - {'✅ PASS' if self.results['standards_engine']['mean_ms'] < 50 else '❌ FAIL'}")
        
    def benchmark_validation(self):
        """Benchmark input validation."""
        print("\n📊 Benchmarking Input Validation...")
        
        # Test data
        test_input = {
            "context": {
                "project_type": "web_application",
                "framework": "react",
                "requirements": ["security", "performance"]
            }
        }
        
        # Warm up
        for _ in range(10):
            self.validator.validate_tool_input("get_applicable_standards", test_input)
            
        # Benchmark
        times = []
        for _ in range(1000):
            start = time.time()
            self.validator.validate_tool_input("get_applicable_standards", test_input)
            elapsed = time.time() - start
            times.append(elapsed)
            
        self.results["validation"] = {
            "operation": "validate_tool_input",
            "iterations": 1000,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000
        }
        
        print(f"  ✓ Mean: {self.results['validation']['mean_ms']:.2f}ms")
        print(f"  ✓ Median: {self.results['validation']['median_ms']:.2f}ms")
        print(f"  ✓ Target: <5ms - {'✅ PASS' if self.results['validation']['mean_ms'] < 5 else '❌ FAIL'}")
        
    def benchmark_auth(self):
        """Benchmark authentication operations."""
        print("\n📊 Benchmarking Authentication...")
        
        # Enable auth for testing
        self.auth_manager.config.enabled = True
        self.auth_manager.config.secret_key = "test_secret_key"
        
        # Generate a token
        token = self.auth_manager.generate_token("test_user")
        
        # Warm up
        for _ in range(10):
            self.auth_manager.verify_token(token)
            
        # Benchmark token verification
        times = []
        for _ in range(1000):
            start = time.time()
            is_valid, payload, error = self.auth_manager.verify_token(token)
            elapsed = time.time() - start
            times.append(elapsed)
            assert is_valid
            
        self.results["auth_verification"] = {
            "operation": "verify_token",
            "iterations": 1000,
            "mean_ms": statistics.mean(times) * 1000,
            "median_ms": statistics.median(times) * 1000,
            "min_ms": min(times) * 1000,
            "max_ms": max(times) * 1000
        }
        
        print(f"  ✓ Mean: {self.results['auth_verification']['mean_ms']:.2f}ms")
        print(f"  ✓ Median: {self.results['auth_verification']['median_ms']:.2f}ms")
        print(f"  ✓ Target: <2ms - {'✅ PASS' if self.results['auth_verification']['mean_ms'] < 2 else '❌ FAIL'}")
        
    async def benchmark_cache(self):
        """Benchmark cache operations."""
        print("\n📊 Benchmarking Cache Operations...")
        
        # Test data
        test_key = "test_key"
        test_value = {"data": "x" * 1000, "nested": {"values": list(range(100))}}
        
        # Benchmark cache set
        times_set = []
        for i in range(100):
            key = f"{test_key}_{i}"
            start = time.time()
            await self.cache.set(key, test_value, ttl=300)
            elapsed = time.time() - start
            times_set.append(elapsed)
            
        # Benchmark cache get (hits)
        times_get = []
        for i in range(100):
            key = f"{test_key}_{i}"
            start = time.time()
            result = await self.cache.get(key)
            elapsed = time.time() - start
            times_get.append(elapsed)
            assert result == test_value
            
        self.results["cache"] = {
            "set_operation": {
                "iterations": 100,
                "mean_ms": statistics.mean(times_set) * 1000,
                "median_ms": statistics.median(times_set) * 1000
            },
            "get_operation": {
                "iterations": 100,
                "mean_ms": statistics.mean(times_get) * 1000,
                "median_ms": statistics.median(times_get) * 1000
            }
        }
        
        print(f"  ✓ Set Mean: {self.results['cache']['set_operation']['mean_ms']:.2f}ms")
        print(f"  ✓ Get Mean: {self.results['cache']['get_operation']['mean_ms']:.2f}ms")
        print(f"  ✓ Target: <10ms - {'✅ PASS' if self.results['cache']['get_operation']['mean_ms'] < 10 else '❌ FAIL'}")
        
    def benchmark_concurrent_operations(self):
        """Benchmark concurrent operation handling."""
        print("\n📊 Benchmarking Concurrent Operations...")
        
        async def concurrent_validation(num_concurrent: int):
            """Run concurrent validations."""
            test_input = {
                "context": {
                    "project_type": "api",
                    "language": "python"
                }
            }
            
            async def single_validation():
                return self.validator.validate_tool_input("get_applicable_standards", test_input)
                
            start = time.time()
            tasks = [single_validation() for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks)
            elapsed = time.time() - start
            
            return elapsed, len(results)
            
        # Test different concurrency levels
        concurrency_results = {}
        
        for concurrent in [1, 10, 50, 100]:
            elapsed, count = asyncio.run(concurrent_validation(concurrent))
            throughput = count / elapsed
            
            concurrency_results[concurrent] = {
                "total_time_s": elapsed,
                "throughput_ops": throughput,
                "avg_latency_ms": (elapsed / count) * 1000
            }
            
            print(f"  ✓ {concurrent} concurrent: {throughput:.0f} ops/s, {concurrency_results[concurrent]['avg_latency_ms']:.2f}ms avg")
            
        self.results["concurrency"] = concurrency_results
        
    def _percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile."""
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        if index >= len(sorted_values):
            index = len(sorted_values) - 1
        return sorted_values[index]
        
    def generate_report(self):
        """Generate performance report."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK REPORT")
        print("="*80)
        
        # Check overall performance
        targets_met = True
        
        # Standards engine target: <50ms
        if self.results.get("standards_engine", {}).get("mean_ms", 100) > 50:
            targets_met = False
            
        # Validation target: <5ms
        if self.results.get("validation", {}).get("mean_ms", 10) > 5:
            targets_met = False
            
        # Auth target: <2ms
        if self.results.get("auth_verification", {}).get("mean_ms", 5) > 2:
            targets_met = False
            
        # Cache target: <10ms
        if self.results.get("cache", {}).get("get_operation", {}).get("mean_ms", 20) > 10:
            targets_met = False
            
        print(f"\n📈 Overall Result: {'✅ ALL PERFORMANCE TARGETS MET' if targets_met else '❌ SOME TARGETS MISSED'}")
        
        # Save detailed results
        with open("benchmark_results_simple.json", "w") as f:
            json.dump(self.results, f, indent=2)
            
        print(f"\n📄 Detailed results saved to: benchmark_results_simple.json")
        
    async def run_all(self):
        """Run all benchmarks."""
        print("🚀 Starting MCP Standards Server Performance Benchmarks...")
        
        await self.benchmark_standards_engine()
        self.benchmark_validation()
        self.benchmark_auth()
        await self.benchmark_cache()
        self.benchmark_concurrent_operations()
        
        self.generate_report()


if __name__ == "__main__":
    benchmark = PerformanceBenchmark()
    asyncio.run(benchmark.run_all())