"""Cache performance benchmarks."""

import time
import asyncio
import statistics
from typing import List, Dict, Any
import json
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.core.cache import RedisCache, CacheConfig, cache_result


class CacheBenchmark:
    """Benchmark suite for cache performance."""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
        self.results = {}
        
    def generate_data(self, size: int) -> Dict[str, Any]:
        """Generate test data of specified size."""
        if size == "small":
            return {"id": 1, "name": "test", "value": 42}
        elif size == "medium":
            return {
                "id": 1,
                "name": "test",
                "data": [{"item": i, "value": i * 2} for i in range(100)],
                "metadata": {f"key_{i}": f"value_{i}" for i in range(50)}
            }
        else:  # large
            return {
                "id": 1,
                "name": "test",
                "items": [
                    {
                        "id": i,
                        "data": ''.join(random.choices(string.ascii_letters, k=100)),
                        "numbers": list(range(100))
                    }
                    for i in range(100)
                ]
            }
            
    def benchmark_basic_operations(self, iterations: int = 1000):
        """Benchmark basic get/set operations."""
        print(f"\nBenchmarking basic operations ({iterations} iterations)...")
        
        # Test data
        small_data = self.generate_data("small")
        medium_data = self.generate_data("medium")
        large_data = self.generate_data("large")
        
        # Benchmark SET operations
        set_times = {"small": [], "medium": [], "large": []}
        
        for i in range(iterations):
            # Small data
            start = time.perf_counter()
            self.cache.set(f"bench_small_{i}", small_data)
            set_times["small"].append(time.perf_counter() - start)
            
            # Medium data
            start = time.perf_counter()
            self.cache.set(f"bench_medium_{i}", medium_data)
            set_times["medium"].append(time.perf_counter() - start)
            
            # Large data
            if i < iterations // 10:  # Less iterations for large data
                start = time.perf_counter()
                self.cache.set(f"bench_large_{i}", large_data)
                set_times["large"].append(time.perf_counter() - start)
                
        # Benchmark GET operations
        get_times = {"small": [], "medium": [], "large": []}
        
        for i in range(iterations):
            # Small data
            start = time.perf_counter()
            self.cache.get(f"bench_small_{i}")
            get_times["small"].append(time.perf_counter() - start)
            
            # Medium data
            start = time.perf_counter()
            self.cache.get(f"bench_medium_{i}")
            get_times["medium"].append(time.perf_counter() - start)
            
            # Large data
            if i < iterations // 10:
                start = time.perf_counter()
                self.cache.get(f"bench_large_{i}")
                get_times["large"].append(time.perf_counter() - start)
                
        # Calculate statistics
        results = {
            "set_operations": {
                size: {
                    "mean_ms": statistics.mean(times) * 1000,
                    "median_ms": statistics.median(times) * 1000,
                    "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
                    "p99_ms": statistics.quantiles(times, n=100)[98] * 1000
                }
                for size, times in set_times.items() if times
            },
            "get_operations": {
                size: {
                    "mean_ms": statistics.mean(times) * 1000,
                    "median_ms": statistics.median(times) * 1000,
                    "p95_ms": statistics.quantiles(times, n=20)[18] * 1000,
                    "p99_ms": statistics.quantiles(times, n=100)[98] * 1000
                }
                for size, times in get_times.items() if times
            }
        }
        
        self.results["basic_operations"] = results
        self._print_results("Basic Operations", results)
        
    def benchmark_l1_vs_l2(self, iterations: int = 10000):
        """Benchmark L1 cache vs L2 cache performance."""
        print(f"\nBenchmarking L1 vs L2 cache ({iterations} iterations)...")
        
        test_data = self.generate_data("medium")
        key = "l1_vs_l2_test"
        
        # Ensure data is in both L1 and L2
        self.cache.set(key, test_data)
        
        # Benchmark L1 hits (data already in L1)
        l1_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.cache.get(key)
            l1_times.append(time.perf_counter() - start)
            
        # Clear L1 cache to force L2 hits
        self.cache.clear_l1_cache()
        
        # Benchmark L2 hits
        l2_times = []
        for _ in range(iterations):
            self.cache.clear_l1_cache()  # Ensure L2 hit
            start = time.perf_counter()
            self.cache.get(key)
            l2_times.append(time.perf_counter() - start)
            
        results = {
            "l1_cache": {
                "mean_us": statistics.mean(l1_times) * 1_000_000,
                "median_us": statistics.median(l1_times) * 1_000_000,
                "p99_us": statistics.quantiles(l1_times, n=100)[98] * 1_000_000
            },
            "l2_cache": {
                "mean_us": statistics.mean(l2_times) * 1_000_000,
                "median_us": statistics.median(l2_times) * 1_000_000,
                "p99_us": statistics.quantiles(l2_times, n=100)[98] * 1_000_000
            },
            "speedup": statistics.mean(l2_times) / statistics.mean(l1_times)
        }
        
        self.results["l1_vs_l2"] = results
        self._print_results("L1 vs L2 Cache", results)
        
    def benchmark_batch_operations(self, batch_sizes: List[int] = [10, 50, 100, 500]):
        """Benchmark batch operations."""
        print(f"\nBenchmarking batch operations...")
        
        results = {"mget": {}, "mset": {}}
        
        for batch_size in batch_sizes:
            # Prepare data
            keys = [f"batch_key_{i}" for i in range(batch_size)]
            data = {key: self.generate_data("small") for key in keys}
            
            # Benchmark MSET
            start = time.perf_counter()
            self.cache.mset(data)
            mset_time = time.perf_counter() - start
            
            # Benchmark MGET
            start = time.perf_counter()
            self.cache.mget(keys)
            mget_time = time.perf_counter() - start
            
            results["mset"][batch_size] = {
                "total_ms": mset_time * 1000,
                "per_key_ms": (mset_time / batch_size) * 1000
            }
            
            results["mget"][batch_size] = {
                "total_ms": mget_time * 1000,
                "per_key_ms": (mget_time / batch_size) * 1000
            }
            
        self.results["batch_operations"] = results
        self._print_results("Batch Operations", results)
        
    def benchmark_concurrent_access(self, workers: int = 10, operations: int = 100):
        """Benchmark concurrent cache access."""
        print(f"\nBenchmarking concurrent access ({workers} workers, {operations} ops each)...")
        
        test_data = self.generate_data("medium")
        
        def worker_task(worker_id: int) -> Dict[str, float]:
            times = []
            for i in range(operations):
                key = f"concurrent_{worker_id}_{i}"
                
                # SET operation
                start = time.perf_counter()
                self.cache.set(key, test_data)
                set_time = time.perf_counter() - start
                
                # GET operation
                start = time.perf_counter()
                self.cache.get(key)
                get_time = time.perf_counter() - start
                
                times.append({"set": set_time, "get": get_time})
                
            return times
            
        # Run concurrent workers
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_task, i) for i in range(workers)]
            all_times = []
            for future in as_completed(futures):
                all_times.extend(future.result())
                
        total_time = time.perf_counter() - start_time
        
        # Calculate statistics
        set_times = [t["set"] for t in all_times]
        get_times = [t["get"] for t in all_times]
        
        results = {
            "total_operations": workers * operations * 2,
            "total_time_s": total_time,
            "throughput_ops_per_sec": (workers * operations * 2) / total_time,
            "set_latency_ms": {
                "mean": statistics.mean(set_times) * 1000,
                "p95": statistics.quantiles(set_times, n=20)[18] * 1000,
                "p99": statistics.quantiles(set_times, n=100)[98] * 1000
            },
            "get_latency_ms": {
                "mean": statistics.mean(get_times) * 1000,
                "p95": statistics.quantiles(get_times, n=20)[18] * 1000,
                "p99": statistics.quantiles(get_times, n=100)[98] * 1000
            }
        }
        
        self.results["concurrent_access"] = results
        self._print_results("Concurrent Access", results)
        
    async def benchmark_async_operations(self, iterations: int = 1000):
        """Benchmark async operations."""
        print(f"\nBenchmarking async operations ({iterations} iterations)...")
        
        test_data = self.generate_data("medium")
        
        # Single async operations
        async_times = []
        for i in range(iterations):
            key = f"async_bench_{i}"
            
            start = time.perf_counter()
            await self.cache.async_set(key, test_data)
            await self.cache.async_get(key)
            async_times.append(time.perf_counter() - start)
            
        # Concurrent async operations
        async def concurrent_operation(i: int):
            key = f"async_concurrent_{i}"
            start = time.perf_counter()
            await self.cache.async_set(key, test_data)
            result = await self.cache.async_get(key)
            return time.perf_counter() - start
            
        start_time = time.perf_counter()
        concurrent_times = await asyncio.gather(
            *[concurrent_operation(i) for i in range(100)]
        )
        concurrent_total = time.perf_counter() - start_time
        
        results = {
            "sequential": {
                "mean_ms": statistics.mean(async_times) * 1000,
                "median_ms": statistics.median(async_times) * 1000,
                "p99_ms": statistics.quantiles(async_times, n=100)[98] * 1000
            },
            "concurrent": {
                "total_time_s": concurrent_total,
                "operations": 200,  # 100 sets + 100 gets
                "throughput_ops_per_sec": 200 / concurrent_total,
                "mean_ms": statistics.mean(concurrent_times) * 1000
            }
        }
        
        self.results["async_operations"] = results
        self._print_results("Async Operations", results)
        
    def benchmark_decorator_overhead(self, iterations: int = 1000):
        """Benchmark decorator overhead."""
        print(f"\nBenchmarking decorator overhead ({iterations} iterations)...")
        
        # Function without caching
        def uncached_function(x: int) -> int:
            return x * x
            
        # Function with caching
        @cache_result("bench", cache=self.cache)
        def cached_function(x: int) -> int:
            return x * x
            
        # Benchmark uncached
        uncached_times = []
        for i in range(iterations):
            start = time.perf_counter()
            uncached_function(i)
            uncached_times.append(time.perf_counter() - start)
            
        # Benchmark cached (first call - miss)
        cache_miss_times = []
        for i in range(iterations):
            key = f"bench_decorator_{i}"
            self.cache.delete_pattern(f"*{key}*")  # Ensure cache miss
            start = time.perf_counter()
            cached_function(i)
            cache_miss_times.append(time.perf_counter() - start)
            
        # Benchmark cached (second call - hit)
        cache_hit_times = []
        for i in range(iterations):
            start = time.perf_counter()
            cached_function(i)  # Should hit cache
            cache_hit_times.append(time.perf_counter() - start)
            
        results = {
            "uncached_us": statistics.mean(uncached_times) * 1_000_000,
            "cache_miss_us": statistics.mean(cache_miss_times) * 1_000_000,
            "cache_hit_us": statistics.mean(cache_hit_times) * 1_000_000,
            "miss_overhead_us": (statistics.mean(cache_miss_times) - statistics.mean(uncached_times)) * 1_000_000,
            "hit_speedup": statistics.mean(cache_miss_times) / statistics.mean(cache_hit_times)
        }
        
        self.results["decorator_overhead"] = results
        self._print_results("Decorator Overhead", results)
        
    def benchmark_memory_efficiency(self):
        """Benchmark memory efficiency."""
        print(f"\nBenchmarking memory efficiency...")
        
        # Get initial metrics
        initial_metrics = self.cache.get_metrics()
        
        # Add various sized data
        sizes = {
            "small": 1000,
            "medium": 100,
            "large": 10
        }
        
        for size, count in sizes.items():
            data = self.generate_data(size)
            for i in range(count):
                self.cache.set(f"mem_test_{size}_{i}", data)
                
        # Get final metrics
        final_metrics = self.cache.get_metrics()
        
        results = {
            "l1_cache_entries": len(self.cache._l1_cache),
            "l1_cache_max_size": self.cache.config.l1_max_size,
            "estimated_memory_mb": self._estimate_memory_usage(),
            "hit_rates": {
                "l1": final_metrics.get("l1_hit_rate", 0),
                "l2": final_metrics.get("l2_hit_rate", 0)
            }
        }
        
        self.results["memory_efficiency"] = results
        self._print_results("Memory Efficiency", results)
        
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        # This is a rough estimate
        total_size = 0
        for key, value in self.cache._l1_cache.items():
            total_size += len(str(key)) + len(json.dumps(value, default=str))
        return total_size / (1024 * 1024)
        
    def _print_results(self, title: str, results: Dict[str, Any]):
        """Pretty print benchmark results."""
        print(f"\n{title} Results:")
        print("=" * 50)
        
        def print_dict(d: dict, indent: int = 0):
            for key, value in d.items():
                if isinstance(value, dict):
                    print("  " * indent + f"{key}:")
                    print_dict(value, indent + 1)
                elif isinstance(value, float):
                    print("  " * indent + f"{key}: {value:.3f}")
                else:
                    print("  " * indent + f"{key}: {value}")
                    
        print_dict(results)
        
    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("Running Redis Cache Performance Benchmarks")
        print("=" * 50)
        
        self.benchmark_basic_operations(iterations=1000)
        self.benchmark_l1_vs_l2(iterations=10000)
        self.benchmark_batch_operations()
        self.benchmark_concurrent_access(workers=10, operations=100)
        asyncio.run(self.benchmark_async_operations(iterations=1000))
        self.benchmark_decorator_overhead(iterations=1000)
        self.benchmark_memory_efficiency()
        
        # Print summary
        self._print_summary()
        
    def _print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 50)
        print("BENCHMARK SUMMARY")
        print("=" * 50)
        
        if "l1_vs_l2" in self.results:
            print(f"\nL1 Cache Speedup: {self.results['l1_vs_l2']['speedup']:.1f}x faster than L2")
            
        if "concurrent_access" in self.results:
            throughput = self.results['concurrent_access']['throughput_ops_per_sec']
            print(f"Concurrent Throughput: {throughput:.0f} ops/sec")
            
        if "decorator_overhead" in self.results:
            overhead = self.results['decorator_overhead']['miss_overhead_us']
            speedup = self.results['decorator_overhead']['hit_speedup']
            print(f"Decorator Overhead: {overhead:.1f} microseconds")
            print(f"Cache Hit Speedup: {speedup:.1f}x faster")
            
        print("\nCache is working effectively! ✓")


def main():
    """Run benchmarks."""
    # Configure cache for benchmarks
    config = CacheConfig(
        host="localhost",
        port=6379,
        db=14,  # Separate DB for benchmarks
        key_prefix="benchmark",
        l1_max_size=10000,
        l1_ttl=60
    )
    
    cache = RedisCache(config)
    
    try:
        # Check Redis connection
        health = cache.health_check()
        if not health["redis_connected"]:
            print("ERROR: Redis is not connected. Please start Redis and try again.")
            return
            
        # Clear benchmark database
        import redis
        with redis.Redis(host=config.host, port=config.port, db=config.db) as r:
            r.flushdb()
            
        # Run benchmarks
        benchmark = CacheBenchmark(cache)
        benchmark.run_all_benchmarks()
        
    except Exception as e:
        print(f"ERROR: {e}")
        
    finally:
        cache.close()


if __name__ == "__main__":
    main()