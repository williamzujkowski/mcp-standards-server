"""Demo script showing Redis cache capabilities."""

import asyncio
import time
from typing import Any

from src.core.cache import CacheConfig, RedisCache, cache_result, invalidate_cache
from src.core.cache.integration import CacheMetricsCollector


# Example: Basic cache operations
def demo_basic_operations():
    """Demonstrate basic cache operations."""
    print("\n=== Basic Cache Operations ===")

    # Initialize cache
    cache = RedisCache(CacheConfig(key_prefix="demo"))

    # Set and get
    print("\n1. Simple set/get:")
    cache.set("user:123", {"name": "John Doe", "role": "admin"})
    user = cache.get("user:123")
    print(f"   Retrieved user: {user}")

    # TTL example
    print("\n2. TTL example:")
    cache.set("temp_token", "abc123", ttl=2)
    print(f"   Token (immediate): {cache.get('temp_token')}")
    time.sleep(3)
    print(f"   Token (after 3s): {cache.get('temp_token')}")

    # Batch operations
    print("\n3. Batch operations:")
    users = {
        "user:1": {"name": "Alice"},
        "user:2": {"name": "Bob"},
        "user:3": {"name": "Charlie"},
    }
    cache.mset(users)

    retrieved = cache.mget(["user:1", "user:2", "user:3", "user:4"])
    print(f"   Retrieved users: {retrieved}")

    # Pattern deletion
    print("\n4. Pattern deletion:")
    cache.set("session:123:data", "data1")
    cache.set("session:123:token", "token1")
    cache.set("session:456:data", "data2")

    deleted = cache.delete_pattern("session:123:*")
    print(f"   Deleted {deleted} keys matching 'session:123:*'")
    print(f"   session:123:data exists: {cache.get('session:123:data') is not None}")
    print(f"   session:456:data exists: {cache.get('session:456:data') is not None}")


# Example: Using decorators
def demo_decorators():
    """Demonstrate cache decorators."""
    print("\n=== Cache Decorators ===")

    # Create a cached function
    call_count = 0

    @cache_result("compute", ttl=60)
    def expensive_computation(n: int) -> int:
        nonlocal call_count
        call_count += 1
        print(f"   Computing factorial({n})...")
        time.sleep(0.5)  # Simulate expensive operation

        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    print("\n1. Function caching:")

    # First call - cache miss
    start = time.time()
    result1 = expensive_computation(10)
    time1 = time.time() - start
    print(f"   First call: {result1} (took {time1:.3f}s)")

    # Second call - cache hit
    start = time.time()
    result2 = expensive_computation(10)
    time2 = time.time() - start
    print(f"   Second call: {result2} (took {time2:.3f}s)")
    print(f"   Speedup: {time1/time2:.1f}x")
    print(f"   Function called {call_count} times")

    # Cache invalidation
    print("\n2. Cache invalidation:")

    @cache_result("data", ttl=300)
    def get_data(key: str) -> str:
        return f"data_for_{key}"

    @invalidate_cache(pattern="data:*")
    def update_all_data():
        print("   Updating all data and invalidating cache...")
        return "updated"

    # Cache some data
    print(f"   Initial data: {get_data('key1')}")

    # Update and invalidate
    update_all_data()

    # Data will be recomputed
    print(f"   After update: {get_data('key1')}")


# Example: Async operations
async def demo_async_operations():
    """Demonstrate async cache operations."""
    print("\n=== Async Cache Operations ===")

    cache = RedisCache(CacheConfig(key_prefix="async_demo"))

    # Async decorator
    @cache_result("async_api", ttl=120)
    async def fetch_api_data(endpoint: str) -> dict[str, Any]:
        print(f"   Fetching data from {endpoint}...")
        await asyncio.sleep(1)  # Simulate API call
        return {
            "endpoint": endpoint,
            "data": f"Response from {endpoint}",
            "timestamp": time.time(),
        }

    print("\n1. Async caching:")

    # First call
    start = time.time()
    data1 = await fetch_api_data("/api/users")
    time1 = time.time() - start
    print(f"   First call took {time1:.3f}s")

    # Cached call
    start = time.time()
    data2 = await fetch_api_data("/api/users")
    time2 = time.time() - start
    print(f"   Cached call took {time2:.3f}s")
    print(f"   Same timestamp: {data1['timestamp'] == data2['timestamp']}")

    # Concurrent operations
    print("\n2. Concurrent operations:")

    async def concurrent_test():
        tasks = []
        for i in range(10):
            tasks.append(cache.async_set(f"concurrent_{i}", f"value_{i}"))

        await asyncio.gather(*tasks)

        # Retrieve all
        keys = [f"concurrent_{i}" for i in range(10)]
        results = await asyncio.gather(*[cache.async_get(k) for k in keys])
        return results

    start = time.time()
    results = await concurrent_test()
    elapsed = time.time() - start
    print(f"   Set and retrieved 10 values concurrently in {elapsed:.3f}s")
    print(
        f"   All values correct: {all(results[i] == f'value_{i}' for i in range(10))}"
    )


# Example: Real-world scenario
class StandardsService:
    """Example service using caching."""

    def __init__(self):
        self.cache = RedisCache(CacheConfig(key_prefix="standards"))

    @cache_result("search", ttl=300)
    def search_standards(self, query: str, category: str = None) -> list[dict]:
        """Search standards with caching."""
        print(f"   Searching for '{query}' in category '{category}'...")
        time.sleep(0.5)  # Simulate database query

        # Mock results
        results = [
            {"id": "ISO27001", "name": "Information Security", "score": 0.95},
            {"id": "NIST-CSF", "name": "Cybersecurity Framework", "score": 0.87},
            {"id": "GDPR", "name": "Data Protection", "score": 0.82},
        ]

        if category:
            results = [r for r in results if category.lower() in r["name"].lower()]

        return results

    @cache_result("standard", ttl=3600, include_self=True)
    def get_standard(self, standard_id: str) -> dict:
        """Get standard details with caching."""
        print(f"   Fetching standard {standard_id}...")
        time.sleep(0.3)  # Simulate database query

        return {
            "id": standard_id,
            "name": f"Standard {standard_id}",
            "version": "1.0",
            "requirements": ["REQ-1", "REQ-2", "REQ-3"],
        }

    @invalidate_cache(pattern="standard:*:{standard_id}:*")
    def update_standard(self, standard_id: str, updates: dict):
        """Update standard and invalidate cache."""
        print(f"   Updating standard {standard_id}...")
        # Update logic here
        return True

    def get_metrics(self) -> dict:
        """Get cache metrics."""
        return self.cache.get_metrics()


def demo_real_world():
    """Demonstrate real-world usage."""
    print("\n=== Real-World Example ===")

    service = StandardsService()

    print("\n1. Search caching:")

    # First search - cache miss
    start = time.time()
    results1 = service.search_standards("security")
    time1 = time.time() - start
    print(f"   First search: {len(results1)} results in {time1:.3f}s")

    # Second search - cache hit
    start = time.time()
    results2 = service.search_standards("security")
    time2 = time.time() - start
    print(f"   Cached search: {len(results2)} results in {time2:.3f}s")
    print(f"   Speedup: {time1/time2:.1f}x")

    print("\n2. Standard retrieval:")

    # Get standard - cache miss
    standard1 = service.get_standard("ISO27001")
    print(f"   Retrieved: {standard1['name']}")

    # Get again - cache hit
    standard2 = service.get_standard("ISO27001")
    print(f"   Cached: Same object = {standard1 == standard2}")

    print("\n3. Cache invalidation:")

    # Update standard
    service.update_standard("ISO27001", {"version": "2.0"})

    # Get again - cache miss after invalidation
    standard3 = service.get_standard("ISO27001")
    print(f"   After update: Recomputed = {standard1 != standard3}")

    print("\n4. Cache metrics:")
    metrics = service.get_metrics()
    print(f"   L1 hit rate: {metrics['l1_hit_rate']:.1%}")
    print(f"   L2 hit rate: {metrics['l2_hit_rate']:.1%}")
    print(
        f"   Total operations: {sum(metrics[k] for k in ['l1_hits', 'l1_misses', 'l2_hits', 'l2_misses'])}"
    )


def demo_monitoring():
    """Demonstrate cache monitoring."""
    print("\n=== Cache Monitoring ===")

    cache = RedisCache(CacheConfig(key_prefix="monitor"))

    # Perform some operations
    for i in range(100):
        cache.set(f"key_{i}", f"value_{i}")
        if i % 3 == 0:
            cache.get(f"key_{i}")  # Some hits
        if i % 5 == 0:
            cache.get(f"missing_{i}")  # Some misses

    # Get metrics
    collector = CacheMetricsCollector(cache)
    report = collector.collect_metrics()

    print("\n1. Cache metrics:")
    metrics = report["cache_metrics"]
    print(f"   L1 hits: {metrics['l1_hits']}")
    print(f"   L1 misses: {metrics['l1_misses']}")
    print(f"   L1 hit rate: {metrics['l1_hit_rate']:.1%}")
    print(f"   Cache efficiency: {report['performance']['cache_efficiency']:.1%}")

    print("\n2. Health status:")
    health = report["health"]
    print(f"   Status: {health['status']}")
    print(f"   Redis connected: {health['redis_connected']}")
    print(f"   Latency: {health.get('latency_ms', 'N/A')}ms")
    print(f"   Circuit breaker: {health['circuit_breaker_state']}")


async def main():
    """Run all demos."""
    print("Redis Cache Demo")
    print("=" * 50)

    try:
        # Check Redis connection
        cache = RedisCache()
        health = cache.health_check()

        if not health["redis_connected"]:
            print("ERROR: Redis is not connected. Please start Redis and try again.")
            print("You can start Redis with: redis-server")
            return

        print(f"✓ Redis connected (latency: {health['latency_ms']:.1f}ms)")

        # Run demos
        demo_basic_operations()
        demo_decorators()
        await demo_async_operations()
        demo_real_world()
        demo_monitoring()

        print("\n" + "=" * 50)
        print("Demo completed successfully! ✓")

    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nMake sure Redis is running on localhost:6379")


if __name__ == "__main__":
    asyncio.run(main())
