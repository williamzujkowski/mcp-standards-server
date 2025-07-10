"""Memory usage profiling for MCP components."""

import asyncio
import gc
import sys
import tracemalloc
from pathlib import Path
from typing import Any

import psutil

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark, MemoryProfiler


class MemoryUsageBenchmark(BaseBenchmark):
    """Profile memory usage for each MCP component."""

    def __init__(self, iterations: int = 10):
        super().__init__("MCP Memory Usage Profile", iterations)
        self.memory_profiler = MemoryProfiler(interval=0.1)
        self.component_profiles: dict[str, dict[str, Any]] = {}

    async def setup(self):
        """Setup profiling environment."""
        # Start tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Force garbage collection
        gc.collect()

        # Record baseline memory
        self.baseline_memory = self._get_memory_info()

    async def run_single_iteration(self) -> dict[str, Any]:
        """Profile memory usage for different components."""
        results = {}

        # Test 1: Server initialization
        results["server_init"] = await self._profile_server_init()

        # Test 2: Standards loading
        results["standards_loading"] = await self._profile_standards_loading()

        # Test 3: Search operations
        results["search_ops"] = await self._profile_search_operations()

        # Test 4: Token optimization
        results["token_ops"] = await self._profile_token_operations()

        # Test 5: Concurrent operations
        results["concurrent_ops"] = await self._profile_concurrent_operations()

        return results

    async def _profile_server_init(self) -> dict[str, Any]:
        """Profile memory during server initialization."""
        gc.collect()
        start_memory = self._get_memory_info()

        # Start memory monitoring
        await self.memory_profiler.start()

        # Initialize server
        MCPStandardsServer({
            "search": {"enabled": True},
            "token_model": "gpt-4"
        })

        # Stop monitoring
        await self.memory_profiler.stop()

        end_memory = self._get_memory_info()

        return {
            "memory_increase_mb": end_memory["rss"] - start_memory["rss"],
            "peak_memory_mb": self.memory_profiler.get_summary()["rss"]["peak_mb"],
            "allocations": self._get_top_allocations()
        }

    async def _profile_standards_loading(self) -> dict[str, Any]:
        """Profile memory when loading standards."""
        gc.collect()

        server = MCPStandardsServer({"search": {"enabled": False}})

        # Create test standards of different sizes
        await self._create_test_standards()

        start_memory = self._get_memory_info()
        await self.memory_profiler.start()

        # Load multiple standards
        standards_loaded = []
        for i in range(20):
            try:
                await server._get_standard_details(f"memory-test-{i}")
                standards_loaded.append(f"memory-test-{i}")
            except Exception:
                pass

        await self.memory_profiler.stop()
        end_memory = self._get_memory_info()

        return {
            "standards_loaded": len(standards_loaded),
            "memory_increase_mb": end_memory["rss"] - start_memory["rss"],
            "memory_per_standard_mb": (end_memory["rss"] - start_memory["rss"]) / len(standards_loaded) if standards_loaded else 0,
            "peak_memory_mb": self.memory_profiler.get_summary()["rss"]["peak_mb"]
        }

    async def _profile_search_operations(self) -> dict[str, Any]:
        """Profile memory during search operations."""
        gc.collect()

        server = MCPStandardsServer({
            "search": {"enabled": True},
            "search_model": "sentence-transformers/all-MiniLM-L6-v2"
        })

        start_memory = self._get_memory_info()
        await self.memory_profiler.start()

        # Perform various searches
        searches = [
            "security authentication",
            "react performance optimization",
            "microservices architecture patterns",
            "testing strategies unit integration",
            "deployment kubernetes docker"
        ]

        for query in searches:
            await server._search_standards(query, limit=20)

        await self.memory_profiler.stop()
        end_memory = self._get_memory_info()

        return {
            "searches_performed": len(searches),
            "memory_increase_mb": end_memory["rss"] - start_memory["rss"],
            "search_index_size_mb": self._estimate_search_index_size(server),
            "peak_memory_mb": self.memory_profiler.get_summary()["rss"]["peak_mb"]
        }

    async def _profile_token_operations(self) -> dict[str, Any]:
        """Profile memory during token optimization."""
        gc.collect()

        server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })

        # Create large standard for testing
        await self._create_large_standard()

        start_memory = self._get_memory_info()
        await self.memory_profiler.start()

        # Test different optimization formats
        formats = ["full", "condensed", "summary", "reference"]

        for fmt in formats:
            await server._get_optimized_standard(
                "large-memory-test",
                format_type=fmt,
                token_budget=5000
            )

        await self.memory_profiler.stop()
        end_memory = self._get_memory_info()

        return {
            "formats_tested": len(formats),
            "memory_increase_mb": end_memory["rss"] - start_memory["rss"],
            "peak_memory_mb": self.memory_profiler.get_summary()["rss"]["peak_mb"],
            "avg_memory_per_format_mb": (end_memory["rss"] - start_memory["rss"]) / len(formats)
        }

    async def _profile_concurrent_operations(self) -> dict[str, Any]:
        """Profile memory under concurrent load."""
        gc.collect()

        server = MCPStandardsServer({"search": {"enabled": False}})

        start_memory = self._get_memory_info()
        await self.memory_profiler.start()

        # Create concurrent tasks
        tasks = []
        for i in range(10):
            if i % 3 == 0:
                task = server._list_available_standards(limit=20)
            elif i % 3 == 1:
                task = server._get_sync_status()
            else:
                task = server._get_applicable_standards({"language": "python"})

            tasks.append(asyncio.create_task(task))

        # Wait for all tasks
        await asyncio.gather(*tasks, return_exceptions=True)

        await self.memory_profiler.stop()
        end_memory = self._get_memory_info()

        return {
            "concurrent_tasks": len(tasks),
            "memory_increase_mb": end_memory["rss"] - start_memory["rss"],
            "peak_memory_mb": self.memory_profiler.get_summary()["rss"]["peak_mb"],
            "memory_per_task_mb": (end_memory["rss"] - start_memory["rss"]) / len(tasks)
        }

    def _get_memory_info(self) -> dict[str, float]:
        """Get current memory information."""
        process = psutil.Process()
        mem_info = process.memory_info()

        return {
            "rss": mem_info.rss / 1024 / 1024,  # MB
            "vms": mem_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        }

    def _get_top_allocations(self, limit: int = 5) -> list[dict[str, Any]]:
        """Get top memory allocations."""
        if not tracemalloc.is_tracing():
            return []

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        allocations = []
        for stat in top_stats[:limit]:
            allocations.append({
                "file": stat.traceback.format()[0],
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            })

        return allocations

    def _estimate_search_index_size(self, server) -> float:
        """Estimate search index size in memory."""
        # This is a rough estimate
        if hasattr(server, 'search') and server.search:
            # Estimate based on loaded embeddings
            return 50.0  # Placeholder
        return 0.0

    async def _create_test_standards(self):
        """Create test standards of varying sizes."""
        import json
        from pathlib import Path

        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        for i in range(20):
            # Vary the size
            content_size = 100 * (i + 1)  # Increasing size

            standard = {
                "id": f"memory-test-{i}",
                "name": f"Memory Test Standard {i}",
                "category": "test",
                "content": {
                    "overview": "x" * content_size,
                    "guidelines": ["y" * 50 for _ in range(i + 1)],
                    "examples": ["z" * 100 for _ in range(i)]
                }
            }

            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)

    async def _create_large_standard(self):
        """Create a large standard for memory testing."""
        import json
        from pathlib import Path

        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Create a ~10MB standard
        large_standard = {
            "id": "large-memory-test",
            "name": "Large Memory Test Standard",
            "category": "test",
            "content": {
                "overview": "x" * 100000,  # 100KB
                "guidelines": ["y" * 10000 for _ in range(50)],  # 5MB
                "examples": ["z" * 10000 for _ in range(50)],  # 5MB
                "details": {
                    "section1": "a" * 50000,
                    "section2": "b" * 50000,
                    "section3": "c" * 50000,
                }
            }
        }

        filepath = cache_dir / f"{large_standard['id']}.json"
        with open(filepath, 'w') as f:
            json.dump(large_standard, f)

    async def teardown(self):
        """Cleanup and generate report."""
        # Stop tracemalloc
        if tracemalloc.is_tracing():
            tracemalloc.stop()

        # Force garbage collection
        gc.collect()

        # Generate memory report
        self.memory_report = self._generate_memory_report()

    def _generate_memory_report(self) -> dict[str, Any]:
        """Generate comprehensive memory report."""
        report = {
            "baseline_memory_mb": self.baseline_memory["rss"],
            "components": self.component_profiles,
            "recommendations": []
        }

        # Add recommendations based on findings
        total_memory = sum(
            profile.get("memory_increase_mb", 0)
            for profile in self.component_profiles.values()
        )

        if total_memory > 100:
            report["recommendations"].append(
                "High memory usage detected - consider implementing memory limits"
            )

        return report
