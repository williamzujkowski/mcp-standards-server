"""Memory leak detection for MCP server."""

import asyncio
import gc
import tracemalloc
import weakref
from typing import Any, Dict, List, Optional, Set

import psutil

from src.mcp_server import MCPStandardsServer
from ..framework import BaseBenchmark, MemoryProfiler


class LeakDetectionBenchmark(BaseBenchmark):
    """Detect memory leaks in MCP server operations."""
    
    def __init__(self, iterations: int = 100):
        super().__init__("MCP Memory Leak Detection", iterations)
        self.memory_profiler = MemoryProfiler(interval=0.5)
        self.object_tracker = ObjectTracker()
        self.leak_candidates: List[Dict[str, Any]] = []
        
    async def setup(self):
        """Setup leak detection."""
        # Start tracemalloc with more frames
        tracemalloc.start(10)
        
        # Force initial garbage collection
        gc.collect()
        
        # Take baseline snapshot
        self.baseline_snapshot = tracemalloc.take_snapshot()
        self.baseline_memory = self._get_memory_info()
        
        # Create test data
        await self._create_test_data()
    
    async def run_single_iteration(self) -> Dict[str, Any]:
        """Run operations that might leak memory."""
        # Track objects before operations
        self.object_tracker.snapshot("before")
        
        # Run various operations
        await self._test_server_lifecycle()
        await self._test_repeated_operations()
        await self._test_error_scenarios()
        await self._test_concurrent_operations()
        
        # Force garbage collection
        gc.collect()
        
        # Track objects after operations
        self.object_tracker.snapshot("after")
        
        # Check for leaks
        leaks = self._check_for_leaks()
        
        return {
            "memory_growth_mb": self._get_memory_info()["rss"] - self.baseline_memory["rss"],
            "leaked_objects": len(leaks),
            "gc_stats": gc.get_stats()[-1] if gc.get_stats() else {}
        }
    
    async def _test_server_lifecycle(self):
        """Test server creation and destruction."""
        servers = []
        
        # Create multiple servers
        for i in range(5):
            server = MCPStandardsServer({
                "search": {"enabled": False},
                "token_model": "gpt-4"
            })
            servers.append(weakref.ref(server))
            
            # Perform some operations
            await server._get_sync_status()
            await server._list_available_standards(limit=10)
        
        # Delete servers
        servers.clear()
        gc.collect()
    
    async def _test_repeated_operations(self):
        """Test repeated operations for memory growth."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Repeat operations many times
        for i in range(20):
            # Get standard details
            try:
                await server._get_standard_details(f"leak-test-{i % 5}")
            except:
                pass
            
            # List standards
            await server._list_available_standards(limit=50)
            
            # Get applicable standards
            await server._get_applicable_standards({
                "language": "python",
                "framework": "fastapi"
            })
            
            # Token optimization
            try:
                await server._get_optimized_standard(
                    f"leak-test-{i % 5}",
                    format_type="condensed"
                )
            except:
                pass
    
    async def _test_error_scenarios(self):
        """Test error handling for memory leaks."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Test various error scenarios
        error_tests = [
            # Non-existent standard
            lambda: server._get_standard_details("non-existent-standard"),
            # Invalid parameters
            lambda: server._search_standards("", limit=-1),
            # Invalid format
            lambda: server._get_optimized_standard("test", format_type="invalid"),
            # Large token budget
            lambda: server._estimate_token_usage(["test"] * 1000),
        ]
        
        for test in error_tests:
            try:
                await test()
            except Exception:
                # Expected - we're testing error handling
                pass
    
    async def _test_concurrent_operations(self):
        """Test concurrent operations for memory leaks."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Create many concurrent tasks
        tasks = []
        for i in range(50):
            task = asyncio.create_task(
                server._get_applicable_standards({"language": f"lang-{i}"})
            )
            tasks.append(task)
        
        # Wait with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=10.0
            )
        except asyncio.TimeoutError:
            # Cancel remaining tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
    
    def _check_for_leaks(self) -> List[Dict[str, Any]]:
        """Check for memory leaks."""
        leaks = []
        
        # Check object growth
        growth = self.object_tracker.get_growth("before", "after")
        for type_name, count in growth.items():
            if count > 10:  # Threshold for leak detection
                leaks.append({
                    "type": type_name,
                    "growth": count,
                    "severity": "high" if count > 100 else "medium"
                })
        
        # Check tracemalloc differences
        current_snapshot = tracemalloc.take_snapshot()
        top_stats = current_snapshot.compare_to(self.baseline_snapshot, 'lineno')
        
        for stat in top_stats[:10]:
            if stat.size_diff > 1024 * 1024:  # 1MB threshold
                leaks.append({
                    "location": str(stat),
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count_diff": stat.count_diff
                })
        
        self.leak_candidates.extend(leaks)
        return leaks
    
    def _get_memory_info(self) -> Dict[str, float]:
        """Get current memory information."""
        process = psutil.Process()
        mem_info = process.memory_info()
        
        return {
            "rss": mem_info.rss / 1024 / 1024,
            "vms": mem_info.vms / 1024 / 1024
        }
    
    async def _create_test_data(self):
        """Create test standards."""
        import json
        from pathlib import Path
        
        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            standard = {
                "id": f"leak-test-{i}",
                "name": f"Leak Test Standard {i}",
                "category": "test",
                "content": {
                    "overview": f"Test standard {i}",
                    "guidelines": [f"Guideline {j}" for j in range(10)]
                }
            }
            
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)
    
    async def teardown(self):
        """Analyze leak detection results."""
        # Stop tracemalloc
        tracemalloc.stop()
        
        # Final garbage collection
        gc.collect()
        
        # Generate leak report
        self.leak_report = self._generate_leak_report()
    
    def _generate_leak_report(self) -> Dict[str, Any]:
        """Generate comprehensive leak report."""
        # Analyze leak patterns
        leak_types = {}
        for leak in self.leak_candidates:
            leak_type = leak.get("type") or "memory"
            if leak_type not in leak_types:
                leak_types[leak_type] = []
            leak_types[leak_type].append(leak)
        
        # Calculate total memory growth
        final_memory = self._get_memory_info()
        total_growth = final_memory["rss"] - self.baseline_memory["rss"]
        
        report = {
            "total_memory_growth_mb": total_growth,
            "growth_per_iteration_mb": total_growth / self.iterations if self.iterations > 0 else 0,
            "leak_types": leak_types,
            "leak_detected": total_growth > 10,  # 10MB threshold
            "recommendations": self._generate_recommendations(leak_types, total_growth)
        }
        
        return report
    
    def _generate_recommendations(self, leak_types: Dict[str, List], growth: float) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if growth > 10:
            recommendations.append("Significant memory growth detected - investigate memory management")
        
        if "MCPStandardsServer" in leak_types:
            recommendations.append("Server instances may not be properly cleaned up")
        
        if "dict" in leak_types or "list" in leak_types:
            recommendations.append("Consider clearing caches and temporary data structures")
        
        if growth > 50:
            recommendations.append("Critical: Memory leak detected - immediate action required")
        
        return recommendations


class ObjectTracker:
    """Track object creation and destruction."""
    
    def __init__(self):
        self.snapshots: Dict[str, Dict[str, int]] = {}
    
    def snapshot(self, name: str):
        """Take a snapshot of current objects."""
        gc.collect()
        
        counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            counts[obj_type] = counts.get(obj_type, 0) + 1
        
        self.snapshots[name] = counts
    
    def get_growth(self, before: str, after: str) -> Dict[str, int]:
        """Get object growth between snapshots."""
        if before not in self.snapshots or after not in self.snapshots:
            return {}
        
        before_counts = self.snapshots[before]
        after_counts = self.snapshots[after]
        
        growth = {}
        
        for obj_type in after_counts:
            before_count = before_counts.get(obj_type, 0)
            after_count = after_counts[obj_type]
            
            if after_count > before_count:
                growth[obj_type] = after_count - before_count
        
        return growth