"""Latency distribution analysis for MCP tools."""

import asyncio
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.mcp_server import MCPStandardsServer
from ..framework import BaseBenchmark


class MCPLatencyBenchmark(BaseBenchmark):
    """Analyze latency distribution for MCP tools."""
    
    def __init__(self, iterations: int = 1000):
        super().__init__("MCP Latency Distribution", iterations)
        self.server: MCPStandardsServer = None
        self.latencies_by_tool: Dict[str, List[float]] = defaultdict(list)
        self.latency_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        
    async def setup(self):
        """Setup MCP server and test data."""
        self.server = MCPStandardsServer({
            "search": {"enabled": True},
            "token_model": "gpt-4"
        })
        
        # Clear latencies
        self.latencies_by_tool.clear()
        
        # Setup test data
        await self._setup_test_data()
    
    async def _setup_test_data(self):
        """Create test standards with varying sizes."""
        import json
        
        cache_dir = self.server.synchronizer.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Small standard
        small_standard = {
            "id": "small-standard",
            "name": "Small Standard",
            "category": "test",
            "content": {"overview": "Small content"}
        }
        
        # Medium standard
        medium_standard = {
            "id": "medium-standard",
            "name": "Medium Standard",
            "category": "test",
            "content": {
                "overview": "Medium content " * 100,
                "guidelines": ["Guideline " * 50 for _ in range(10)]
            }
        }
        
        # Large standard
        large_standard = {
            "id": "large-standard",
            "name": "Large Standard",
            "category": "test",
            "content": {
                "overview": "Large content " * 1000,
                "guidelines": ["Guideline " * 100 for _ in range(50)],
                "examples": ["Example " * 200 for _ in range(20)]
            }
        }
        
        for standard in [small_standard, medium_standard, large_standard]:
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)
    
    async def run_single_iteration(self) -> Dict[str, Any]:
        """Measure latency for different operations."""
        # Define test operations
        operations = [
            ("get_standard_small", self._test_get_standard_small),
            ("get_standard_medium", self._test_get_standard_medium),
            ("get_standard_large", self._test_get_standard_large),
            ("search_simple", self._test_search_simple),
            ("search_complex", self._test_search_complex),
            ("list_standards", self._test_list_standards),
            ("token_optimize", self._test_token_optimization),
            ("get_applicable", self._test_get_applicable_standards),
        ]
        
        # Run each operation and measure latency
        for op_name, op_func in operations:
            start = time.perf_counter()
            try:
                await op_func()
                end = time.perf_counter()
                latency = end - start
                self.latencies_by_tool[op_name].append(latency)
            except Exception as e:
                # Record error as max latency
                self.latencies_by_tool[op_name].append(1.0)
        
        # Return current iteration metrics
        return {
            "iteration_complete": True,
            "operations_tested": len(operations)
        }
    
    async def _test_get_standard_small(self):
        """Test getting a small standard."""
        await self.server._get_standard_details("small-standard")
    
    async def _test_get_standard_medium(self):
        """Test getting a medium standard."""
        await self.server._get_standard_details("medium-standard")
    
    async def _test_get_standard_large(self):
        """Test getting a large standard."""
        await self.server._get_standard_details("large-standard")
    
    async def _test_search_simple(self):
        """Test simple search."""
        await self.server._search_standards("test", limit=5)
    
    async def _test_search_complex(self):
        """Test complex search."""
        await self.server._search_standards(
            "security authentication oauth jwt token",
            limit=20,
            min_relevance=0.5
        )
    
    async def _test_list_standards(self):
        """Test listing standards."""
        await self.server._list_available_standards(limit=10)
    
    async def _test_token_optimization(self):
        """Test token optimization."""
        await self.server._get_optimized_standard(
            "medium-standard",
            format_type="condensed",
            token_budget=1000
        )
    
    async def _test_get_applicable_standards(self):
        """Test getting applicable standards."""
        await self.server._get_applicable_standards({
            "language": "javascript",
            "framework": "react",
            "project_type": "web_app"
        })
    
    async def teardown(self):
        """Analyze latency distribution."""
        # Calculate distribution metrics
        self.latency_distribution = self._calculate_distributions()
    
    def _calculate_distributions(self) -> Dict[str, Any]:
        """Calculate latency distributions for all operations."""
        from ..framework.stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()
        
        distributions = {}
        
        for op_name, latencies in self.latencies_by_tool.items():
            if not latencies:
                continue
            
            # Basic statistics
            dist = {
                "count": len(latencies),
                "min": min(latencies),
                "max": max(latencies),
                "mean": stats.mean(latencies),
                "median": stats.median(latencies),
                "std_dev": stats.std_dev(latencies),
                "percentiles": stats.percentiles(latencies, [50, 90, 95, 99, 99.9]),
            }
            
            # Histogram buckets
            bucket_counts = defaultdict(int)
            for latency in latencies:
                for bucket in self.latency_buckets:
                    if latency <= bucket:
                        bucket_counts[bucket] += 1
                        break
                else:
                    bucket_counts["inf"] += 1
            
            dist["histogram"] = dict(bucket_counts)
            
            # Calculate SLO compliance
            dist["slo_compliance"] = {
                "under_10ms": sum(1 for l in latencies if l < 0.01) / len(latencies),
                "under_50ms": sum(1 for l in latencies if l < 0.05) / len(latencies),
                "under_100ms": sum(1 for l in latencies if l < 0.1) / len(latencies),
                "under_1s": sum(1 for l in latencies if l < 1.0) / len(latencies),
            }
            
            distributions[op_name] = dist
        
        return distributions
    
    def get_latency_summary(self) -> str:
        """Generate a summary of latency characteristics."""
        if not hasattr(self, 'latency_distribution'):
            return "No latency data available"
        
        lines = ["# Latency Distribution Summary\n"]
        
        for op_name, dist in self.latency_distribution.items():
            lines.append(f"\n## {op_name}")
            lines.append(f"- Samples: {dist['count']}")
            lines.append(f"- Mean: {dist['mean']*1000:.2f}ms")
            lines.append(f"- Median: {dist['median']*1000:.2f}ms")
            lines.append(f"- P95: {dist['percentiles'][95]*1000:.2f}ms")
            lines.append(f"- P99: {dist['percentiles'][99]*1000:.2f}ms")
            lines.append(f"- SLO (<100ms): {dist['slo_compliance']['under_100ms']*100:.1f}%")
        
        return "\n".join(lines)


class MCPLatencyUnderLoadBenchmark(BaseBenchmark):
    """Analyze how latency changes under different load conditions."""
    
    def __init__(
        self,
        load_levels: List[int] = [1, 5, 10, 20, 50],
        duration_per_level: int = 30
    ):
        super().__init__("MCP Latency Under Load", len(load_levels))
        self.load_levels = load_levels
        self.duration_per_level = duration_per_level
        self.server: MCPStandardsServer = None
        self.results_by_load: Dict[int, Dict[str, List[float]]] = {}
        
    async def setup(self):
        """Setup server."""
        self.server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })
        
        # Setup test data
        await self._setup_test_data()
    
    async def _setup_test_data(self):
        """Create test data."""
        import json
        
        cache_dir = self.server.synchronizer.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a standard test set
        for i in range(10):
            standard = {
                "id": f"load-test-{i}",
                "name": f"Load Test Standard {i}",
                "category": "test",
                "content": {
                    "overview": f"Content for standard {i}" * 50,
                    "guidelines": [f"Guideline {j}" for j in range(20)]
                }
            }
            
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)
    
    async def run_single_iteration(self) -> Dict[str, Any]:
        """Run latency test at current load level."""
        # Get current load level
        current_load = self.load_levels[len(self.results_by_load)]
        
        print(f"\nTesting latency with {current_load} concurrent requests...")
        
        # Collect latencies
        latencies_by_operation = defaultdict(list)
        
        # Create concurrent tasks
        tasks = []
        for _ in range(current_load):
            task = asyncio.create_task(
                self._measure_operations(latencies_by_operation)
            )
            tasks.append(task)
        
        # Run for duration
        end_time = time.time() + self.duration_per_level
        
        while time.time() < end_time:
            await asyncio.sleep(0.1)
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Store results
        self.results_by_load[current_load] = dict(latencies_by_operation)
        
        # Return summary for this load level
        from ..framework.stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()
        
        summary = {}
        for op, latencies in latencies_by_operation.items():
            if latencies:
                summary[f"{op}_p95"] = stats.percentiles(latencies, [95])[95]
                summary[f"{op}_mean"] = stats.mean(latencies)
        
        return summary
    
    async def _measure_operations(self, latencies_dict: Dict[str, List[float]]):
        """Continuously measure operation latencies."""
        operations = [
            ("get_standard", self._op_get_standard),
            ("list_standards", self._op_list_standards),
            ("search", self._op_search),
        ]
        
        while True:
            for op_name, op_func in operations:
                try:
                    start = time.perf_counter()
                    await op_func()
                    end = time.perf_counter()
                    
                    latencies_dict[op_name].append(end - start)
                    
                except asyncio.CancelledError:
                    return
                except Exception:
                    # Record error as high latency
                    latencies_dict[op_name].append(1.0)
            
            # Small delay between operations
            await asyncio.sleep(0.01)
    
    async def _op_get_standard(self):
        """Get a random standard."""
        std_id = f"load-test-{random.randint(0, 9)}"
        await self.server._get_standard_details(std_id)
    
    async def _op_list_standards(self):
        """List standards."""
        await self.server._list_available_standards(limit=20)
    
    async def _op_search(self):
        """Search standards."""
        queries = ["test", "standard", "content", "guideline"]
        await self.server._search_standards(random.choice(queries), limit=10)
    
    async def teardown(self):
        """Analyze how latency changes with load."""
        self.load_analysis = self._analyze_load_impact()
    
    def _analyze_load_impact(self) -> Dict[str, Any]:
        """Analyze the impact of load on latency."""
        from ..framework.stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()
        
        analysis = {
            "load_levels": self.load_levels[:len(self.results_by_load)],
            "operations": {}
        }
        
        # Analyze each operation
        operations = set()
        for load_data in self.results_by_load.values():
            operations.update(load_data.keys())
        
        for op in operations:
            op_analysis = {
                "latencies_by_load": {},
                "degradation": {}
            }
            
            for load, data in self.results_by_load.items():
                if op in data and data[op]:
                    latencies = data[op]
                    op_analysis["latencies_by_load"][load] = {
                        "p50": stats.percentiles(latencies, [50])[50],
                        "p95": stats.percentiles(latencies, [95])[95],
                        "p99": stats.percentiles(latencies, [99])[99],
                        "mean": stats.mean(latencies),
                    }
            
            # Calculate degradation
            if op_analysis["latencies_by_load"]:
                loads = sorted(op_analysis["latencies_by_load"].keys())
                if len(loads) >= 2:
                    base_p95 = op_analysis["latencies_by_load"][loads[0]]["p95"]
                    for load in loads[1:]:
                        current_p95 = op_analysis["latencies_by_load"][load]["p95"]
                        degradation = (current_p95 - base_p95) / base_p95 if base_p95 > 0 else 0
                        op_analysis["degradation"][load] = degradation
            
            analysis["operations"][op] = op_analysis
        
        return analysis