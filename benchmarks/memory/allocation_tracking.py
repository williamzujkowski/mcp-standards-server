"""Track memory allocations by component."""

import asyncio
import gc
import sys
import tracemalloc
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.mcp_server import MCPStandardsServer
from ..framework import BaseBenchmark


class AllocationTrackingBenchmark(BaseBenchmark):
    """Track and analyze memory allocations by component."""
    
    def __init__(self, iterations: int = 50):
        super().__init__("MCP Allocation Tracking", iterations)
        self.allocation_stats: Dict[str, Dict[str, Any]] = {}
        self.component_snapshots: Dict[str, tracemalloc.Snapshot] = {}
        
    async def setup(self):
        """Setup allocation tracking."""
        # Start tracemalloc with more frames for better tracking
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        tracemalloc.start(25)
        
        # Clear previous data
        self.allocation_stats.clear()
        self.component_snapshots.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Create test data
        await self._create_test_data()
    
    async def run_single_iteration(self) -> Dict[str, Any]:
        """Track allocations for different components."""
        results = {}
        
        # Track each component separately
        components = [
            ("rule_engine", self._track_rule_engine),
            ("token_optimizer", self._track_token_optimizer),
            ("search_engine", self._track_search_engine),
            ("sync_manager", self._track_sync_manager),
            ("cache_operations", self._track_cache_operations),
        ]
        
        for component_name, track_func in components:
            gc.collect()
            tracemalloc.clear_traces()
            
            # Take snapshot before
            snapshot_before = tracemalloc.take_snapshot()
            
            # Run component operations
            await track_func()
            
            # Take snapshot after
            snapshot_after = tracemalloc.take_snapshot()
            
            # Analyze allocations
            stats = self._analyze_allocations(
                component_name,
                snapshot_before,
                snapshot_after
            )
            
            results[component_name] = stats
            self.allocation_stats[component_name] = stats
        
        return results
    
    async def _track_rule_engine(self):
        """Track rule engine allocations."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Various rule engine operations
        contexts = [
            {"language": "python"},
            {"language": "javascript", "framework": "react"},
            {"language": "java", "framework": "spring", "project_type": "api"},
            {
                "language": "typescript",
                "framework": "angular",
                "project_type": "enterprise",
                "team_size": "large",
                "deployment": "kubernetes",
                "security_level": "high"
            }
        ]
        
        for context in contexts:
            await server._get_applicable_standards(context, include_resolution_details=True)
    
    async def _track_token_optimizer(self):
        """Track token optimizer allocations."""
        server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })
        
        # Test different optimization scenarios
        standard_ids = ["alloc-test-0", "alloc-test-1", "alloc-test-2"]
        
        # Single standard optimization
        for std_id in standard_ids:
            for format_type in ["full", "condensed", "summary"]:
                try:
                    await server._get_optimized_standard(
                        std_id,
                        format_type=format_type,
                        token_budget=3000
                    )
                except:
                    pass
        
        # Multi-standard optimization
        await server._auto_optimize_standards(
            standard_ids,
            total_token_budget=10000
        )
        
        # Progressive loading
        for std_id in standard_ids[:1]:
            try:
                await server._progressive_load_standard(
                    std_id,
                    initial_sections=["overview", "guidelines"],
                    max_depth=3
                )
            except:
                pass
    
    async def _track_search_engine(self):
        """Track search engine allocations."""
        server = MCPStandardsServer({
            "search": {"enabled": True},
            "search_model": "sentence-transformers/all-MiniLM-L6-v2"
        })
        
        # Various search queries
        queries = [
            "security",
            "authentication oauth jwt token refresh",
            "react hooks performance optimization memoization",
            "microservices distributed tracing observability",
            "kubernetes deployment scaling horizontal pod autoscaler"
        ]
        
        for query in queries:
            await server._search_standards(
                query,
                limit=20,
                min_relevance=0.3
            )
    
    async def _track_sync_manager(self):
        """Track sync manager allocations."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Sync operations
        await server._get_sync_status()
        
        # Simulate sync (without actual network calls)
        # This tests the sync infrastructure
        try:
            await server._sync_standards(force=False)
        except:
            # Expected if no actual remote
            pass
    
    async def _track_cache_operations(self):
        """Track cache-related allocations."""
        server = MCPStandardsServer({"search": {"enabled": False}})
        
        # Load many standards to test caching
        for i in range(10):
            try:
                await server._get_standard_details(f"alloc-test-{i}")
            except:
                pass
        
        # List operations that might cache results
        await server._list_available_standards(limit=100)
        await server._list_available_standards(category="test", limit=50)
    
    def _analyze_allocations(
        self,
        component: str,
        before: tracemalloc.Snapshot,
        after: tracemalloc.Snapshot
    ) -> Dict[str, Any]:
        """Analyze memory allocations for a component."""
        # Get top differences
        top_stats = after.compare_to(before, 'lineno')
        
        # Calculate total allocations
        total_allocated = sum(stat.size_diff for stat in top_stats if stat.size_diff > 0)
        total_freed = sum(-stat.size_diff for stat in top_stats if stat.size_diff < 0)
        
        # Get top allocating locations
        top_allocations = []
        for stat in sorted(top_stats, key=lambda x: x.size_diff, reverse=True)[:10]:
            if stat.size_diff > 0:
                top_allocations.append({
                    "file": stat.traceback[0].filename,
                    "line": stat.traceback[0].lineno,
                    "size_kb": stat.size_diff / 1024,
                    "count": stat.count_diff
                })
        
        # Get allocation by file
        allocations_by_file = defaultdict(int)
        for stat in top_stats:
            if stat.size_diff > 0:
                filename = stat.traceback[0].filename
                # Simplify path
                if "site-packages" in filename:
                    filename = filename.split("site-packages/")[-1]
                elif "mcp-standards-server" in filename:
                    filename = filename.split("mcp-standards-server/")[-1]
                
                allocations_by_file[filename] += stat.size_diff
        
        # Get top allocating files
        top_files = sorted(
            allocations_by_file.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            "total_allocated_kb": total_allocated / 1024,
            "total_freed_kb": total_freed / 1024,
            "net_allocation_kb": (total_allocated - total_freed) / 1024,
            "top_allocations": top_allocations,
            "top_files": [
                {"file": f, "size_kb": s / 1024}
                for f, s in top_files
            ],
            "allocation_count": sum(1 for stat in top_stats if stat.size_diff > 0)
        }
    
    async def _create_test_data(self):
        """Create test data for allocation tracking."""
        import json
        from pathlib import Path
        
        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create standards with varying complexity
        for i in range(10):
            standard = {
                "id": f"alloc-test-{i}",
                "name": f"Allocation Test {i}",
                "category": "test",
                "tags": [f"tag{j}" for j in range(i + 1)],
                "content": {
                    "overview": "x" * (100 * (i + 1)),
                    "guidelines": [f"Guideline {j}" * 10 for j in range(i + 5)],
                    "examples": [f"Example {j}" * 20 for j in range(i + 2)],
                    "sections": {
                        f"section_{j}": f"Content {j}" * 50
                        for j in range(i)
                    }
                }
            }
            
            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)
    
    async def teardown(self):
        """Generate allocation report."""
        # Stop tracemalloc
        tracemalloc.stop()
        
        # Generate report
        self.allocation_report = self._generate_allocation_report()
    
    def _generate_allocation_report(self) -> Dict[str, Any]:
        """Generate comprehensive allocation report."""
        report = {
            "components": {},
            "summary": {
                "total_allocated_kb": 0,
                "total_net_kb": 0,
            },
            "hotspots": [],
            "recommendations": []
        }
        
        # Aggregate component data
        for component, stats in self.allocation_stats.items():
            report["components"][component] = {
                "net_allocation_kb": stats["net_allocation_kb"],
                "allocation_count": stats["allocation_count"],
                "top_file": stats["top_files"][0]["file"] if stats["top_files"] else "N/A"
            }
            
            report["summary"]["total_allocated_kb"] += stats["total_allocated_kb"]
            report["summary"]["total_net_kb"] += stats["net_allocation_kb"]
        
        # Find hotspots
        all_allocations = []
        for component, stats in self.allocation_stats.items():
            for alloc in stats["top_allocations"]:
                alloc["component"] = component
                all_allocations.append(alloc)
        
        # Sort by size
        hotspots = sorted(all_allocations, key=lambda x: x["size_kb"], reverse=True)[:10]
        report["hotspots"] = hotspots
        
        # Generate recommendations
        if report["summary"]["total_net_kb"] > 1000:  # 1MB
            report["recommendations"].append(
                "High memory allocation detected - review top allocating components"
            )
        
        # Check for specific component issues
        for component, stats in self.allocation_stats.items():
            if stats["net_allocation_kb"] > 500:  # 500KB
                report["recommendations"].append(
                    f"{component}: High memory allocation ({stats['net_allocation_kb']:.0f}KB)"
                )
        
        return report
    
    def print_allocation_summary(self):
        """Print a formatted allocation summary."""
        if not hasattr(self, 'allocation_report'):
            print("No allocation report available")
            return
        
        report = self.allocation_report
        
        print("\n" + "=" * 80)
        print("MEMORY ALLOCATION ANALYSIS")
        print("=" * 80)
        
        print("\n## Component Summary")
        print(f"{'Component':<20} {'Net Alloc (KB)':<15} {'Allocations':<12} {'Top File':<30}")
        print("-" * 80)
        
        for component, data in report["components"].items():
            print(
                f"{component:<20} "
                f"{data['net_allocation_kb']:<15.1f} "
                f"{data['allocation_count']:<12} "
                f"{data['top_file']:<30}"
            )
        
        print(f"\n{'TOTAL':<20} {report['summary']['total_net_kb']:<15.1f}")
        
        print("\n## Top Allocation Hotspots")
        print(f"{'Component':<15} {'File:Line':<40} {'Size (KB)':<10}")
        print("-" * 65)
        
        for hotspot in report["hotspots"][:5]:
            location = f"{hotspot['file'].split('/')[-1]}:{hotspot['line']}"
            print(
                f"{hotspot['component']:<15} "
                f"{location:<40} "
                f"{hotspot['size_kb']:<10.1f}"
            )
        
        if report["recommendations"]:
            print("\n## Recommendations")
            for rec in report["recommendations"]:
                print(f"- {rec}")