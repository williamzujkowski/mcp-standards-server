#!/usr/bin/env python3
"""Simple benchmark runner for MCP Standards Server."""

import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp_server import MCPStandardsServer


class SimpleBenchmark:
    """Simple benchmark for MCP tools."""
    
    def __init__(self):
        self.results = []
        self.server = None
    
    async def setup(self):
        """Setup MCP server."""
        config = {
            "search": {"enabled": False},
            "token_model": "gpt-4",
            "default_token_budget": 8000
        }
        self.server = MCPStandardsServer(config)
        print("✓ MCP Server initialized")
    
    async def benchmark_tool(self, tool_name: str, args: dict, iterations: int = 10) -> dict:
        """Benchmark a specific MCP tool."""
        print(f"\nBenchmarking {tool_name}...")
        
        method_name = f"_{tool_name}"
        method = getattr(self.server, method_name, None)
        
        if not method:
            print(f"✗ Tool {tool_name} not found")
            return {"error": f"Tool {tool_name} not found"}
        
        times = []
        errors = []
        
        # Warmup
        for _ in range(3):
            try:
                await method(**args)
            except:
                pass
        
        # Actual benchmark
        for i in range(iterations):
            try:
                start = time.perf_counter()
                result = await method(**args)
                end = time.perf_counter()
                
                elapsed = end - start
                times.append(elapsed)
                
                if i == 0:
                    print(f"  First run: {elapsed:.4f}s")
                
            except Exception as e:
                errors.append(str(e))
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  ✓ Completed {len(times)} iterations")
            print(f"  Average: {avg_time:.4f}s")
            print(f"  Min: {min_time:.4f}s")
            print(f"  Max: {max_time:.4f}s")
            
            return {
                "tool": tool_name,
                "iterations": len(times),
                "avg_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "times": times,
                "errors": errors
            }
        else:
            print(f"  ✗ All iterations failed")
            return {
                "tool": tool_name,
                "iterations": 0,
                "errors": errors
            }
    
    async def run_benchmarks(self):
        """Run benchmarks for all MCP tools."""
        print("MCP Standards Server - Performance Benchmarks")
        print("=" * 50)
        
        await self.setup()
        
        # Define test scenarios
        test_scenarios = [
            {
                "tool": "get_sync_status",
                "args": {}
            },
            {
                "tool": "list_available_standards",
                "args": {"limit": 10}
            },
            {
                "tool": "get_standard_details",
                "args": {"standard_id": "CODING_STANDARDS"}
            },
            {
                "tool": "search_standards",
                "args": {
                    "query": "security authentication",
                    "limit": 5,
                    "min_relevance": 0.5
                }
            },
            {
                "tool": "get_applicable_standards",
                "args": {
                    "context": {
                        "language": "python",
                        "framework": "fastapi",
                        "project_type": "api"
                    }
                }
            },
            {
                "tool": "estimate_token_usage",
                "args": {
                    "standard_ids": ["CODING_STANDARDS"],
                    "format_types": ["full", "condensed"]
                }
            }
        ]
        
        # Run benchmarks
        for scenario in test_scenarios:
            result = await self.benchmark_tool(
                scenario["tool"],
                scenario["args"],
                iterations=20
            )
            self.results.append(result)
        
        # Save results
        self.save_results()
        
        print("\n" + "=" * 50)
        print("Benchmarks completed!")
    
    def save_results(self):
        """Save benchmark results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(project_root) / "benchmark_results" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / "results.json", "w") as f:
            json.dump(self.results, f, indent=2)
        
        # Create summary
        summary = {
            "timestamp": timestamp,
            "total_tools": len(self.results),
            "successful_tools": len([r for r in self.results if r.get("iterations", 0) > 0]),
            "tool_performance": {}
        }
        
        for result in self.results:
            if result.get("iterations", 0) > 0:
                summary["tool_performance"][result["tool"]] = {
                    "avg_time": f"{result['avg_time']:.4f}s",
                    "min_time": f"{result['min_time']:.4f}s",
                    "max_time": f"{result['max_time']:.4f}s"
                }
        
        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Create baseline if it doesn't exist
        baseline_dir = Path(project_root) / "benchmark_results" / "baseline"
        if not baseline_dir.exists():
            baseline_dir.mkdir(parents=True, exist_ok=True)
            with open(baseline_dir / "results.json", "w") as f:
                json.dump(self.results, f, indent=2)
            with open(baseline_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\n✓ Created baseline in: {baseline_dir}")
        
        print(f"\n✓ Results saved to: {output_dir}")


async def main():
    """Main entry point."""
    benchmark = SimpleBenchmark()
    await benchmark.run_benchmarks()


if __name__ == "__main__":
    asyncio.run(main())