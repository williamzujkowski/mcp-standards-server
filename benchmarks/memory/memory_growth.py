"""Memory growth analysis over time."""

import asyncio
import gc
import time
from typing import Any

import psutil

from src.mcp_server import MCPStandardsServer

from ..framework import BaseBenchmark


class MemoryGrowthBenchmark(BaseBenchmark):
    """Analyze memory growth patterns over extended periods."""

    def __init__(self, duration_minutes: int = 10, sample_interval: float = 1.0):
        super().__init__("MCP Memory Growth Analysis", 1)
        self.duration_seconds = duration_minutes * 60
        self.sample_interval = sample_interval
        self.memory_samples: list[tuple[float, float]] = []  # (timestamp, memory_mb)
        self.operation_counts: dict[str, int] = {}

    async def setup(self):
        """Setup for memory growth analysis."""
        # Initialize server
        self.server = MCPStandardsServer({
            "search": {"enabled": False},
            "token_model": "gpt-4"
        })

        # Create test data
        await self._create_test_data()

        # Clear samples
        self.memory_samples.clear()
        self.operation_counts.clear()

        # Force garbage collection
        gc.collect()

        # Record baseline
        self.start_time = time.time()
        self.baseline_memory = self._get_memory_mb()

    async def run_single_iteration(self) -> dict[str, Any]:
        """Run continuous operations while monitoring memory."""
        # Start memory monitoring task
        monitor_task = asyncio.create_task(self._monitor_memory())

        # Start workload tasks
        workload_tasks = [
            asyncio.create_task(self._continuous_reads()),
            asyncio.create_task(self._continuous_searches()),
            asyncio.create_task(self._continuous_optimizations()),
            asyncio.create_task(self._continuous_validations()),
        ]

        # Run for specified duration
        await asyncio.sleep(self.duration_seconds)

        # Stop tasks
        monitor_task.cancel()
        for task in workload_tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(monitor_task, *workload_tasks, return_exceptions=True)

        # Analyze growth
        analysis = self._analyze_growth()

        return {
            "duration_seconds": self.duration_seconds,
            "total_operations": sum(self.operation_counts.values()),
            "memory_growth_mb": analysis["total_growth_mb"],
            "growth_rate_mb_per_hour": analysis["growth_rate_mb_per_hour"],
            "growth_pattern": analysis["pattern"],
            "operations": dict(self.operation_counts)
        }

    async def _monitor_memory(self):
        """Continuously monitor memory usage."""
        while True:
            try:
                current_time = time.time() - self.start_time
                memory_mb = self._get_memory_mb()
                self.memory_samples.append((current_time, memory_mb))

                await asyncio.sleep(self.sample_interval)
            except asyncio.CancelledError:
                break

    async def _continuous_reads(self):
        """Continuously read standards."""
        while True:
            try:
                for i in range(10):
                    try:
                        await self.server._get_standard_details(f"growth-test-{i}")
                        self.operation_counts["reads"] = self.operation_counts.get("reads", 0) + 1
                    except Exception:
                        pass

                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break

    async def _continuous_searches(self):
        """Continuously perform searches."""
        queries = ["test", "standard", "growth", "memory", "performance"]

        while True:
            try:
                for query in queries:
                    await self.server._search_standards(query, limit=10)
                    self.operation_counts["searches"] = self.operation_counts.get("searches", 0) + 1

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break

    async def _continuous_optimizations(self):
        """Continuously optimize standards."""
        while True:
            try:
                for i in range(5):
                    try:
                        await self.server._get_optimized_standard(
                            f"growth-test-{i}",
                            format_type="condensed",
                            token_budget=2000
                        )
                        self.operation_counts["optimizations"] = self.operation_counts.get("optimizations", 0) + 1
                    except Exception:
                        pass

                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                break

    async def _continuous_validations(self):
        """Continuously validate code."""
        code_samples = [
            "def hello(): pass",
            "class Test: pass",
            "import os\nprint('test')",
        ]

        while True:
            try:
                for code in code_samples:
                    await self.server._validate_against_standard(
                        code,
                        "python-pep8",
                        "python"
                    )
                    self.operation_counts["validations"] = self.operation_counts.get("validations", 0) + 1

                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                break

    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

    def _analyze_growth(self) -> dict[str, Any]:
        """Analyze memory growth pattern."""
        if len(self.memory_samples) < 2:
            return {
                "total_growth_mb": 0,
                "growth_rate_mb_per_hour": 0,
                "pattern": "insufficient_data"
            }

        # Calculate total growth
        initial_memory = self.memory_samples[0][1]
        final_memory = self.memory_samples[-1][1]
        total_growth = final_memory - initial_memory

        # Calculate growth rate
        duration_hours = self.duration_seconds / 3600
        growth_rate = total_growth / duration_hours if duration_hours > 0 else 0

        # Analyze pattern
        pattern = self._detect_growth_pattern()

        # Find memory spikes
        spikes = self._find_memory_spikes()

        return {
            "total_growth_mb": total_growth,
            "growth_rate_mb_per_hour": growth_rate,
            "pattern": pattern,
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "peak_memory_mb": max(m[1] for m in self.memory_samples),
            "memory_spikes": spikes,
            "samples_collected": len(self.memory_samples)
        }

    def _detect_growth_pattern(self) -> str:
        """Detect the pattern of memory growth."""
        if len(self.memory_samples) < 10:
            return "insufficient_data"

        # Extract memory values
        memories = [m[1] for m in self.memory_samples]

        # Calculate linear regression
        n = len(memories)
        x_mean = (n - 1) / 2
        y_mean = sum(memories) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(memories))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return "stable"

        slope = numerator / denominator

        # Calculate R-squared
        y_pred = [slope * i + (y_mean - slope * x_mean) for i in range(n)]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(memories, y_pred, strict=False))
        ss_tot = sum((y - y_mean) ** 2 for y in memories)

        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Determine pattern
        if r_squared > 0.8 and slope > 0.1:
            return "linear_growth"
        elif r_squared > 0.8 and slope < -0.1:
            return "decreasing"
        elif r_squared < 0.3:
            return "erratic"
        else:
            # Check for step pattern
            if self._has_step_pattern(memories):
                return "step_growth"
            else:
                return "stable"

    def _has_step_pattern(self, memories: list[float]) -> bool:
        """Check if memory shows step-like growth."""
        # Look for sudden jumps
        threshold = 5.0  # 5MB jump
        jumps = 0

        for i in range(1, len(memories)):
            if memories[i] - memories[i-1] > threshold:
                jumps += 1

        return jumps > 2

    def _find_memory_spikes(self) -> list[dict[str, Any]]:
        """Find significant memory spikes."""
        if len(self.memory_samples) < 3:
            return []

        spikes = []
        threshold = 10.0  # 10MB spike

        for i in range(1, len(self.memory_samples) - 1):
            prev_mem = self.memory_samples[i-1][1]
            curr_mem = self.memory_samples[i][1]
            next_mem = self.memory_samples[i+1][1]

            # Check for spike (sudden increase then decrease)
            if curr_mem - prev_mem > threshold and curr_mem - next_mem > threshold / 2:
                spikes.append({
                    "timestamp": self.memory_samples[i][0],
                    "memory_mb": curr_mem,
                    "spike_size_mb": curr_mem - prev_mem
                })

        return spikes

    async def _create_test_data(self):
        """Create test data for growth analysis."""
        import json
        from pathlib import Path

        cache_dir = Path("data/standards/cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            standard = {
                "id": f"growth-test-{i}",
                "name": f"Growth Test Standard {i}",
                "category": "test",
                "content": {
                    "overview": "x" * 1000,
                    "guidelines": ["y" * 100 for _ in range(10)],
                    "examples": ["z" * 200 for _ in range(5)]
                }
            }

            filepath = cache_dir / f"{standard['id']}.json"
            with open(filepath, 'w') as f:
                json.dump(standard, f)

    async def teardown(self):
        """Generate growth analysis report."""
        self.growth_report = self._generate_growth_report()

    def _generate_growth_report(self) -> str:
        """Generate a detailed growth report."""
        analysis = self._analyze_growth()

        lines = [
            "# Memory Growth Analysis Report",
            "",
            f"Duration: {self.duration_seconds / 60:.1f} minutes",
            f"Total Operations: {sum(self.operation_counts.values())}",
            "",
            "## Memory Growth Summary",
            f"- Initial Memory: {analysis['initial_memory_mb']:.1f} MB",
            f"- Final Memory: {analysis['final_memory_mb']:.1f} MB",
            f"- Total Growth: {analysis['total_growth_mb']:.1f} MB",
            f"- Growth Rate: {analysis['growth_rate_mb_per_hour']:.2f} MB/hour",
            f"- Pattern: {analysis['pattern']}",
            "",
            "## Operations Performed",
        ]

        for op, count in self.operation_counts.items():
            rate = count / (self.duration_seconds / 60)  # per minute
            lines.append(f"- {op}: {count} total ({rate:.1f}/min)")

        if analysis.get("memory_spikes"):
            lines.extend([
                "",
                "## Memory Spikes Detected",
            ])
            for spike in analysis["memory_spikes"]:
                lines.append(
                    f"- Time: {spike['timestamp']:.1f}s, "
                    f"Size: {spike['spike_size_mb']:.1f} MB"
                )

        # Add recommendations
        lines.extend([
            "",
            "## Recommendations",
        ])

        if analysis["growth_rate_mb_per_hour"] > 100:
            lines.append("- CRITICAL: High memory growth rate detected")
        elif analysis["growth_rate_mb_per_hour"] > 10:
            lines.append("- WARNING: Moderate memory growth detected")

        if analysis["pattern"] == "linear_growth":
            lines.append("- Linear growth pattern suggests memory leak")
        elif analysis["pattern"] == "step_growth":
            lines.append("- Step growth pattern suggests accumulating cache")

        return "\n".join(lines)
