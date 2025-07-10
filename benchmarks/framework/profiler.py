"""Memory and time profiling utilities."""

import asyncio
import functools
import gc
import time
import tracemalloc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class MemorySnapshot:
    """Memory usage snapshot."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    shared_mb: float
    available_mb: float
    percent: float

    # From tracemalloc
    current_mb: float = 0.0
    peak_mb: float = 0.0

    def __str__(self):
        return (
            f"Memory: RSS={self.rss_mb:.1f}MB, "
            f"VMS={self.vms_mb:.1f}MB, "
            f"Available={self.available_mb:.1f}MB ({self.percent:.1f}%)"
        )


@dataclass
class AllocationTrace:
    """Memory allocation trace from tracemalloc."""

    filename: str
    lineno: int
    size_mb: float
    count: int
    traceback: list[str]


class MemoryProfiler:
    """Profile memory usage during benchmark execution."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.snapshots: list[MemorySnapshot] = []
        self.allocations: list[AllocationTrace] = []
        self._monitoring = False
        self._process = psutil.Process()
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start memory monitoring."""
        self._monitoring = True
        self.snapshots.clear()

        # Start tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Start monitoring task
        self._task = asyncio.create_task(self._monitor_memory())

    async def stop(self):
        """Stop memory monitoring."""
        self._monitoring = False

        if self._task:
            await self._task
            self._task = None

        # Capture allocation traces
        self._capture_allocations()

    async def _monitor_memory(self):
        """Monitor memory usage periodically."""
        while self._monitoring:
            snapshot = self._take_snapshot()
            self.snapshots.append(snapshot)
            await asyncio.sleep(self.interval)

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory snapshot."""
        mem_info = self._process.memory_info()
        vm = psutil.virtual_memory()

        current, peak = 0.0, 0.0
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            current = current / 1024 / 1024  # Convert to MB
            peak = peak / 1024 / 1024

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=mem_info.rss / 1024 / 1024,
            vms_mb=mem_info.vms / 1024 / 1024,
            shared_mb=getattr(mem_info, "shared", 0) / 1024 / 1024,
            available_mb=vm.available / 1024 / 1024,
            percent=vm.percent,
            current_mb=current,
            peak_mb=peak,
        )

    def _capture_allocations(self, limit: int = 10):
        """Capture top memory allocations."""
        if not tracemalloc.is_tracing():
            return

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("traceback")

        self.allocations.clear()

        for stat in top_stats[:limit]:
            # Get traceback
            tb_lines = []
            for frame in stat.traceback:
                tb_lines.append(f"{frame.filename}:{frame.lineno}")

            self.allocations.append(
                AllocationTrace(
                    filename=stat.traceback[0].filename,
                    lineno=stat.traceback[0].lineno,
                    size_mb=stat.size / 1024 / 1024,
                    count=stat.count,
                    traceback=tb_lines,
                )
            )

    def get_summary(self) -> dict[str, Any]:
        """Get memory profiling summary."""
        if not self.snapshots:
            return {}

        rss_values = [s.rss_mb for s in self.snapshots]

        return {
            "samples": len(self.snapshots),
            "duration_seconds": self.snapshots[-1].timestamp
            - self.snapshots[0].timestamp,
            "rss": {
                "min_mb": min(rss_values),
                "max_mb": max(rss_values),
                "avg_mb": sum(rss_values) / len(rss_values),
                "peak_mb": max(rss_values),
            },
            "tracemalloc": {
                "peak_mb": max((s.peak_mb for s in self.snapshots), default=0),
                "final_mb": self.snapshots[-1].current_mb if self.snapshots else 0,
            },
            "top_allocations": [
                {
                    "location": f"{a.filename}:{a.lineno}",
                    "size_mb": a.size_mb,
                    "count": a.count,
                }
                for a in self.allocations[:5]
            ],
        }

    def detect_memory_leak(self, threshold_mb: float = 10.0) -> bool:
        """Detect potential memory leak based on growth."""
        if len(self.snapshots) < 10:
            return False

        # Check if memory consistently increases
        rss_values = [s.rss_mb for s in self.snapshots]

        # Use linear regression to detect trend
        n = len(rss_values)
        x_mean = n / 2
        y_mean = sum(rss_values) / n

        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(rss_values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return False

        slope = numerator / denominator

        # Check if growth exceeds threshold
        total_growth = slope * n
        return total_growth > threshold_mb

    @contextmanager
    def profile_block(self, name: str):
        """Context manager for profiling a code block."""
        gc.collect()
        start_snapshot = self._take_snapshot()

        yield

        gc.collect()
        end_snapshot = self._take_snapshot()

        print(f"\nMemory Profile - {name}:")
        print(f"  RSS Change: {end_snapshot.rss_mb - start_snapshot.rss_mb:.2f} MB")
        print(f"  Peak: {end_snapshot.peak_mb:.2f} MB")


class TimeProfiler:
    """Profile execution time with high precision."""

    def __init__(self):
        self.timings: dict[str, list[float]] = {}

    @contextmanager
    def profile(self, name: str):
        """Context manager for timing a code block."""
        start = time.perf_counter()

        try:
            yield
        finally:
            elapsed = time.perf_counter() - start

            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)

    def time_async(self, name: str | None = None):
        """Decorator for timing async functions."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                label = name or func.__name__

                start = time.perf_counter()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start

                    if label not in self.timings:
                        self.timings[label] = []
                    self.timings[label].append(elapsed)

            return wrapper

        return decorator

    def time_sync(self, name: str | None = None):
        """Decorator for timing sync functions."""

        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                label = name or func.__name__

                start = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    elapsed = time.perf_counter() - start

                    if label not in self.timings:
                        self.timings[label] = []
                    self.timings[label].append(elapsed)

            return wrapper

        return decorator

    def get_summary(self) -> dict[str, dict[str, float]]:
        """Get timing summary."""
        from .stats import StatisticalAnalyzer

        stats = StatisticalAnalyzer()

        summary = {}

        for name, times in self.timings.items():
            if times:
                summary[name] = {
                    "count": len(times),
                    "total": sum(times),
                    "min": min(times),
                    "max": max(times),
                    "mean": stats.mean(times),
                    "median": stats.median(times),
                    "std_dev": stats.std_dev(times) if len(times) > 1 else 0,
                    "p95": (
                        stats.percentiles(times, [95])[95]
                        if len(times) > 1
                        else times[0]
                    ),
                }

        return summary

    def print_summary(self, top_n: int = 10):
        """Print timing summary."""
        summary = self.get_summary()

        if not summary:
            print("No timings recorded.")
            return

        # Sort by total time
        sorted_items = sorted(
            summary.items(), key=lambda x: x[1]["total"], reverse=True
        )[:top_n]

        print("\nTiming Summary:")
        print("-" * 80)
        print(
            f"{'Function':<40} {'Count':>8} {'Total(s)':>10} {'Mean(ms)':>10} {'p95(ms)':>10}"
        )
        print("-" * 80)

        for name, data in sorted_items:
            print(
                f"{name[:40]:<40} "
                f"{data['count']:>8} "
                f"{data['total']:>10.3f} "
                f"{data['mean']*1000:>10.2f} "
                f"{data['p95']*1000:>10.2f}"
            )

    def clear(self):
        """Clear all timings."""
        self.timings.clear()


# Global profiler instances
memory_profiler = MemoryProfiler()
time_profiler = TimeProfiler()
