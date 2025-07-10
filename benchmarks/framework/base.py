"""Base classes for benchmarking framework."""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    name: str
    timestamp: datetime
    duration: float  # seconds
    iterations: int

    # Performance metrics
    min_time: float
    max_time: float
    mean_time: float
    median_time: float
    std_dev: float
    percentiles: dict[int, float]  # 50th, 90th, 95th, 99th

    # Memory metrics
    peak_memory_mb: float
    avg_memory_mb: float
    memory_samples: list[float] = field(default_factory=list)

    # Additional metrics
    throughput: float | None = None  # operations per second
    latency_distribution: dict[str, float] | None = None
    custom_metrics: dict[str, Any] = field(default_factory=dict)

    # Metadata
    tags: list[str] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration,
            "iterations": self.iterations,
            "performance": {
                "min_time": self.min_time,
                "max_time": self.max_time,
                "mean_time": self.mean_time,
                "median_time": self.median_time,
                "std_dev": self.std_dev,
                "percentiles": self.percentiles,
                "throughput": self.throughput,
            },
            "memory": {
                "peak_mb": self.peak_memory_mb,
                "avg_mb": self.avg_memory_mb,
                "samples": self.memory_samples,
            },
            "latency_distribution": self.latency_distribution,
            "custom_metrics": self.custom_metrics,
            "metadata": {
                "tags": self.tags,
                "environment": self.environment,
                "errors": self.errors,
            }
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration=data["duration"],
            iterations=data["iterations"],
            min_time=data["performance"]["min_time"],
            max_time=data["performance"]["max_time"],
            mean_time=data["performance"]["mean_time"],
            median_time=data["performance"]["median_time"],
            std_dev=data["performance"]["std_dev"],
            percentiles=data["performance"]["percentiles"],
            peak_memory_mb=data["memory"]["peak_mb"],
            avg_memory_mb=data["memory"]["avg_mb"],
            memory_samples=data["memory"]["samples"],
            throughput=data["performance"].get("throughput"),
            latency_distribution=data.get("latency_distribution"),
            custom_metrics=data.get("custom_metrics", {}),
            tags=data["metadata"]["tags"],
            environment=data["metadata"]["environment"],
            errors=data["metadata"]["errors"],
        )


class BaseBenchmark(ABC):
    """Base class for all benchmarks."""

    def __init__(self, name: str, iterations: int = 100):
        self.name = name
        self.iterations = iterations
        self.results: list[BenchmarkResult] = []
        self._process = psutil.Process()

    @abstractmethod
    async def setup(self):
        """Setup before benchmark runs."""
        pass

    @abstractmethod
    async def run_single_iteration(self) -> dict[str, Any]:
        """Run a single iteration of the benchmark."""
        pass

    @abstractmethod
    async def teardown(self):
        """Cleanup after benchmark runs."""
        pass

    async def warmup(self, iterations: int = 5):
        """Warmup runs to stabilize performance."""
        for _ in range(iterations):
            await self.run_single_iteration()

    async def run(self, warmup_iterations: int = 5) -> BenchmarkResult:
        """Run the complete benchmark."""
        # Setup
        await self.setup()

        # Warmup
        if warmup_iterations > 0:
            await self.warmup(warmup_iterations)

        # Collect baseline memory
        baseline_memory = self._process.memory_info().rss / 1024 / 1024  # MB

        # Run benchmark
        times = []
        memory_samples = []
        custom_metrics_list = []
        errors = []

        start_time = time.time()

        for i in range(self.iterations):
            try:
                # Time the iteration
                iter_start = time.perf_counter()
                custom_metrics = await self.run_single_iteration()
                iter_end = time.perf_counter()

                times.append(iter_end - iter_start)

                # Sample memory
                current_memory = self._process.memory_info().rss / 1024 / 1024
                memory_samples.append(current_memory - baseline_memory)

                # Collect custom metrics
                if custom_metrics:
                    custom_metrics_list.append(custom_metrics)

            except Exception as e:
                errors.append(f"Iteration {i}: {str(e)}")

        total_duration = time.time() - start_time

        # Teardown
        await self.teardown()

        # Calculate statistics
        from .stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()

        result = BenchmarkResult(
            name=self.name,
            timestamp=datetime.now(),
            duration=total_duration,
            iterations=len(times),
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0,
            mean_time=stats.mean(times) if times else 0,
            median_time=stats.median(times) if times else 0,
            std_dev=stats.std_dev(times) if times else 0,
            percentiles=stats.percentiles(times, [50, 90, 95, 99]) if times else {},
            peak_memory_mb=max(memory_samples) if memory_samples else 0,
            avg_memory_mb=stats.mean(memory_samples) if memory_samples else 0,
            memory_samples=memory_samples,
            throughput=len(times) / total_duration if total_duration > 0 else 0,
            errors=errors,
            environment=self._get_environment_info()
        )

        # Aggregate custom metrics
        if custom_metrics_list:
            result.custom_metrics = self._aggregate_custom_metrics(custom_metrics_list)

        self.results.append(result)
        return result

    def _get_environment_info(self) -> dict[str, Any]:
        """Collect environment information."""
        return {
            "python_version": psutil.version_info,
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024 / 1024 / 1024,
            "platform": {
                "system": psutil.LINUX if hasattr(psutil, "LINUX") else "unknown",
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        }

    def _aggregate_custom_metrics(self, metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate custom metrics from all iterations."""
        if not metrics_list:
            return {}

        aggregated = {}

        # Group by metric name
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        from .stats import StatisticalAnalyzer
        stats = StatisticalAnalyzer()

        for key in all_keys:
            values = [m.get(key) for m in metrics_list if key in m]

            if all(isinstance(v, int | float) for v in values):
                # Numeric metrics - calculate statistics
                aggregated[key] = {
                    "mean": stats.mean(values),
                    "median": stats.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std_dev": stats.std_dev(values)
                }
            else:
                # Non-numeric - just store all values
                aggregated[key] = values

        return aggregated

    def save_results(self, filepath: Path):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2
            )

    def load_results(self, filepath: Path):
        """Load results from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
            self.results = [BenchmarkResult.from_dict(r) for r in data]


class BenchmarkSuite:
    """Suite for running multiple benchmarks."""

    def __init__(self, name: str):
        self.name = name
        self.benchmarks: list[BaseBenchmark] = []
        self.results: dict[str, list[BenchmarkResult]] = {}

    def add_benchmark(self, benchmark: BaseBenchmark):
        """Add a benchmark to the suite."""
        self.benchmarks.append(benchmark)

    async def run_all(self, warmup_iterations: int = 5) -> dict[str, list[BenchmarkResult]]:
        """Run all benchmarks in the suite."""
        print(f"\nRunning benchmark suite: {self.name}")
        print("=" * 80)

        for benchmark in self.benchmarks:
            print(f"\nRunning {benchmark.name}...")
            try:
                result = await benchmark.run(warmup_iterations)

                if benchmark.name not in self.results:
                    self.results[benchmark.name] = []
                self.results[benchmark.name].append(result)

                # Print summary
                print(f"  ✓ Completed in {result.duration:.2f}s")
                print(f"  ✓ Mean time: {result.mean_time:.4f}s")
                print(f"  ✓ Throughput: {result.throughput:.2f} ops/s")

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")

        return self.results

    def save_results(self, directory: Path):
        """Save all results to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        # Save individual benchmark results
        for name, results in self.results.items():
            filepath = directory / f"{name.replace(' ', '_')}.json"
            with open(filepath, 'w') as f:
                json.dump(
                    [r.to_dict() for r in results],
                    f,
                    indent=2
                )

        # Save summary
        summary = {
            "suite": self.name,
            "timestamp": datetime.now().isoformat(),
            "benchmarks": list(self.results.keys()),
            "summary": {}
        }

        for name, results in self.results.items():
            if results:
                latest = results[-1]
                summary["summary"][name] = {
                    "mean_time": latest.mean_time,
                    "throughput": latest.throughput,
                    "peak_memory_mb": latest.peak_memory_mb,
                    "errors": len(latest.errors)
                }

        with open(directory / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
