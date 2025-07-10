"""Real-time metrics collection for MCP server."""

import asyncio
import json
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and storage."""
    name: str
    type: str  # counter, gauge, histogram
    description: str
    unit: str = ""
    data_points: deque[MetricPoint] = field(default_factory=lambda: deque(maxlen=1000))

    def add_point(self, value: float, labels: dict[str, str] | None = None):
        """Add a data point to the metric."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        self.data_points.append(point)

    def get_latest(self) -> float | None:
        """Get the latest value."""
        if self.data_points:
            return self.data_points[-1].value
        return None

    def get_stats(self, window_seconds: int = 60) -> dict[str, float]:
        """Get statistics for a time window."""
        cutoff = time.time() - window_seconds
        values = [
            p.value for p in self.data_points
            if p.timestamp >= cutoff
        ]

        if not values:
            return {}

        return {
            "count": len(values),
            "sum": sum(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else 0
        }


class MetricsCollector:
    """Collect and manage performance metrics."""

    def __init__(self):
        self.metrics: dict[str, Metric] = {}
        self._collectors: list[Callable] = []
        self._running = False
        self._task: asyncio.Task | None = None
        self._process = psutil.Process()

        # Register default metrics
        self._register_default_metrics()

    def _register_default_metrics(self):
        """Register default system metrics."""
        # System metrics
        self.register_metric(
            "system_cpu_percent",
            "gauge",
            "CPU usage percentage",
            "%"
        )
        self.register_metric(
            "system_memory_rss_mb",
            "gauge",
            "Resident set size",
            "MB"
        )
        self.register_metric(
            "system_memory_percent",
            "gauge",
            "Memory usage percentage",
            "%"
        )

        # MCP operation metrics
        self.register_metric(
            "mcp_request_duration_ms",
            "histogram",
            "Request duration",
            "ms"
        )
        self.register_metric(
            "mcp_request_count",
            "counter",
            "Total requests",
            ""
        )
        self.register_metric(
            "mcp_error_count",
            "counter",
            "Total errors",
            ""
        )

        # Component-specific metrics
        self.register_metric(
            "rule_engine_evaluation_time_ms",
            "histogram",
            "Rule evaluation time",
            "ms"
        )
        self.register_metric(
            "token_optimizer_compression_ratio",
            "gauge",
            "Token compression ratio",
            ""
        )
        self.register_metric(
            "search_query_time_ms",
            "histogram",
            "Search query time",
            "ms"
        )
        self.register_metric(
            "cache_hit_rate",
            "gauge",
            "Cache hit rate",
            "%"
        )

    def register_metric(
        self,
        name: str,
        metric_type: str,
        description: str,
        unit: str = ""
    ) -> Metric:
        """Register a new metric."""
        metric = Metric(
            name=name,
            type=metric_type,
            description=description,
            unit=unit
        )
        self.metrics[name] = metric
        return metric

    def record(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str] | None = None
    ):
        """Record a metric value."""
        if metric_name in self.metrics:
            self.metrics[metric_name].add_point(value, labels)

    def increment(
        self,
        metric_name: str,
        value: float = 1,
        labels: dict[str, str] | None = None
    ):
        """Increment a counter metric."""
        if metric_name in self.metrics:
            metric = self.metrics[metric_name]
            if metric.type == "counter":
                current = metric.get_latest() or 0
                metric.add_point(current + value, labels)

    async def start_collection(self, interval: float = 1.0):
        """Start collecting metrics."""
        self._running = True
        self._task = asyncio.create_task(self._collect_loop(interval))

    async def stop_collection(self):
        """Stop collecting metrics."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _collect_loop(self, interval: float):
        """Main collection loop."""
        while self._running:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Run custom collectors
                for collector in self._collectors:
                    try:
                        if asyncio.iscoroutinefunction(collector):
                            await collector()
                        else:
                            collector()
                    except Exception as e:
                        print(f"Collector error: {e}")

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break

    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        # CPU
        cpu_percent = self._process.cpu_percent()
        self.record("system_cpu_percent", cpu_percent)

        # Memory
        mem_info = self._process.memory_info()
        self.record("system_memory_rss_mb", mem_info.rss / 1024 / 1024)
        self.record("system_memory_percent", self._process.memory_percent())

    def add_collector(self, collector: Callable):
        """Add a custom metric collector."""
        self._collectors.append(collector)

    def get_metrics_summary(self, window_seconds: int = 60) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}

        for name, metric in self.metrics.items():
            stats = metric.get_stats(window_seconds)
            if stats:
                summary[name] = {
                    "type": metric.type,
                    "unit": metric.unit,
                    "stats": stats
                }

        return summary

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, metric in self.metrics.items():
            # Add metric info
            lines.append(f"# HELP {name} {metric.description}")
            lines.append(f"# TYPE {name} {metric.type}")

            # Add latest value
            latest = metric.get_latest()
            if latest is not None:
                lines.append(f"{name} {latest}")

        return "\n".join(lines)

    def save_snapshot(self, filepath: Path):
        """Save current metrics snapshot."""
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }

        for name, metric in self.metrics.items():
            # Get recent data points
            recent_points = []
            cutoff = time.time() - 300  # Last 5 minutes

            for point in metric.data_points:
                if point.timestamp >= cutoff:
                    recent_points.append({
                        "timestamp": point.timestamp,
                        "value": point.value,
                        "labels": point.labels
                    })

            snapshot["metrics"][name] = {
                "type": metric.type,
                "description": metric.description,
                "unit": metric.unit,
                "data_points": recent_points,
                "stats": metric.get_stats(300)
            }

        with open(filepath, 'w') as f:
            json.dump(snapshot, f, indent=2)

    def load_snapshot(self, filepath: Path):
        """Load metrics from snapshot."""
        with open(filepath) as f:
            snapshot = json.load(f)

        for name, data in snapshot["metrics"].items():
            # Create metric if doesn't exist
            if name not in self.metrics:
                self.register_metric(
                    name,
                    data["type"],
                    data["description"],
                    data["unit"]
                )

            # Load data points
            metric = self.metrics[name]
            for point_data in data["data_points"]:
                point = MetricPoint(
                    timestamp=point_data["timestamp"],
                    value=point_data["value"],
                    labels=point_data["labels"]
                )
                metric.data_points.append(point)


class MCPMetricsCollector(MetricsCollector):
    """MCP-specific metrics collector."""

    def __init__(self, mcp_server):
        super().__init__()
        self.mcp_server = mcp_server

        # Add MCP-specific collectors
        self.add_collector(self._collect_mcp_metrics)

    async def _collect_mcp_metrics(self):
        """Collect MCP-specific metrics."""
        # This would hook into actual MCP server internals
        # For now, we'll simulate some metrics

        # Simulate cache metrics
        cache_hits = 85.0  # Simulated
        self.record("cache_hit_rate", cache_hits)

        # Simulate active connections
        # In real implementation, this would query server state
        pass

    def record_request(
        self,
        tool_name: str,
        duration_ms: float,
        success: bool,
        error: str | None = None
    ):
        """Record an MCP request."""
        # Record duration
        self.record(
            "mcp_request_duration_ms",
            duration_ms,
            {"tool": tool_name, "success": str(success)}
        )

        # Increment counters
        self.increment("mcp_request_count", labels={"tool": tool_name})

        if not success:
            self.increment("mcp_error_count", labels={
                "tool": tool_name,
                "error": error or "unknown"
            })
