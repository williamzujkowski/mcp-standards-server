"""
Performance monitoring and metrics collection for MCP server.

Provides comprehensive metrics tracking with export capabilities.
"""

import asyncio
import logging
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    """Types of metrics collected."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point."""

    timestamp: float
    value: float
    labels: dict[str, str]


@dataclass
class MetricSummary:
    """Summary statistics for a metric."""

    count: int
    sum: float
    min: float
    max: float
    avg: float
    p50: float
    p95: float
    p99: float


class MetricsCollector:
    """Collects and aggregates performance metrics."""

    def __init__(
        self,
        window_size: int = 300,  # 5 minutes
        max_samples: int = 10000,
        export_interval: int = 60,  # Export every minute
    ):
        """
        Initialize metrics collector.

        Args:
            window_size: Time window for metrics aggregation in seconds
            max_samples: Maximum samples to keep per metric
            export_interval: Interval for exporting metrics in seconds
        """
        self.window_size = window_size
        self.max_samples = max_samples
        self.export_interval = export_interval

        # Metric storage
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_samples)
        )
        self._timers: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))

        # Labels for metrics
        self._labels: dict[str, dict[str, str]] = defaultdict(dict)

        # Export handlers
        self._export_handlers: list[Callable] = []

        # Background export task
        self._export_task: asyncio.Task | None = None

    def increment(
        self, name: str, value: float = 1.0, labels: dict[str, str] | None = None
    ) -> None:
        """Increment a counter metric."""
        key = self._make_key(name, labels)
        self._counters[key] += value
        if labels:
            self._labels[key] = labels

    def gauge(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Set a gauge metric."""
        key = self._make_key(name, labels)
        self._gauges[key] = value
        if labels:
            self._labels[key] = labels

    def histogram(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a histogram value."""
        key = self._make_key(name, labels)
        self._histograms[key].append(MetricPoint(time.time(), value, labels or {}))

    def timer(self, name: str, labels: dict[str, str] | None = None) -> "TimerContext":
        """Context manager for timing operations."""
        return TimerContext(self, name, labels)

    def record_duration(
        self, name: str, duration: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a duration measurement."""
        key = self._make_key(name, labels)
        self._timers[key].append(MetricPoint(time.time(), duration, labels or {}))

    def get_summary(
        self, name: str, metric_type: MetricType, labels: dict[str, str] | None = None
    ) -> MetricSummary | None:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)

        if metric_type == MetricType.COUNTER:
            value = self._counters.get(key, 0)
            return MetricSummary(
                count=1,
                sum=value,
                min=value,
                max=value,
                avg=value,
                p50=value,
                p95=value,
                p99=value,
            )

        elif metric_type == MetricType.GAUGE:
            value = self._gauges.get(key, 0)
            return MetricSummary(
                count=1,
                sum=value,
                min=value,
                max=value,
                avg=value,
                p50=value,
                p95=value,
                p99=value,
            )

        elif metric_type in [MetricType.HISTOGRAM, MetricType.TIMER]:
            data = (
                self._histograms[key]
                if metric_type == MetricType.HISTOGRAM
                else self._timers[key]
            )
            if not data:
                return None

            # Filter to window
            current_time = time.time()
            values = [
                p.value for p in data if current_time - p.timestamp <= self.window_size
            ]

            if not values:
                return None

            values.sort()
            return MetricSummary(
                count=len(values),
                sum=sum(values),
                min=min(values),
                max=max(values),
                avg=statistics.mean(values),
                p50=self._percentile(values, 50),
                p95=self._percentile(values, 95),
                p99=self._percentile(values, 99),
            )

        return None

    def get_all_metrics(self) -> dict[str, Any]:
        """Get all metrics as a dictionary."""
        metrics: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
            "timers": {},
        }

        # Export counters
        for key, value in self._counters.items():
            metrics["counters"][key] = {
                "value": value,
                "labels": self._labels.get(key, {}),
            }

        # Export gauges
        for key, value in self._gauges.items():
            metrics["gauges"][key] = {
                "value": value,
                "labels": self._labels.get(key, {}),
            }

        # Export histograms (create a copy to avoid dictionary change during iteration)
        for key, _data in list(self._histograms.items()):
            summary = self.get_summary(
                key.split(":")[0], MetricType.HISTOGRAM, self._labels.get(key)
            )
            if summary:
                metrics["histograms"][key] = {
                    "summary": asdict(summary),
                    "labels": self._labels.get(key, {}),
                }

        # Export timers (create a copy to avoid dictionary change during iteration)
        for key, _data in list(self._timers.items()):
            summary = self.get_summary(
                key.split(":")[0], MetricType.TIMER, self._labels.get(key)
            )
            if summary:
                metrics["timers"][key] = {
                    "summary": asdict(summary),
                    "labels": self._labels.get(key, {}),
                }

        return metrics

    def export_to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        timestamp = int(time.time() * 1000)

        # Counters
        for key, value in self._counters.items():
            name = key.split(":")[0]
            labels = self._format_labels(self._labels.get(key, {}))
            lines.append(f"{name}_total{labels} {value} {timestamp}")

        # Gauges
        for key, value in self._gauges.items():
            name = key.split(":")[0]
            labels = self._format_labels(self._labels.get(key, {}))
            lines.append(f"{name}{labels} {value} {timestamp}")

        # Histograms
        for key in self._histograms:
            name = key.split(":")[0]
            summary = self.get_summary(
                name, MetricType.HISTOGRAM, self._labels.get(key)
            )
            if summary:
                labels = self._format_labels(self._labels.get(key, {}))
                lines.extend(
                    [
                        f"{name}_count{labels} {summary.count} {timestamp}",
                        f"{name}_sum{labels} {summary.sum} {timestamp}",
                        f'{name}{labels.rstrip("}")},quantile="0.5"}} {summary.p50} {timestamp}',
                        f'{name}{labels.rstrip("}")},quantile="0.95"}} {summary.p95} {timestamp}',
                        f'{name}{labels.rstrip("}")},quantile="0.99"}} {summary.p99} {timestamp}',
                    ]
                )

        return "\n".join(lines)

    def add_export_handler(self, handler: Callable) -> None:
        """Add a handler for metric exports."""
        self._export_handlers.append(handler)

    async def start_export_task(self) -> None:
        """Start background metric export task."""
        if self._export_task is None:
            self._export_task = asyncio.create_task(self._export_loop())

    async def stop_export_task(self) -> None:
        """Stop background metric export task."""
        if self._export_task:
            self._export_task.cancel()
            try:
                await self._export_task
            except asyncio.CancelledError:
                pass
            self._export_task = None

    async def _export_loop(self) -> None:
        """Background task to export metrics periodically."""
        while True:
            try:
                await asyncio.sleep(self.export_interval)
                metrics = self.get_all_metrics()

                # Call export handlers
                for handler in self._export_handlers:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(metrics)
                        else:
                            handler(metrics)
                    except Exception as e:
                        logger.error(f"Error in metric export handler: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric export loop: {e}")

    def _make_key(self, name: str, labels: dict[str, str] | None = None) -> str:
        """Create a unique key for a metric with labels."""
        if not labels:
            return name

        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}:{label_str}"

    def _format_labels(self, labels: dict[str, str]) -> str:
        """Format labels for Prometheus export."""
        if not labels:
            return ""

        label_parts = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_parts) + "}"

    def _percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        index = int(len(values) * percentile / 100)
        if index >= len(values):
            index = len(values) - 1
        return values[index]


class TimerContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        collector: MetricsCollector,
        name: str,
        labels: dict[str, str] | None = None,
    ):
        self.collector = collector
        self.name = name
        self.labels = labels
        self.start_time: float | None = None

    def __enter__(self) -> "TimerContext":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_duration(self.name, duration, self.labels)


class MCPMetrics:
    """MCP-specific metrics collection."""

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        """Initialize MCP metrics."""
        self.collector = collector or MetricsCollector()

        # Pre-defined metric names
        self.TOOL_CALL_DURATION = "mcp_tool_call_duration_seconds"
        self.TOOL_CALL_TOTAL = "mcp_tool_calls_total"
        self.TOOL_CALL_ERRORS = "mcp_tool_call_errors_total"
        self.AUTH_ATTEMPTS = "mcp_auth_attempts_total"
        self.AUTH_FAILURES = "mcp_auth_failures_total"
        self.RATE_LIMIT_HITS = "mcp_rate_limit_hits_total"
        self.CACHE_HITS = "mcp_cache_hits_total"
        self.CACHE_MISSES = "mcp_cache_misses_total"
        self.ACTIVE_CONNECTIONS = "mcp_active_connections"
        self.REQUEST_SIZE = "mcp_request_size_bytes"
        self.RESPONSE_SIZE = "mcp_response_size_bytes"
        self.ERRORS_TOTAL = "mcp_errors_total"
        self.HTTP_REQUESTS_TOTAL = "mcp_http_requests_total"
        self.HTTP_REQUEST_DURATION = "mcp_http_request_duration_seconds"

    def record_tool_call(
        self,
        tool_name: str,
        duration: float,
        success: bool,
        error_type: str | None = None,
    ) -> None:
        """Record metrics for a tool call."""
        labels = {"tool": tool_name, "success": str(success).lower()}

        self.collector.increment(self.TOOL_CALL_TOTAL, labels=labels)
        self.collector.record_duration(self.TOOL_CALL_DURATION, duration, labels=labels)

        if not success and error_type:
            error_labels = {"tool": tool_name, "error_type": error_type}
            self.collector.increment(self.TOOL_CALL_ERRORS, labels=error_labels)

    def record_auth_attempt(self, auth_type: str, success: bool) -> None:
        """Record authentication attempt."""
        labels = {"type": auth_type, "success": str(success).lower()}
        self.collector.increment(self.AUTH_ATTEMPTS, labels=labels)

        if not success:
            self.collector.increment(self.AUTH_FAILURES, labels={"type": auth_type})

    def record_rate_limit_hit(self, identifier: str, tier: str) -> None:
        """Record rate limit hit."""
        labels = {"tier": tier}
        self.collector.increment(self.RATE_LIMIT_HITS, labels=labels)

    def record_cache_access(self, tool_name: str, hit: bool) -> None:
        """Record cache access."""
        labels = {"tool": tool_name}
        if hit:
            self.collector.increment(self.CACHE_HITS, labels=labels)
        else:
            self.collector.increment(self.CACHE_MISSES, labels=labels)

    def update_active_connections(self, count: int) -> None:
        """Update active connections gauge."""
        self.collector.gauge(self.ACTIVE_CONNECTIONS, count)

    def record_request_size(self, size: int, tool_name: str) -> None:
        """Record request size."""
        labels = {"tool": tool_name}
        self.collector.histogram(self.REQUEST_SIZE, size, labels=labels)

    def record_response_size(self, size: int, tool_name: str) -> None:
        """Record response size."""
        labels = {"tool": tool_name}
        self.collector.histogram(self.RESPONSE_SIZE, size, labels=labels)

    def record_error(
        self, error_type: str, error_code: str, function: str | None = None
    ) -> None:
        """Record an error occurrence."""
        labels = {"error_type": error_type, "error_code": error_code}
        if function:
            labels["function"] = function
        self.collector.increment(self.ERRORS_TOTAL, labels=labels)

    def record_http_request(
        self, method: str, path: str, status: int, duration: float, error: bool = False
    ) -> None:
        """Record HTTP request metrics."""
        labels = {
            "method": method,
            "path": path,
            "status": str(status),
            "error": str(error).lower(),
        }
        self.collector.increment(self.HTTP_REQUESTS_TOTAL, labels=labels)
        self.collector.record_duration(
            self.HTTP_REQUEST_DURATION, duration, labels=labels
        )

    def record_operation(
        self,
        operation: str,
        success: bool,
        error_type: str | None = None,
        labels: dict[str, str] | None = None,
    ) -> None:
        """Record operation metrics."""
        operation_labels = {"operation": operation, "success": str(success).lower()}
        if error_type:
            operation_labels["error_type"] = error_type
        if labels:
            operation_labels.update(labels)

        self.collector.increment(f"{operation}_total", labels=operation_labels)

    def record_duration(
        self, metric: str, duration: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record duration metric."""
        self.collector.record_duration(metric, duration, labels=labels)

    def get_dashboard_metrics(self) -> dict[str, Any]:
        """Get metrics formatted for dashboard display."""
        all_metrics = self.collector.get_all_metrics()

        # Calculate derived metrics
        total_calls = sum(
            m["value"]
            for k, m in all_metrics["counters"].items()
            if k.startswith(self.TOOL_CALL_TOTAL)
        )

        total_errors = sum(
            m["value"]
            for k, m in all_metrics["counters"].items()
            if k.startswith(self.TOOL_CALL_ERRORS)
        )

        error_rate = (total_errors / total_calls * 100) if total_calls > 0 else 0

        # Get cache stats
        cache_hits = sum(
            m["value"]
            for k, m in all_metrics["counters"].items()
            if k.startswith(self.CACHE_HITS)
        )

        cache_misses = sum(
            m["value"]
            for k, m in all_metrics["counters"].items()
            if k.startswith(self.CACHE_MISSES)
        )

        cache_hit_rate = (
            (cache_hits / (cache_hits + cache_misses) * 100)
            if (cache_hits + cache_misses) > 0
            else 0
        )

        return {
            "summary": {
                "total_calls": total_calls,
                "error_rate": round(error_rate, 2),
                "cache_hit_rate": round(cache_hit_rate, 2),
                "active_connections": all_metrics["gauges"]
                .get(self.ACTIVE_CONNECTIONS, {})
                .get("value", 0),
            },
            "tool_performance": self._get_tool_performance(all_metrics),
            "rate_limits": self._get_rate_limit_stats(all_metrics),
            "auth_stats": self._get_auth_stats(all_metrics),
            "raw_metrics": all_metrics,
        }

    def _get_tool_performance(self, metrics: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract tool performance metrics."""
        tool_stats = {}

        # Aggregate by tool
        for key, data in metrics["timers"].items():
            if key.startswith(self.TOOL_CALL_DURATION):
                tool = data["labels"].get("tool", "unknown")
                if tool not in tool_stats:
                    tool_stats[tool] = {
                        "tool": tool,
                        "calls": 0,
                        "avg_duration": 0,
                        "p95_duration": 0,
                        "p99_duration": 0,
                    }

                summary = data["summary"]
                tool_stats[tool]["calls"] += summary["count"]
                tool_stats[tool]["avg_duration"] = summary["avg"]
                tool_stats[tool]["p95_duration"] = summary["p95"]
                tool_stats[tool]["p99_duration"] = summary["p99"]

        return list(tool_stats.values())

    def _get_rate_limit_stats(self, metrics: dict[str, Any]) -> dict[str, int]:
        """Extract rate limit statistics."""
        stats = {}

        for key, data in metrics["counters"].items():
            if key.startswith(self.RATE_LIMIT_HITS):
                tier = data["labels"].get("tier", "unknown")
                stats[tier] = data["value"]

        return stats

    def _get_auth_stats(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Extract authentication statistics."""
        attempts = 0
        failures = 0

        for key, data in metrics["counters"].items():
            if key.startswith(self.AUTH_ATTEMPTS):
                attempts += data["value"]
            elif key.startswith(self.AUTH_FAILURES):
                failures += data["value"]

        return {
            "total_attempts": attempts,
            "total_failures": failures,
            "success_rate": (
                round((1 - failures / attempts) * 100, 2) if attempts > 0 else 100
            ),
        }


# Singleton instance
_mcp_metrics: MCPMetrics | None = None


def get_mcp_metrics() -> MCPMetrics:
    """Get the singleton MCP metrics instance."""
    global _mcp_metrics
    if _mcp_metrics is None:
        _mcp_metrics = MCPMetrics()
    return _mcp_metrics
