"""
Comprehensive performance monitoring and metrics collection system.

This module provides:
- Real-time performance metrics collection
- Historical performance tracking
- Performance alerting and thresholds
- Metrics aggregation and analysis
- Dashboard data preparation
- Performance benchmarking
"""

import asyncio
import logging
import statistics
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

import numpy as np
import psutil
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from ..cache.redis_client import RedisCache

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    TIMER = "timer"


class AlertLevel(Enum):
    """Alert levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric."""

    name: str
    type: MetricType
    description: str
    labels: list[str] = field(default_factory=list)
    buckets: list[float] | None = None  # For histograms
    quantiles: list[float] | None = None  # For summaries
    unit: str = ""


@dataclass
class MetricValue:
    """A metric value with timestamp."""

    value: float
    timestamp: float
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "eq"
    alert_level: AlertLevel
    message: str
    cooldown_seconds: int = 300  # 5 minutes


@dataclass
class PerformanceAlert:
    """Performance alert."""

    metric_name: str
    current_value: float
    threshold: PerformanceThreshold
    timestamp: float
    resolved: bool = False
    resolved_at: float | None = None


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring."""

    # Collection settings
    collection_interval: float = 30.0  # seconds
    retention_period: int = 86400  # 24 hours in seconds
    max_samples_per_metric: int = 2880  # 30 days at 30s intervals

    # Metrics settings
    enable_system_metrics: bool = True
    enable_application_metrics: bool = True
    enable_custom_metrics: bool = True

    # Alerting settings
    enable_alerting: bool = True
    alert_cooldown: int = 300  # 5 minutes

    # Storage settings
    enable_redis_storage: bool = True
    redis_key_prefix: str = "metrics"
    redis_ttl: int = 86400  # 24 hours

    # Prometheus settings
    enable_prometheus: bool = True
    prometheus_port: int = 8000

    # Performance optimization
    async_collection: bool = True
    batch_size: int = 100
    max_workers: int = 4


class MetricCollector:
    """Collects and stores performance metrics."""

    def __init__(self, config: PerformanceConfig) -> None:
        self.config = config
        self.metrics_registry: dict[str, Any] = {}
        self.prometheus_registry = CollectorRegistry()
        self.prometheus_metrics: dict[str, Any] = {}

        # Data storage
        self.metric_data: defaultdict[str, Any] = defaultdict(
            lambda: deque(maxlen=config.max_samples_per_metric)
        )
        self.metric_locks: defaultdict[str, Any] = defaultdict(threading.Lock)

        # Alerting
        self.thresholds: dict[str, Any] = {}
        self.active_alerts: dict[str, Any] = {}
        self.alert_callbacks: list[Callable[[str, Any], None]] = []

        # Task management
        self.collection_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()

        # Redis storage
        self.redis_cache: RedisCache | None = None

        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)

    def set_redis_cache(self, redis_cache: RedisCache) -> None:
        """Set Redis cache for metric storage."""
        self.redis_cache = redis_cache

    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric definition."""
        self.metrics_registry[definition.name] = definition

        # Create Prometheus metric if enabled
        if self.config.enable_prometheus:
            self._create_prometheus_metric(definition)

    def _create_prometheus_metric(self, definition: MetricDefinition) -> None:
        """Create Prometheus metric."""
        metric_kwargs = {
            "name": definition.name,
            "documentation": definition.description,
            "labelnames": definition.labels,
            "registry": self.prometheus_registry,
        }

        metric: Any
        if definition.type == MetricType.COUNTER:
            metric = Counter(
                name=str(metric_kwargs["name"]),
                documentation=str(metric_kwargs["documentation"]),
                labelnames=cast(list[str], metric_kwargs["labelnames"]),
                registry=cast(CollectorRegistry, metric_kwargs["registry"]),
            )
        elif definition.type == MetricType.GAUGE:
            metric = Gauge(
                name=str(metric_kwargs["name"]),
                documentation=str(metric_kwargs["documentation"]),
                labelnames=cast(list[str], metric_kwargs["labelnames"]),
                registry=cast(CollectorRegistry, metric_kwargs["registry"]),
            )
        elif definition.type == MetricType.HISTOGRAM:
            if definition.buckets:
                metric_kwargs["buckets"] = definition.buckets
            metric = Histogram(
                name=str(metric_kwargs["name"]),
                documentation=str(metric_kwargs["documentation"]),
                labelnames=cast(list[str], metric_kwargs["labelnames"]),
                registry=cast(CollectorRegistry, metric_kwargs["registry"]),
                buckets=cast(
                    list[float], metric_kwargs.get("buckets", Histogram.DEFAULT_BUCKETS)
                ),
            )
        elif definition.type == MetricType.SUMMARY:
            if definition.quantiles:
                metric_kwargs["quantiles"] = definition.quantiles
            metric = Summary(
                name=str(metric_kwargs["name"]),
                documentation=str(metric_kwargs["documentation"]),
                labelnames=cast(list[str], metric_kwargs["labelnames"]),
                registry=cast(CollectorRegistry, metric_kwargs["registry"]),
            )
        else:
            return

        self.prometheus_metrics[definition.name] = metric

    def record_metric(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a metric value."""
        if name not in self.metrics_registry:
            logger.warning(f"Metric {name} not registered")
            return

        labels = labels or {}
        timestamp = time.time()

        # Store in local data
        metric_value = MetricValue(value=value, timestamp=timestamp, labels=labels)

        with self.metric_locks[name]:
            self.metric_data[name].append(metric_value)

        # Update Prometheus metric
        if self.config.enable_prometheus and name in self.prometheus_metrics:
            prometheus_metric = self.prometheus_metrics[name]
            definition = self.metrics_registry[name]

            if definition.type == MetricType.COUNTER:
                prometheus_metric.labels(**labels).inc(value)
            elif definition.type == MetricType.GAUGE:
                prometheus_metric.labels(**labels).set(value)
            elif definition.type in [MetricType.HISTOGRAM, MetricType.SUMMARY]:
                prometheus_metric.labels(**labels).observe(value)

        # Store in Redis if enabled
        if self.config.enable_redis_storage and self.redis_cache:
            asyncio.create_task(self._store_metric_in_redis(name, metric_value))

        # Check thresholds
        if self.config.enable_alerting:
            self._check_thresholds(name, value)

    async def _store_metric_in_redis(
        self, name: str, metric_value: MetricValue
    ) -> None:
        """Store metric in Redis."""
        try:
            key = f"{self.config.redis_key_prefix}:{name}:{int(metric_value.timestamp)}"
            data = {
                "value": metric_value.value,
                "timestamp": metric_value.timestamp,
                "labels": metric_value.labels,
            }

            if self.redis_cache is not None:
                await self.redis_cache.async_set(key, data, ttl=self.config.redis_ttl)
        except Exception as e:
            logger.error(f"Failed to store metric in Redis: {e}")

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold."""
        if threshold.metric_name not in self.thresholds:
            self.thresholds[threshold.metric_name] = []
        self.thresholds[threshold.metric_name].append(threshold)

    def _check_thresholds(self, metric_name: str, value: float) -> None:
        """Check if metric value exceeds thresholds."""
        if metric_name not in self.thresholds:
            return

        current_time = time.time()

        for threshold in self.thresholds[metric_name]:
            # Check if threshold is exceeded
            exceeded = False

            if threshold.comparison == "gt" and value > threshold.threshold_value:
                exceeded = True
            elif threshold.comparison == "lt" and value < threshold.threshold_value:
                exceeded = True
            elif (
                threshold.comparison == "eq"
                and abs(value - threshold.threshold_value) < 0.001
            ):
                exceeded = True

            alert_key = f"{metric_name}:{threshold.alert_level.value}"

            if exceeded:
                # Check cooldown
                if alert_key in self.active_alerts:
                    last_alert = self.active_alerts[alert_key]
                    if current_time - last_alert.timestamp < threshold.cooldown_seconds:
                        continue

                # Create alert
                alert = PerformanceAlert(
                    metric_name=metric_name,
                    current_value=value,
                    threshold=threshold,
                    timestamp=current_time,
                )

                self.active_alerts[alert_key] = alert

                # Trigger alert callbacks
                self._trigger_alert(alert)
            else:
                # Resolve alert if it was active
                if alert_key in self.active_alerts:
                    alert = self.active_alerts[alert_key]
                    if not alert.resolved:
                        alert.resolved = True
                        alert.resolved_at = current_time
                        self._trigger_alert(alert)

    def _trigger_alert(self, alert: PerformanceAlert) -> None:
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                # Convert alert to string for callback compatibility
                alert_str = f"{alert.metric_name}: {alert.current_value} (threshold: {alert.threshold})"
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(alert_str, alert))
                else:
                    callback(alert_str, alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback."""
        self.alert_callbacks.append(callback)

    def remove_alert_callback(self, callback: Callable) -> None:
        """Remove alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)

    def get_metric_data(
        self,
        metric_name: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[MetricValue]:
        """Get metric data for a time range."""
        if metric_name not in self.metric_data:
            return []

        with self.metric_locks[metric_name]:
            data = list(self.metric_data[metric_name])

        # Filter by time range
        if start_time is not None:
            data = [d for d in data if d.timestamp >= start_time]
        if end_time is not None:
            data = [d for d in data if d.timestamp <= end_time]

        return data

    def get_metric_statistics(
        self,
        metric_name: str,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> dict[str, Any]:
        """Get statistics for a metric."""
        data = self.get_metric_data(metric_name, start_time, end_time)

        if not data:
            return {}

        values = [d.value for d in data]

        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "first_timestamp": data[0].timestamp,
            "last_timestamp": data[-1].timestamp,
        }

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        if not self.config.enable_prometheus:
            return ""

        result: str = generate_latest(self.prometheus_registry).decode("utf-8")
        return result

    async def start(self) -> None:
        """Start metric collection."""
        if self.config.async_collection:
            self.collection_task = asyncio.create_task(self._collection_worker())

        logger.info("Metric collector started")

    async def stop(self) -> None:
        """Stop metric collection."""
        self.shutdown_event.set()

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        self.executor.shutdown(wait=True)
        logger.info("Metric collector stopped")

    async def _collection_worker(self) -> None:
        """Worker task for automatic metric collection."""
        while not self.shutdown_event.is_set():
            try:
                if self.config.enable_system_metrics:
                    await self._collect_system_metrics()

                await asyncio.sleep(self.config.collection_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metric collection: {e}")
                await asyncio.sleep(self.config.collection_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("system_cpu_percent", cpu_percent)

            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("system_memory_used_bytes", memory.used)
            self.record_metric("system_memory_available_bytes", memory.available)
            self.record_metric("system_memory_percent", memory.percent)

            # Disk metrics
            disk = psutil.disk_usage("/")
            self.record_metric("system_disk_used_bytes", disk.used)
            self.record_metric("system_disk_free_bytes", disk.free)
            self.record_metric("system_disk_percent", disk.used / disk.total * 100)

            # Network metrics
            net_io = psutil.net_io_counters()
            self.record_metric("system_network_bytes_sent", net_io.bytes_sent)
            self.record_metric("system_network_bytes_recv", net_io.bytes_recv)

        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class PerformanceMonitor:
    """Main performance monitoring system."""

    def __init__(self, config: PerformanceConfig | None = None) -> None:
        self.config = config or PerformanceConfig()
        self.collector = MetricCollector(self.config)
        self.benchmarks: dict[str, Benchmark] = {}
        self.running = False

        # Initialize standard metrics
        self._register_standard_metrics()

        # Initialize standard thresholds
        self._register_standard_thresholds()

    def _register_standard_metrics(self) -> None:
        """Register standard application metrics."""
        metrics = [
            # System metrics
            MetricDefinition(
                "system_cpu_percent", MetricType.GAUGE, "CPU usage percentage"
            ),
            MetricDefinition(
                "system_memory_used_bytes", MetricType.GAUGE, "Memory used in bytes"
            ),
            MetricDefinition(
                "system_memory_available_bytes",
                MetricType.GAUGE,
                "Available memory in bytes",
            ),
            MetricDefinition(
                "system_memory_percent", MetricType.GAUGE, "Memory usage percentage"
            ),
            MetricDefinition(
                "system_disk_used_bytes", MetricType.GAUGE, "Disk space used in bytes"
            ),
            MetricDefinition(
                "system_disk_free_bytes", MetricType.GAUGE, "Free disk space in bytes"
            ),
            MetricDefinition(
                "system_disk_percent", MetricType.GAUGE, "Disk usage percentage"
            ),
            MetricDefinition(
                "system_network_bytes_sent", MetricType.COUNTER, "Network bytes sent"
            ),
            MetricDefinition(
                "system_network_bytes_recv",
                MetricType.COUNTER,
                "Network bytes received",
            ),
            # Application metrics
            MetricDefinition(
                "app_request_count",
                MetricType.COUNTER,
                "Total requests",
                ["method", "endpoint"],
            ),
            MetricDefinition(
                "app_request_duration_seconds",
                MetricType.HISTOGRAM,
                "Request duration",
                ["method", "endpoint"],
            ),
            MetricDefinition(
                "app_error_count", MetricType.COUNTER, "Total errors", ["error_type"]
            ),
            MetricDefinition(
                "app_cache_hits", MetricType.COUNTER, "Cache hits", ["cache_type"]
            ),
            MetricDefinition(
                "app_cache_misses", MetricType.COUNTER, "Cache misses", ["cache_type"]
            ),
            MetricDefinition(
                "app_db_query_duration_seconds",
                MetricType.HISTOGRAM,
                "Database query duration",
                ["query_type"],
            ),
            MetricDefinition(
                "app_active_connections",
                MetricType.GAUGE,
                "Active connections",
                ["connection_type"],
            ),
            # Search metrics
            MetricDefinition(
                "search_query_count",
                MetricType.COUNTER,
                "Search queries",
                ["query_type"],
            ),
            MetricDefinition(
                "search_query_duration_seconds",
                MetricType.HISTOGRAM,
                "Search query duration",
                ["query_type"],
            ),
            MetricDefinition(
                "search_results_count",
                MetricType.HISTOGRAM,
                "Search results count",
                ["query_type"],
            ),
            MetricDefinition(
                "search_cache_hits", MetricType.COUNTER, "Search cache hits"
            ),
            MetricDefinition(
                "search_cache_misses", MetricType.COUNTER, "Search cache misses"
            ),
            # MCP metrics
            MetricDefinition(
                "mcp_tool_calls", MetricType.COUNTER, "MCP tool calls", ["tool_name"]
            ),
            MetricDefinition(
                "mcp_tool_duration_seconds",
                MetricType.HISTOGRAM,
                "MCP tool duration",
                ["tool_name"],
            ),
            MetricDefinition(
                "mcp_tool_errors",
                MetricType.COUNTER,
                "MCP tool errors",
                ["tool_name", "error_type"],
            ),
        ]

        for metric in metrics:
            self.collector.register_metric(metric)

    def _register_standard_thresholds(self) -> None:
        """Register standard performance thresholds."""
        thresholds = [
            # System thresholds
            PerformanceThreshold(
                "system_cpu_percent", 80.0, "gt", AlertLevel.WARNING, "High CPU usage"
            ),
            PerformanceThreshold(
                "system_cpu_percent",
                95.0,
                "gt",
                AlertLevel.CRITICAL,
                "Critical CPU usage",
            ),
            PerformanceThreshold(
                "system_memory_percent",
                80.0,
                "gt",
                AlertLevel.WARNING,
                "High memory usage",
            ),
            PerformanceThreshold(
                "system_memory_percent",
                95.0,
                "gt",
                AlertLevel.CRITICAL,
                "Critical memory usage",
            ),
            PerformanceThreshold(
                "system_disk_percent", 80.0, "gt", AlertLevel.WARNING, "High disk usage"
            ),
            PerformanceThreshold(
                "system_disk_percent",
                95.0,
                "gt",
                AlertLevel.CRITICAL,
                "Critical disk usage",
            ),
            # Application thresholds
            PerformanceThreshold(
                "app_request_duration_seconds",
                1.0,
                "gt",
                AlertLevel.WARNING,
                "Slow request",
            ),
            PerformanceThreshold(
                "app_request_duration_seconds",
                5.0,
                "gt",
                AlertLevel.CRITICAL,
                "Very slow request",
            ),
            PerformanceThreshold(
                "app_db_query_duration_seconds",
                0.5,
                "gt",
                AlertLevel.WARNING,
                "Slow database query",
            ),
            PerformanceThreshold(
                "app_db_query_duration_seconds",
                2.0,
                "gt",
                AlertLevel.CRITICAL,
                "Very slow database query",
            ),
            # Search thresholds
            PerformanceThreshold(
                "search_query_duration_seconds",
                0.5,
                "gt",
                AlertLevel.WARNING,
                "Slow search query",
            ),
            PerformanceThreshold(
                "search_query_duration_seconds",
                2.0,
                "gt",
                AlertLevel.CRITICAL,
                "Very slow search query",
            ),
        ]

        for threshold in thresholds:
            self.collector.add_threshold(threshold)

    def set_redis_cache(self, redis_cache: RedisCache) -> None:
        """Set Redis cache for metric storage."""
        self.collector.set_redis_cache(redis_cache)

    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a custom metric."""
        self.collector.register_metric(definition)

    def record_metric(
        self, name: str, value: float, labels: dict[str, str] | None = None
    ) -> None:
        """Record a metric value."""
        self.collector.record_metric(name, value, labels)

    def add_threshold(self, threshold: PerformanceThreshold) -> None:
        """Add a performance threshold."""
        self.collector.add_threshold(threshold)

    def add_alert_callback(self, callback: Callable) -> None:
        """Add alert callback."""
        self.collector.add_alert_callback(callback)

    def time_operation(
        self, metric_name: str, labels: dict[str, str] | None = None
    ) -> "TimingContext":
        """Context manager for timing operations."""
        return TimingContext(self, metric_name, labels)

    def create_benchmark(self, name: str, operation: Callable) -> "Benchmark":
        """Create a performance benchmark."""
        benchmark = Benchmark(name, operation, self)
        self.benchmarks[name] = benchmark
        return benchmark

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get summary of all metrics."""
        summary = {}

        for metric_name in self.collector.metrics_registry.keys():
            stats = self.collector.get_metric_statistics(metric_name)
            if stats:
                summary[metric_name] = stats

        return summary

    def get_dashboard_data(self) -> dict[str, Any]:
        """Get data formatted for dashboard display."""
        current_time = time.time()
        hour_ago = current_time - 3600

        dashboard: dict[str, Any] = {
            "timestamp": current_time,
            "system_metrics": {},
            "application_metrics": {},
            "alerts": [],
        }

        # System metrics
        for metric_name in [
            "system_cpu_percent",
            "system_memory_percent",
            "system_disk_percent",
        ]:
            stats = self.collector.get_metric_statistics(metric_name, hour_ago)
            if stats:
                dashboard["system_metrics"][metric_name] = {
                    "current": stats["mean"],
                    "max": stats["max"],
                    "min": stats["min"],
                    "trend": self._calculate_trend(metric_name, hour_ago),
                }

        # Application metrics
        for metric_name in [
            "app_request_count",
            "app_request_duration_seconds",
            "app_error_count",
        ]:
            stats = self.collector.get_metric_statistics(metric_name, hour_ago)
            if stats:
                dashboard["application_metrics"][metric_name] = stats

        # Active alerts
        dashboard["alerts"] = [
            {
                "metric": alert.metric_name,
                "level": alert.threshold.alert_level.value,
                "message": alert.threshold.message,
                "value": alert.current_value,
                "threshold": alert.threshold.threshold_value,
                "timestamp": alert.timestamp,
                "resolved": alert.resolved,
            }
            for alert in self.collector.active_alerts.values()
        ]

        return dashboard

    def _calculate_trend(self, metric_name: str, start_time: float) -> str:
        """Calculate trend for a metric."""
        data = self.collector.get_metric_data(metric_name, start_time)
        if len(data) < 2:
            return "stable"

        # Simple trend calculation
        first_half = data[: len(data) // 2]
        second_half = data[len(data) // 2 :]

        first_avg = sum(d.value for d in first_half) / len(first_half)
        second_avg = sum(d.value for d in second_half) / len(second_half)

        if second_avg > first_avg * 1.1:
            return "increasing"
        elif second_avg < first_avg * 0.9:
            return "decreasing"
        else:
            return "stable"

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics."""
        return self.collector.get_prometheus_metrics()

    async def start(self) -> None:
        """Start performance monitoring."""
        await self.collector.start()
        self.running = True
        logger.info("Performance monitor started")

    async def stop(self) -> None:
        """Stop performance monitoring."""
        await self.collector.stop()
        self.running = False
        logger.info("Performance monitor stopped")


class TimingContext:
    """Context manager for timing operations."""

    def __init__(
        self,
        monitor: PerformanceMonitor,
        metric_name: str,
        labels: dict[str, str] | None = None,
    ):
        self.monitor = monitor
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time: float | None = None

    def __enter__(self) -> "TimingContext":
        self.start_time = time.time()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_metric(self.metric_name, duration, self.labels)


class Benchmark:
    """Performance benchmark."""

    def __init__(
        self, name: str, operation: Callable, monitor: PerformanceMonitor
    ) -> None:
        self.name = name
        self.operation = operation
        self.monitor = monitor
        self.results: list[dict[str, Any]] = []

    def run(self, iterations: int = 100) -> dict[str, Any]:
        """Run benchmark."""
        self.results = []

        for i in range(iterations):
            start_time = time.time()

            try:
                result = self.operation()
                duration = time.time() - start_time
                self.results.append(
                    {
                        "iteration": i,
                        "duration": duration,
                        "success": True,
                        "result": result,
                    }
                )
            except Exception as e:
                duration = time.time() - start_time
                self.results.append(
                    {
                        "iteration": i,
                        "duration": duration,
                        "success": False,
                        "error": str(e),
                    }
                )

        return self.get_summary()

    def get_summary(self) -> dict[str, Any]:
        """Get benchmark summary."""
        if not self.results:
            return {}

        durations = [r["duration"] for r in self.results]
        successes = [r for r in self.results if r["success"]]

        return {
            "name": self.name,
            "iterations": len(self.results),
            "success_rate": len(successes) / len(self.results),
            "total_duration": sum(durations),
            "average_duration": statistics.mean(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "median_duration": statistics.median(durations),
            "std_duration": statistics.stdev(durations) if len(durations) > 1 else 0,
            "p95_duration": np.percentile(durations, 95),
            "p99_duration": np.percentile(durations, 99),
            "throughput": (
                len(self.results) / sum(durations) if sum(durations) > 0 else 0
            ),
        }


# Global performance monitor
_global_monitor: PerformanceMonitor | None = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


async def initialize_performance_monitor(
    config: PerformanceConfig | None = None,
) -> PerformanceMonitor:
    """Initialize and start global performance monitor."""
    global _global_monitor
    _global_monitor = PerformanceMonitor(config)
    await _global_monitor.start()
    return _global_monitor


async def shutdown_performance_monitor() -> None:
    """Shutdown global performance monitor."""
    global _global_monitor
    if _global_monitor:
        await _global_monitor.stop()
        _global_monitor = None


# Convenience functions
def record_metric(
    name: str, value: float, labels: dict[str, str] | None = None
) -> None:
    """Record a metric using global monitor."""
    monitor = get_performance_monitor()
    monitor.record_metric(name, value, labels)


def time_operation(
    metric_name: str, labels: dict[str, str] | None = None
) -> TimingContext:
    """Time an operation using global monitor."""
    monitor = get_performance_monitor()
    return monitor.time_operation(metric_name, labels)
