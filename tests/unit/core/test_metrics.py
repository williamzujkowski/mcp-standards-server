"""
Unit tests for metrics collection module.
"""

import asyncio
import time

import pytest

from src.core.metrics import MCPMetrics, MetricsCollector, MetricType, get_mcp_metrics


class TestMetricsCollector:
    """Test basic metrics collector functionality."""

    @pytest.fixture
    def collector(self):
        """Create metrics collector instance."""
        return MetricsCollector(window_size=60, max_samples=100)

    def test_counter_increment(self, collector):
        """Test counter metric increments."""
        collector.increment("test_counter")
        collector.increment("test_counter", 2.0)
        collector.increment("test_counter", labels={"env": "test"})

        # Check values
        assert collector._counters["test_counter"] == 3.0
        assert collector._counters["test_counter:env=test"] == 1.0

    def test_gauge_set(self, collector):
        """Test gauge metric setting."""
        collector.gauge("test_gauge", 42.0)
        collector.gauge("test_gauge", 50.0)  # Should overwrite
        collector.gauge("test_gauge", 10.0, labels={"type": "cpu"})

        assert collector._gauges["test_gauge"] == 50.0
        assert collector._gauges["test_gauge:type=cpu"] == 10.0

    def test_histogram_recording(self, collector):
        """Test histogram value recording."""
        for i in range(10):
            collector.histogram("test_histogram", float(i))

        key = "test_histogram"
        assert len(collector._histograms[key]) == 10

        # Test summary calculation
        summary = collector.get_summary("test_histogram", MetricType.HISTOGRAM)
        assert summary.count == 10
        assert summary.min == 0.0
        assert summary.max == 9.0
        assert summary.avg == 4.5
        assert summary.p50 == 5.0  # Median of 0-9 is 5

    def test_timer_context(self, collector):
        """Test timer context manager."""
        with collector.timer("test_timer"):
            time.sleep(0.01)  # Sleep 10ms

        key = "test_timer"
        assert len(collector._timers[key]) == 1
        assert collector._timers[key][0].value >= 0.01

    def test_timer_with_labels(self, collector):
        """Test timer with labels."""
        labels = {"operation": "fetch", "status": "success"}

        with collector.timer("db_query", labels=labels):
            time.sleep(0.01)

        key = "db_query:operation=fetch,status=success"
        assert len(collector._timers[key]) == 1

    def test_get_all_metrics(self, collector):
        """Test getting all metrics."""
        # Add various metrics
        collector.increment("requests_total", labels={"method": "GET"})
        collector.gauge("cpu_usage", 75.5)
        collector.histogram("response_size", 1024)
        collector.record_duration("request_duration", 0.123)

        metrics = collector.get_all_metrics()

        assert "timestamp" in metrics
        assert len(metrics["counters"]) == 1
        assert len(metrics["gauges"]) == 1
        assert len(metrics["histograms"]) == 1
        assert len(metrics["timers"]) == 1

    def test_prometheus_export(self, collector):
        """Test Prometheus format export."""
        collector.increment("http_requests", labels={"method": "GET"})
        collector.gauge("memory_usage", 1024.5)

        prometheus_output = collector.export_to_prometheus()

        assert 'http_requests_total{method="GET"}' in prometheus_output
        assert "memory_usage" in prometheus_output
        assert "1024.5" in prometheus_output

    def test_window_filtering(self, collector):
        """Test that old metrics are filtered out."""
        # Add old metric
        old_point = collector._histograms["test_metric"]
        from src.core.metrics import MetricPoint

        old_point.append(MetricPoint(time.time() - 120, 100, {}))  # 2 minutes ago

        # Add recent metric
        collector.histogram("test_metric", 200)

        # Get summary with 60s window
        summary = collector.get_summary("test_metric", MetricType.HISTOGRAM)

        # Should only include recent value
        assert summary.count == 1
        assert summary.min == 200
        assert summary.max == 200

    @pytest.mark.asyncio
    async def test_export_handlers(self, collector):
        """Test export handler functionality."""
        exported_metrics = None

        def handler(metrics):
            nonlocal exported_metrics
            exported_metrics = metrics

        collector.add_export_handler(handler)
        collector.export_interval = 0.01  # 10ms for testing

        await collector.start_export_task()
        await asyncio.sleep(0.02)  # Wait for export
        await collector.stop_export_task()

        assert exported_metrics is not None
        assert "timestamp" in exported_metrics


class TestMCPMetrics:
    """Test MCP-specific metrics."""

    @pytest.fixture
    def mcp_metrics(self):
        """Create MCP metrics instance."""
        return MCPMetrics()

    def test_record_tool_call_success(self, mcp_metrics):
        """Test recording successful tool call."""
        mcp_metrics.record_tool_call("get_standards", 0.123, success=True)

        # Check counters
        key = "mcp_tool_calls_total:success=true,tool=get_standards"
        assert mcp_metrics.collector._counters[key] == 1.0

        # Check timer
        timer_key = "mcp_tool_call_duration_seconds:success=true,tool=get_standards"
        assert len(mcp_metrics.collector._timers[timer_key]) == 1

    def test_record_tool_call_failure(self, mcp_metrics):
        """Test recording failed tool call."""
        mcp_metrics.record_tool_call(
            "validate_code", 0.456, success=False, error_type="validation_error"
        )

        # Check error counter
        error_key = (
            "mcp_tool_call_errors_total:error_type=validation_error,tool=validate_code"
        )
        assert mcp_metrics.collector._counters[error_key] == 1.0

    def test_record_auth_attempts(self, mcp_metrics):
        """Test recording authentication attempts."""
        mcp_metrics.record_auth_attempt("jwt", success=True)
        mcp_metrics.record_auth_attempt("jwt", success=False)
        mcp_metrics.record_auth_attempt("api_key", success=True)

        # Check counters
        assert (
            mcp_metrics.collector._counters[
                "mcp_auth_attempts_total:success=true,type=jwt"
            ]
            == 1.0
        )
        assert (
            mcp_metrics.collector._counters[
                "mcp_auth_attempts_total:success=false,type=jwt"
            ]
            == 1.0
        )
        assert (
            mcp_metrics.collector._counters["mcp_auth_failures_total:type=jwt"] == 1.0
        )

    def test_record_cache_access(self, mcp_metrics):
        """Test recording cache hits and misses."""
        mcp_metrics.record_cache_access("get_standards", hit=True)
        mcp_metrics.record_cache_access("get_standards", hit=True)
        mcp_metrics.record_cache_access("get_standards", hit=False)

        assert (
            mcp_metrics.collector._counters["mcp_cache_hits_total:tool=get_standards"]
            == 2.0
        )
        assert (
            mcp_metrics.collector._counters["mcp_cache_misses_total:tool=get_standards"]
            == 1.0
        )

    def test_update_active_connections(self, mcp_metrics):
        """Test updating active connections gauge."""
        mcp_metrics.update_active_connections(5)
        mcp_metrics.update_active_connections(10)

        assert mcp_metrics.collector._gauges["mcp_active_connections"] == 10

    def test_record_sizes(self, mcp_metrics):
        """Test recording request/response sizes."""
        mcp_metrics.record_request_size(1024, "upload_file")
        mcp_metrics.record_response_size(2048, "get_standards")

        assert (
            len(
                mcp_metrics.collector._histograms[
                    "mcp_request_size_bytes:tool=upload_file"
                ]
            )
            == 1
        )
        assert (
            len(
                mcp_metrics.collector._histograms[
                    "mcp_response_size_bytes:tool=get_standards"
                ]
            )
            == 1
        )

    def test_dashboard_metrics(self, mcp_metrics):
        """Test dashboard metrics aggregation."""
        # Simulate some activity
        mcp_metrics.record_tool_call("tool1", 0.1, success=True)
        mcp_metrics.record_tool_call("tool1", 0.2, success=True)
        mcp_metrics.record_tool_call("tool2", 0.3, success=False, error_type="timeout")

        mcp_metrics.record_cache_access("tool1", hit=True)
        mcp_metrics.record_cache_access("tool1", hit=False)

        mcp_metrics.record_auth_attempt("jwt", success=True)
        mcp_metrics.record_auth_attempt("jwt", success=False)

        mcp_metrics.update_active_connections(3)

        # Get dashboard metrics
        dashboard = mcp_metrics.get_dashboard_metrics()

        assert dashboard["summary"]["total_calls"] == 3
        assert dashboard["summary"]["error_rate"] == 33.33  # 1 error out of 3
        assert dashboard["summary"]["cache_hit_rate"] == 50.0  # 1 hit out of 2
        assert dashboard["summary"]["active_connections"] == 3

        assert dashboard["auth_stats"]["total_attempts"] == 2
        assert dashboard["auth_stats"]["total_failures"] == 1
        assert dashboard["auth_stats"]["success_rate"] == 50.0

    def test_singleton_instance(self):
        """Test that get_mcp_metrics returns singleton."""
        metrics1 = get_mcp_metrics()
        metrics2 = get_mcp_metrics()

        assert metrics1 is metrics2
