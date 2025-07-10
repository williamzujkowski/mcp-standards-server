"""Real-time performance dashboard."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Optional matplotlib imports - gracefully handle missing dependency
try:
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Create dummy classes/functions for when matplotlib is not available
    mdates = None
    plt = None
    FuncAnimation = None
    Figure = None

from .metrics import MetricsCollector


class PerformanceDashboard:
    """Real-time performance monitoring dashboard."""

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        update_interval: int = 1000  # milliseconds
    ):
        self.metrics_collector = metrics_collector
        self.update_interval = update_interval
        self.figures: list[Figure] = []
        self.animations: list[FuncAnimation] = []

    def create_live_dashboard(self) -> Figure:
        """Create a live updating dashboard."""
        if not MATPLOTLIB_AVAILABLE:
            print("WARNING: matplotlib not available. Dashboard visualization disabled.")
            return None
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('MCP Standards Server - Performance Dashboard', fontsize=16)

        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. CPU and Memory usage (top left)
        ax_system = fig.add_subplot(gs[0, 0])
        ax_system.set_title('System Resources')
        ax_system.set_ylabel('Usage %')

        # 2. Request rate (top middle)
        ax_requests = fig.add_subplot(gs[0, 1])
        ax_requests.set_title('Request Rate')
        ax_requests.set_ylabel('Requests/sec')

        # 3. Response time (top right)
        ax_latency = fig.add_subplot(gs[0, 2])
        ax_latency.set_title('Response Time')
        ax_latency.set_ylabel('Latency (ms)')

        # 4. Error rate (middle left)
        ax_errors = fig.add_subplot(gs[1, 0])
        ax_errors.set_title('Error Rate')
        ax_errors.set_ylabel('Errors/min')

        # 5. Cache performance (middle middle)
        ax_cache = fig.add_subplot(gs[1, 1])
        ax_cache.set_title('Cache Hit Rate')
        ax_cache.set_ylabel('Hit Rate %')

        # 6. Memory growth (middle right)
        ax_memory = fig.add_subplot(gs[1, 2])
        ax_memory.set_title('Memory Usage')
        ax_memory.set_ylabel('Memory (MB)')

        # 7. Throughput by operation (bottom, full width)
        ax_throughput = fig.add_subplot(gs[2, :])
        ax_throughput.set_title('Operation Throughput')
        ax_throughput.set_ylabel('Operations/sec')

        # Store axes for updates
        self.axes = {
            'system': ax_system,
            'requests': ax_requests,
            'latency': ax_latency,
            'errors': ax_errors,
            'cache': ax_cache,
            'memory': ax_memory,
            'throughput': ax_throughput
        }

        # Initialize data storage
        self.data = {
            'timestamps': [],
            'cpu': [],
            'memory_percent': [],
            'memory_mb': [],
            'request_rate': [],
            'latency_p95': [],
            'error_rate': [],
            'cache_hit_rate': [],
            'throughput_by_op': {}
        }

        # Create animation
        anim = FuncAnimation(
            fig,
            self._update_dashboard,
            interval=self.update_interval,
            blit=False
        )

        self.animations.append(anim)
        self.figures.append(fig)

        return fig

    def _update_dashboard(self, frame):
        """Update dashboard with latest metrics."""
        # Get current metrics
        summary = self.metrics_collector.get_metrics_summary(window_seconds=60)

        # Update timestamp
        now = datetime.now()
        self.data['timestamps'].append(now)

        # Limit data to last 5 minutes
        cutoff = now - timedelta(minutes=5)
        self._trim_old_data(cutoff)

        # Update system metrics
        cpu = summary.get('system_cpu_percent', {}).get('stats', {}).get('latest', 0)
        mem_pct = summary.get('system_memory_percent', {}).get('stats', {}).get('latest', 0)
        mem_mb = summary.get('system_memory_rss_mb', {}).get('stats', {}).get('latest', 0)

        self.data['cpu'].append(cpu)
        self.data['memory_percent'].append(mem_pct)
        self.data['memory_mb'].append(mem_mb)

        # Update request metrics
        req_stats = summary.get('mcp_request_count', {}).get('stats', {})
        req_rate = req_stats.get('count', 0) / 60.0 if req_stats else 0
        self.data['request_rate'].append(req_rate)

        # Update latency
        latency_stats = summary.get('mcp_request_duration_ms', {}).get('stats', {})
        latency_p95 = self._calculate_percentile(latency_stats, 95) if latency_stats else 0
        self.data['latency_p95'].append(latency_p95)

        # Update error rate
        error_stats = summary.get('mcp_error_count', {}).get('stats', {})
        error_rate = error_stats.get('count', 0) if error_stats else 0
        self.data['error_rate'].append(error_rate)

        # Update cache hit rate
        cache_rate = summary.get('cache_hit_rate', {}).get('stats', {}).get('latest', 0)
        self.data['cache_hit_rate'].append(cache_rate)

        # Update plots
        self._update_plots()

    def _update_plots(self):
        """Update all plots with current data."""
        timestamps = self.data['timestamps']

        if not timestamps:
            return

        # Update system resources
        ax = self.axes['system']
        ax.clear()
        ax.plot(timestamps, self.data['cpu'], 'b-', label='CPU %')
        ax.plot(timestamps, self.data['memory_percent'], 'r-', label='Memory %')
        ax.legend()
        ax.set_ylim(0, 100)
        self._format_time_axis(ax)

        # Update request rate
        ax = self.axes['requests']
        ax.clear()
        ax.plot(timestamps, self.data['request_rate'], 'g-', linewidth=2)
        ax.fill_between(timestamps, 0, self.data['request_rate'], alpha=0.3, color='green')
        self._format_time_axis(ax)

        # Update latency
        ax = self.axes['latency']
        ax.clear()
        ax.plot(timestamps, self.data['latency_p95'], 'orange', linewidth=2)
        ax.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='SLO (100ms)')
        ax.legend()
        self._format_time_axis(ax)

        # Update error rate
        ax = self.axes['errors']
        ax.clear()
        ax.plot(timestamps, self.data['error_rate'], 'r-', linewidth=2)
        ax.fill_between(timestamps, 0, self.data['error_rate'], alpha=0.3, color='red')
        self._format_time_axis(ax)

        # Update cache hit rate
        ax = self.axes['cache']
        ax.clear()
        ax.plot(timestamps, self.data['cache_hit_rate'], 'cyan', linewidth=2)
        ax.set_ylim(0, 100)
        ax.axhline(y=80, color='g', linestyle='--', alpha=0.5, label='Target (80%)')
        ax.legend()
        self._format_time_axis(ax)

        # Update memory usage
        ax = self.axes['memory']
        ax.clear()
        ax.plot(timestamps, self.data['memory_mb'], 'purple', linewidth=2)
        ax.fill_between(timestamps, 0, self.data['memory_mb'], alpha=0.3, color='purple')
        self._format_time_axis(ax)

        # Update throughput by operation
        # This would show breakdown by MCP tool
        ax = self.axes['throughput']
        ax.clear()
        # Placeholder - would show stacked area chart of throughput by operation
        ax.text(0.5, 0.5, 'Throughput by Operation\n(To be implemented)',
                ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()

    def _format_time_axis(self, ax):
        """Format time axis for better readability."""
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.SecondLocator(interval=30))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _trim_old_data(self, cutoff: datetime):
        """Remove data older than cutoff."""
        if not self.data['timestamps']:
            return

        # Find cutoff index
        cutoff_idx = 0
        for i, ts in enumerate(self.data['timestamps']):
            if ts >= cutoff:
                cutoff_idx = i
                break

        # Trim all data arrays
        if cutoff_idx > 0:
            for key in ['timestamps', 'cpu', 'memory_percent', 'memory_mb',
                       'request_rate', 'latency_p95', 'error_rate', 'cache_hit_rate']:
                self.data[key] = self.data[key][cutoff_idx:]

    def _calculate_percentile(self, stats: dict[str, Any], percentile: int) -> float:
        """Calculate percentile from stats (simplified)."""
        # In real implementation, would maintain histogram
        # For now, use max as approximation
        return stats.get('max', 0)

    def save_snapshot(self, filepath: Path):
        """Save dashboard snapshot as image."""
        if not MATPLOTLIB_AVAILABLE:
            print("WARNING: matplotlib not available. Cannot save snapshot.")
            return
        
        if self.figures:
            self.figures[0].savefig(filepath, dpi=150, bbox_inches='tight')

    def generate_html_dashboard(self, output_dir: Path):
        """Generate static HTML dashboard."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get current metrics
        summary = self.metrics_collector.get_metrics_summary(window_seconds=300)

        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>MCP Performance Dashboard</title>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="5">
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #333; color: white; padding: 20px; border-radius: 5px; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px; }}
        .metric-card {{ background: white; padding: 20px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
        .metric-value {{ font-size: 36px; font-weight: bold; color: #333; }}
        .metric-label {{ color: #666; margin-bottom: 10px; }}
        .metric-unit {{ font-size: 18px; color: #999; }}
        .status-good {{ color: #4CAF50; }}
        .status-warning {{ color: #FF9800; }}
        .status-error {{ color: #F44336; }}
        .timestamp {{ text-align: right; color: #999; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>MCP Standards Server - Performance Dashboard</h1>
            <p class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="metrics-grid">
            {self._generate_metric_cards(summary)}
        </div>
    </div>
</body>
</html>
        """

        # Save HTML
        with open(output_dir / 'dashboard.html', 'w') as f:
            f.write(html_content)

        # Save metrics data as JSON
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def _generate_metric_cards(self, summary: dict[str, Any]) -> str:
        """Generate HTML for metric cards."""
        cards = []

        # CPU usage
        cpu = summary.get('system_cpu_percent', {}).get('stats', {}).get('latest', 0)
        cpu_status = self._get_status_class(cpu, 80, 90)
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value {cpu_status}">{cpu:.1f}<span class="metric-unit">%</span></div>
            </div>
        """)

        # Memory usage
        mem_mb = summary.get('system_memory_rss_mb', {}).get('stats', {}).get('latest', 0)
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value">{mem_mb:.1f}<span class="metric-unit">MB</span></div>
            </div>
        """)

        # Request rate
        req_stats = summary.get('mcp_request_count', {}).get('stats', {})
        req_rate = req_stats.get('count', 0) / 60.0 if req_stats else 0
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">Request Rate</div>
                <div class="metric-value">{req_rate:.1f}<span class="metric-unit">req/s</span></div>
            </div>
        """)

        # Error rate
        error_stats = summary.get('mcp_error_count', {}).get('stats', {})
        error_count = error_stats.get('count', 0) if error_stats else 0
        error_status = self._get_status_class(error_count, 1, 5, inverse=True)
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">Errors (last 5 min)</div>
                <div class="metric-value {error_status}">{error_count}</div>
            </div>
        """)

        # Cache hit rate
        cache_rate = summary.get('cache_hit_rate', {}).get('stats', {}).get('latest', 0)
        cache_status = self._get_status_class(cache_rate, 70, 50, higher_is_better=True)
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value {cache_status}">{cache_rate:.1f}<span class="metric-unit">%</span></div>
            </div>
        """)

        # Response time
        latency_stats = summary.get('mcp_request_duration_ms', {}).get('stats', {})
        avg_latency = latency_stats.get('avg', 0) if latency_stats else 0
        latency_status = self._get_status_class(avg_latency, 100, 200, inverse=True)
        cards.append(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Response Time</div>
                <div class="metric-value {latency_status}">{avg_latency:.1f}<span class="metric-unit">ms</span></div>
            </div>
        """)

        return '\n'.join(cards)

    def _get_status_class(
        self,
        value: float,
        warning_threshold: float,
        error_threshold: float,
        inverse: bool = False,
        higher_is_better: bool = False
    ) -> str:
        """Get CSS class based on value and thresholds."""
        if inverse:
            if value >= error_threshold:
                return "status-error"
            elif value >= warning_threshold:
                return "status-warning"
            else:
                return "status-good"
        elif higher_is_better:
            if value <= error_threshold:
                return "status-error"
            elif value <= warning_threshold:
                return "status-warning"
            else:
                return "status-good"
        else:
            if value >= error_threshold:
                return "status-error"
            elif value >= warning_threshold:
                return "status-warning"
            else:
                return "status-good"
