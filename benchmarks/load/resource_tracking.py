"""Resource utilization tracking during load tests."""

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import psutil


@dataclass
class ResourceSnapshot:
    """System resource snapshot."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    open_files: int
    threads: int


class ResourceTracker:
    """Track system resource utilization."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: list[ResourceSnapshot] = []
        self._tracking = False
        self._task: asyncio.Task | None = None
        self._process = psutil.Process()
        self._start_io_counters = None
        self._start_net_counters = None

    async def start_tracking(self):
        """Start resource tracking."""
        self._tracking = True
        self.snapshots.clear()

        # Get initial IO counters
        self._start_io_counters = psutil.disk_io_counters()
        self._start_net_counters = psutil.net_io_counters()

        self._task = asyncio.create_task(self._track_loop())

    async def stop_tracking(self):
        """Stop resource tracking."""
        self._tracking = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _track_loop(self):
        """Main tracking loop."""
        while self._tracking:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                await asyncio.sleep(self.interval)
            except asyncio.CancelledError:
                break

    def _take_snapshot(self) -> ResourceSnapshot:
        """Take a resource snapshot."""
        # CPU and memory
        cpu_percent = self._process.cpu_percent()
        mem_info = self._process.memory_info()
        mem_percent = self._process.memory_percent()

        # Disk IO
        current_io = psutil.disk_io_counters()
        if self._start_io_counters and current_io:
            disk_read_mb = (
                (current_io.read_bytes - self._start_io_counters.read_bytes)
                / 1024
                / 1024
            )
            disk_write_mb = (
                (current_io.write_bytes - self._start_io_counters.write_bytes)
                / 1024
                / 1024
            )
        else:
            disk_read_mb = 0
            disk_write_mb = 0

        # Network IO
        current_net = psutil.net_io_counters()
        if self._start_net_counters and current_net:
            net_sent_mb = (
                (current_net.bytes_sent - self._start_net_counters.bytes_sent)
                / 1024
                / 1024
            )
            net_recv_mb = (
                (current_net.bytes_recv - self._start_net_counters.bytes_recv)
                / 1024
                / 1024
            )
        else:
            net_sent_mb = 0
            net_recv_mb = 0

        # File descriptors and threads
        try:
            open_files = len(self._process.open_files())
        except Exception:
            open_files = 0

        try:
            threads = self._process.num_threads()
        except Exception:
            threads = 0

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_percent=mem_percent,
            memory_mb=mem_info.rss / 1024 / 1024,
            disk_io_read_mb=disk_read_mb,
            disk_io_write_mb=disk_write_mb,
            network_sent_mb=net_sent_mb,
            network_recv_mb=net_recv_mb,
            open_files=open_files,
            threads=threads,
        )

    def get_summary(self) -> dict[str, Any]:
        """Get resource utilization summary."""
        if not self.snapshots:
            return {}

        cpu_values = [s.cpu_percent for s in self.snapshots]
        memory_values = [s.memory_mb for s in self.snapshots]

        return {
            "duration_seconds": self.snapshots[-1].timestamp
            - self.snapshots[0].timestamp,
            "samples": len(self.snapshots),
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "peak": max(cpu_values),
            },
            "memory": {
                "min_mb": min(memory_values),
                "max_mb": max(memory_values),
                "avg_mb": sum(memory_values) / len(memory_values),
                "growth_mb": memory_values[-1] - memory_values[0],
            },
            "io": {
                "total_disk_read_mb": self.snapshots[-1].disk_io_read_mb,
                "total_disk_write_mb": self.snapshots[-1].disk_io_write_mb,
                "total_network_sent_mb": self.snapshots[-1].network_sent_mb,
                "total_network_recv_mb": self.snapshots[-1].network_recv_mb,
            },
            "resources": {
                "max_open_files": max(s.open_files for s in self.snapshots),
                "max_threads": max(s.threads for s in self.snapshots),
            },
        }

    def detect_resource_issues(self) -> list[str]:
        """Detect potential resource issues."""
        issues = []

        if not self.snapshots:
            return issues

        summary = self.get_summary()

        # CPU issues
        if summary["cpu"]["avg"] > 80:
            issues.append(f"High average CPU usage: {summary['cpu']['avg']:.1f}%")

        if summary["cpu"]["peak"] > 95:
            issues.append(
                f"CPU saturation detected: {summary['cpu']['peak']:.1f}% peak"
            )

        # Memory issues
        if summary["memory"]["growth_mb"] > 100:
            issues.append(
                f"Significant memory growth: {summary['memory']['growth_mb']:.1f}MB"
            )

        # Check for memory leak pattern
        if len(self.snapshots) > 10:
            # Simple linear regression to detect trend
            x = list(range(len(self.snapshots)))
            y = [s.memory_mb for s in self.snapshots]

            # Calculate slope
            n = len(x)
            x_mean = sum(x) / n
            y_mean = sum(y) / n

            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator > 0:
                slope = numerator / denominator
                # If memory growing more than 1MB per sample
                if slope > 1:
                    issues.append(
                        f"Potential memory leak: {slope:.2f}MB/sample growth rate"
                    )

        # File descriptor issues
        if summary["resources"]["max_open_files"] > 1000:
            issues.append(
                f"High file descriptor usage: {summary['resources']['max_open_files']}"
            )

        return issues
