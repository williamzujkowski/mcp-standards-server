"""Benchmarking suite for MCP Standards Server.

This package contains comprehensive benchmarking tools for performance analysis,
load testing, memory profiling, and regression detection.
"""

from .framework import (
    BaseBenchmark,
    BenchmarkResult,
    BenchmarkSuite,
    BenchmarkVisualizer,
    RegressionDetector,
)
from .load import StressTestBenchmark
from .mcp_tools import (
    MCPColdStartBenchmark,
    MCPLatencyBenchmark,
    MCPResponseTimeBenchmark,
    MCPThroughputBenchmark,
)
from .memory import (
    AllocationTrackingBenchmark,
    LeakDetectionBenchmark,
    MemoryGrowthBenchmark,
    MemoryUsageBenchmark,
)
from .monitoring import (
    AlertSystem,
    MetricsCollector,
    PerformanceDashboard,
)

__all__ = [
    # Framework
    "BaseBenchmark",
    "BenchmarkResult",
    "BenchmarkSuite",
    "BenchmarkVisualizer",
    "RegressionDetector",
    # MCP Tools
    "MCPColdStartBenchmark",
    "MCPLatencyBenchmark",
    "MCPResponseTimeBenchmark",
    "MCPThroughputBenchmark",
    # Memory
    "AllocationTrackingBenchmark",
    "LeakDetectionBenchmark",
    "MemoryGrowthBenchmark",
    "MemoryUsageBenchmark",
    # Load
    "StressTestBenchmark",
    # Monitoring
    "AlertSystem",
    "MetricsCollector",
    "PerformanceDashboard",
]
