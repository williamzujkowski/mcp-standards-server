# MCP Standards Server Performance Benchmarking Suite

A comprehensive performance benchmarking suite for the MCP Standards Server, providing detailed insights into performance characteristics, bottlenecks, and optimization opportunities.

## Overview

This benchmarking suite includes:

- **Framework**: Base classes and utilities for consistent benchmarking
- **MCP Tools**: Response time, throughput, latency, and cold start benchmarks
- **Memory Profiling**: Usage analysis, leak detection, and allocation tracking
- **Continuous Monitoring**: Real-time metrics, dashboards, and alerts
- **Load Testing**: Stress tests, scalability analysis, and breaking point detection
- **Reporting**: HTML reports, visualizations, and regression detection

## Quick Start

### Run Quick Benchmarks (CI/CD)
```bash
python benchmarks/run_benchmarks.py --mode quick
```

### Run Full Benchmark Suite
```bash
python benchmarks/run_benchmarks.py --mode full
```

### Run Stress Tests
```bash
python benchmarks/run_benchmarks.py --mode stress
```

### Run Continuous Monitoring
```bash
python benchmarks/run_benchmarks.py --mode monitor --duration 30
```

## Benchmark Components

### 1. Framework (`benchmarks/framework/`)

Core benchmarking infrastructure:

- **BaseBenchmark**: Abstract base class for all benchmarks
- **BenchmarkResult**: Standardized result container
- **StatisticalAnalyzer**: Statistical analysis utilities
- **BenchmarkVisualizer**: Chart and dashboard generation
- **RegressionDetector**: Performance regression detection

### 2. MCP Tool Benchmarks (`benchmarks/mcp_tools/`)

#### Response Time Benchmark
Measures response times for all MCP tools:
```python
benchmark = MCPResponseTimeBenchmark(iterations=100)
result = await benchmark.run()
```

#### Throughput Benchmark
Tests throughput under concurrent load:
```python
benchmark = MCPThroughputBenchmark(
    concurrent_clients=50,
    duration_seconds=60
)
```

#### Latency Distribution
Analyzes latency patterns and percentiles:
```python
benchmark = MCPLatencyBenchmark(iterations=1000)
```

#### Cold Start Analysis
Compares cold vs warm start performance:
```python
benchmark = MCPColdStartBenchmark(iterations=20)
```

### 3. Memory Profiling (`benchmarks/memory/`)

#### Memory Usage Profiling
Tracks memory usage by component:
```python
benchmark = MemoryUsageBenchmark()
```

#### Leak Detection
Identifies potential memory leaks:
```python
benchmark = LeakDetectionBenchmark(iterations=100)
```

#### Growth Analysis
Monitors memory growth over time:
```python
benchmark = MemoryGrowthBenchmark(
    duration_minutes=10,
    sample_interval=1.0
)
```

#### Allocation Tracking
Tracks memory allocations by source:
```python
benchmark = AllocationTrackingBenchmark()
```

### 4. Continuous Monitoring (`benchmarks/monitoring/`)

#### Metrics Collection
Real-time metrics with Prometheus-compatible export:
```python
collector = MetricsCollector()
await collector.start_collection()
```

#### Performance Dashboard
Live updating HTML dashboard:
```python
dashboard = PerformanceDashboard(collector)
dashboard.generate_html_dashboard(output_dir)
```

#### Alert System
Configurable performance alerts:
```python
alerts = AlertSystem(collector)
alerts.add_rule(AlertRule(
    name="high_cpu",
    metric="system_cpu_percent",
    condition="> 80",
    threshold=80,
    severity=AlertSeverity.WARNING
))
```

### 5. Load Testing (`benchmarks/load/`)

#### Stress Testing
Simulates realistic user load:
```python
benchmark = StressTestBenchmark(
    user_count=100,
    spawn_rate=5.0,
    test_duration=300,
    scenario="mixed"
)
```

Available scenarios:
- `mixed`: Balanced workload
- `read_heavy`: Focus on read operations
- `compute_heavy`: CPU-intensive operations
- `search_heavy`: Search-focused load

## Metrics Collected

### Performance Metrics
- Response time (mean, median, percentiles)
- Throughput (operations/second)
- Latency distribution
- Error rates
- Cold start penalties

### Resource Metrics
- CPU usage
- Memory usage (RSS, VMS)
- Memory growth rate
- Allocation patterns
- Garbage collection statistics

### Application Metrics
- Cache hit rates
- Token optimization efficiency
- Rule engine performance
- Search query times

## Output and Reports

### Directory Structure
```
benchmark_results/
├── YYYYMMDD_HHMMSS/
│   ├── summary.json
│   ├── [benchmark_name].json
│   ├── [benchmark_name]_dashboard.png
│   └── report.html
└── baseline/
    └── ...
```

### HTML Reports
Comprehensive HTML reports include:
- Executive summary
- Performance metrics
- Trend analysis
- Visualizations
- Recommendations

### Visualizations
- Timing distribution histograms
- Memory usage over time
- Latency percentile charts
- Performance comparison charts
- Trend analysis graphs

## CI/CD Integration

### GitHub Actions Example
```yaml
- name: Run Performance Benchmarks
  run: |
    python benchmarks/run_benchmarks.py --mode quick
    
- name: Check for Regressions
  run: |
    python benchmarks/check_regressions.py \
      --baseline benchmark_results/baseline \
      --current benchmark_results/latest
```

### Performance Gates
Set thresholds for CI/CD:
```python
# In check_regressions.py
if result.mean_time > baseline.mean_time * 1.1:  # 10% regression
    sys.exit(1)  # Fail the build
```

## Best Practices

### 1. Consistent Environment
- Run benchmarks on dedicated hardware
- Disable CPU frequency scaling
- Close unnecessary applications
- Use consistent Python versions

### 2. Statistical Validity
- Run sufficient iterations (minimum 50-100)
- Include warmup runs
- Check for outliers
- Calculate confidence intervals

### 3. Regression Detection
- Maintain baseline results
- Set reasonable thresholds
- Consider variance in measurements
- Track trends over time

### 4. Memory Profiling
- Force garbage collection before tests
- Use tracemalloc for detailed tracking
- Monitor both RSS and heap usage
- Check for reference cycles

## Troubleshooting

### High Variance in Results
- Increase number of iterations
- Check for background processes
- Ensure consistent system state
- Look for GC interference

### Memory Profiling Issues
- Ensure tracemalloc is started
- Check for circular imports
- Verify cleanup in teardown
- Monitor system memory limits

### Load Test Failures
- Start with lower user counts
- Check resource limits (ulimit)
- Monitor system resources
- Verify test data exists

## Advanced Usage

### Custom Benchmarks
Create custom benchmarks by extending BaseBenchmark:

```python
class MyCustomBenchmark(BaseBenchmark):
    async def setup(self):
        # Initialize resources
        pass
    
    async def run_single_iteration(self):
        # Run benchmark logic
        return {"custom_metric": value}
    
    async def teardown(self):
        # Cleanup
        pass
```

### Custom Metrics
Add application-specific metrics:

```python
collector.register_metric(
    "my_custom_metric",
    "gauge",
    "Description of metric",
    "unit"
)
```

### Custom Visualizations
Extend BenchmarkVisualizer for custom charts:

```python
class MyVisualizer(BenchmarkVisualizer):
    def plot_custom_chart(self, result):
        # Custom visualization logic
        pass
```

## Contributing

When adding new benchmarks:
1. Extend appropriate base class
2. Follow naming conventions
3. Document metrics collected
4. Add to appropriate category
5. Update this README

## License

See main project LICENSE file.