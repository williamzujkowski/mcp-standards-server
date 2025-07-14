# Performance Reference

Comprehensive guide to optimizing MCP Standards Server performance.

## Performance Overview

The MCP Standards Server is designed for high performance with:
- Sub-100ms response times for standard operations
- Horizontal scalability
- Efficient resource utilization
- Intelligent caching strategies

## Performance Metrics

### Key Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Response Time (p50) | <50ms | Median response time |
| Response Time (p95) | <100ms | 95th percentile |
| Response Time (p99) | <200ms | 99th percentile |
| Throughput | >1000 req/s | Requests per second |
| CPU Usage | <70% | Average CPU utilization |
| Memory Usage | <512MB | Process memory |
| Cache Hit Rate | >90% | L1+L2 cache hits |

### Monitoring

```python
from src.core.performance.metrics import PerformanceMonitor

monitor = PerformanceMonitor()

# Track operation
with monitor.track("validation"):
    result = validate_code(code)

# Get metrics
metrics = monitor.get_metrics()
print(f"Average time: {metrics.avg_time}ms")
print(f"Operations/sec: {metrics.throughput}")
```

## Optimization Strategies

### 1. Caching Optimization

```yaml
# Aggressive caching configuration
cache:
  l1:
    max_size: 50000
    ttl: 600
    preload: true
  
  l2:
    pipeline_size: 100
    compression: true
    
  warming:
    parallel: true
    batch_size: 1000
```

### 2. Database Optimization

```python
# Use connection pooling
from sqlalchemy import create_engine

engine = create_engine(
    "postgresql://localhost/mcp",
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)

# Batch operations
def bulk_insert_standards(standards):
    with engine.begin() as conn:
        conn.execute(
            insert(StandardsTable),
            standards
        )
```

### 3. Async Operations

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncValidator:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=10)
    
    async def validate_files(self, files):
        """Validate files concurrently."""
        tasks = [
            self.validate_file(file)
            for file in files
        ]
        return await asyncio.gather(*tasks)
    
    async def validate_file(self, file):
        """Validate single file asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self._validate_sync,
            file
        )
```

## Code Optimization

### 1. Efficient Pattern Matching

```python
# Bad - Multiple regex compilations
def check_patterns(code):
    if re.search(r'TODO', code):
        violations.append('todo')
    if re.search(r'FIXME', code):
        violations.append('fixme')

# Good - Compile once, reuse
class PatternChecker:
    def __init__(self):
        self.patterns = {
            'todo': re.compile(r'TODO'),
            'fixme': re.compile(r'FIXME')
        }
    
    def check_patterns(self, code):
        return [
            name for name, pattern in self.patterns.items()
            if pattern.search(code)
        ]
```

### 2. Memory-Efficient Processing

```python
# Bad - Load entire file
def process_large_file(path):
    content = open(path).read()
    return process(content)

# Good - Stream processing
def process_large_file(path):
    results = []
    with open(path) as f:
        for chunk in iter(lambda: f.read(4096), ''):
            results.extend(process_chunk(chunk))
    return results
```

### 3. Lazy Loading

```python
class StandardsRepository:
    def __init__(self):
        self._standards = None
        self._index = None
    
    @property
    def standards(self):
        """Lazy load standards."""
        if self._standards is None:
            self._standards = self._load_standards()
        return self._standards
    
    @property
    def search_index(self):
        """Lazy load search index."""
        if self._index is None:
            self._index = self._build_index()
        return self._index
```

## Profiling

### CPU Profiling

```python
import cProfile
import pstats

def profile_operation():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Operation to profile
    validate_directory("src/")
    
    profiler.disable()
    
    # Analysis
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
```

### Memory Profiling

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    # Track memory usage line by line
    large_data = load_standards()  # +50MB
    processed = process_data(large_data)  # +30MB
    return compress(processed)  # -60MB
```

### Line Profiling

```python
from line_profiler import LineProfiler

def profile_critical_path():
    lp = LineProfiler()
    lp.add_function(critical_function)
    
    # Run with profiling
    lp.enable()
    result = critical_function()
    lp.disable()
    
    # Show results
    lp.print_stats()
```

## Benchmarking

### Micro-benchmarks

```python
import timeit

# Compare implementations
def benchmark_implementations():
    implementations = {
        'regex': lambda: regex_validate(code),
        'ast': lambda: ast_validate(code),
        'hybrid': lambda: hybrid_validate(code)
    }
    
    for name, func in implementations.items():
        time = timeit.timeit(func, number=1000)
        print(f"{name}: {time:.4f}s")
```

### Load Testing

```python
# locustfile.py for load testing
from locust import HttpUser, task, between

class MCPUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def validate_code(self):
        self.client.post("/api/validate", json={
            "code": "def test(): pass",
            "standard": "python-best-practices"
        })
    
    @task(1)
    def get_standards(self):
        self.client.get("/api/standards")
```

## Configuration Tuning

### High-Performance Configuration

```yaml
performance:
  # Worker configuration
  workers: ${CPU_COUNT}
  threads_per_worker: 4
  
  # Request handling
  request_timeout: 30
  keepalive_timeout: 5
  
  # Resource limits
  max_request_size: 10485760  # 10MB
  max_memory_per_request: 104857600  # 100MB
  
  # Optimization flags
  enable_jit: true
  enable_async: true
  enable_caching: true
  
  # Garbage collection
  gc:
    threshold0: 700
    threshold1: 10
    threshold2: 10
```

### Database Performance

```yaml
database:
  # Connection pool
  pool_size: 20
  max_overflow: 40
  pool_timeout: 30
  pool_recycle: 3600
  
  # Query optimization
  echo: false
  statement_timeout: 5000
  
  # Indexes
  auto_create_indexes: true
  index_cache_size: 100000
```

## Scaling Strategies

### Horizontal Scaling

```yaml
# docker-compose.yml
version: '3.8'

services:
  mcp-server:
    image: mcp-standards-server
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2'
          memory: 1G
    environment:
      - REDIS_URL=redis://redis:6379
      - DATABASE_URL=postgresql://db:5432/mcp
  
  nginx:
    image: nginx
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - mcp-server
```

### Vertical Scaling

```python
# Optimize for multi-core
import multiprocessing

def parallel_validation(files):
    cpu_count = multiprocessing.cpu_count()
    
    with multiprocessing.Pool(cpu_count) as pool:
        results = pool.map(validate_file, files)
    
    return results
```

## Performance Best Practices

### 1. Minimize I/O

```python
# Bad - Multiple file reads
for standard_id in standard_ids:
    with open(f"standards/{standard_id}.yaml") as f:
        standards.append(yaml.load(f))

# Good - Batch read
all_standards = load_all_standards()  # Single I/O
standards = [all_standards[sid] for sid in standard_ids]
```

### 2. Use Appropriate Data Structures

```python
# Bad - O(n) lookup
violations = []
for rule in rules:
    if rule.id in violations:
        continue

# Good - O(1) lookup
violations = set()
for rule in rules:
    if rule.id in violations:
        continue
```

### 3. Avoid Premature Optimization

```python
# Profile first
with profile_context():
    result = operation()

# Then optimize hot paths only
if is_hot_path:
    result = optimized_operation()
else:
    result = simple_operation()
```

## Troubleshooting Performance

### Common Issues

1. **Slow Response Times**
   - Check cache hit rates
   - Profile database queries
   - Review async operation usage

2. **High Memory Usage**
   - Implement pagination
   - Use generators for large datasets
   - Check for memory leaks

3. **CPU Bottlenecks**
   - Parallelize CPU-intensive tasks
   - Optimize regex patterns
   - Consider caching computed results

### Performance Debugging

```python
import logging
import time

logging.basicConfig(level=logging.DEBUG)

class PerformanceDebugger:
    def __init__(self):
        self.timings = {}
    
    def track(self, operation):
        start = time.time()
        yield
        duration = time.time() - start
        
        self.timings[operation] = duration
        
        if duration > 0.1:  # Log slow operations
            logging.warning(
                f"Slow operation: {operation} took {duration:.3f}s"
            )
```

## Related Documentation

- [Caching Reference](./caching.md)
- [Benchmarking Guide](../../benchmarks/README.md)
- [Monitoring Setup](../../monitoring/README.md)