# Standards Synchronization Test Suite

This comprehensive test suite provides extensive coverage for the standards synchronization functionality, including unit tests, integration tests, performance tests, and security tests.

## Test Structure

### Unit Tests

#### `/tests/unit/core/standards/test_sync_comprehensive.py`
Comprehensive unit tests covering:
- **Error Handling**: Network timeouts, disk errors, API failures
- **Edge Cases**: Empty repos, large files, Unicode filenames, concurrent operations
- **Rate Limiting**: Boundary conditions, malformed headers, clock skew
- **File Filtering**: Complex glob patterns, case sensitivity, Unicode support
- **Partial Sync**: Mixed success/failure scenarios, recovery mechanisms
- **Cache Management**: Metadata persistence, corruption recovery, TTL checking

#### `/tests/unit/core/standards/test_sync_security.py`
Security-focused tests covering:
- **Path Traversal Prevention**: Absolute paths, parent directory traversal, symlinks
- **Content Validation**: Hash verification, size limits, content type validation
- **Credential Security**: Token handling, secure transmission, no logging
- **Input Sanitization**: Filename validation, URL validation, response validation
- **Secure File Operations**: Atomic writes, directory permissions, safe replacement

### Integration Tests

#### `/tests/integration/test_sync_integration.py`
End-to-end integration tests covering:
- **Full Sync Workflows**: Initial sync, incremental sync, updates
- **CLI Integration**: Command-line interface testing
- **Cross-Platform**: Windows paths, Unicode paths
- **Error Recovery**: Partial failures, retry mechanisms
- **Rate Limit Handling**: Waiting behavior, header updates
- **Concurrent Operations**: Parallel downloads, metadata updates

### Performance Tests

#### `/tests/performance/test_sync_performance.py`
Performance benchmarks covering:
- **Large Repository**: Syncing 1000+ files efficiently
- **Concurrency**: Optimal parallel download testing
- **Memory Usage**: Memory profiling during large syncs
- **Cache Performance**: Lookup scaling, serialization speed
- **Network Performance**: Retry overhead, rate limit impact

## Running the Tests

### Run All Sync Tests
```bash
# Run all sync-related tests
pytest tests/ -k "sync" -v

# Run with coverage
pytest tests/ -k "sync" --cov=src.core.standards.sync --cov-report=html
```

### Run Specific Test Categories

#### Unit Tests Only
```bash
pytest tests/unit/core/standards/test_sync*.py -v
```

#### Integration Tests Only
```bash
pytest tests/integration/test_sync_integration.py -v
```

#### Performance Tests Only
```bash
pytest tests/performance/test_sync_performance.py -v --benchmark
```

#### Security Tests Only
```bash
pytest tests/unit/core/standards/test_sync_security.py -v
```

### Run with Specific Markers
```bash
# Run only benchmark tests
pytest -m benchmark -v

# Run only async tests
pytest -m asyncio -v

# Skip tests requiring network
pytest -m "not requires_network" -v
```

## Test Configuration

### Environment Variables
- `GITHUB_TOKEN`: Set to test authenticated API access
- `MCP_TEST_MODE`: Automatically set by test framework
- `CI`: Set in CI environments to skip certain tests

### Performance Benchmarks
Performance tests include benchmarks that measure:
- Sync duration for various file counts
- Memory usage during operations
- Concurrent download efficiency
- Cache lookup performance

Results are printed to console with detailed metrics.

### Mock Infrastructure
The test suite includes comprehensive mocking for:
- GitHub API responses
- Network conditions (timeouts, errors)
- File system operations
- Rate limiting scenarios

## Test Data

### Mock Repository Structure
Integration tests use a mock repository with:
- Various file types (`.md`, `.yaml`, `.json`)
- Nested directory structures
- Files of different sizes
- Special cases (drafts, hidden files)

### Edge Cases Covered
- Unicode filenames and content
- Path traversal attempts
- Malformed API responses
- Concurrent operations
- Large file handling
- Rate limit scenarios

## Debugging Failed Tests

### Enable Debug Logging
```bash
pytest tests/unit/core/standards/test_sync_comprehensive.py -v -s --log-cli-level=DEBUG
```

### Run Specific Test
```bash
pytest tests/unit/core/standards/test_sync_comprehensive.py::TestErrorHandling::test_network_timeout -v
```

### Performance Profiling
```bash
pytest tests/performance/test_sync_performance.py --profile
```

## Continuous Integration

The test suite is designed to run in CI environments with:
- Automatic marker detection
- Network test skipping when appropriate
- Performance baseline tracking
- Security vulnerability scanning

## Contributing

When adding new sync functionality:
1. Add corresponding unit tests to `test_sync_comprehensive.py`
2. Add integration tests if the feature involves external systems
3. Add performance tests for any performance-critical code
4. Add security tests for any security-sensitive features
5. Update this README with new test categories

## Test Coverage Goals

- Unit test coverage: > 90%
- Integration test coverage: All major workflows
- Performance regression detection: < 10% variance
- Security test coverage: All input validation and file operations