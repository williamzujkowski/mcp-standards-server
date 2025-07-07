# End-to-End Tests for MCP Standards Server

This directory contains comprehensive end-to-end tests for the MCP Standards Server, covering all aspects of functionality, performance, and reliability.

## Test Structure

```
e2e/
├── test_mcp_server.py      # Main E2E test suite
├── test_performance.py     # Performance and scalability tests
├── fixtures.py            # Test fixtures and utilities
└── README.md             # This file
```

## Test Categories

### 1. **Functional Tests** (`test_mcp_server.py`)

#### Server Lifecycle
- Server startup and shutdown
- Graceful shutdown handling
- Multiple client connections

#### MCP Tools Testing
- `get_applicable_standards` - Context-based standard selection
- `validate_against_standard` - Code validation
- `search_standards` - Semantic search functionality
- `get_standard_details` - Standard retrieval
- `list_available_standards` - Standard listing
- `suggest_improvements` - Code improvement suggestions

#### Integration Tests
- Standards synchronization workflow
- Rule engine integration
- Semantic search accuracy
- Cache management

#### Error Handling
- Invalid tool names
- Missing parameters
- Malformed data
- Timeout scenarios

### 2. **Performance Tests** (`test_performance.py`)

#### Load Testing
- Concurrent request handling (100+ requests)
- Mixed workload scenarios
- Throughput measurements

#### Memory Testing
- Memory usage under load
- Memory leak detection
- Cache memory management

#### Response Time Benchmarks
- Standard selection benchmarks
- Search operation benchmarks
- Validation benchmarks

#### Scalability Tests
- Maximum concurrent connections
- Large response handling
- Resource utilization

## Running the Tests

### Basic Usage

```bash
# Run all E2E tests
pytest tests/e2e/

# Run specific test file
pytest tests/e2e/test_mcp_server.py

# Run with coverage
pytest tests/e2e/ --cov=src --cov-report=html

# Run performance tests only
pytest tests/e2e/ -m performance

# Run with detailed output
pytest tests/e2e/ -vv -s
```

### Using the Test Runner Script

```bash
# Run all E2E tests with coverage
./scripts/run_e2e_tests.sh

# Run performance tests
./scripts/run_e2e_tests.sh --performance

# Run in parallel
./scripts/run_e2e_tests.sh --parallel

# Run without coverage
./scripts/run_e2e_tests.sh --no-coverage
```

## Test Fixtures

The `fixtures.py` file provides:

### Sample Contexts
- `react_web_app` - React web application context
- `python_api` - Python API context
- `mobile_app` - Mobile application context
- `microservice` - Microservice context
- `data_pipeline` - Data pipeline context
- `ml_project` - Machine learning project context
- `mcp_server` - MCP server development context

### Mock Data
- `MOCK_STANDARDS` - Sample standards for testing
- `MOCK_RULES` - Sample rules for rule engine
- `MockStandardsRepository` - In-memory standards repository

### Test Utilities
- `TestDataGenerator` - Generate test code samples
- `create_test_mcp_config` - Create test configuration

## Performance Metrics

Performance tests track:
- **Response Times**: min, max, mean, median, p95, p99
- **Memory Usage**: min, max, mean, growth over time
- **CPU Usage**: utilization percentages
- **Throughput**: requests per second
- **Error Rates**: percentage of failed requests

## CI/CD Integration

### GitHub Actions Workflow

The `.github/workflows/e2e-tests.yml` workflow:
1. Runs on multiple OS (Ubuntu, macOS, Windows)
2. Tests multiple Python versions (3.10, 3.11, 3.12)
3. Generates coverage reports
4. Runs performance benchmarks
5. Detects memory leaks
6. Performs security scans

### Test Reports

- **Coverage Reports**: Published to GitHub Pages
- **Performance Results**: Tracked with benchmark-action
- **Test Results**: Available as artifacts
- **PR Comments**: Automatic test summaries

## Writing New Tests

### Test Template

```python
@pytest.mark.asyncio
async def test_new_functionality(mcp_client):
    """Test description."""
    # Arrange
    test_data = {...}
    
    # Act
    result = await mcp_client.call_tool(
        "tool_name",
        test_data
    )
    
    # Assert
    assert "expected_key" in result
    assert result["expected_key"] == expected_value
```

### Best Practices

1. **Use Descriptive Names**: Test names should clearly describe what's being tested
2. **Follow AAA Pattern**: Arrange, Act, Assert
3. **Test One Thing**: Each test should focus on a single behavior
4. **Use Fixtures**: Leverage provided fixtures for common data
5. **Mark Tests Appropriately**: Use `@pytest.mark` for categorization
6. **Handle Async Properly**: Use `@pytest.mark.asyncio` for async tests

## Troubleshooting

### Common Issues

1. **MCP Not Installed**
   ```bash
   pip install mcp
   ```

2. **Redis Not Running**
   ```bash
   # Linux/macOS
   redis-server --daemonize yes
   
   # Windows
   redis-server --service-start
   ```

3. **Timeout Errors**
   - Increase timeout: `pytest --timeout=600`
   - Check system resources

4. **Import Errors**
   - Ensure PYTHONPATH includes project root
   - Install in development mode: `pip install -e .`

### Debug Mode

Run tests with debug output:
```bash
pytest tests/e2e/ -vv -s --log-cli-level=DEBUG
```

## Performance Baselines

Expected performance metrics:

| Metric | Target | Actual |
|--------|--------|--------|
| p95 Response Time | < 1.0s | TBD |
| p99 Response Time | < 2.0s | TBD |
| Throughput | > 10 req/s | TBD |
| Memory Growth | < 50MB | TBD |
| Error Rate | < 1% | TBD |

## Contributing

When adding new E2E tests:

1. Place tests in appropriate test class
2. Use provided fixtures and utilities
3. Add performance benchmarks for new features
4. Update this README with new test descriptions
5. Ensure tests pass locally before pushing
6. Monitor CI results and fix any issues