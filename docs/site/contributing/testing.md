# Testing Guidelines

Comprehensive testing ensures the reliability and quality of MCP Standards Server. This guide covers testing strategies, tools, and best practices.

## Testing Philosophy

- **Test First**: Write tests before implementation (TDD)
- **Full Coverage**: Aim for >80% code coverage
- **Fast Feedback**: Tests should run quickly
- **Isolated Tests**: No dependencies between tests
- **Clear Failures**: Tests should clearly indicate what failed

## Test Structure

```
tests/
├── unit/               # Fast, isolated unit tests
├── integration/        # Component integration tests
├── e2e/               # End-to-end workflow tests
├── performance/       # Performance benchmarks
├── fixtures/          # Test data and mocks
└── conftest.py       # Shared pytest configuration
```

## Writing Unit Tests

### Basic Test Structure

```python
# tests/unit/test_feature.py
import pytest
from unittest.mock import Mock, patch

from src.module import YourClass

class TestYourClass:
    """Test suite for YourClass."""
    
    def setup_method(self):
        """Set up test dependencies."""
        self.instance = YourClass()
    
    def teardown_method(self):
        """Clean up after tests."""
        # Cleanup if needed
        pass
    
    def test_basic_functionality(self):
        """Test basic functionality works as expected."""
        # Arrange
        input_data = "test"
        expected = "TEST"
        
        # Act
        result = self.instance.process(input_data)
        
        # Assert
        assert result == expected
    
    def test_edge_case(self):
        """Test edge cases are handled properly."""
        with pytest.raises(ValueError):
            self.instance.process(None)
```

### Testing with Mocks

```python
class TestExternalIntegration:
    @patch('src.module.external_api')
    def test_api_call(self, mock_api):
        """Test external API integration."""
        # Configure mock
        mock_api.return_value = {'status': 'success'}
        
        # Test your code
        result = your_function()
        
        # Verify mock was called correctly
        mock_api.assert_called_once_with('expected', 'args')
        assert result['status'] == 'success'
    
    @patch('src.module.redis_client')
    def test_caching(self, mock_redis):
        """Test caching behavior."""
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        
        # First call should cache
        result1 = cached_function('key')
        mock_redis.set.assert_called_once()
        
        # Second call should use cache
        mock_redis.get.return_value = result1
        result2 = cached_function('key')
        assert result1 == result2
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input_str,expected", [
    ("hello", "HELLO"),
    ("Hello World", "HELLO WORLD"),
    ("123", "123"),
    ("", ""),
    ("  spaces  ", "  SPACES  "),
])
def test_uppercase_conversion(input_str, expected):
    """Test uppercase conversion 1.0.0
    assert input_str.upper() == expected

@pytest.mark.parametrize("code,language,expected_issues", [
    ("print('hello')", "python", 0),
    ("console.log('hello')", "javascript", 1),
    ("fmt.Println('hello')", "go", 0),
])
def test_language_validation(code, language, expected_issues):
    """Test validation across different languages."""
    validator = get_validator(language)
    result = validator.analyze(code)
    assert len(result.violations) == expected_issues
```

## Writing Integration Tests

### Database Integration

```python
# tests/integration/test_database.py
import pytest
from sqlalchemy import create_engine

@pytest.fixture
def test_db():
    """Create test database."""
    engine = create_engine('sqlite:///:memory:')
    # Set up schema
    Base.metadata.create_all(engine)
    
    yield engine
    
    # Cleanup
    engine.dispose()

def test_database_operations(test_db):
    """Test database CRUD operations."""
    with test_db.connect() as conn:
        # Test insert
        result = conn.execute(insert_query)
        assert result.rowcount == 1
        
        # Test select
        rows = conn.execute(select_query).fetchall()
        assert len(rows) == 1
```

### Redis Integration

```python
# tests/integration/test_cache.py
import pytest
import fakeredis

@pytest.fixture
def redis_client():
    """Create fake Redis client for testing."""
    return fakeredis.FakeRedis()

def test_cache_operations(redis_client):
    """Test cache operations."""
    from src.core.cache import MCPCache
    
    cache = MCPCache(redis_client)
    
    # Test set and get
    cache.set('key', 'value', ttl=60)
    assert cache.get('key') == 'value'
    
    # Test expiration
    cache.set('temp', 'data', ttl=1)
    import time
    time.sleep(2)
    assert cache.get('temp') is None
```

## Writing E2E Tests

### MCP Server E2E Test

```python
# tests/e2e/test_mcp_workflow.py
import asyncio
import pytest
from src.mcp_server import MCPServer

@pytest.mark.asyncio
async def test_complete_workflow():
    """Test complete MCP workflow."""
    # Start server
    server = MCPServer()
    await server.start()
    
    try:
        # Connect client
        client = MCPClient('localhost:8080')
        await client.connect()
        
        # Test getting standards
        standards = await client.get_applicable_standards({
            'project_type': 'web',
            'language': 'python'
        })
        assert len(standards) > 0
        
        # Test validation
        result = await client.validate_code(
            'def bad_function(): pass',
            standard_id='python-best-practices'
        )
        assert not result['passed']
        assert len(result['violations']) > 0
        
    finally:
        await server.stop()
```

### CLI E2E Test

```python
# tests/e2e/test_cli.py
import subprocess
import tempfile

def test_cli_validation():
    """Test CLI validation workflow."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py') as f:
        # Write test code
        f.write('''
def calculate_sum(numbers):
    total = 0
    for n in numbers:
        total += n
    return total
        ''')
        f.flush()
        
        # Run validation
        result = subprocess.run(
            ['mcp-standards', 'validate', f.name],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert 'Validation passed' in result.stdout
```

## Performance Testing

### Benchmark Tests

```python
# tests/performance/test_benchmarks.py
import pytest

@pytest.mark.benchmark
def test_validation_performance(benchmark):
    """Benchmark validation performance."""
    from src.core.standards import StandardsEngine
    
    engine = StandardsEngine()
    code = "def test(): pass\n" * 1000  # 1000 functions
    
    # Benchmark the validation
    result = benchmark(engine.validate_code, code)
    
    # Assert performance requirements
    assert benchmark.stats['mean'] < 0.1  # Less than 100ms

@pytest.mark.benchmark
def test_search_performance(benchmark):
    """Benchmark search performance."""
    from src.core.standards.semantic_search import SemanticSearch
    
    search = SemanticSearch()
    
    # Benchmark search
    result = benchmark(search.search, "security best practices")
    
    assert len(result) > 0
    assert benchmark.stats['mean'] < 0.05  # Less than 50ms
```

### Memory Testing

```python
# tests/performance/test_memory.py
import pytest
import psutil
import os

def test_memory_usage():
    """Test memory usage stays within limits."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Run memory-intensive operation
    from src.core.standards import StandardsEngine
    engine = StandardsEngine()
    
    # Load all standards
    for i in range(100):
        engine.load_standard(f'standard-{i}')
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_increase = final_memory - initial_memory
    
    # Assert memory usage is reasonable
    assert memory_increase < 100  # Less than 100MB increase

@pytest.mark.memprof
def test_memory_leaks():
    """Test for memory leaks."""
    # This test will generate memory profile
    for i in range(1000):
        obj = create_large_object()
        process_object(obj)
        # Object should be garbage collected
```

## Test Fixtures

### Shared Fixtures

```python
# tests/conftest.py
import pytest
from pathlib import Path

@pytest.fixture(scope='session')
def test_data_dir():
    """Path to test data directory."""
    return Path(__file__).parent / 'fixtures'

@pytest.fixture
def sample_code():
    """Sample code for testing."""
    return {
        'python': '''
def calculate_average(numbers):
    """Calculate average of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
        ''',
        'javascript': '''
function calculateAverage(numbers) {
    if (!numbers.length) return 0;
    return numbers.reduce((a, b) => a + b) / numbers.length;
}
        '''
    }

@pytest.fixture
def mock_standards():
    """Mock standards for testing."""
    return [
        {
            'id': 'test-standard',
            'name': 'Test Standard',
            'rules': [
                {'id': 'rule1', 'severity': 'error'},
                {'id': 'rule2', 'severity': 'warning'}
            ]
        }
    ]
```

## Test Configuration

### pytest.ini

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Markers
markers =
    slow: marks tests as slow
    integration: marks tests as integration tests
    benchmark: marks tests as benchmarks
    asyncio: marks tests as async

# Coverage
addopts = 
    --cov=src
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80

# Timeouts
timeout = 60
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_validator.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run only fast tests
pytest -m "not slow"

# Run in parallel
pytest -n auto

# Run with verbose output
pytest -vv

# Run and stop on first failure
pytest -x

# Run with debugging
pytest --pdb
```

## Continuous Integration

### GitHub Actions Test Job

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ['3.9', '3.10', '3.11', '3.12']
  
  steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python-version 1.0.0
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        pytest --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

## Best Practices

### 1. Test Naming

```python
# Good: Descriptive test names
def test_validator_detects_sql_injection_in_f_strings():
    pass

def test_cache_returns_none_for_expired_keys():
    pass

# Bad: Vague test names
def test_validator():
    pass

def test_cache():
    pass
```

### 2. Test Independence

```python
# Good: Each test sets up its own data
def test_one():
    data = create_test_data()
    assert process(data) == expected

def test_two():
    data = create_test_data()  # Fresh data
    assert process(data) == expected

# Bad: Tests depend on shared state
shared_data = create_test_data()

def test_one():
    assert process(shared_data) == expected

def test_two():
    # May fail if test_one modifies shared_data
    assert process(shared_data) == expected
```

### 3. Assertion Messages

```python
# Good: Clear assertion messages
assert len(results) == 5, f"Expected 5 results, got {len(results)}"

# Bad: No context on failure
assert len(results) == 5
```

## Debugging Failed Tests

1. **Run Single Test**
   ```bash
   pytest tests/unit/test_file.py::TestClass::test_method -vv
   ```

2. **Enable Debugging**
   ```bash
   pytest --pdb  # Drop into debugger on failure
   ```

3. **Print Debugging**
   ```python
   def test_complex_logic():
       result = complex_function()
       print(f"Result: {result}")  # Use -s flag to see prints
       assert result == expected
   ```

4. **Check Test Logs**
   ```bash
   pytest --log-cli-level=DEBUG
   ```

## Related Documentation

- [Development Setup](./setup.md)
- [Code Standards](./standards.md)
- [CI/CD Integration](../guides/cicd-integration.md)

Happy testing! 🧪