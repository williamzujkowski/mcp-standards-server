# Test Organization

This directory contains all tests for the MCP Standards Server, organized by test type and following the source code structure.

## Structure

```
tests/
├── unit/                      # Unit tests (isolated component tests)
│   ├── analyzers/            # Tests for src/analyzers/
│   ├── cli/                  # Tests for src/cli/
│   ├── core/                 # Tests for src/core/
│   │   ├── mcp/             # Tests for src/core/mcp/
│   │   └── standards/       # Tests for src/core/standards/
│   └── test_server.py       # Tests for src/server.py
├── integration/              # Integration tests (multiple components)
│   └── test_mcp_integration.py
└── e2e/                     # End-to-end tests (future)
```

## Naming Convention

- Test files are named `test_<module>.py` matching the source module
- Test classes follow `Test<ClassName>` pattern
- Test methods follow `test_<functionality>` pattern

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest tests/unit

# Run only integration tests
pytest tests/integration

# Run tests for a specific module
pytest tests/unit/core/mcp/test_models.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run in verbose mode
pytest -v
```

## Test Types

### Unit Tests
- Test individual components in isolation
- Mock external dependencies
- Fast execution
- Located in `tests/unit/`

### Integration Tests
- Test interaction between multiple components
- May use real dependencies
- Slower than unit tests
- Located in `tests/integration/`

### End-to-End Tests
- Test complete workflows
- Use real system components
- Slowest execution
- Located in `tests/e2e/` (future)

## Adding New Tests

1. Create test file matching source module structure
2. Import the module being tested
3. Follow existing patterns for test organization
4. Ensure NIST control annotations are included
5. Run tests locally before committing

## Test Coverage

We maintain a minimum of 80% test coverage. Check coverage with:

```bash
pytest --cov=src --cov-report=term-missing
```