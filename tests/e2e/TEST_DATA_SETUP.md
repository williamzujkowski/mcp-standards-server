# E2E Test Data Setup

This document describes the test data setup for E2E integration tests.

## Overview

The E2E tests require a minimal set of test standards and rules to function properly. The test data is created dynamically during test execution but can also be pre-created.

## Test Data Structure

```
data/
└── standards/
    ├── cache/          # Test standards JSON files
    │   ├── react-18-patterns.json
    │   ├── python-testing.json
    │   └── javascript-es6-standards.json
    ├── meta/           # Rules and metadata
    │   └── enhanced-selection-rules.json
    └── sync_config.yaml
```

## Test Standards

The following test standards are created:

1. **react-18-patterns** - React 18 best practices
2. **python-testing** - Python testing standards
3. **javascript-es6-standards** - JavaScript ES6+ standards

## Test Rules

Five rules are created to test the rule engine:

1. **react-web-rule** - Matches React web applications (priority: 10)
2. **javascript-web-rule** - Matches JavaScript web applications (priority: 5)
3. **general-javascript-rule** - Matches general JavaScript projects (priority: 3)
4. **python-api-rule** - Matches Python API applications (priority: 8)
5. **mobile-app-rule** - Matches React Native mobile apps (priority: 7)

## Setup Process

The test data is automatically created by the `test_data_setup.py` module when tests run. The conftest.py fixture calls this setup function.

## Running E2E Tests

Due to a configuration issue with pytest.ini files, use the following approach:

```bash
# Run with clean configuration
python -m pytest tests/e2e/test_coverage_basic.py -v --asyncio-mode=auto

# Or create a temporary clean config
cat > temp_pytest.ini << EOF
[pytest]
testpaths = tests/e2e
addopts = -v --tb=short --asyncio-mode=auto
timeout = 60
asyncio_default_fixture_loop_scope = function
EOF

python -m pytest -c temp_pytest.ini tests/e2e/
```

## Verified Functionality

The following E2E test scenarios have been verified:

- ✅ MCP server startup and shutdown
- ✅ Client connection establishment
- ✅ List available standards (25 standards found)
- ✅ Get applicable standards based on context
- ✅ Get standard details
- ✅ Rule engine evaluation
- ✅ Standards caching

## Troubleshooting

1. **"file or directory not found: #" error** - This is caused by a configuration parsing issue. Use the clean config approach above.

2. **Scope mismatch errors** - Fixed by changing the mcp_server fixture from session scope to function scope.

3. **Test hanging** - The MCP server runs as a subprocess and communicates via stdio. Long timeouts may occur if the server fails to start properly.