# Test Reorganization Plan

## Current Test Files Analysis

### 1. MCP Server Tests
- **test_mcp_server.py**: Tests `src/server.py` (main server entry point)
  - Functions: initialize_server, list_tools, call_tool, handle_* functions
  - Keep as: `tests/unit/test_server.py`

- **test_core_mcp_server.py**: Tests `src/core/mcp/server.py` (MCP protocol server)
  - Classes: MCPServer, create_app, WebSocket handling
  - Keep as: `tests/unit/core/mcp/test_server.py`

- **test_server_additional.py**: Additional server tests
  - Merge into: `tests/unit/test_server.py`

### 2. Model Tests
- **test_models.py**: Tests MCP models from `src/core/mcp/models.py`
  - Classes: MCPMessage, MCPResponse, SessionInfo, etc.
  - Keep as: `tests/unit/core/mcp/test_models.py`

- **test_models_additional.py**: Tests both MCP and Standards models
  - Split between:
    - `tests/unit/core/mcp/test_models.py` (MCP models)
    - `tests/unit/core/standards/test_models.py` (Standards models)

- **test_models_coverage.py**: More model tests for coverage
  - Merge into respective model test files

### 3. Standards Engine Tests
- **test_standards_engine.py**: Comprehensive tests (660 lines, 26 tests)
  - Keep as: `tests/unit/core/standards/test_engine.py`

- **test_core_standards_engine.py**: Basic tests (89 lines, 1 test)
  - Merge into above or delete (less comprehensive)

### 4. Handler Tests
- **test_core_mcp_handlers.py**: Tests `src/core/mcp/handlers.py`
  - Keep as: `tests/unit/core/mcp/test_handlers.py`

- **test_core_standards_handlers.py**: Tests `src/core/standards/handlers.py`
  - Keep as: `tests/unit/core/standards/test_handlers.py`

### 5. CLI Tests
- **test_cli.py**: Tests main CLI commands
  - Keep as: `tests/unit/cli/test_main.py`

- **test_cli_standards.py**: Tests standards CLI commands
  - Keep as: `tests/unit/cli/test_standards_commands.py`

### 6. Other Tests
- **test_logging.py**: Tests `src/core/logging.py`
  - Keep as: `tests/unit/core/test_logging.py`

- **test_enhanced_patterns.py**: Tests pattern detection
  - Keep as: `tests/unit/analyzers/test_enhanced_patterns.py`

- **test_versioning.py**: Tests standards versioning
  - Keep as: `tests/unit/core/standards/test_versioning.py`

### 7. Integration Tests
- **test_mcp_integration.py**: Already in correct location
  - Keep as: `tests/integration/test_mcp_integration.py`

## Proposed Directory Structure

```
tests/
├── conftest.py
├── __init__.py
├── unit/
│   ├── __init__.py
│   ├── test_server.py                    # Main server tests (merged)
│   ├── analyzers/
│   │   ├── __init__.py
│   │   └── test_enhanced_patterns.py     # Pattern detection tests
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── test_main.py                  # Main CLI tests
│   │   └── test_standards_commands.py    # Standards CLI tests
│   └── core/
│       ├── __init__.py
│       ├── test_logging.py               # Logging tests
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── test_handlers.py          # MCP handler tests
│       │   ├── test_models.py            # MCP model tests (merged)
│       │   └── test_server.py            # Core MCP server tests
│       └── standards/
│           ├── __init__.py
│           ├── test_engine.py            # Standards engine tests (comprehensive)
│           ├── test_handlers.py          # Standards handler tests
│           ├── test_models.py            # Standards model tests (split from additional)
│           └── test_versioning.py        # Versioning tests
├── integration/
│   ├── __init__.py
│   └── test_mcp_integration.py          # Existing integration tests
└── e2e/
    └── __init__.py                       # For future end-to-end tests
```

## Benefits

1. **Clear organization**: Mirrors source code structure
2. **No ambiguity**: Each file has a single, clear purpose
3. **Easy to find**: Test location matches source location
4. **No more "additional" or "coverage" files**: All tests consolidated
5. **Proper test levels**: Unit vs Integration vs E2E

## Migration Steps

1. Create new directory structure
2. Move files according to mapping
3. Merge duplicate test content
4. Update imports in moved files
5. Update any CI/CD test paths
6. Run all tests to verify
7. Delete old test files

## Import Updates Needed

When moving files, update imports:
- Change relative imports to match new locations
- Update any test discovery patterns in pyproject.toml
- Update coverage configuration if needed