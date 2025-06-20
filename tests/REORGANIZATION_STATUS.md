# Test Reorganization Status

## ✅ Completed

### Directory Structure Created
- `tests/unit/` - Unit tests organized by module
- `tests/unit/core/mcp/` - MCP-related tests  
- `tests/unit/core/standards/` - Standards-related tests
- `tests/unit/cli/` - CLI tests
- `tests/unit/analyzers/` - Analyzer tests
- `tests/integration/` - Integration tests
- `tests/e2e/` - End-to-end tests (future)

### Files Moved
1. `test_mcp_server.py` → `tests/unit/test_server.py`
2. `test_core_mcp_server.py` → `tests/unit/core/mcp/test_server.py`
3. `test_core_mcp_handlers.py` → `tests/unit/core/mcp/test_handlers.py`
4. `test_models.py` → `tests/unit/core/mcp/test_models.py`
5. `test_core_standards_handlers.py` → `tests/unit/core/standards/test_handlers.py`
6. `test_versioning.py` → `tests/unit/core/standards/test_versioning.py`
7. `test_cli.py` → `tests/unit/cli/test_main.py`
8. `test_cli_standards.py` → `tests/unit/cli/test_standards_commands.py`
9. `test_logging.py` → `tests/unit/core/test_logging.py`
10. `test_enhanced_patterns.py` → `tests/unit/analyzers/test_enhanced_patterns.py`

### Files Merged
1. `test_server_additional.py` → Merged into `tests/unit/test_server.py`
2. `test_models_additional.py` → Split between:
   - MCP models → `tests/unit/core/mcp/test_models.py`
   - Standards models → `tests/unit/core/standards/test_models.py` (new file)

### New Files Created
- `tests/unit/core/standards/test_models.py` - Standards-specific model tests

## ⚠️ Needs Resolution

### Standards Engine Tests
We have two versions:
1. `tests/unit/core/standards/test_engine.py` (88 lines, basic)
2. `tests/test_standards_engine.py` (702 lines, comprehensive)

**Recommendation**: Use the comprehensive version as the main test file.

### Files to Delete (after verification)
In root tests/ directory:
- `test_server_additional.py` (merged)
- `test_models_additional.py` (split and merged)
- `test_models_coverage.py` (merged)
- `test_standards_engine.py` (after moving)
- `test_core_standards_engine.py` (less comprehensive)

## 🔧 Next Steps

1. **Choose standards engine test file**:
   ```bash
   # Option 1: Replace with comprehensive version
   mv tests/test_standards_engine.py tests/unit/core/standards/test_engine.py
   
   # Option 2: Keep both with different names
   mv tests/test_standards_engine.py tests/unit/core/standards/test_engine_comprehensive.py
   ```

2. **Update imports in all moved files**:
   - Check for relative imports that need updating
   - Update any cross-test imports

3. **Update pyproject.toml**:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests/unit", "tests/integration", "tests/e2e"]
   ```

4. **Run all tests** to ensure nothing broke:
   ```bash
   pytest -xvs
   ```

5. **Delete old files** after confirming all tests pass:
   ```bash
   rm tests/test_*.py
   ```

## 📊 Summary

- **Before**: 16 test files with unclear naming and duplicates
- **After**: Organized structure mirroring source code
- **Benefits**: 
  - Clear what each test file tests
  - No more "additional" or "coverage" suffixes
  - Easy to find tests for any module
  - Proper separation of test levels