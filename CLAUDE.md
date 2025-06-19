# CLAUDE.md - MCP Standards Server

This file provides LLM-specific instructions for working with the MCP Standards Server codebase.

## Project Overview

This is an MCP (Model Context Protocol) server that provides NIST 800-53r5 compliance checking and standards enforcement. The server analyzes code for security control implementations and generates compliance documentation.

## Key Components

### Core Modules
- `src/core/mcp/`: MCP protocol implementation with WebSocket support
- `src/core/standards/`: Standards loading engine with caching
- `src/core/compliance/`: NIST control mapping and OSCAL generation

### Analyzers
- `src/analyzers/`: Language-specific code analyzers (Python, JS, Go, Java)
- Uses AST parsing for deep analysis
- Pattern matching for security control detection

### CLI
- `src/cli/`: Typer-based command-line interface
- Commands: init, scan, server, generate, validate, ssp

## NIST Control Annotations

When adding security-relevant code, ALWAYS include NIST annotations:

```python
# @nist-controls: AC-3, AU-2
# @evidence: Role-based access control with audit logging
# @oscal-component: api-gateway
```

## Development Guidelines

1. **Security First**: All new features must consider security implications
2. **Type Safety**: Use type hints and run mypy
3. **Test Coverage**: Maintain >80% test coverage
4. **Documentation**: Update API docs for new endpoints

## Common Tasks

### Adding a New Analyzer
1. Create analyzer in `src/analyzers/{language}_analyzer.py`
2. Inherit from `BaseAnalyzer`
3. Implement required methods
4. Add tests in `tests/analyzers/`

### Adding NIST Controls
1. Update mappings in `NISTControlMapper`
2. Add patterns to `_load_mappings()`
3. Document evidence templates

### Generating Compliant Code
Templates in `templates/` include NIST annotations. When creating new templates:
1. Include relevant @nist-controls
2. Add @evidence descriptions
3. Follow security best practices

## Testing

```bash
# Run all tests
poetry run pytest

# Test specific module
poetry run pytest tests/core/test_nist_mapper.py -v

# With coverage
poetry run pytest --cov=src --cov-report=html
```

## Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check WebSocket connections:
```bash
wscat -c ws://localhost:8000/mcp -H "Authorization: Bearer $TOKEN"
```

## Performance Considerations

1. Use Redis caching for standards
2. Batch analyze files in projects
3. Optimize AST parsing for large codebases

## Security Checklist

Before committing:
- [ ] Add NIST annotations to security code
- [ ] No hardcoded secrets
- [ ] Input validation on all endpoints
- [ ] Error messages don't leak information
- [ ] Authentication required for sensitive operations