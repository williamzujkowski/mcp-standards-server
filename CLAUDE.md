# CLAUDE.md - MCP Standards Server

This file provides LLM-specific instructions for working with the MCP Standards Server codebase.

## Project Overview

This is a Model Context Protocol (MCP) server built with the official Python SDK that provides NIST 800-53r5 compliance checking and standards enforcement. The server exposes tools, resources, and prompts for LLMs to analyze code for security control implementations and generate compliance documentation.

## Current Implementation Status

### ✅ Completed Features
- **OSCAL Handler**: Full OSCAL 1.0.0 support with SSP generation and integrity checking
- **Language Analyzers**: Python, JavaScript/TypeScript, Go, and Java with deep AST analysis
- **Standards Engine**: Complete YAML loading, Redis caching, natural language mapping
- **CLI Commands**: init, scan, ssp, server, version, generate, validate (all functional)
- **Standards Import**: 17 standards documents imported from williamzujkowski/standards
- **MCP Resources**: 20+ dynamic resource endpoints with real-time loading
- **MCP Prompts**: 5 specialized prompt templates for compliance scenarios
- **Code Templates**: NIST-compliant templates for API, auth, logging, encryption, database
- **Git Integration**: Automated hooks for pre-commit and pre-push compliance checking
- **VS Code Support**: Integrated settings and workflow configuration
- **Example Projects**: Python API, JavaScript frontend with comprehensive documentation
- **Test Coverage**: 91.56% (exceeding 80% target)
- **GitHub Workflows**: CI/CD pipelines with security scanning

### 🚧 Remaining Tasks (Low Priority)
- Standards versioning and automated updates
- REST API endpoints for non-MCP access
- Additional language support (Ruby, PHP, C++, Rust)

### 📋 Future Enhancements
- Advanced MCP features (elicitation, progress tracking, cancellation)
- Machine learning for automatic control suggestions
- Compliance drift detection and alerting
- Real-time compliance dashboard

## Technology Stack

- **Package Manager**: uv (not Poetry)
- **MCP SDK**: Official Python SDK from modelcontextprotocol
- **Python**: 3.11+
- **Key Dependencies**: mcp, pydantic, redis, tree-sitter, typer

## Key Components

### MCP Server (`src/server.py`)
- Main entry point using MCP Python SDK
- Implements tools, resources, and prompts
- Handles all MCP protocol communication

### Standards Engine (`src/core/standards/`)
- Natural language to standard notation mapping
- Token-aware standard loading
- Redis caching support

### Compliance Module (`src/compliance/`)
- NIST control scanning
- OSCAL generation
- Compliance validation

### Analyzers (`src/analyzers/`)
- Language-specific code analyzers (Python, JS, Go, Java)
- AST parsing for deep analysis
- Pattern matching for security control detection

## MCP Tools Implementation

When implementing new MCP tools:

1. Add tool definition in `@app.list_tools()`
2. Add handler in `@app.call_tool()`
3. Include proper input schema validation
4. Always return `List[TextContent | ImageContent]`
5. Add NIST control annotations

Example:
```python
Tool(
    name="new_tool",
    description="Tool description",
    inputSchema={
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        },
        "required": ["param"]
    }
)
```

## NIST Control Annotations

When adding security-relevant code, ALWAYS include NIST annotations:

```python
# @nist-controls: AC-3, AU-2
# @evidence: Role-based access control with audit logging
# @oscal-component: api-gateway
```

## Development Guidelines

1. **Use uv, not Poetry**: All dependency management through uv
2. **MCP SDK Patterns**: Follow official SDK patterns for tools/resources/prompts
3. **Type Safety**: Use type hints and run mypy
4. **Test Coverage**: Maintain >80% test coverage
5. **Security First**: All new features must consider security implications

## Common Tasks

### Adding a New MCP Tool
1. Define tool in `list_tools()`
2. Add handler in `call_tool()`
3. Create helper function for complex logic
4. Add tests in `tests/test_mcp_tools.py`
5. Update README with tool documentation

### Adding NIST Controls
1. Update mappings in `NaturalLanguageMapper`
2. Add patterns to compliance scanner
3. Document evidence templates
4. Test with real code examples

### Working with Standards
1. Standards are loaded via `StandardsEngine`
2. Natural language queries are mapped to notation
3. Token limits are respected for LLM contexts
4. Caching via Redis when available

## Testing

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Test specific module
pytest tests/test_server.py -v

# With coverage
pytest --cov=src --cov-report=html
```

## Debugging MCP Server

1. Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

2. Test with MCP inspector:
```bash
npx @modelcontextprotocol/inspector mcp-standards-server
```

3. Check tool schemas are valid JSON Schema
4. Verify resource URIs follow standards:// pattern

## Performance Considerations

1. Use Redis caching for standards
2. Implement token budgets for LLM contexts
3. Batch operations where possible
4. Lazy load heavy resources

## Security Checklist

Before committing:
- [ ] Add NIST annotations to security code
- [ ] No hardcoded secrets
- [ ] Input validation on all tool parameters
- [ ] Error messages don't leak sensitive info
- [ ] All tools have proper descriptions

## Common Issues

1. **Import Errors**: Ensure using `mcp` not custom protocol
2. **Tool Schema Validation**: Must be valid JSON Schema
3. **Async/Await**: All MCP handlers must be async
4. **Return Types**: Tools must return TextContent/ImageContent list