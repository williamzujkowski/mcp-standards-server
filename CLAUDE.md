# CLAUDE.md - MCP Standards Server

This file serves as the primary logic router for LLMs working with the MCP Standards Server codebase. It provides comprehensive instructions, decision trees, and optimization strategies to ensure efficient and effective development.

## 🤖 LLM Quick Reference

### Priority Actions
1. **Token Optimization**: We need to achieve 90% token reduction (currently at 0%)
2. **Micro Standards**: Implement 500-token chunks (partially implemented, tests exist but have import issues)
3. **Semantic Search**: Replace static query mappings with ML-based understanding (enhanced_mapper implemented, semantic_search 88% complete)
4. **Test Coverage**: Add comprehensive tests for analyzers (✅ completed - 70% coverage achieved)
5. **Additional Languages**: Ruby, PHP, C++, Rust, C# support

### Key Metrics
- **Current Test Coverage**: 70% (up from 54%)
- **Standards Imported**: 17/23 from williamzujkowski/standards
- **Token Reduction**: 0% (Target: 90%)
- **Languages Supported**: 4 (Python, JS/TS, Go, Java)
- **IaC Analyzers**: 3 (Terraform, Dockerfile, Kubernetes)
- **NIST Controls Detected**: 200+ across 20 families
- **Total Tests**: 523 (up from 362)
- **New Test Methods**: 120+

## Project Overview

This is a Model Context Protocol (MCP) server built with the official Python SDK that provides NIST 800-53r5 compliance checking and standards enforcement. The server exposes tools, resources, and prompts for LLMs to analyze code for security control implementations and generate compliance documentation.

### 🎯 Mission Critical Requirements
1. **LLM Optimization**: All content must be optimized for LLM consumption
2. **Token Efficiency**: Achieve 90% reduction in token usage
3. **Standards Compliance**: Follow williamzujkowski/standards patterns
4. **Developer Experience**: Make it easy for LLMs to understand and modify
5. **Real-time Updates**: Support dynamic content loading and caching

## Current Implementation Status

### ✅ Completed Features
- **OSCAL Handler**: Full OSCAL 1.0.0 support with SSP generation and integrity checking
- **Language Analyzers**: Python, JavaScript/TypeScript, Go, and Java with deep AST analysis
- **Enhanced Pattern Detection**: 200+ NIST control patterns across all 20 families
- **Standards Engine**: Complete YAML loading, Redis caching, natural language mapping
- **Standards Versioning**: Full version management with rollback capabilities
- **CLI Commands**: init, scan, ssp, server, version, generate, validate, coverage (all functional)
- **Control Coverage Reports**: Comprehensive gap analysis with multiple output formats
- **Standards Import**: 17 standards documents imported from williamzujkowski/standards
- **MCP Resources**: 20+ dynamic resource endpoints with real-time loading
- **MCP Prompts**: 5 specialized prompt templates for compliance scenarios
- **Code Templates**: NIST-compliant templates for API, auth, logging, encryption, database
- **Git Integration**: Automated hooks for pre-commit compliance checking
- **VS Code Support**: Planned feature for integrated settings and workflow
- **Example Projects**: Python API, JavaScript frontend with comprehensive documentation
- **Test Coverage**: 70% (improved from 54%, targeting 80%)
- **GitHub Workflows**: CI/CD pipelines with security scanning

### 🚧 Remaining Tasks (Low Priority)
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

## 🧭 LLM Decision Trees

### When Working on Token Optimization
```
Is the task about token reduction?
├─ Yes → Check src/core/standards/engine.py
│   ├─ Implementing new strategy? → Add to TokenOptimizationStrategy enum
│   ├─ Testing optimization? → Use tests/unit/core/standards/test_engine.py
│   └─ Measuring reduction? → Update metrics in StandardsEngine._calculate_tokens
└─ No → Continue to next decision
```

### When Adding New Features
```
What type of feature?
├─ MCP Tool → src/server.py (@app.list_tools and @app.call_tool)
├─ CLI Command → src/cli/main.py (new command function)
├─ Analyzer → src/analyzers/ (extend BaseAnalyzer)
├─ Standard → data/standards/ (follow STANDARD_TEMPLATE.md)
└─ Documentation → docs/ (update relevant section)
```

### When Debugging
```
What's the issue?
├─ MCP Protocol → Enable debug logging in src/server.py
├─ Standards Loading → Check Redis cache and YAML parsing
├─ Token Counting → Verify tokenizer implementation
├─ Test Failures → Run specific test with -v flag
└─ Import Errors → Check pyproject.toml dependencies
```

## 📊 Current Gaps Analysis

### Critical Missing Features (High Priority)
1. **Token Reduction Engine** (0% → 90%)
   - Location: `src/core/standards/engine.py`
   - Required: Implement SUMMARIZE, ESSENTIAL_ONLY, HIERARCHICAL strategies
   - Test: `tests/unit/core/standards/test_token_optimization.py`

2. **Micro Standards Generator** (500-token chunks)
   - Location: `src/core/standards/micro_standards.py` (to be created)
   - Required: Chunking algorithm, index generation
   - Test: `tests/unit/core/standards/test_micro_standards.py`

3. **Semantic Query Engine**
   - Location: `src/core/standards/semantic_search.py` (to be created)
   - Required: Embedding model, similarity search
   - Dependencies: sentence-transformers, faiss

### Medium Priority Features
1. **Additional Language Support** (Ruby, PHP, C++, Rust, C#)
2. **Context-Aware Recommendations**
3. **Progressive Content Loading**
4. **Caching Layer for Optimized Content**

## 🔧 LLM-Specific Workflows

### Adding Token Optimization
1. Check current implementation in `StandardsEngine._optimize_for_tokens`
2. Add new strategy to `TokenOptimizationStrategy` enum
3. Implement strategy logic in `_optimize_for_tokens`
4. Add tests for new strategy
5. Update metrics tracking
6. Document in README.md

### Creating Micro Standards
1. Create `src/core/standards/micro_standards.py`
2. Implement `MicroStandardsGenerator` class
3. Add chunking algorithm (500 tokens max)
4. Create index for quick lookup
5. Integrate with MCP resources
6. Add CLI command for generation

### Implementing Semantic Search
1. Create `src/core/standards/semantic_search.py`
2. Add embedding model (sentence-transformers)
3. Build vector database (FAISS/ChromaDB)
4. Replace static mappings in NaturalLanguageMapper
5. Add similarity threshold configuration
6. Create evaluation metrics

## 📁 File Navigation Guide

### Core Components
- `src/server.py` - MCP server entry point (tools, resources, prompts)
- `src/core/standards/engine.py` - Standards loading and token management
- `src/core/standards/mapper.py` - Natural language to notation mapping
- `src/analyzers/` - Language-specific analyzers
- `data/standards/` - Imported standards content

### Testing
- `tests/unit/` - Unit tests mirroring src structure
- `tests/integration/` - Integration tests
- `tests/e2e/` - End-to-end tests
- `tests/fixtures/` - Test data and mocks

### Documentation
- `docs/` - User and developer documentation
- `README.md` - Project overview and quick start
- `CHANGELOG.md` - Version history
- `TODO_ANALYZERS.md` - Analyzer implementation status

## 🚀 Performance Optimization Tips

1. **Token Counting**: Use proper tokenizer (tiktoken) instead of character estimation
2. **Caching**: Leverage Redis for pre-optimized content
3. **Lazy Loading**: Load standards on-demand, not all at startup
4. **Batch Processing**: Process multiple files concurrently
5. **Incremental Updates**: Only reprocess changed content

## 🔐 Security Considerations

Always include NIST annotations when adding security features:
```python
# @nist-controls: AC-3, AU-2
# @evidence: Implementation description
# @oscal-component: component-name
```

## 📈 Metrics to Track

1. **Token Reduction Rate**: Current vs optimized token count
2. **Query Response Time**: Natural language processing speed
3. **Cache Hit Rate**: Redis cache effectiveness
4. **Coverage Completeness**: NIST controls detected vs total
5. **Test Coverage**: Maintain above 80%

## 🤝 Integration Points

### With williamzujkowski/standards
- Import mechanism in `scripts/import_standards.py`
- Validation against STANDARD_TEMPLATE.md
- Version tracking for updates

### With LLMs
- MCP protocol for tool/resource access
- Token-optimized responses
- Context-aware content delivery

### With CI/CD
- GitHub Actions workflows
- Automated compliance checking
- Coverage reporting