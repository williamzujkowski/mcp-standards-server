# MCP Standards Server Documentation

Welcome to the comprehensive documentation for the MCP Standards Server - a production-ready Model Context Protocol server that provides intelligent NIST 800-53r5 compliance checking, automated code analysis, and standards enforcement with real standards content from the official [williamzujkowski/standards](https://github.com/williamzujkowski/standards) repository.

## 🚀 Current Status: Production Ready

**91.56% Test Coverage** | **17 Standards Imported** | **All CLI Commands Functional** | **Complete MCP Integration**

## Documentation Structure

### 📐 [Architecture](./architecture/)
- System design and architecture decisions
- [Architecture Decision Records (ADRs)](./architecture/decisions/)
- Component diagrams and data flow
- Standards engine architecture

### 🔌 [API Reference](./api/)
- [MCP Tools Documentation](./api/mcp-tools.md) - 6 comprehensive tools
- [MCP Resources Guide](./api/mcp-resources.md) - 20+ dynamic endpoints
- [MCP Prompts Reference](./api/mcp-prompts.md) - 5 specialized templates
- REST API endpoints (future)

### 🛡️ [NIST Compliance](./nist/)
- [NIST 800-53r5 Control Mappings](./nist/controls.md)
- [Control Implementation Guides](./nist/implementation.md)
- [OSCAL 1.0.0 Documentation](./nist/oscal.md)
- Evidence generation and SSP creation

### 📚 [User Guides](./guides/)
- [Getting Started Guide](./guides/installation.md)
- [CLI Usage Guide](./guides/cli.md)
- [Quick Start Tutorial](./guides/quickstart.md)
- [Integration Examples](./guides/integration.md)

## 🎯 Key Features

### 🔍 **Enhanced Multi-Language Analysis**
- Complete AST-based analysis for Python, JavaScript/TypeScript, Go, and Java
- **200+ NIST control patterns** across all 20 families
- Advanced security pattern detection with confidence scoring
- Control relationship suggestions and gap analysis

### 📊 **Complete Standards Library**
- **17 imported standards** from official repository
- **Real-time YAML loading** with JSON indexing
- **Natural language queries** for standards content
- **Token-aware loading** for LLM optimization
- **Standards versioning** and update management

### 📋 **OSCAL 1.0.0 Compliance**
- Generate complete System Security Plans (SSPs)
- Component-based architecture modeling
- SHA256 integrity checking for exports
- NIST profile support (low/moderate/high)

### 📈 **Control Coverage Analysis**
- Comprehensive coverage reports in markdown/JSON/HTML
- Family-level coverage percentages
- Control relationship mapping
- Missing control suggestions
- High-confidence control identification

### 🤖 **Complete MCP Integration**
- **6 MCP Tools**: load_standards, analyze_code, suggest_controls, generate_template, validate_compliance, scan_with_llm
- **20+ MCP Resources**: Dynamic standards access by category and document
- **5 MCP Prompts**: Specialized templates for compliance scenarios

### 🛠️ **Production CLI**
- **init**: Project initialization with Git hooks and VS Code integration
- **scan**: Comprehensive codebase analysis with multiple output formats
- **generate**: NIST-compliant code template generation
- **validate**: Standards validation with detailed reporting
- **ssp**: OSCAL System Security Plan generation
- **server**: MCP server with full protocol support

### 🔧 **Developer Integration**
- **Git Hooks**: Automated pre-commit and pre-push compliance checking
- **VS Code Support**: Integrated settings and workflow configuration
- **Template System**: 5 template types (API, auth, logging, encryption, database)
- **Example Projects**: Production-ready Python API and JavaScript frontend

## 📖 Quick Start Links

- **[Installation Guide](./guides/installation.md)** - Complete setup instructions
- **[Quick Start Tutorial](./guides/quickstart.md)** - 5-minute getting started
- **[CLI Commands Guide](./guides/cli.md)** - All command documentation
- **[MCP Tools Reference](./api/mcp-tools.md)** - Tool usage examples
- **[NIST Control Reference](./nist/controls.md)** - Implementation guidance
- **[Example Projects](../examples/)** - Real-world implementations

## 🏆 Implementation Highlights

### ✅ **Core Compliance (Phase 1 - Complete)**
- NIST control mapping engine with 91.56% test coverage
- Multi-language analyzers with deep AST analysis
- OSCAL 1.0.0 integration with integrity checking
- Real standards content with natural language queries

### ✅ **CLI & Integration (Phase 2 - Complete)**
- Complete CLI with all commands functional
- Git hooks integration for automated compliance
- VS Code settings and workflow configuration
- Template generation with NIST compliance

### ✅ **MCP Protocol (Phase 3 - Complete)**
- Full MCP server implementation with official SDK
- Dynamic resource providers with real-time loading
- Specialized prompt templates for compliance scenarios
- Production-ready with comprehensive error handling

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/williamzujkowski/mcp-standards-server/issues)
- **Discussions**: [GitHub Discussions](https://github.com/williamzujkowski/mcp-standards-server/discussions)
- **Security**: See [SECURITY.md](../SECURITY.md)

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to this project.