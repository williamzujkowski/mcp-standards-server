# MCP Standards Server Documentation

Welcome to the MCP Standards Server documentation! This comprehensive guide will help you integrate development standards into your workflow using the Model Context Protocol.

## What is MCP Standards Server?

MCP Standards Server is a powerful tool that helps development teams:

- 📚 **Manage Standards** - Sync and cache development standards from repositories
- 🔍 **Query Standards** - Find applicable standards based on project context
- ✅ **Validate Code** - Check code against standards with auto-fix capabilities
- 🤖 **IDE Integration** - Real-time validation and suggestions in your editor
- 🚀 **CI/CD Ready** - Integrate standards checking into your pipeline
- 🎯 **MCP Protocol** - Use standards via Model Context Protocol for AI assistants

## Quick Start

```bash
# Install
pip install mcp-standards-server

# Initialize configuration
mcp-standards config --init

# Sync standards
mcp-standards sync

# Validate your code
mcp-standards validate .

# Start MCP server
mcp-standards serve
```

## Documentation Overview

### 🚀 Getting Started
- [Installation Guide](./guides/installation.md)
- [Quick Start Tutorial](./guides/quickstart.md)
- [Configuration Guide](./guides/configuration.md)

### 📖 User Guides
- [CLI Commands Reference](./reference/cli-commands.md)
- [Common Workflows](./guides/workflows.md)
- [IDE Integration](./guides/ide-integration.md)
- [CI/CD Integration](./guides/cicd-integration.md)

### 🔧 API Reference
- [MCP Tools Reference](./api/mcp-tools.md)
- [Configuration Schema](./api/config-schema.md)
- [Standards Format](./api/standards-format.md)
- [Validation Rules](./api/validation-rules.md)

### 💡 Examples
- [Project Setup Examples](./examples/project-setup.md)
- [Validation Examples](./examples/validation.md)
- [Custom Standards](./examples/custom-standards.md)
- [Integration Scripts](./examples/scripts.md)

### 🏗️ Architecture
- [System Overview](./architecture/overview.md)
- [Standards Engine](./architecture/standards-engine.md)
- [Token Optimization](./architecture/token-optimization.md)
- [MCP Integration](./architecture/mcp-integration.md)

### 🤝 Contributing
- [Development Setup](./contributing/setup.md)
- [Adding Standards](./contributing/standards.md)
- [Writing Validators](./contributing/validators.md)
- [Testing Guide](./contributing/testing.md)

## Key Features

### 🎯 Intelligent Standard Selection

The server automatically selects applicable standards based on:
- Project type (web app, API, CLI, etc.)
- Programming languages
- Frameworks and libraries
- Special requirements (security, accessibility, performance)

### 🔧 Flexible Validation

- **Auto-fix capabilities** for common issues
- **Multiple output formats** (JSON, JUnit, SARIF)
- **Configurable severity levels**
- **Incremental validation** for large codebases

### 🤖 MCP Protocol Support

Use standards through AI assistants and tools:
- Query standards with natural language
- Get code suggestions based on standards
- Validate code in real-time
- Generate compliant code templates

### 📊 Comprehensive Reporting

- Detailed validation reports
- Compliance dashboards
- Historical tracking
- Team metrics

## Community

- 💬 [Discord Server](https://discord.gg/mcp-standards)
- 🐛 [Issue Tracker](https://github.com/williamzujkowski/mcp-standards-server/issues)
- 📢 [Announcements](https://github.com/williamzujkowski/mcp-standards-server/discussions)
- 🌟 [Star on GitHub](https://github.com/williamzujkowski/mcp-standards-server)

## License

MCP Standards Server is open source software licensed under the MIT License.

---

Ready to get started? Check out the [Installation Guide](./guides/installation.md) or jump right into the [Quick Start Tutorial](./guides/quickstart.md)!