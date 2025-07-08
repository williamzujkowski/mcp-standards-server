# MCP Standards Server Documentation

Welcome to the comprehensive documentation for the MCP Standards Server. This documentation is organized to help different types of users find the information they need quickly.

## 📚 Documentation Structure

### [CLI Documentation](./cli/)
Command-line interface documentation for end users.

- **[CLI Overview](./cli/README.md)** - Introduction to the CLI
- **[Command Reference](./cli/commands/)** - Detailed documentation for each command
- **[Configuration Guide](./cli/configuration.md)** - How to configure the server
- **[Troubleshooting](./cli/troubleshooting.md)** - Common issues and solutions
- **[Examples](./cli/examples/)** - Practical usage examples

#### Tutorials
- **[Getting Started](./cli/tutorials/getting-started.md)** - Quick start guide
- **[Common Workflows](./cli/tutorials/common-workflows.md)** - Typical usage patterns
- **[IDE Integration](./cli/tutorials/ide-integration.md)** - Editor setup guides
- **[CI/CD Integration](./cli/tutorials/cicd-integration.md)** - Pipeline integration

### [Man Pages](./man/)
Traditional Unix manual pages for system-wide installation.

- **[Installation Instructions](./man/README.md)** - How to install man pages
- **Manual Pages** - Formatted documentation for each command

### [Documentation Website](./site/)
Web-based documentation with search and navigation.

- **[Home](./site/index.md)** - Documentation homepage
- **[Installation Guide](./site/guides/installation.md)** - Detailed installation instructions
- **[API Reference](./site/api/)** - Programmatic interface documentation
  - **[MCP Tools](./site/api/mcp-tools.md)** - Model Context Protocol tools
  - **[Configuration Schema](./site/api/config-schema.md)** - Configuration file format
  - **[Standards Format](./site/api/standards-format.md)** - Standards file specification
- **[Architecture](./site/architecture/)** - System design documentation

### [Internal Documentation](./internal/)
Technical documentation for developers and contributors.

- **[Token Optimization](./token-optimization.md)** - Token management system
- **[Cache Design](./cache/)** - Caching architecture

## 🎯 Quick Links by Role

### For Users
1. Start with [Getting Started](./cli/tutorials/getting-started.md)
2. Learn [Common Workflows](./cli/tutorials/common-workflows.md)
3. Reference [Command Documentation](./cli/commands/)
4. Troubleshoot with [Troubleshooting Guide](./cli/troubleshooting.md)

### For Developers
1. Read [Installation Guide](./site/guides/installation.md)
2. Integrate using [MCP Tools Reference](./site/api/mcp-tools.md)
3. Configure with [Configuration Guide](./cli/configuration.md)
4. Automate with [CI/CD Integration](./cli/tutorials/cicd-integration.md)

### For Contributors
1. Understand [Architecture](./site/architecture/)
2. Review [API Documentation](./site/api/)
3. Follow [Contributing Guidelines](../CONTRIBUTING.md)
4. Run [Tests](../tests/README.md)

## 📖 Documentation Formats

### Online Documentation
- **Searchable website** - Full documentation with search
- **GitHub Pages** - Hosted documentation site
- **In-editor help** - Context-sensitive help in IDEs

### Offline Documentation
- **Man pages** - Traditional Unix manual pages
- **PDF export** - Printable documentation
- **Markdown files** - Version-controlled docs

### Interactive Documentation
- **CLI help** - Built-in `--help` for all commands
- **Shell completion** - Tab completion with descriptions
- **Example scripts** - Runnable examples

## 🔍 Finding Information

### By Topic
- **Installation** → [Installation Guide](./site/guides/installation.md)
- **Configuration** → [Configuration Guide](./cli/configuration.md)
- **Commands** → [Command Reference](./cli/commands/)
- **Integration** → [IDE](./cli/tutorials/ide-integration.md) / [CI/CD](./cli/tutorials/cicd-integration.md)
- **API** → [MCP Tools](./site/api/mcp-tools.md)
- **Troubleshooting** → [Troubleshooting Guide](./cli/troubleshooting.md)

### By Task
- **"How do I install?"** → [Installation Guide](./site/guides/installation.md)
- **"How do I validate code?"** → [validate command](./cli/commands/validate.md)
- **"How do I integrate with VS Code?"** → [IDE Integration](./cli/tutorials/ide-integration.md)
- **"How do I use in CI/CD?"** → [CI/CD Integration](./cli/tutorials/cicd-integration.md)
- **"How do I query standards?"** → [query command](./cli/commands/query.md)

### By User Type
- **New users** → [Getting Started](./cli/tutorials/getting-started.md)
- **CLI users** → [CLI Documentation](./cli/)
- **API users** → [API Reference](./site/api/)
- **DevOps** → [CI/CD Integration](./cli/tutorials/cicd-integration.md)
- **Contributors** → [Contributing Guidelines](../CONTRIBUTING.md)

## 🛠️ Building Documentation

### Prerequisites
```bash
# For building the website
npm install -g @docusaurus/core

# For building man pages
apt-get install pandoc  # or brew install pandoc

# For PDF generation
apt-get install texlive  # or brew install --cask mactex
```

### Build Commands
```bash
# Build website
cd docs/site && npm run build

# Build man pages
make -C docs/man

# Generate PDF
pandoc docs/cli/README.md -o mcp-standards-manual.pdf
```

## 📝 Contributing to Documentation

We welcome documentation contributions! Please:

1. Follow the existing structure and style
2. Include examples for new features
3. Update the table of contents
4. Test all code examples
5. Check for broken links
6. Submit a pull request

See [Contributing Guidelines](../CONTRIBUTING.md) for more details.

## 📞 Getting Help

- **Documentation issues** → [GitHub Issues](https://github.com/williamzujkowski/mcp-standards-server/issues)
- **Questions** → [GitHub Discussions](https://github.com/williamzujkowski/mcp-standards-server/discussions)
- **Live chat** → [Discord Server](https://discord.gg/mcp-standards)

---

Last updated: January 2025 | Version: 1.0.0