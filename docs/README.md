# MCP Standards Server Documentation Index

**Last Updated:** 2025-07-16  
**Project Status:** ✅ Fully Operational - Complete standards ecosystem with verified components

Welcome to the comprehensive documentation for the MCP Standards Server. This documentation is organized to help different types of users find the information they need quickly.

## 📚 Documentation Structure

### 🎯 Essential Documents
Start here for project overview and current status:

- **[CLAUDE.md](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CLAUDE.md)** - Complete project overview and implementation status
- **[README.md](https://github.com/williamzujkowski/mcp-standards-server/blob/main/README.md)** - Quick start guide and basic usage
- **[Standards Complete Catalog](https://github.com/williamzujkowski/mcp-standards-server/blob/main/STANDARDS_COMPLETE_CATALOG.md)** - All 25 available standards
- **[Web UI Verification Report](https://github.com/williamzujkowski/mcp-standards-server/blob/main/WEB_UI_DEPLOYMENT_VERIFICATION_REPORT.md)** - Web interface deployment guide
- **[Performance Baseline](https://github.com/williamzujkowski/mcp-standards-server/blob/main/benchmarks/PERFORMANCE_BASELINE.md)** - Current system performance metrics

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

### [Security Documentation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/SECURITY.md)
Security measures and implementation details.

- **[Security Implementation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/SECURITY_IMPLEMENTATION.md)** - Security protocols and measures
- **[Environment Variables](https://github.com/williamzujkowski/mcp-standards-server/blob/main/ENVIRONMENT_VARIABLES.md)** - Secure configuration management

### [Implementation Status & Reports](./reports/)
Project status, evaluations, and implementation summaries.

- **[MCP Evaluation Plan](https://github.com/williamzujkowski/mcp-standards-server/blob/main/MCP_EVALUATION_PLAN.md)** - Evaluation methodology
- **[MCP Evaluation Report](https://github.com/williamzujkowski/mcp-standards-server/blob/main/MCP_EVALUATION_REPORT.md)** - Implementation results
- **[Standards Ecosystem](https://github.com/williamzujkowski/mcp-standards-server/blob/main/STANDARDS_ECOSYSTEM.md)** - Complete ecosystem overview
- **[CI/CD Implementation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CICD_IMPLEMENTATION_SUMMARY.md)** - Deployment automation
- **[Comprehensive Test Report](https://github.com/williamzujkowski/mcp-standards-server/blob/main/COMPREHENSIVE_TEST_EXECUTION_REPORT.md)** - Test execution summary
- **[Metrics Integration](https://github.com/williamzujkowski/mcp-standards-server/blob/main/METRICS_INTEGRATION.md)** - Performance monitoring setup

### [Technical Deep Dives](./technical/)
Detailed technical documentation for developers.

- **[Rule Engine](https://github.com/williamzujkowski/mcp-standards-server/blob/main/src/core/standards/README_RULE_ENGINE.md)** - Standards selection logic
- **[Semantic Search](https://github.com/williamzujkowski/mcp-standards-server/blob/main/src/core/standards/README_SEMANTIC_SEARCH.md)** - Search implementation  
- **[Standards Generation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/STANDARDS_GENERATION_GUIDE.md)** - Creating new standards
- **[Analyzers Framework](https://github.com/williamzujkowski/mcp-standards-server/blob/main/src/analyzers/README.md)** - Code analysis system
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
3. Follow [Contributing Guidelines](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CONTRIBUTING_STANDARDS.md)
4. Run [Tests](https://github.com/williamzujkowski/mcp-standards-server/blob/main/tests/README_SYNC_TESTS.md)
5. Study [Standards Generation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/STANDARDS_GENERATION_GUIDE.md)
6. Review [Security Implementation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/SECURITY_IMPLEMENTATION.md)

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
- **Contributors** → [Contributing Guidelines](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CONTRIBUTING_STANDARDS.md)
- **Security Engineers** → [Security Documentation](https://github.com/williamzujkowski/mcp-standards-server/blob/main/SECURITY.md)
- **Standards Authors** → [Standards Generation Guide](https://github.com/williamzujkowski/mcp-standards-server/blob/main/STANDARDS_GENERATION_GUIDE.md)

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

See [Contributing Guidelines](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CONTRIBUTING_STANDARDS.md) for more details.

## 📞 Getting Help

- **Documentation issues** → [GitHub Issues](https://github.com/williamzujkowski/mcp-standards-server/issues)
- **Questions** → [GitHub Discussions](https://github.com/williamzujkowski/mcp-standards-server/discussions)
- **Live chat** → [Discord Server](https://discord.gg/mcp-standards)

---

Last updated: January 2025 | Version: 1.0.0