# MCP Standards Server

A Model Context Protocol (MCP) server that provides intelligent, context-aware access to development standards. This system enables LLMs to automatically select and apply appropriate standards based on project requirements.

## Features

### Core Capabilities
- **25 Comprehensive Standards**: Complete coverage of software development lifecycle
- **Intelligent Standard Selection**: Rule-based engine with 40+ detection rules
- **MCP Server Implementation**: Full Model Context Protocol support with multiple tools
- **Standards Generation System**: Template-based creation with quality assurance
- **Hybrid Vector Storage**: ChromaDB + in-memory for semantic search
- **Multi-Language Analyzers**: Python, JavaScript, Go, Java, Rust, TypeScript support

### Advanced Features
- **Redis Caching Layer**: L1/L2 architecture for performance optimization
- **Web UI**: React/TypeScript interface for browsing and testing standards
- **CLI Tools**: Comprehensive command-line interface with documentation
- **Performance Benchmarking**: Continuous monitoring and optimization
- **Token Optimization**: Multiple compression formats for LLM efficiency
- **NIST Compliance**: NIST 800-53r5 control mapping and validation
- **Community Features**: Review process, contribution guidelines, analytics

## Requirements

- Python 3.8 or higher
- Redis (optional, for caching)
- Node.js 16+ (optional, for web UI)

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[full]"

# For development with testing tools
pip install -e ".[test]"

# Install Redis (optional, for caching)
# macOS: brew install redis
# Ubuntu: sudo apt-get install redis-server
# Windows: Use WSL or Docker
```

### Basic Usage

```python
from pathlib import Path
from src.core.standards.rule_engine import RuleEngine

# Load the rule engine
rules_path = Path("data/standards/meta/standard-selection-rules.json")
engine = RuleEngine(rules_path)

# Define your project context
context = {
    "project_type": "web_application",
    "framework": "react",
    "language": "javascript",
    "requirements": ["accessibility", "performance"]
}

# Get applicable standards
result = engine.evaluate(context)
print(f"Selected standards: {result['resolved_standards']}")
```

### Running the MCP Server

```bash
# Start the MCP server
python -m src.server

# Or use the CLI
mcp-standards --help

# Run the web UI
mcp-standards web
```

### Using the Universal Project Kickstart

```bash
# Copy the kickstart prompt for any LLM
cat kickstart.md
```

### Synchronizing Standards

The server can automatically sync standards from the GitHub repository:

```bash
# Check for updates
mcp-standards sync --check

# Perform synchronization
mcp-standards sync

# Force sync all files (ignore cache)
mcp-standards sync --force

# View sync status
mcp-standards status

# Manage cache
mcp-standards cache --list
mcp-standards cache --clear
```

Configure synchronization in `data/standards/sync_config.yaml`.

## Rule Configuration

Rules are defined in JSON format in `data/standards/meta/standard-selection-rules.json`. Each rule specifies:

- Conditions for when it applies
- Standards to apply when matched
- Priority for conflict resolution
- Tags for categorization

Example rule:

```json
{
  "id": "react-web-app",
  "name": "React Web Application Standards",
  "priority": 10,
  "conditions": {
    "logic": "AND",
    "conditions": [
      {
        "field": "project_type",
        "operator": "equals",
        "value": "web_application"
      },
      {
        "field": "framework",
        "operator": "in",
        "value": ["react", "next.js", "gatsby"]
      }
    ]
  },
  "standards": [
    "react-18-patterns",
    "javascript-es2025",
    "frontend-accessibility"
  ],
  "tags": ["frontend", "react", "web"]
}
```

## Available Standards

The system includes 25 comprehensive standards:

### Specialty Domains (8)
- AI/ML Operations, Blockchain/Web3, IoT/Edge Computing, Gaming Development
- AR/VR Development, Advanced API Design, Database Optimization, Green Computing

### Testing & Quality (3)
- Advanced Testing, Code Review, Performance Optimization

### Security & Compliance (3)
- Security Review & Audit, Data Privacy, Business Continuity

### Documentation & Communication (4)
- Technical Content, Documentation Writing, Team Collaboration, Project Planning

### Operations & Infrastructure (4)
- Deployment & Release, Monitoring & Incident Response, SRE, Technical Debt

### User Experience (3)
- Advanced Accessibility, Internationalization, Developer Experience

See [STANDARDS_COMPLETE_CATALOG.md](./STANDARDS_COMPLETE_CATALOG.md) for details.

## Architecture

```
mcp-standards-server/
├── src/
│   ├── core/
│   │   ├── mcp/              # MCP server implementation
│   │   ├── standards/        # Standards engine & storage
│   │   └── compliance/       # NIST compliance mapping
│   ├── analyzers/            # Language-specific analyzers
│   ├── generators/           # Standards generation system
│   ├── cli/                  # CLI interface
│   └── web/                  # React/TypeScript UI
├── data/
│   └── standards/            # 25 comprehensive standards
│       ├── meta/            # Rule engine configuration
│       └── cache/           # Redis-backed cache
├── templates/                # Standard generation templates
├── tests/                    # Comprehensive test suite
└── benchmarks/              # Performance benchmarking
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/unit/core/standards/test_rule_engine.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Documentation

### Quick Start
- [Universal Project Kickstart](./kickstart.md) - Copy-paste prompt for any LLM
- [Standards Complete Catalog](./STANDARDS_COMPLETE_CATALOG.md) - All 25 standards
- [Creating Standards Guide](./docs/CREATING_STANDARDS_GUIDE.md) - How to create new standards

### Technical Documentation
- [Claude Integration Guide](CLAUDE.md) - Main system documentation
- [Rule Engine Documentation](src/core/standards/README_RULE_ENGINE.md)
- [API Documentation](./docs/api/mcp-tools.md)
- [Project Plan](project_plan.md)

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project is part of the williamzujkowski/standards ecosystem, designed to improve code quality and consistency through intelligent standard selection and application.