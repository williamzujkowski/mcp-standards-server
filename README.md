# MCP Standards Server

A Model Context Protocol (MCP) server that provides intelligent, context-aware access to development standards. This system enables LLMs to automatically select and apply appropriate standards based on project requirements.

## Features

- **Intelligent Standard Selection**: Rule-based engine automatically selects relevant standards based on project context
- **Flexible Rule System**: Define complex conditions using AND/OR/NOT logic with various operators
- **Priority-Based Conflict Resolution**: Handles competing standards through configurable priorities
- **Automatic Standards Synchronization**: Fetches and caches standards from GitHub repository with version tracking
- **Extensible Architecture**: Easy to add new rules, standards, and evaluation logic
- **Decision Tree Visualization**: Understand rule relationships and decision paths
- **Token Optimization**: Efficient standard delivery optimized for LLM consumption
- **Rate Limit Handling**: Intelligent GitHub API rate limit management with automatic retry

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

### Running the Demo

```bash
python examples/rule_engine_demo.py
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

## Supported Project Types

The rule engine supports automatic standard selection for:

- **Web Applications**: React, Vue, Angular, and vanilla JavaScript
- **APIs**: REST and GraphQL APIs in Python, Node.js, and other languages
- **Mobile Apps**: React Native and other mobile frameworks
- **Microservices**: Cloud-native and containerized applications
- **Data Pipelines**: ETL and data processing workflows
- **Machine Learning**: ML project structure and deployment
- **MCP Servers**: Model Context Protocol server development

## Architecture

```
mcp-standards-server/
├── src/
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                # CLI interface
│   └── core/
│       └── standards/
│           ├── rule_engine.py      # Main rule engine implementation
│           ├── sync.py             # Standards synchronization module
│           └── __init__.py
├── data/
│   └── standards/
│       ├── cache/                  # Cached standards files
│       ├── meta/
│       │   └── standard-selection-rules.json  # Rule definitions
│       └── sync_config.yaml        # Sync configuration
├── tests/
│   └── unit/
│       └── core/
│           └── standards/
│               ├── test_rule_engine.py  # Rule engine tests
│               └── test_sync.py         # Sync module tests
└── examples/
    └── rule_engine_demo.py  # Usage examples
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

- [Rule Engine Documentation](src/core/standards/README_RULE_ENGINE.md)
- [Project Plan](project_plan.md)
- [Claude Integration Guide](CLAUDE.md)

## License

MIT License - see LICENSE file for details

## Acknowledgments

This project is part of the williamzujkowski/standards ecosystem, designed to improve code quality and consistency through intelligent standard selection and application.