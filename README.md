# MCP Standards Server

A Model Context Protocol (MCP) server that provides intelligent NIST 800-53r5 compliance checking, code analysis, and standards enforcement for modern development workflows. Built using the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk).

## Features

- 🔒 **NIST 800-53r5 Compliance**: Automated control mapping and evidence generation
- 🤖 **MCP Protocol**: Native MCP server implementation for LLM integration
- 📊 **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Java
- 🚀 **Standards Engine**: Natural language queries for standards and compliance
- 📝 **OSCAL Support**: Generate System Security Plans (SSPs) automatically
- 🔍 **Code Analysis**: Deep AST-based pattern recognition for security controls

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mcp-standards-server.git
cd mcp-standards-server
```

2. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

4. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your settings
```

### Running the MCP Server

The server implements the Model Context Protocol and can be used with any MCP-compatible client:

```bash
# Run the MCP server
mcp-standards-server

# Or run directly with Python
python -m src.server
```

### Using with Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "/path/to/venv/bin/mcp-standards-server"
    }
  }
}
```

## MCP Tools

The server provides the following MCP tools:

### `load_standards`
Load standards based on natural language or notation queries:
```
Query: "secure api design"
Response: Loads CS:api, SEC:api, and related standards
```

### `analyze_code`
Analyze code for NIST control implementations:
```
Input: Python code snippet
Output: Detected NIST controls, recommendations, and compliance score
```

### `suggest_controls`
Get NIST control recommendations based on requirements:
```
Input: "Building user authentication system"
Output: Suggests IA-2, IA-5, AC-7, and related controls
```

### `generate_template`
Generate NIST-compliant code templates:
```
Template types: api-endpoint, auth-module, logging-setup, encryption-utils
Languages: Python, JavaScript, TypeScript, Go, Java
```

### `validate_compliance`
Validate code/project against NIST compliance requirements:
```
Input: File or directory path
Output: Compliance report with gaps and recommendations
```

## MCP Resources

The server exposes these resources:

- `standards://catalog` - Complete catalog of available standards
- `standards://nist-controls` - NIST 800-53r5 control catalog
- `standards://templates` - Code templates library

## MCP Prompts

Pre-configured prompts for common scenarios:

- `secure-api-design` - Design secure APIs with NIST compliance
- `compliance-checklist` - Generate project-specific compliance checklists

## NIST Control Annotations

Add NIST control annotations to your code:

```python
def authenticate_user(username: str, password: str) -> User:
    """
    Authenticate user with multi-factor authentication
    @nist-controls: IA-2, IA-2(1), IA-5
    @evidence: MFA implementation using TOTP
    @oscal-component: authentication-service
    """
    # Implementation here
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   MCP Client    │────▶│   MCP Server     │────▶│ Standards Engine│
│  (LLM/Claude)   │     │  (This Project)  │     │                 │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Code Analyzers   │     │ NIST Mapper     │
                        └──────────────────┘     └─────────────────┘
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_models.py
```

### Code Quality

```bash
# Format and lint
ruff check src/ tests/
ruff format src/ tests/

# Type checking
mypy src/
```

### Docker

Build and run with Docker:

```bash
# Build image
docker build -t mcp-standards-server .

# Run container
docker run -it mcp-standards-server
```

## Configuration

The server can be configured through environment variables:

- `STANDARDS_PATH` - Path to standards repository (default: `./data/standards`)
- `REDIS_URL` - Redis connection URL for caching (optional)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `MCP_SERVER_NAME` - Server name for MCP protocol (default: `mcp-standards-server`)

## Security Considerations

This project implements the following security controls:

- **Authentication** (IA-2): Integration with MCP client authentication
- **Access Control** (AC-3): Tool-level access control
- **Encryption** (SC-8, SC-13): Secure communication via MCP protocol
- **Audit Logging** (AU-2, AU-3): Comprehensive security event logging
- **Input Validation** (SI-10): All tool inputs validated

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add NIST control annotations to new security-relevant code
4. Ensure all tests pass and coverage remains above 80%
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with the [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- Based on the [williamzujkowski/standards](https://github.com/williamzujkowski/standards) repository
- Implements NIST 800-53r5 security controls