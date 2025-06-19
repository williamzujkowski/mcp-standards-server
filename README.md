# MCP Standards Server

A Model Context Protocol (MCP) server that provides intelligent NIST 800-53r5 compliance checking, code analysis, and standards enforcement for modern development workflows.

## Features

- 🔒 **NIST 800-53r5 Compliance**: Automated control mapping and evidence generation
- 🤖 **LLM Integration**: Natural language queries for standards and compliance
- 📊 **Multi-Language Support**: Python, JavaScript, TypeScript, Go, Java
- 🚀 **Real-time Analysis**: WebSocket-based MCP protocol implementation
- 📝 **OSCAL Support**: Generate System Security Plans (SSPs) automatically
- 🔍 **Deep Code Analysis**: AST-based pattern recognition for security controls

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.6+
- Docker & Docker Compose (optional)
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mcp-standards-server.git
cd mcp-standards-server
```

2. Install dependencies:
```bash
poetry install
```

3. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your settings
```

4. Initialize a project:
```bash
poetry run mcp-standards init
```

### Usage

#### Start the MCP Server

```bash
# Development mode
poetry run mcp-standards server --reload

# Production mode with Docker
docker-compose up -d
```

#### Scan Your Codebase

```bash
# Basic scan
poetry run mcp-standards scan

# Deep analysis with OSCAL output
poetry run mcp-standards scan --deep --output-format oscal --output-file ssp.json
```

#### Generate Compliant Code

```bash
# Generate a secure API endpoint
poetry run mcp-standards generate api-endpoint --controls AC-3,AU-2

# Generate authentication module
poetry run mcp-standards generate auth-module --language python
```

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
│   LLM Client    │────▶│   MCP Server     │────▶│ Standards Engine│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │ Code Analyzers   │     │ NIST Mapper     │
                        └──────────────────┘     └─────────────────┘
```

## Configuration

The project uses a hierarchical configuration system:

1. **Project Config** (`.mcp-standards/config.yaml`):
   - NIST profile selection
   - Language preferences
   - Scan exclusions

2. **Environment Variables** (`.env`):
   - Server settings
   - Database connections
   - Security keys

3. **Standards Mapping** (`src/data/standards/`):
   - Control definitions
   - Pattern mappings
   - Evidence templates

## API Documentation

### MCP Protocol Methods

- `load_standards`: Load standards based on natural language or notation
- `analyze_code`: Analyze code for NIST control implementations
- `suggest_controls`: Get control recommendations for code patterns
- `generate_ssp`: Create OSCAL-compliant System Security Plans

### REST API Endpoints

- `GET /health`: Health check
- `POST /api/scan`: Trigger code scan
- `GET /api/controls`: List implemented controls
- `POST /api/generate`: Generate compliant code

## Development

### Running Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=src --cov-report=html

# Specific test file
poetry run pytest tests/test_nist_mapper.py
```

### Code Quality

```bash
# Format code
poetry run black src/ tests/

# Lint
poetry run ruff check src/ tests/

# Type checking
poetry run mypy src/
```

### Pre-commit Hooks

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Security Considerations

This project implements the following security controls:

- **Authentication** (IA-2): JWT-based authentication with MFA support
- **Access Control** (AC-3): Role-based access control for API endpoints
- **Encryption** (SC-8, SC-13): TLS for data in transit, AES-256 for data at rest
- **Audit Logging** (AU-2, AU-3): Comprehensive security event logging
- **Input Validation** (SI-10): All inputs validated and sanitized

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Add NIST control annotations to new security-relevant code
4. Ensure all tests pass and coverage remains above 80%
5. Submit a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the [williamzujkowski/standards](https://github.com/williamzujkowski/standards) repository
- Implements NIST 800-53r5 security controls
- Uses the Model Context Protocol (MCP) specification