# MCP Standards Server

A comprehensive Model Context Protocol (MCP) server that provides intelligent NIST 800-53r5 compliance checking, automated code analysis, and standards enforcement for modern development workflows. Built using the official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) with real standards content from [williamzujkowski/standards](https://github.com/williamzujkowski/standards).

## 🚀 Features

### Core Compliance
- 🔒 **NIST 800-53r5 Compliance**: Detects 200+ controls across all 20 families with 91.56% test coverage
- 📊 **17 Standards Imported**: Complete standards library from official repository
- 📝 **OSCAL 1.0.0 Support**: Generate System Security Plans (SSPs) automatically
- 🔍 **Multi-Language Analysis**: Python, JavaScript/TypeScript, Go, Java with enhanced AST parsing

### MCP Integration
- 🤖 **Native MCP Server**: Official SDK implementation with full protocol support
- 🌐 **Dynamic Resources**: Real-time standards access with 20+ resource endpoints
- 💬 **Smart Prompts**: 5 specialized prompt templates for compliance queries
- 🔄 **Live Standards**: Direct access to current standards documentation with versioning

### Developer Tools
- 🛠️ **Complete CLI**: init, scan, generate, validate, ssp, coverage, standards commands
- 📋 **Code Templates**: NIST-compliant templates for common patterns
- 🔧 **Git Integration**: Automated hooks for pre-commit compliance checking
- 🎯 **VS Code Support**: Integrated settings and workflow
- 📈 **Coverage Analysis**: Comprehensive control coverage reports with gap analysis

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip
- Redis (optional, for caching)

### Installation

#### Using pip
```bash
pip install mcp-standards-server
```

#### From source
1. Clone the repository:
```bash
git clone https://github.com/williamzujkowski/mcp-standards-server.git
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

### Basic Usage

#### Initialize a Project
```bash
# Initialize with NIST compliance structure
mcp-standards init --profile moderate --setup-hooks

# This creates:
# - .mcp-standards/config.yaml (project configuration)
# - compliance/ (documentation structure)
# - Git hooks for automated validation
# - VS Code settings
```

#### Scan for Compliance
```bash
# Scan current directory
mcp-standards scan

# Scan with specific profile
mcp-standards scan --profile high --output-format json

# Validate against specific controls
mcp-standards validate --controls "AC-3,AU-2,IA-2"

# Generate control coverage report
mcp-standards coverage --output-format markdown

# Export coverage report
mcp-standards coverage --output-format html --output-file coverage.html
```

#### Generate Secure Code
```bash
# Generate Python API template
mcp-standards generate api --language python --controls "AC-3,IA-2"

# Generate authentication module
mcp-standards generate auth --language python --output auth.py

# Generate logging setup
mcp-standards generate logging --language python
```

#### Create System Security Plan
```bash
# Generate OSCAL-compliant SSP
mcp-standards ssp --output system-security-plan.json --profile moderate

# Generate for specific components
mcp-standards ssp --path ./src --format oscal
```

#### Start MCP Server
```bash
# Start the MCP server
mcp-standards server --host 127.0.0.1 --port 8000

# Or run as module
python -m src.server
```

### Using with Claude Desktop

Add to your Claude Desktop configuration:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"],
      "env": {
        "STANDARDS_PATH": "/path/to/standards"
      }
    }
  }
}
```

## 🛠️ MCP Tools

The server provides comprehensive MCP tools for compliance workflows:

### `load_standards`
Load standards based on natural language or notation queries:
```json
{
  "query": "secure api design",
  "context": "Building REST API with OAuth2",
  "token_limit": 10000
}
```
Returns relevant standards sections with metadata and token counting.

### `analyze_code` 
Analyze code for NIST control implementations:
```json
{
  "code": "def authenticate_user(username, password):",
  "language": "python",
  "filename": "auth.py"
}
```
Returns detected controls, evidence, suggestions, and compliance score.

### `suggest_controls`
Get NIST control recommendations based on requirements:
```json
{
  "description": "Building user authentication with MFA",
  "components": ["web-app", "api", "database"],
  "security_level": "high"
}
```
Returns recommended controls with implementation guidance.

### `generate_template`
Generate NIST-compliant code templates:
```json
{
  "template_type": "api",
  "language": "python", 
  "controls": ["AC-3", "AU-2", "SI-10"]
}
```
Supports: api, auth, logging, encryption, database templates.

### `validate_compliance`
Validate code/project against NIST requirements:
```json
{
  "path": "/project/src",
  "profile": "moderate",
  "controls": ["AC-3", "AU-2"]
}
```
Returns compliance report with gaps and recommendations.

### `scan_with_llm`
Enhanced scanning with LLM analysis:
```json
{
  "path": "/project",
  "focus_areas": ["authentication", "encryption"],
  "output_format": "detailed"
}
```
Provides deep insights with LLM-powered analysis.

## 🌐 MCP Resources

Dynamic resource access to standards content:

### Standards Categories
- `standards://category/core` - Core unified standards
- `standards://category/development` - Development standards  
- `standards://category/security` - Security standards
- `standards://category/cloud` - Cloud-native standards
- `standards://category/data` - Data engineering standards

### Individual Standards
- `standards://document/unified_standards` - Master unified standards
- `standards://document/coding_standards` - Coding best practices
- `standards://document/modern_security_standards` - Security patterns
- `standards://document/testing_standards` - Testing frameworks

### Catalogs & References
- `standards://catalog` - Complete standards catalog with indexing
- `standards://nist-controls` - NIST 800-53r5 control catalog with families
- `standards://templates` - Available code templates with metadata

## 💬 MCP Prompts

Specialized prompt templates for compliance scenarios:

### `secure-api-design`
Design secure APIs with NIST compliance:
```json
{
  "api_type": "REST"
}
```

### `compliance-checklist` 
Generate project-specific compliance checklists:
```json
{
  "project_type": "web application",
  "profile": "moderate"
}
```

### `security-review`
Perform comprehensive security reviews:
```json
{
  "code_context": "User authentication system",
  "focus_areas": "authentication, session management"
}
```

### `control-implementation`
Get implementation guidance for NIST controls:
```json
{
  "control_id": "AC-3",
  "technology": "Python Flask"
}
```

### `standards-query`
Query standards for specific requirements:
```json
{
  "topic": "API security",
  "domain": "development"
}
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

## 📁 Project Structure

```
mcp-standards-server/
├── src/
│   ├── cli/                 # CLI commands (init, scan, generate, validate, ssp)
│   ├── core/
│   │   ├── compliance/      # OSCAL handler, SSP generation
│   │   ├── standards/       # Standards engine, natural language mapping
│   │   └── templates.py     # NIST-compliant code templates
│   ├── analyzers/          # Multi-language code analyzers
│   ├── compliance/         # Compliance scanning and reporting
│   └── server.py           # Main MCP server implementation
├── data/
│   └── standards/          # Imported standards (17 files)
├── examples/               # Example implementations
│   ├── python-api/         # Flask API with NIST compliance
│   ├── javascript-frontend/ # Secure frontend SPA
│   └── secure-database/    # Database security patterns
├── docs/                   # Comprehensive documentation
├── tests/                  # Test suite (91.56% coverage)
└── scripts/                # Utility scripts
```

## 🔧 Configuration

### Environment Variables
- `STANDARDS_PATH` - Path to standards repository (default: `./data/standards`)
- `REDIS_URL` - Redis connection URL for caching (optional)
- `LOG_LEVEL` - Logging level (default: `INFO`)
- `MCP_SERVER_NAME` - Server name for MCP protocol

### Project Configuration (.mcp-standards/config.yaml)
```yaml
version: "1.0.0"
profile: "moderate"  # low, moderate, high
language: "python"
scanning:
  include_patterns: ["*.py", "*.js", "*.ts", "*.go", "*.java"]
  exclude_patterns: ["node_modules/**", "venv/**", ".git/**"]
compliance:
  required_controls: ["AC-3", "AU-2", "IA-2", "SC-8", "SI-10"]
  nist_profile: "moderate"
```

## 🛡️ Security & Compliance

### Comprehensive NIST 800-53r5 Coverage
The server now detects **200+ controls** across all 20 NIST families:

#### Key Control Families
- **Access Control (AC)**: 25 controls including RBAC, least privilege, session management
- **Audit & Accountability (AU)**: 16 controls for comprehensive logging and audit trails
- **System & Communications Protection (SC)**: 45 controls including encryption, boundary protection
- **System & Information Integrity (SI)**: 23 controls for input validation, malware protection
- **Identification & Authentication (IA)**: 12 controls including MFA, federated identity
- **Configuration Management (CM)**: 12 controls for baselines and change control
- **Contingency Planning (CP)**: 13 controls for backup and disaster recovery
- **Risk Assessment (RA)**: 10 controls for vulnerability scanning and risk analysis
- **Incident Response (IR)**: 10 controls for incident handling and forensics
- **Supply Chain Risk Management (SR)**: 12 controls including SBOM generation

#### High-Priority Controls Detected
- **IA-2(1)**: Multi-factor Authentication
- **SC-8/SC-13**: Encryption in Transit and at Rest
- **SI-10**: Input Validation and SQL Injection Prevention
- **AU-2/AU-3**: Security Event Logging with Full Context
- **AC-3/AC-6**: Role-Based Access Control with Least Privilege
- **CP-9/CP-10**: Information System Backup and Recovery
- **RA-5**: Vulnerability Scanning
- **MP-6**: Media Sanitization

### Security Features
- **91.56% Test Coverage** - Exceeding industry standards
- **Real Standards Content** - Official williamzujkowski/standards repository
- **OSCAL 1.0.0 Compliance** - Generate compliant SSPs
- **Multi-Language Analysis** - AST-based security pattern detection
- **Automated Workflows** - Git hooks for continuous compliance

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