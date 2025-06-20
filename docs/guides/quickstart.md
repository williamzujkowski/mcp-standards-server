# Quick Start Guide - MCP Standards Server

Get up and running with MCP Standards Server in under 5 minutes! This guide covers installation, basic usage, and key workflows for NIST 800-53r5 compliance.

## 🚀 1-Minute Setup

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Installation

**Option 1: From PyPI (when published)**
```bash
pip install mcp-standards-server
```

**Option 2: From source (current)**
```bash
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server
uv venv && source .venv/bin/activate
uv pip install -e .
```

### Quick Test
```bash
mcp-standards --version
# Should show: MCP Standards Server v0.1.0
```

## 🎯 Core Workflows

### Initialize New Project
```bash
# Basic initialization
mcp-standards init

# Full setup with Git hooks and VS Code integration
mcp-standards init --profile moderate --setup-hooks --setup-vscode

# Creates:
# - .mcp-standards/config.yaml (project configuration)
# - compliance/ (documentation structure)
# - Git hooks for automated validation
# - VS Code settings and tasks
```

### Code Analysis & Scanning
```bash
# Quick scan of current directory
mcp-standards scan

# Scan with specific NIST profile
mcp-standards scan --profile high --output-format json

# Validate specific controls
mcp-standards validate --controls "AC-3,AU-2,IA-2" --output-format table
```

### Generate Secure Code Templates
```bash
# Generate Python API with NIST compliance
mcp-standards generate api --language python --controls "AC-3,IA-2" --output secure_api.py

# Generate authentication module
mcp-standards generate auth --language python --output auth_module.py

# Generate logging setup
mcp-standards generate logging --language javascript --output security_logger.js

# Available templates: api, auth, logging, encryption, database
```

### System Security Plan (SSP) Generation
```bash
# Generate OSCAL-compliant SSP
mcp-standards ssp --output system-security-plan.json --profile moderate

# Generate for specific components
mcp-standards ssp --path ./src --format oscal --output backend-ssp.json
```

### MCP Server
```bash
# Start MCP server for Claude/LLM integration
mcp-standards server --host 127.0.0.1 --port 8000

# With debug logging
mcp-standards server --log-level debug
```

## 💻 Example Workflows

### 1. API Development Workflow
```bash
# 1. Initialize project with Git hooks
mcp-standards init --profile moderate --setup-hooks

# 2. Generate secure API template
mcp-standards generate api --language python --controls "AC-3,AU-2,IA-2" --output app.py

# 3. Scan for compliance as you code
mcp-standards scan --output-format json --output-file compliance-report.json

# 4. Generate SSP when ready
mcp-standards ssp --output api-security-plan.json
```

### 2. Frontend Security Workflow
```bash
# 1. Generate secure frontend template
mcp-standards generate auth --language javascript --output secure-auth.js

# 2. Validate client-side security
mcp-standards validate --controls "AC-3,SI-10,SI-11"

# 3. Scan for XSS and security issues
mcp-standards scan frontend/ --output-format oscal
```

### 3. Database Security Workflow
```bash
# 1. Generate secure database patterns
mcp-standards generate database --language python --controls "AC-3,AU-2" --output db_security.py

# 2. Validate SQL injection prevention
mcp-standards validate --controls "SI-10" --path database/

# 3. Generate compliance documentation
mcp-standards ssp database/ --output database-ssp.json
```

## 🔌 Integration Examples

### Claude Desktop Integration
Add to `~/.config/Claude/claude_desktop_config.json` (Linux) or equivalent:

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

### Git Hooks Integration
```bash
# Automatic setup during init
mcp-standards init --setup-hooks

# Manual setup
echo "mcp-standards validate --fail-on-missing" > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

### VS Code Integration
```bash
# Automatic setup during init
mcp-standards init --setup-vscode

# Manual configuration in .vscode/settings.json:
{
  "python.linting.enabled": true,
  "python.linting.pylintArgs": ["--load-plugins=mcp_standards_linter"],
  "files.associations": {
    "*.py": "python-nist"
  }
}
```

## 📝 NIST Control Annotations

Add compliance annotations to your code:

```python
def authenticate_user(username: str, password: str) -> User:
    """
    Authenticate user with secure password verification
    @nist-controls: IA-2, IA-5
    @evidence: Secure password authentication with bcrypt
    @oscal-component: authentication-service
    """
    # Implementation with proper hashing
    return verify_credentials(username, password)

@audit_log
def access_resource(user: User, resource: str) -> bool:
    """
    Enforce role-based access control
    @nist-controls: AC-3
    @evidence: RBAC implementation with audit logging
    @oscal-component: authorization-service
    """
    return user.has_permission(resource)
```

## 🛠️ Advanced Usage

### Custom Standards Loading
```bash
# Load specific standards
export STANDARDS_PATH="/custom/path/to/standards"
mcp-standards scan --standards-query "secure API design"
```

### Multi-Language Analysis
```bash
# Scan mixed-language project
mcp-standards scan \
  --include-patterns "*.py,*.js,*.ts,*.go,*.java" \
  --exclude-patterns "node_modules/**,venv/**" \
  --output-format oscal
```

### Performance Optimization
```bash
# Enable Redis caching
export REDIS_URL="redis://localhost:6379/0"
mcp-standards server

# Use token limits for large codebases
mcp-standards scan --token-limit 50000
```

## 🎯 Quick Reference

### All CLI Commands
```bash
mcp-standards init        # Initialize project
mcp-standards scan        # Analyze code for compliance
mcp-standards generate    # Generate secure code templates
mcp-standards validate    # Validate against standards
mcp-standards ssp         # Generate System Security Plan
mcp-standards server      # Start MCP server
mcp-standards version     # Show version info
```

### Key File Locations
- Configuration: `.mcp-standards/config.yaml`
- Standards: `data/standards/` (17 imported documents)
- Examples: `examples/` (Python API, JS frontend, etc.)
- Documentation: `compliance/` (generated during init)

## 🆘 Troubleshooting

### Common Issues
1. **Import errors**: Ensure virtual environment is activated
2. **Standards not found**: Check `STANDARDS_PATH` environment variable
3. **Git hooks failing**: Run `mcp-standards validate` manually first
4. **MCP server not starting**: Check port availability and permissions

### Get Help
```bash
mcp-standards --help
mcp-standards scan --help
mcp-standards generate --help
```

## 🚀 Next Steps

- **[CLI Reference](./cli.md)** - Complete command documentation
- **[MCP Tools Guide](../api/mcp-tools.md)** - Use with Claude/LLMs
- **[NIST Controls](../nist/controls.md)** - Implementation guidance
- **[Example Projects](../../examples/)** - Production-ready implementations
- **[Architecture Guide](../architecture/)** - System design and patterns