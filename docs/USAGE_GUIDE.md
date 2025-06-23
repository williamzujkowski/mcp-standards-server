# MCP Standards Server - Complete Usage Guide

**Version:** 1.0.0  
**Last Updated:** 2025-06-23  
**Status:** Active  
**Standard Code:** USG  

**Summary:** Comprehensive usage guide with all CLI commands and MCP tools  
**Tokens:** ~4500 (helps AI plan context usage)  
**Priority:** high  

This guide provides comprehensive instructions for using the MCP Standards Server with Claude CLI and documents all available commands.

## 📋 Table of Contents

1. [Using with Claude CLI](#using-with-claude-cli)
2. [MCP Server Configuration](#mcp-server-configuration)
3. [CLI Commands Reference](#cli-commands-reference)
4. [MCP Tools Reference](#mcp-tools-reference)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Using with Claude CLI

### Prerequisites

1. **Install Required Dependencies:**

**Redis (Required for caching tier):**
```bash
# macOS
brew install redis && brew services start redis

# Ubuntu/Debian
sudo apt update && sudo apt install redis-server
sudo systemctl start redis-server

# Windows (WSL)
sudo apt install redis-server && sudo service redis-server start
```

2. **Install Claude CLI:**
```bash
npm install -g @anthropic-ai/claude-cli
```

3. **Install MCP Standards Server:**
```bash
# Using pip (automatically installs FAISS and ChromaDB as core dependencies)
pip install mcp-standards-server

# Or from source
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server
uv venv
source .venv/bin/activate
uv pip install -e .
```

4. **Verify Installation:**
```bash
# Test Redis
redis-cli ping  # Should return: PONG

# Test MCP server
mcp-standards cache status  # Should show all three tiers
```

### Configuration for Claude CLI

1. Create or edit your Claude configuration file:

**macOS/Linux**:
```bash
mkdir -p ~/.config/claude
nano ~/.config/claude/claude_desktop_config.json
```

**Windows**:
```powershell
mkdir %APPDATA%\Claude
notepad %APPDATA%\Claude\claude_desktop_config.json
```

2. Add the MCP Standards Server configuration:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"],
      "env": {
        "STANDARDS_PATH": "/path/to/your/standards",
        "REDIS_URL": "redis://localhost:6379",
        "LOG_LEVEL": "INFO"
      }
    }
  }
}
```

### Alternative: Using Python Path

If `mcp-standards` is not in your PATH, use the full Python command:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/mcp-standards-server",
      "env": {
        "PYTHONPATH": "/path/to/mcp-standards-server",
        "STANDARDS_PATH": "/path/to/standards"
      }
    }
  }
}
```

### Starting Claude with MCP Server

1. Start the MCP server manually (for testing):
```bash
mcp-standards server
# Or
python -m src.server
```

2. Launch Claude Desktop - it will automatically connect to configured MCP servers

3. Verify connection by asking Claude:
```
What MCP tools are available?
```

## MCP Server Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STANDARDS_PATH` | Path to standards directory | `./data/standards` |
| `REDIS_URL` | Redis connection URL (required) | `redis://localhost:6379` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `CACHE_TTL` | Redis cache time-to-live (seconds) | `3600` |
| `MAX_TOKENS` | Maximum tokens per response | `10000` |
| `VECTOR_STORE_TYPE` | Always `hybrid` (three-tier) | `hybrid` |
| `FAISS_HOT_CACHE_SIZE` | FAISS tier capacity | `1000` |
| `CHROMADB_PERSIST_PATH` | ChromaDB storage location | `./.chroma_db` |

### Configuration File

Create `.mcp-standards/config.yaml`:

```yaml
# Project configuration
project:
  name: "My Secure Project"
  profile: moderate  # low, moderate, high
  languages:
    - python
    - javascript
  
# Standards configuration  
standards:
  path: ./data/standards
  auto_import: true
  version_control: true

# Analysis settings
analysis:
  recursive: true
  ignore_patterns:
    - "*.test.js"
    - "__pycache__"
  frameworks:
    - django
    - react

# MCP settings with three-tier architecture
mcp:
  max_tokens: 10000
  vector_store:
    type: hybrid  # Always hybrid (FAISS + ChromaDB + Redis)
    # Tier 1: Redis Query Cache
    redis:
      url: redis://localhost:6379
      ttl: 3600
    # Tier 2: FAISS Hot Cache  
    faiss:
      hot_cache_size: 1000
      access_threshold: 10
    # Tier 3: ChromaDB Persistent Storage
    chromadb:
      persist_path: ./.chroma_db
      batch_size: 100
```

## CLI Commands Reference

### 🚀 `init` - Initialize Project

Initialize MCP standards compliance for a project.

```bash
mcp-standards init [OPTIONS] [PROJECT_PATH]
```

**Arguments:**
- `PROJECT_PATH`: Path to initialize (default: current directory)

**Options:**
- `--profile`: NIST profile: low, moderate, high (default: moderate)
- `--language`: Primary language (auto-detect if not specified)
- `--setup-hooks/--no-setup-hooks`: Setup Git hooks (default: true)

**Example:**
```bash
# Initialize with moderate profile and Git hooks
mcp-standards init --profile moderate --setup-hooks

# Initialize for Python project with high security
mcp-standards init --profile high --language python
```

**Creates:**
- `.mcp-standards/config.yaml` - Project configuration
- `compliance/` - Documentation structure
- `.git/hooks/pre-commit` - Automated validation hooks

### 🔍 `scan` - Scan for Compliance

Scan code for NIST control implementations.

```bash
mcp-standards scan [OPTIONS] [PATH]
```

**Arguments:**
- `PATH`: Path to scan (default: current directory)

**Options:**
- `--output-format`: Output format: table, json, yaml, oscal (default: table)
- `--recursive/--no-recursive`: Scan recursively (default: true)
- `--profile`: Filter by NIST profile
- `--save-report`: Save report to file

**Example:**
```bash
# Basic scan with table output
mcp-standards scan

# Scan with JSON output
mcp-standards scan --output-format json

# Scan specific directory with report
mcp-standards scan ./src --save-report compliance-report.json
```

### 📝 `ssp` - Generate System Security Plan

Generate OSCAL-compliant System Security Plan.

```bash
mcp-standards ssp [OPTIONS]
```

**Options:**
- `--output`: Output file (default: ssp.json)
- `--format`: Output format: oscal, json (default: oscal)
- `--profile`: NIST profile to use
- `--metadata-file`: Custom metadata JSON file

**Example:**
```bash
# Generate OSCAL SSP
mcp-standards ssp --output my-ssp.json

# Generate with custom metadata
mcp-standards ssp --metadata-file project-metadata.json
```

### 🛡️ `validate` - Validate Compliance

Validate code against specific NIST requirements.

```bash
mcp-standards validate [OPTIONS] [PATH]
```

**Arguments:**
- `PATH`: File or directory to validate

**Options:**
- `--profile`: NIST profile: low, moderate, high
- `--controls`: Comma-separated control IDs (e.g., "AC-3,AU-2")
- `--strict/--no-strict`: Fail on warnings (default: false)
- `--output-format`: Output format: table, json

**Example:**
```bash
# Validate against moderate profile
mcp-standards validate --profile moderate

# Validate specific controls
mcp-standards validate --controls "AC-3,AU-2,IA-2" --strict
```

### 📊 `coverage` - Generate Coverage Report

Analyze and report on NIST control coverage.

```bash
mcp-standards coverage [OPTIONS] [PATH]
```

**Arguments:**
- `PATH`: Path to analyze (default: current directory)

**Options:**
- `--output-format`: Output format: markdown, json, html (default: markdown)
- `--output-file`: Save report to file
- `--include-gaps/--no-include-gaps`: Include gap analysis (default: true)
- `--min-confidence`: Minimum confidence threshold (0.0-1.0)

**Example:**
```bash
# Generate markdown coverage report
mcp-standards coverage

# Generate HTML report with gaps
mcp-standards coverage --output-format html --output-file coverage.html

# Coverage with high confidence only
mcp-standards coverage --min-confidence 0.8
```

### 🏗️ `generate` - Generate Secure Code

Generate NIST-compliant code templates.

```bash
mcp-standards generate [TEMPLATE_TYPE] [OPTIONS]
```

**Arguments:**
- `TEMPLATE_TYPE`: Template to generate: api, auth, logging, encryption, database

**Options:**
- `--language`: Target language: python, javascript, go, java
- `--controls`: Specific controls to implement
- `--framework`: Framework-specific template (e.g., django, express)
- `--output`: Output directory

**Example:**
```bash
# Generate Python API with authentication
mcp-standards generate api --language python --controls "AC-3,IA-2"

# Generate Django authentication module
mcp-standards generate auth --language python --framework django

# Generate secure logging module
mcp-standards generate logging --language javascript --controls "AU-2,AU-3"
```

### 📦 `standards` - Standards Management

Manage imported standards and versions.

```bash
mcp-standards standards [COMMAND] [OPTIONS]
```

**Subcommands:**
- `list`: List all imported standards
- `import`: Import standards from repository
- `update`: Update standards to latest version
- `rollback`: Rollback to previous version
- `search`: Search standards content

**Example:**
```bash
# List all standards
mcp-standards standards list

# Import new standards
mcp-standards standards import --source williamzujkowski/standards

# Search for specific content
mcp-standards standards search "authentication requirements"

# Rollback to previous version
mcp-standards standards rollback --version 1.2.0
```

### 🗄️ `cache` - Cache Management

Manage the three-tier cache system.

```bash
mcp-standards cache [ACTION] [OPTIONS]
```

**Arguments:**
- `ACTION`: Action to perform: status, clear, optimize

**Options:**
- `--tier`: Specific tier: redis, faiss, chromadb, all
- `--force`: Force operation without confirmation

**Example:**
```bash
# Check cache status
mcp-standards cache status

# Clear all caches
mcp-standards cache clear --tier all

# Optimize FAISS hot cache
mcp-standards cache optimize --tier faiss
```

### 🖥️ `server` - Start MCP Server

Start the MCP server for Claude integration.

```bash
mcp-standards server [OPTIONS]
```

**Options:**
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--reload`: Auto-reload on changes
- `--log-level`: Logging level: DEBUG, INFO, WARNING, ERROR

**Example:**
```bash
# Start with defaults
mcp-standards server

# Start with debug logging
mcp-standards server --log-level DEBUG

# Start with auto-reload for development
mcp-standards server --reload
```

### ℹ️ `version` - Show Version

Display version information.

```bash
mcp-standards version
```

## MCP Tools Reference

When using Claude with the MCP server, these tools are available:

### 🔍 `load_standards`

Load relevant standards based on query.

**Parameters:**
```json
{
  "query": "string",        // Natural language or notation query
  "context": "string",      // Optional context for better results
  "token_limit": 10000,     // Maximum tokens to return
  "filters": {              // Optional filters
    "language": "python",
    "framework": "django",
    "nist_family": "AC"
  }
}
```

### 💻 `analyze_code`

Analyze code for NIST control implementations.

**Parameters:**
```json
{
  "code": "string",         // Code to analyze
  "language": "string",     // Programming language
  "filename": "string",     // Optional filename
  "context": {              // Optional context
    "framework": "string",
    "purpose": "string"
  }
}
```

### 🎯 `suggest_controls`

Get NIST control recommendations.

**Parameters:**
```json
{
  "description": "string",      // System/feature description
  "components": ["string"],     // System components
  "security_level": "string",   // low, moderate, high
  "existing_controls": ["string"] // Already implemented controls
}
```

### 📝 `generate_template`

Generate secure code templates.

**Parameters:**
```json
{
  "template_type": "string",    // api, auth, logging, etc.
  "language": "string",         // Target language
  "controls": ["string"],       // NIST controls to implement
  "options": {                  // Template options
    "framework": "string",
    "async": boolean,
    "testing": boolean
  }
}
```

### ✅ `validate_compliance`

Validate against NIST requirements.

**Parameters:**
```json
{
  "path": "string",             // Path to validate
  "profile": "string",          // NIST profile
  "controls": ["string"],       // Specific controls
  "recursive": true             // Scan recursively
}
```

### 🤖 `scan_with_llm`

Enhanced scanning with LLM analysis.

**Parameters:**
```json
{
  "path": "string",             // Path to scan
  "focus_areas": ["string"],    // Areas to focus on
  "output_format": "string",    // detailed, summary
  "include_suggestions": true   // Include remediation suggestions
}
```

### 🔎 `semantic_search`

Search standards using natural language.

**Parameters:**
```json
{
  "query": "string",            // Search query
  "limit": 10,                  // Maximum results
  "threshold": 0.7              // Similarity threshold
}
```

### 📊 `cache_stats`

Get cache performance statistics.

**Parameters:**
```json
{
  "tier": "string",             // all, redis, faiss, chromadb
  "include_metrics": true       // Include detailed metrics
}
```

## Advanced Usage

### Using with CI/CD

**GitHub Actions Example:**
```yaml
name: NIST Compliance Check
on: [push, pull_request]

jobs:
  compliance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install MCP Standards
        run: pip install mcp-standards-server
      
      - name: Run Compliance Scan
        run: |
          mcp-standards scan --output-format json --save-report compliance.json
          
      - name: Validate Critical Controls
        run: |
          mcp-standards validate --controls "AC-3,AU-2,IA-2" --strict
          
      - name: Generate Coverage Report
        run: |
          mcp-standards coverage --output-format html --output-file coverage.html
          
      - name: Upload Reports
        uses: actions/upload-artifact@v3
        with:
          name: compliance-reports
          path: |
            compliance.json
            coverage.html
```

### Pre-commit Hook Integration

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: local
    hooks:
      - id: mcp-standards-validate
        name: NIST Compliance Validation
        entry: mcp-standards validate
        language: system
        pass_filenames: false
        always_run: true
        
      - id: mcp-standards-scan
        name: Security Control Scan
        entry: mcp-standards scan --output-format json
        language: system
        pass_filenames: false
        files: \.(py|js|go|java)$
```

### Programmatic Usage

```python
from mcp_standards_server import ComplianceScanner, StandardsEngine

async def analyze_project():
    # Initialize scanner
    scanner = ComplianceScanner()
    
    # Scan project
    results = await scanner.scan_directory("./src")
    
    # Get coverage metrics
    coverage = scanner.get_coverage_report()
    
    # Generate SSP
    ssp = scanner.generate_ssp(
        profile="moderate",
        metadata={"system_name": "My App"}
    )
    
    return results, coverage, ssp
```

## Troubleshooting

### Common Issues

**1. MCP Server Not Connecting**
```bash
# Check if server is running
mcp-standards server --log-level DEBUG

# Verify configuration
cat ~/.config/claude/claude_desktop_config.json

# Test connection
curl http://localhost:8000/health
```

**2. Standards Not Loading**
```bash
# Verify standards path
ls -la $STANDARDS_PATH

# Re-import standards
mcp-standards standards import --source williamzujkowski/standards

# Clear cache
mcp-standards cache clear --tier all
```

**3. Three-Tier Cache Issues**
```bash
# Check all tiers status
mcp-standards cache status

# Test Redis (required tier)
redis-cli ping  # Must return: PONG

# Fix Redis if not running
# macOS: brew services start redis
# Ubuntu: sudo systemctl start redis-server

# Test vector stores (auto-installed as core dependencies)
python -c "import faiss; import chromadb; print('Vector stores OK')"

# Clear corrupted caches
mcp-standards cache clear --tier all
```

**4. Import Errors**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
uv pip install -e .

# Check Python path
python -c "import src.server; print('OK')"
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
# Set environment variable
export LOG_LEVEL=DEBUG

# Or use command line
mcp-standards server --log-level DEBUG

# Enable MCP protocol debugging
export MCP_DEBUG=1
```

### Getting Help

1. **Check logs:**
   ```bash
   tail -f ~/.mcp-standards/logs/server.log
   ```

2. **Run diagnostics:**
   ```bash
   mcp-standards cache status
   mcp-standards standards list
   mcp-standards version
   ```

3. **Report issues:**
   - GitHub Issues: https://github.com/williamzujkowski/mcp-standards-server/issues
   - Include: Version, OS, Python version, error logs

## Best Practices

1. **Regular Updates:**
   ```bash
   # Update standards monthly
   mcp-standards standards update
   
   # Update the package
   pip install --upgrade mcp-standards-server
   ```

2. **Cache Management:**
   ```bash
   # Weekly optimization
   mcp-standards cache optimize --tier all
   
   # Monitor cache performance
   mcp-standards cache status
   ```

3. **Profile Selection:**
   - `low`: Basic web applications, non-sensitive data
   - `moderate`: Business applications, PII handling
   - `high`: Financial, healthcare, government systems

4. **CI/CD Integration:**
   - Run `validate` on every commit
   - Run `coverage` weekly
   - Generate `ssp` for releases

5. **Performance Tuning:**
   ```yaml
   # Optimize for large codebases
   mcp:
     vector_store:
       faiss_size: 2000      # Increase hot cache
       chromadb_batch: 500   # Larger batches
     cache_ttl: 7200         # Longer cache life
   ```