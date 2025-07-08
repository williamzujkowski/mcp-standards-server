# Getting Started with MCP Standards Server

This guide will help you get up and running with the MCP Standards Server in just a few minutes.

## Prerequisites

- Python 3.8 or higher
- pip or pipx package manager
- Git (for development installation)
- Internet connection for syncing standards

## Installation

### Quick Install (Recommended)

Using pipx for isolated installation:

```bash
# Install pipx if you don't have it
python -m pip install --user pipx
python -m pipx ensurepath

# Install MCP Standards Server
pipx install mcp-standards-server
```

### Standard Install

Using pip:

```bash
pip install mcp-standards-server
```

### Development Install

From source:

```bash
git clone https://github.com/williamzujkowski/mcp-standards-server
cd mcp-standards-server
pip install -e .
```

### Verify Installation

```bash
mcp-standards --version
```

Expected output:
```
MCP Standards Server v1.0.0
```

## Initial Setup

### 1. Initialize Configuration

Run the interactive setup wizard:

```bash
mcp-standards config --init
```

You'll be prompted for:
- Repository details (default: williamzujkowski/standards)
- Cache location (default: ~/.cache/mcp-standards)
- GitHub authentication (optional but recommended)
- Search features (recommended: yes)

Example session:
```
Creating configuration file: /home/user/.config/mcp-standards/config.yaml

? Repository owner: williamzujkowski
? Repository name: standards
? Branch: main
? Path in repository: standards
? Cache directory: (~/.cache/mcp-standards) 
? Configure GitHub authentication? Yes
? Authentication type: token
? GitHub personal access token: **********************
? Enable semantic search? Yes

Configuration file created successfully!
```

### 2. Sync Standards

Download standards from the repository:

```bash
mcp-standards sync
```

Expected output:
```
Starting standards synchronization...
Fetching file list from williamzujkowski/standards...
Files to sync: 45
Downloading: web-development-standards.yaml... [OK]
Downloading: api-design-standards.yaml... [OK]
...
Sync completed with status: success
Duration: 15.23 seconds
Files synced: 45/45
```

### 3. Verify Setup

Check that everything is working:

```bash
mcp-standards status
```

Expected output:
```
MCP Standards Server - Sync Status

Total files cached: 45
Total cache size: 3.42 MB

GitHub API Rate Limit:
  Remaining: 4998/5000
  Resets at: 2025-07-08 15:30:00

Repository: williamzujkowski/standards
Branch: main
Path: standards
```

## Your First Query

### Find Applicable Standards

Let's find standards for a React project:

```bash
mcp-standards query --project-type web-application --framework react
```

Output:
```
Applicable Standards Found: 7

1. React 18 Patterns (react-18-patterns.yaml)
   Tags: frontend, react, javascript, components
   Priority: HIGH
   Summary: Modern React patterns including hooks, Server Components

2. JavaScript ES2025 Standards (javascript-es2025.yaml)
   Tags: javascript, ecmascript, language
   Priority: HIGH
   Summary: Modern JavaScript language features and best practices

3. Web Accessibility Standards (wcag-2.2-accessibility.yaml)
   Tags: accessibility, a11y, web
   Priority: MEDIUM
   Summary: WCAG 2.2 compliance guidelines and ARIA patterns

Use --detailed to see full content
```

### Get Detailed Standards

View complete standard content:

```bash
mcp-standards query --project-type web-application --framework react --detailed | less
```

### Export Standards

Save standards for your project:

```bash
mcp-standards query --project-type api --language python --format markdown > project-standards.md
```

## Basic Validation

### Validate a Single File

```bash
# Create a sample file
cat > button.jsx << 'EOF'
const Button = ({onClick}) => {
  return <button onClick={onClick}>Click</button>
}
export default Button
EOF

# Validate it
mcp-standards validate button.jsx
```

Output:
```
Results:
========

button.jsx
  Line 2: ERROR - Missing accessible label
    Standard: wcag-2.2-accessibility
    Rule: interactive-elements-labels
    
    Fix: Add aria-label or visible text content

Summary:
  Files scanned: 1
  Issues found: 1
    Errors: 1
```

### Auto-Fix Issues

```bash
mcp-standards validate --fix button.jsx
```

## Start the MCP Server

### Basic Server

Start the server for tool integration:

```bash
mcp-standards serve
```

Output:
```
Starting MCP Standards Server v1.0.0
✓ Loaded 45 standards files
✓ Server listening on http://localhost:3000

Available MCP tools:
  - get_applicable_standards
  - validate_code
  - search_standards
  - get_standard_content
```

### Test the Server

In another terminal:

```bash
# Check health
curl http://localhost:3000/health

# Test MCP tool
curl -X POST http://localhost:3000/tools/search_standards \
  -H "Content-Type: application/json" \
  -d '{"query": "How to implement authentication?"}'
```

## IDE Integration

### VS Code

1. Install the MCP Standards extension (when available)
2. Or add to settings.json:

```json
{
  "mcp-standards.server.url": "http://localhost:3000",
  "mcp-standards.validation.onSave": true
}
```

### Command Line Integration

Add to your shell profile:

```bash
# ~/.bashrc or ~/.zshrc

# Alias for quick queries
alias mcp-query='mcp-standards query --format json | jq'

# Function to validate current directory
mcp-validate() {
  mcp-standards validate ${1:-.} --fail-on error
}

# Auto-sync on shell start (optional)
if command -v mcp-standards &> /dev/null; then
  mcp-standards sync --check > /dev/null 2>&1 || echo "Standards outdated. Run 'mcp-standards sync'"
fi
```

## Next Steps

### 1. Explore Commands

Learn about all available commands:

```bash
mcp-standards --help
mcp-standards query --help
mcp-standards validate --help
```

### 2. Configure for Your Project

Create a project-specific configuration:

```bash
# In your project root
cat > .mcp-standards.yaml << 'EOF'
# Project-specific standards configuration
validation:
  include_patterns:
    - "src/**/*.js"
    - "src/**/*.jsx"
  rules:
    overrides:
      max-line-length:
        options:
          limit: 120
EOF
```

### 3. Set Up CI/CD

Add to your GitHub Actions:

```yaml
# .github/workflows/standards.yml
name: Standards Check
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install MCP Standards
        run: pip install mcp-standards-server
      
      - name: Sync Standards
        run: mcp-standards sync
      
      - name: Validate Code
        run: mcp-standards validate . --fail-on error
```

### 4. Explore Advanced Features

- [Token Optimization](../guides/token-optimization.md)
- [Custom Validators](../guides/custom-validators.md)
- [MCP Tool Development](../guides/mcp-tools.md)

## Common Tasks

### Update Standards

```bash
# Check for updates
mcp-standards sync --check

# Update if needed
mcp-standards sync
```

### Clean Cache

```bash
# Remove outdated files
mcp-standards cache --clear-outdated

# Clear everything
mcp-standards cache --clear
```

### Debug Issues

```bash
# Verbose output
mcp-standards -v sync

# Check configuration
mcp-standards config --validate
```

## Getting Help

- Run `mcp-standards <command> --help` for command help
- Check [Troubleshooting Guide](../troubleshooting.md) for common issues
- Visit [GitHub Issues](https://github.com/williamzujkowski/mcp-standards-server/issues) for support

## Summary

You've now:
- ✅ Installed MCP Standards Server
- ✅ Configured repository access
- ✅ Synced standards files
- ✅ Performed your first query
- ✅ Validated code against standards
- ✅ Started the MCP server

Continue exploring the documentation to learn about advanced features and integrations!