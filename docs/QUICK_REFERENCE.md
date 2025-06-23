# MCP Standards Server - Quick Reference Card

**Version:** 1.0.0  
**Last Updated:** 2025-06-23  
**Status:** Active  
**Standard Code:** QRF  

**Summary:** Command cheat sheet for quick lookup  
**Tokens:** ~1400 (helps AI plan context usage)  
**Priority:** high

## 🚀 Quick Setup with Claude CLI

```bash
# 1. Install Redis (Required)
# macOS: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo systemctl start redis-server

# 2. Install MCP Standards Server (includes FAISS & ChromaDB)
pip install mcp-standards-server

# 3. Verify installation
redis-cli ping  # Should return: PONG
mcp-standards cache status  # Should show all tiers

# 4. Configure Claude Desktop
# Edit: ~/.config/claude/claude_desktop_config.json (macOS/Linux)
#   or: %APPDATA%\Claude\claude_desktop_config.json (Windows)

{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"]
    }
  }
}

# 3. Start Claude Desktop - it will auto-connect to the MCP server
```

## 📋 Essential Commands

### Initialize Project
```bash
mcp-standards init                              # Initialize with defaults (moderate profile)
mcp-standards init --profile high --setup-hooks # High security with Git hooks
```

### Scan for Compliance
```bash
mcp-standards scan                              # Scan current directory
mcp-standards scan --output-format json         # JSON output
mcp-standards scan --save-report report.json    # Save to file
```

### Generate System Security Plan
```bash
mcp-standards ssp                               # Generate SSP (default: ssp.json)
mcp-standards ssp --profile high                # High security profile
```

### Validate Specific Controls
```bash
mcp-standards validate                          # Validate entire project
mcp-standards validate --controls "AC-3,AU-2"   # Check specific controls
```

### Generate Coverage Report
```bash
mcp-standards coverage                          # Markdown report to console
mcp-standards coverage --output-format html --output-file coverage.html
```

### Generate Secure Code Templates
```bash
mcp-standards generate api --language python    # Generate API template
mcp-standards generate auth --controls "IA-2"  # Auth with specific controls
```

### Manage Standards
```bash
mcp-standards standards list                    # List imported standards
mcp-standards standards search "encryption"     # Search standards content
```

### Cache Management
```bash
mcp-standards cache status                      # View cache statistics
mcp-standards cache clear --tier all            # Clear all caches
mcp-standards cache optimize                    # Optimize tier placement
```

### Start MCP Server
```bash
mcp-standards server                            # Start on localhost:8000
mcp-standards server --log-level DEBUG          # Debug mode
```

## 🛠️ MCP Tools for Claude

When using Claude with the MCP server connected, you can ask:

- "Load standards about API security"
- "Analyze this code for NIST compliance"
- "Generate a secure authentication module"
- "What controls are missing for high security?"
- "Create a coverage report for my project"

## 🔍 Common Workflows

### First Time Setup
```bash
# 1. Initialize project
mcp-standards init --profile moderate

# 2. Scan existing code
mcp-standards scan

# 3. Generate coverage report
mcp-standards coverage --output-file initial-coverage.md

# 4. Generate SSP
mcp-standards ssp
```

### CI/CD Integration
```yaml
# .github/workflows/compliance.yml
- name: Run Compliance Check
  run: |
    mcp-standards scan --output-format json --save-report compliance.json
    mcp-standards validate --controls "AC-3,AU-2,IA-2" --strict
```

### Pre-commit Hook
```bash
# Automatically installed with: mcp-standards init --setup-hooks
# Or manually add to .git/hooks/pre-commit:
#!/bin/bash
mcp-standards validate --output-format json
```

## 📊 Output Formats

- **table** - Human-readable tables (default)
- **json** - Machine-readable JSON
- **yaml** - YAML format
- **oscal** - OSCAL-compliant (SSP only)
- **markdown** - Markdown (coverage reports)
- **html** - HTML (coverage reports)

## 🎯 NIST Profiles

- **low** - Basic security controls
- **moderate** - Standard business applications (default)
- **high** - Critical systems, healthcare, finance

## 💡 Pro Tips

1. **Use JSON output for scripts**: 
   ```bash
   mcp-standards scan --output-format json | jq '.summary'
   ```

2. **Filter by language**:
   ```bash
   mcp-standards scan ./src --output-format json | jq '.files[] | select(.language == "python")'
   ```

3. **Check specific directories**:
   ```bash
   mcp-standards validate ./api --controls "AC-3,SC-8"
   ```

4. **Generate templates with multiple controls**:
   ```bash
   mcp-standards generate api --controls "AC-3,AU-2,IA-2,SC-8"
   ```

5. **Monitor cache performance**:
   ```bash
   watch -n 5 'mcp-standards cache status'
   ```

## 🆘 Troubleshooting

```bash
# Check version
mcp-standards version

# Verify three-tier architecture
mcp-standards cache status
# Should show Redis, FAISS, and ChromaDB status

# Test Redis (required - core dependency)
redis-cli ping  # Must return: PONG

# Test vector stores (auto-installed as core dependencies)
python -c "import faiss; import chromadb; print('Vector stores OK')"

# Debug MCP connection
mcp-standards server --log-level DEBUG

# Clear corrupted caches
mcp-standards cache clear --tier all --force

# Verify standards are loaded
mcp-standards standards list
```

## 📚 More Information

- Full documentation: `docs/USAGE_GUIDE.md`
- Claude integration: Configure in `claude_desktop_config.json`
- GitHub: https://github.com/williamzujkowski/mcp-standards-server