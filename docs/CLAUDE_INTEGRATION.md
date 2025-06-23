# Using MCP Standards Server with Claude CLI

This guide provides detailed instructions for integrating the MCP Standards Server with Claude CLI to enable NIST compliance checking and secure code generation directly within your Claude conversations.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Starting the Server](#starting-the-server)
4. [Using MCP Tools in Claude](#using-mcp-tools-in-claude)
5. [Example Conversations](#example-conversations)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites
- Python 3.11 or higher
- Claude Desktop application
- **Redis** (required for three-tier hybrid caching)
- **FAISS & ChromaDB** (auto-installed with MCP Standards Server)

### Install Dependencies

#### 1. Install Redis (Required)
**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

**Windows (WSL):**
```bash
sudo apt install redis-server
sudo service redis-server start
```

#### 2. Install MCP Standards Server

**Option 1: From PyPI (Recommended)**
```bash
# This automatically installs FAISS and ChromaDB
pip install mcp-standards-server
```

**Option 2: From Source**
```bash
git clone https://github.com/williamzujkowski/mcp-standards-server.git
cd mcp-standards-server
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```

### Verify Installation
```bash
mcp-standards version
# Should output: MCP Standards Server Version: 0.1.0
```

## Configuration

### 1. Locate Claude Configuration File

The configuration file location depends on your operating system:

- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Linux**: `~/.config/claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

### 2. Basic Configuration

Edit the configuration file to add the MCP Standards Server:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"]
    }
  }
}
```

### 3. Advanced Configuration

For more control, you can specify environment variables and working directory:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server", "--log-level", "INFO"],
      "env": {
        "STANDARDS_PATH": "/path/to/your/standards",
        "REDIS_URL": "redis://localhost:6379",
        "MAX_TOKENS": "10000",
        "CACHE_TTL": "3600",
        "VECTOR_STORE_TYPE": "hybrid"
      },
      "cwd": "/path/to/your/project"
    }
  }
}
```

### 4. Using Virtual Environment

If you installed in a virtual environment, use the full path to the Python executable:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "/path/to/venv/bin/python",
      "args": ["-m", "src.server"],
      "cwd": "/path/to/mcp-standards-server",
      "env": {
        "PYTHONPATH": "/path/to/mcp-standards-server"
      }
    }
  }
}
```

### 5. Multiple Servers

You can configure multiple MCP servers:

```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"]
    },
    "other-server": {
      "command": "other-mcp-server",
      "args": ["--option", "value"]
    }
  }
}
```

## Starting the Server

### Option 1: Let Claude Start It (Recommended)

Simply launch Claude Desktop. It will automatically start all configured MCP servers.

### Option 2: Manual Start (For Testing)

```bash
# Basic start
mcp-standards server

# With debug logging
mcp-standards server --log-level DEBUG

# Custom host/port
mcp-standards server --host 0.0.0.0 --port 8080
```

### Verify Connection

In Claude, ask:
```
What MCP tools are available?
```

You should see the MCP Standards tools listed.

## Using MCP Tools in Claude

### Available Tools

Once connected, you can use these tools in your Claude conversations:

1. **load_standards** - Load relevant standards based on queries
2. **analyze_code** - Analyze code for NIST compliance
3. **suggest_controls** - Get control recommendations
4. **generate_template** - Generate secure code templates
5. **validate_compliance** - Validate against requirements
6. **scan_with_llm** - Enhanced scanning with AI analysis
7. **semantic_search** - Natural language standards search
8. **cache_stats** - View cache performance

### Tool Usage Examples

#### Loading Standards
```
Can you load standards about secure API design?
```

#### Analyzing Code
```
Please analyze this Python code for NIST compliance:
[paste your code]
```

#### Generating Templates
```
Generate a secure authentication module for Python with NIST controls AC-3 and IA-2
```

#### Validating Compliance
```
Validate my project at /home/user/myproject against the moderate NIST profile
```

## Example Conversations

### Example 1: Project Initialization
```
User: I'm starting a new Python web application. Can you help me set up NIST compliance?

Claude: I'll help you set up NIST compliance for your Python web application. Let me first load relevant standards and then guide you through the setup.

[Claude uses load_standards tool]
[Claude uses generate_template tool]

Based on the standards, here's what I recommend:
1. Initialize your project with moderate security profile
2. Here's a secure API template with authentication...
[Generated code with NIST annotations]
```

### Example 2: Code Review
```
User: Can you review this authentication code for security compliance?

Claude: I'll analyze your authentication code for NIST compliance.

[Claude uses analyze_code tool]

Analysis Results:
- Detected Controls: IA-2, IA-5
- Missing Controls: IA-2(1) for multi-factor authentication
- Suggestions: Add MFA support for high-value operations...
```

### Example 3: Compliance Report
```
User: Generate a compliance report for my entire project

Claude: I'll scan your project and generate a comprehensive compliance report.

[Claude uses scan_with_llm tool]
[Claude uses semantic_search for additional context]

Compliance Report Summary:
- Total Controls Implemented: 45
- Coverage: 78%
- High-Priority Gaps: AC-6 (Least Privilege), AU-12 (Audit Generation)
...
```

## Troubleshooting

### Server Not Connecting

1. **Check Claude logs**:
   - Help → Toggle Developer Tools → Console

2. **Verify server is running**:
   ```bash
   ps aux | grep mcp-standards
   ```

3. **Test manual connection**:
   ```bash
   mcp-standards server --log-level DEBUG
   ```

### Tools Not Available

1. **Restart Claude Desktop**
   - Fully quit and restart the application

2. **Check configuration syntax**:
   ```bash
   # Validate JSON
   python -m json.tool < ~/.config/claude/claude_desktop_config.json
   ```

3. **Verify PATH**:
   ```bash
   which mcp-standards
   ```

### Performance Issues

1. **Verify three-tier system is working**:
   ```bash
   # Check all tiers status
   mcp-standards cache status
   
   # Should show:
   # - Redis Query Cache: Connected
   # - FAISS Hot Cache: Active with utilization %
   # - ChromaDB Persistent: Status with document count
   ```

2. **Fix Redis connection issues**:
   ```bash
   # Test Redis connection
   redis-cli ping  # Should return: PONG
   
   # Start Redis if stopped
   # macOS: brew services start redis
   # Ubuntu: sudo systemctl start redis-server
   
   # Configure Redis URL if needed
   export REDIS_URL="redis://localhost:6379"
   ```

3. **Optimize the three-tier cache**:
   ```bash
   # Run tier optimization
   mcp-standards cache optimize
   
   # Clear caches if corrupted
   mcp-standards cache clear --tier all
   ```

4. **Check FAISS/ChromaDB installation**:
   ```bash
   # Test if properly installed
   python -c "import faiss; import chromadb; print('Vector stores available')"
   ```

### Debug Mode

For detailed debugging:

1. **Enable debug logging in config**:
   ```json
   {
     "mcpServers": {
       "mcp-standards": {
         "command": "mcp-standards",
         "args": ["server", "--log-level", "DEBUG"],
         "env": {
           "MCP_DEBUG": "1"
         }
       }
     }
   }
   ```

2. **Check server logs**:
   ```bash
   tail -f ~/.mcp-standards/logs/server.log
   ```

## Best Practices

1. **Initialize Projects First**:
   ```bash
   mcp-standards init --profile moderate
   ```

2. **Ensure Three-Tier Architecture is Active**:
   - **Redis must be running**: Required for query caching tier
   - **FAISS hot cache**: Automatically manages top 1000 standards
   - **ChromaDB persistence**: Handles full corpus with metadata
   - Check status: `mcp-standards cache status`

3. **Regular Updates**:
   ```bash
   pip install --upgrade mcp-standards-server
   mcp-standards standards update
   ```

4. **Configure Project-Specific Settings**:
   Create `.mcp-standards/config.yaml` in your project root

5. **Use Profiles Appropriately**:
   - `low`: Development and testing
   - `moderate`: Production applications
   - `high`: Financial, healthcare, government

## Advanced Features

### Custom Standards Path
```json
{
  "mcpServers": {
    "mcp-standards": {
      "command": "mcp-standards",
      "args": ["server"],
      "env": {
        "STANDARDS_PATH": "/opt/custom-standards"
      }
    }
  }
}
```

### Project-Specific Servers
```json
{
  "mcpServers": {
    "project-a": {
      "command": "mcp-standards",
      "args": ["server"],
      "cwd": "/home/user/project-a",
      "env": {
        "STANDARDS_PATH": "/home/user/project-a/standards"
      }
    },
    "project-b": {
      "command": "mcp-standards",
      "args": ["server"],
      "cwd": "/home/user/project-b",
      "env": {
        "STANDARDS_PATH": "/home/user/project-b/standards"
      }
    }
  }
}
```

### Integration with VS Code
If you're using VS Code, you can also configure the MCP server there:

1. Install the Claude extension for VS Code
2. Configure in `.vscode/settings.json`:
   ```json
   {
     "claude.mcpServers": {
       "mcp-standards": {
         "command": "mcp-standards",
         "args": ["server"]
       }
     }
   }
   ```

## Getting Help

- **Documentation**: See `docs/USAGE_GUIDE.md` for complete command reference
- **Quick Reference**: See `docs/QUICK_REFERENCE.md` for command cheat sheet
- **Issues**: Report at https://github.com/williamzujkowski/mcp-standards-server/issues
- **Community**: Join discussions on GitHub Discussions

## Next Steps

1. Initialize your first project: `mcp-standards init`
2. Scan existing code: `mcp-standards scan`
3. Generate secure templates: `mcp-standards generate api`
4. Create compliance reports: `mcp-standards coverage`

With the MCP Standards Server integrated into Claude, you can seamlessly incorporate NIST compliance checking and secure code generation into your development workflow!