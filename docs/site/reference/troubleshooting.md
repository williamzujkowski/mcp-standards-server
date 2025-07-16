# Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### Python Version Compatibility

**Problem:** Installation fails with Python version 1.0.0

**Solution:**
```bash
# Check Python version
python --version

# MCP Standards Server requires Python 3.11+
# Install compatible Python version
pyenv install 3.11.0
pyenv global 3.11.0
```

### Dependencies Installation Failed

**Problem:** `pip install` fails with dependency conflicts.

**Solution:**
```bash
# Clean install with virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install mcp-standards-server
```

## Server Issues

### Port Already in Use

**Problem:** `[Errno 48] Address already in use`

**Solution:**
```bash
# Find process using port 8080
lsof -i :8080

# Kill the process
kill -9 <PID>

# Or use different port
mcp-standards serve --port 8081
```

### Permission Denied

**Problem:** `Permission denied` when accessing cache directory.

**Solution:**
```bash
# Fix permissions
sudo chown -R $USER ~/.mcp-standards/
chmod 755 ~/.mcp-standards/cache

# Or use custom cache directory
mcp-standards serve --cache-dir ./cache
```

### Memory Issues

**Problem:** Server runs out of memory.

**Solution:**
```bash
# Limit memory usage
export MCP_MAX_MEMORY=1024  # MB
mcp-standards serve

# Or clear cache
mcp-standards cache clear
```

## Configuration Issues

### Invalid YAML Syntax

**Problem:** Configuration file parsing error.

**Solution:**
```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('~/.mcp-standards/config.yaml'))"

# Reset to default configuration
mcp-standards config init --force
```

### Redis Connection Failed

**Problem:** Cannot connect to Redis server.

**Solution:**
```bash
# Check Redis status
redis-cli ping

# Start Redis (macOS with Homebrew)
brew services start redis

# Start Redis (Linux with systemd)
sudo systemctl start redis

# Disable Redis in config if not needed
mcp-standards config set standards.redis.enabled false
```

## Standards Sync Issues

### Network Connection Failed

**Problem:** Cannot download standards from repository.

**Solution:**
```bash
# Check network connectivity
curl -I https://github.com/williamzujkowski/standards

# Use proxy if needed
export HTTPS_PROXY=http://proxy.company.com:8080
mcp-standards sync

# Use local standards directory
mcp-standards config set standards.local_directory /path/to/local/standards
```

### Git Authentication Failed

**Problem:** Cannot access private standards repository.

**Solution:**
```bash
# Set up SSH key
ssh-keygen -t ed25519 -C "your.email@example.com"
cat ~/.ssh/id_ed25519.pub  # Add to GitHub

# Or use personal access token
export GITHUB_TOKEN=your_token_here
mcp-standards sync
```

### Standards Format Error

**Problem:** Invalid standards file format.

**Solution:**
```bash
# Validate standards format
mcp-standards validate-standards ./standards/

# Force resync to get latest format
mcp-standards sync --force
```

## Validation Issues

### No Standards Found

**Problem:** `No applicable standards found for project`.

**Solution:**
```bash
# Check available standards
mcp-standards query list

# Sync standards first
mcp-standards sync

# Manually specify standards
mcp-standards validate --standard python-pep8 .
```

### Language Detection Failed

**Problem:** Cannot detect project language automatically.

**Solution:**
```bash
# Manually specify language
mcp-standards validate --language python .

# Check project structure
find . -name "*.py" -o -name "*.js" -o -name "*.go" | head -10
```

### Performance Issues

**Problem:** Validation is very slow.

**Solution:**
```bash
# Use parallel processing
mcp-standards validate --workers 4 .

# Exclude large directories
echo "node_modules/" >> .mcpignore
echo "*.min.js" >> .mcpignore

# Use incremental validation
mcp-standards validate --incremental .
```

## API and Integration Issues

### MCP Protocol Errors

**Problem:** MCP client cannot connect to server.

**Solution:**
```bash
# Check server status
curl http://localhost:8080/health

# Enable CORS for web clients
mcp-standards config set server.enable_cors true

# Check server logs
tail -f ~/.mcp-standards/mcp-server.log
```

### IDE Integration Failed

**Problem:** VS Code extension not working.

**Solution:**
1. Check extension is installed and enabled
2. Verify server is running: `mcp-standards status`
3. Check extension logs in VS Code Developer Tools
4. Restart VS Code and server

### CI/CD Integration Issues

**Problem:** GitHub Actions workflow fails.

**Solution:**
```yaml
# Add to GitHub Actions workflow
- name: Setup MCP Standards
  run: |
    pip install mcp-standards-server
    mcp-standards sync
    
- name: Validate Code
  run: |
    mcp-standards validate --format sarif --output results.sarif
  continue-on-error: true
```

## Logging and Debugging

### Enable Debug Logging

```bash
# Temporary debug mode
mcp-standards --verbose serve

# Persistent debug configuration
mcp-standards config set logging.level DEBUG
```

### Log File Locations

- **Server logs:** `~/.mcp-standards/mcp-server.log`
- **Validation logs:** `~/.mcp-standards/validation.log`
- **Sync logs:** `~/.mcp-standards/sync.log`

### Collect Diagnostic Information

```bash
# Generate diagnostic report
mcp-standards diagnostic > diagnostic-report.txt

# Include in bug reports:
# - OS and Python version
# - Configuration file
# - Recent log entries
# - Steps to reproduce
```

## Performance Optimization

### Cache Optimization

```bash
# Warm cache for common standards
mcp-standards cache warm

# Monitor cache hit rate
mcp-standards cache stats

# Increase cache size
mcp-standards config set standards.max_cache_size 2GB
```

### Memory Optimization

```bash
# Limit concurrent validations
mcp-standards config set performance.max_workers 2

# Enable garbage collection tuning
mcp-standards config set performance.gc_threshold 0.8
```

## Getting Help

### Community Support

- 💬 [Discord Community](https://discord.gg/mcp-standards)
- 🐛 [GitHub Issues](https://github.com/williamzujkowski/mcp-standards-server/issues)
- 📚 [Documentation](https://mcp-standards-server.readthedocs.io/)

### Bug Reports

When reporting bugs, please include:

1. MCP Standards Server version: `mcp-standards --version`
2. Operating system and Python version
3. Complete error message
4. Steps to reproduce
5. Configuration file (remove sensitive data)
6. Relevant log entries

### Feature Requests

Feature requests are welcome! Please:

1. Check existing issues first
2. Describe the use case
3. Provide implementation ideas if possible
4. Consider contributing the feature

---

Still having issues? Join our [Discord community](https://discord.gg/mcp-standards) for real-time help!
