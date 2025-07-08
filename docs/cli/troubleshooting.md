# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the MCP Standards Server CLI.

## Quick Diagnostics

Run the built-in diagnostic command to check for common issues:

```bash
mcp-standards diagnose
```

This will check:
- Configuration validity
- Network connectivity
- GitHub API access
- Cache permissions
- Dependencies
- System resources

## Common Issues

### Installation Issues

#### Command Not Found

**Problem**: `mcp-standards: command not found`

**Solutions**:

1. **Check installation**:
   ```bash
   pip show mcp-standards-server
   ```

2. **Verify PATH**:
   ```bash
   echo $PATH
   which mcp-standards
   ```

3. **Reinstall**:
   ```bash
   pip uninstall mcp-standards-server
   pip install --user mcp-standards-server
   # Add to PATH if needed
   export PATH="$HOME/.local/bin:$PATH"
   ```

4. **Use pipx** (recommended):
   ```bash
   pipx install mcp-standards-server
   ```

#### Permission Denied

**Problem**: `Permission denied` when installing

**Solution**:
```bash
# Install in user directory
pip install --user mcp-standards-server

# Or use virtual environment
python -m venv venv
source venv/bin/activate
pip install mcp-standards-server
```

### Sync Issues

#### GitHub API Rate Limit

**Problem**: `Error: API rate limit exceeded (60 requests per hour)`

**Solutions**:

1. **Configure authentication**:
   ```bash
   # Set GitHub token
   export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=ghp_xxxxxxxxxxxx
   mcp-standards sync
   ```

2. **Check rate limit status**:
   ```bash
   mcp-standards status --json | jq '.rate_limit'
   ```

3. **Wait for reset**:
   ```bash
   # Show when rate limit resets
   mcp-standards status | grep "Resets at"
   ```

#### Network Connection Errors

**Problem**: `Failed to connect to GitHub API`

**Diagnostics**:
```bash
# Test GitHub connectivity
curl -I https://api.github.com

# Check proxy settings
echo $HTTP_PROXY $HTTPS_PROXY

# Test with verbose output
mcp-standards -v sync
```

**Solutions**:

1. **Configure proxy**:
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   mcp-standards sync
   ```

2. **Use custom GitHub API URL**:
   ```bash
   export MCP_STANDARDS_GITHUB_API_URL=https://github.company.com/api/v3
   ```

#### Sync Hanging or Timing Out

**Problem**: Sync process hangs or times out

**Solutions**:

1. **Reduce parallel downloads**:
   ```bash
   mcp-standards sync --parallel 1
   ```

2. **Increase timeout**:
   ```bash
   mcp-standards sync --timeout 120
   ```

3. **Enable debug logging**:
   ```bash
   MCP_LOG_LEVEL=DEBUG mcp-standards -v sync
   ```

### Cache Issues

#### Cache Directory Not Writable

**Problem**: `Error: Cannot write to cache directory`

**Solutions**:

1. **Check permissions**:
   ```bash
   ls -la ~/.cache/mcp-standards
   # Fix permissions
   chmod 755 ~/.cache/mcp-standards
   ```

2. **Use different cache directory**:
   ```bash
   export MCP_STANDARDS_CACHE_DIRECTORY=/tmp/mcp-cache
   mcp-standards sync
   ```

3. **Clear corrupted cache**:
   ```bash
   rm -rf ~/.cache/mcp-standards
   mcp-standards sync --force
   ```

#### Cache Size Exceeded

**Problem**: `Warning: Cache size exceeds limit`

**Solutions**:

1. **Clear old files**:
   ```bash
   mcp-standards cache --clear-outdated
   ```

2. **Increase cache limit**:
   ```bash
   mcp-standards config --set cache.max_size_mb 1000
   ```

3. **Enable auto-cleanup**:
   ```yaml
   # .mcp-standards.yaml
   cache:
     auto_cleanup:
       enabled: true
       threshold_percent: 80
   ```

### Configuration Issues

#### Configuration Not Loading

**Problem**: Custom configuration ignored

**Diagnostics**:
```bash
# Check which config is loaded
mcp-standards config --which

# Debug config loading
MCP_DEBUG_CONFIG=1 mcp-standards config --show

# Validate config syntax
mcp-standards config --validate
```

**Solutions**:

1. **Check file location**:
   ```bash
   # Correct locations:
   ~/.config/mcp-standards/config.yaml  # User config
   ./.mcp-standards.yaml                # Project config
   ```

2. **Fix YAML syntax**:
   ```bash
   # Validate YAML
   yamllint .mcp-standards.yaml
   
   # Common issues:
   # - Tabs instead of spaces
   # - Missing colons
   # - Incorrect indentation
   ```

3. **Use explicit config**:
   ```bash
   mcp-standards -c /path/to/config.yaml sync
   ```

#### Environment Variables Not Working

**Problem**: Environment variables ignored

**Solutions**:

1. **Check variable naming**:
   ```bash
   # Correct format: MCP_STANDARDS_<SECTION>_<KEY>
   export MCP_STANDARDS_REPOSITORY_OWNER=williamzujkowski
   export MCP_STANDARDS_SYNC_CACHE_TTL_HOURS=48
   ```

2. **Verify export**:
   ```bash
   # Check if exported
   env | grep MCP_STANDARDS
   ```

3. **Debug environment loading**:
   ```bash
   MCP_DEBUG_ENV=1 mcp-standards config --show
   ```

### Validation Issues

#### False Positives

**Problem**: Validation reports errors incorrectly

**Solutions**:

1. **Update standards**:
   ```bash
   mcp-standards sync --force
   ```

2. **Adjust rule severity**:
   ```yaml
   # .mcp-validate.yaml
   rules:
     overrides:
       rule-name:
         severity: warning  # or 'off'
   ```

3. **Ignore specific files**:
   ```bash
   mcp-standards validate --ignore "*.generated.js"
   ```

#### Validation Not Finding Files

**Problem**: No files validated

**Solutions**:

1. **Check patterns**:
   ```bash
   # List files that would be validated
   mcp-standards validate --dry-run --verbose
   ```

2. **Adjust include patterns**:
   ```yaml
   validation:
     include_patterns:
       - "**/*.js"
       - "**/*.jsx"
       - "src/**/*.ts"
   ```

### Server Issues

#### Port Already in Use

**Problem**: `Error: Address already in use`

**Solutions**:

1. **Find process using port**:
   ```bash
   lsof -i :3000
   # or
   netstat -tulpn | grep 3000
   ```

2. **Kill existing process**:
   ```bash
   kill -9 <PID>
   ```

3. **Use different port**:
   ```bash
   mcp-standards serve --port 3001
   ```

#### Server Crashes

**Problem**: Server crashes or restarts frequently

**Diagnostics**:
```bash
# Check system resources
free -h
df -h
top

# Check logs
tail -f /var/log/mcp-standards.log

# Run in debug mode
mcp-standards serve --log-level debug
```

**Solutions**:

1. **Increase memory limit**:
   ```yaml
   server:
     memory:
       max_heap_mb: 2048
   ```

2. **Reduce workers**:
   ```bash
   mcp-standards serve --workers 2
   ```

3. **Enable crash dumps**:
   ```bash
   export MCP_ENABLE_CRASH_DUMPS=1
   mcp-standards serve
   ```

### Performance Issues

#### Slow Sync

**Problem**: Sync takes too long

**Solutions**:

1. **Increase parallel downloads**:
   ```bash
   mcp-standards sync --parallel 10
   ```

2. **Sync specific files**:
   ```bash
   mcp-standards sync --include "web-*.yaml"
   ```

3. **Use incremental sync**:
   ```bash
   # Only sync outdated files
   mcp-standards sync  # Default behavior
   ```

#### High Memory Usage

**Problem**: Excessive memory consumption

**Solutions**:

1. **Disable search indexing**:
   ```yaml
   search:
     enabled: false
   ```

2. **Reduce cache size**:
   ```bash
   mcp-standards cache --clear
   mcp-standards config --set cache.max_size_mb 100
   ```

3. **Use token optimization**:
   ```yaml
   token_optimization:
     compression:
       enabled: true
   ```

## Debug Mode

Enable comprehensive debugging:

```bash
# Maximum verbosity
export MCP_DEBUG=1
export MCP_LOG_LEVEL=DEBUG
export MCP_DEBUG_CONFIG=1
export MCP_DEBUG_ENV=1
export MCP_TRACE_REQUESTS=1

mcp-standards -v sync
```

## Log Analysis

### Log Locations

- **User logs**: `~/.cache/mcp-standards/logs/`
- **System logs**: `/var/log/mcp-standards.log`
- **Crash dumps**: `~/.cache/mcp-standards/crashes/`

### Analyzing Logs

```bash
# Search for errors
grep ERROR ~/.cache/mcp-standards/logs/mcp-standards.log

# Recent errors with context
grep -B5 -A5 ERROR ~/.cache/mcp-standards/logs/mcp-standards.log | tail -50

# Parse JSON logs
cat /var/log/mcp-standards.log | jq 'select(.level == "ERROR")'
```

## Getting Help

### Generate Diagnostic Report

```bash
mcp-standards diagnose --report > diagnostic-report.txt
```

This report includes:
- System information
- Configuration (sanitized)
- Recent errors
- Performance metrics
- Dependency versions

### Community Support

1. **GitHub Issues**: https://github.com/williamzujkowski/mcp-standards-server/issues
2. **Discord**: Join the MCP community Discord
3. **Stack Overflow**: Tag questions with `mcp-standards`

### Enterprise Support

For enterprise support options:
```bash
mcp-standards support --info
```

## Recovery Procedures

### Reset to Clean State

```bash
#!/bin/bash
# reset-mcp-standards.sh

echo "Backing up configuration..."
cp ~/.config/mcp-standards/config.yaml ~/mcp-standards-config.backup.yaml

echo "Clearing cache..."
rm -rf ~/.cache/mcp-standards

echo "Removing user configuration..."
rm -rf ~/.config/mcp-standards

echo "Reinstalling..."
pip uninstall -y mcp-standards-server
pip install mcp-standards-server

echo "Initializing fresh configuration..."
mcp-standards config --init

echo "Reset complete. Restore config from ~/mcp-standards-config.backup.yaml if needed."
```

### Emergency Mode

If the server won't start normally:

```bash
# Start in safe mode
mcp-standards serve --safe-mode

# This disables:
# - Plugins
# - Search indexing  
# - Auto-sync
# - Custom validators
```

## Monitoring and Alerts

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

# Check if server is responsive
if ! curl -f http://localhost:3000/health > /dev/null 2>&1; then
    echo "Server not responding"
    # Send alert or restart
    systemctl restart mcp-standards
fi

# Check cache size
CACHE_SIZE=$(du -sm ~/.cache/mcp-standards | cut -f1)
if [ $CACHE_SIZE -gt 500 ]; then
    echo "Cache size warning: ${CACHE_SIZE}MB"
    mcp-standards cache --clear-outdated
fi

# Check for sync failures
if mcp-standards status --json | jq -e '.last_sync_error' > /dev/null; then
    echo "Sync errors detected"
    mcp-standards sync --force
fi
```

## Known Issues

### Platform-Specific Issues

#### macOS
- File watching may require additional permissions
- Homebrew Python may have PATH issues

#### Windows
- Path separators in config files must use forward slashes
- Long path names may cause issues

#### Linux
- SELinux may block cache writes
- systemd service may need additional capabilities

### Version-Specific Issues

Check the [CHANGELOG](https://github.com/williamzujkowski/mcp-standards-server/blob/main/CHANGELOG.md) for version-specific known issues and fixes.