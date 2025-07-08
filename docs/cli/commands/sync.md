# sync Command

Synchronize standards from the configured GitHub repository.

## Synopsis

```bash
mcp-standards sync [options]
```

## Description

The `sync` command downloads standards files from a GitHub repository and caches them locally. It supports incremental updates, force synchronization, and checking for updates without downloading.

## Options

### `-f, --force`
Force synchronization of all files, ignoring cache TTL.

```bash
mcp-standards sync --force
```

### `--check`
Check for updates without downloading files.

```bash
mcp-standards sync --check
```

### `--include <pattern>`
Include only files matching the pattern (glob syntax).

```bash
mcp-standards sync --include "*.yaml"
mcp-standards sync --include "web-*"
```

### `--exclude <pattern>`
Exclude files matching the pattern.

```bash
mcp-standards sync --exclude "*.draft.yaml"
```

### `--parallel <n>`
Number of parallel downloads (default: 5).

```bash
mcp-standards sync --parallel 10
```

### `--retry <n>`
Number of retry attempts for failed downloads (default: 3).

```bash
mcp-standards sync --retry 5
```

### `--timeout <seconds>`
Timeout for each file download (default: 30).

```bash
mcp-standards sync --timeout 60
```

## Examples

### Basic Sync

```bash
# Sync all standards files
mcp-standards sync
```

Output:
```
Starting standards synchronization...
Fetching file list from williamzujkowski/standards...
Files to sync: 25
Downloading: web-development-standards.yaml... [OK]
Downloading: api-design-standards.yaml... [OK]
...
Sync completed with status: success
Duration: 12.34 seconds
Files synced: 25/25
```

### Check for Updates

```bash
# Check which files need updating
mcp-standards sync --check
```

Output:
```
Checking for updates...

Outdated files (3):
  - web-development-standards.yaml (last synced 48.2 hours ago)
  - testing-standards.yaml (last synced 72.5 hours ago)
  - security-standards.yaml (last synced 96.1 hours ago)

Total cached files: 25
Cache TTL: 24 hours
```

### Force Sync Specific Files

```bash
# Force sync only web-related standards
mcp-standards sync --force --include "web-*.yaml"
```

### Sync with Custom Configuration

```bash
# Use project-specific sync configuration
mcp-standards -c .mcp-standards.yaml sync
```

## Configuration

The sync command uses the following configuration section:

```yaml
# .mcp-standards.yaml
repository:
  owner: williamzujkowski
  repo: standards
  branch: main
  path: standards

sync:
  cache_ttl_hours: 24
  parallel_downloads: 5
  retry_attempts: 3
  timeout_seconds: 30
  include_patterns:
    - "*.yaml"
    - "*.md"
  exclude_patterns:
    - "*.draft.*"
    - ".git*"

cache:
  directory: ~/.cache/mcp-standards
  max_size_mb: 500
```

## Error Handling

The sync command handles various error scenarios:

### Network Errors

```bash
mcp-standards sync
# Error: Failed to connect to GitHub API
# Suggestion: Check internet connection and GitHub status
```

### Rate Limiting

```bash
mcp-standards sync
# Warning: GitHub API rate limit reached (60/60)
# Suggestion: Wait until 2025-07-08 15:30:00 or configure authentication
```

### Permission Errors

```bash
mcp-standards sync
# Error: Permission denied writing to cache directory
# Suggestion: Check directory permissions or use --cache-dir
```

## Performance Considerations

- **Parallel Downloads**: Increase `--parallel` for faster syncs on good connections
- **Caching**: Files are cached based on TTL to minimize API calls
- **Incremental Sync**: Only outdated files are downloaded by default
- **Compression**: Files are compressed in cache to save disk space

## Integration Examples

### CI/CD Pipeline

```bash
#!/bin/bash
# ci-sync-standards.sh

# Check if sync is needed
if mcp-standards sync --check | grep -q "Outdated files"; then
    echo "Syncing outdated standards..."
    mcp-standards sync --retry 5 --timeout 60
else
    echo "Standards are up to date"
fi
```

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Ensure standards are synced before commit
mcp-standards sync --check || {
    echo "Standards are outdated. Run 'mcp-standards sync' first."
    exit 1
}
```

### Scheduled Sync

```cron
# Crontab entry - sync standards daily at 2 AM
0 2 * * * /usr/local/bin/mcp-standards sync >> /var/log/mcp-standards-sync.log 2>&1
```

## Related Commands

- [status](./status.md) - Check sync status
- [cache](./cache.md) - Manage cached files
- [config](./config.md) - View sync configuration