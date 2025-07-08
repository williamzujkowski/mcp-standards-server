# status Command

Show synchronization status and statistics for the MCP Standards Server.

## Synopsis

```bash
mcp-standards status [options]
```

## Description

The `status` command displays comprehensive information about the current state of the standards cache, synchronization history, and system health.

## Options

### `--json`
Output status in JSON format for programmatic access.

```bash
mcp-standards status --json
```

### `--detailed`
Show detailed information including all cached files.

```bash
mcp-standards status --detailed
```

### `--check-health`
Perform health checks and report issues.

```bash
mcp-standards status --check-health
```

### `--summary`
Show only summary information (default: false).

```bash
mcp-standards status --summary
```

## Examples

### Basic Status

```bash
mcp-standards status
```

Output:
```
MCP Standards Server - Sync Status

Total files cached: 25
Total cache size: 3.42 MB

GitHub API Rate Limit:
  Remaining: 58/60
  Resets at: 2025-07-08 15:30:00

Recent syncs:
  - web-development-standards.yaml: 2025-07-08 10:15:30
  - api-design-standards.yaml: 2025-07-08 10:15:31
  - testing-standards.yaml: 2025-07-08 10:15:32
  - security-standards.yaml: 2025-07-08 10:15:33
  - mcp-server-patterns.yaml: 2025-07-08 10:15:34

Repository: williamzujkowski/standards
Branch: main
Path: standards
```

### JSON Output

```bash
mcp-standards status --json | jq .
```

Output:
```json
{
  "total_files": 25,
  "total_size_mb": 3.42,
  "rate_limit": {
    "remaining": 58,
    "limit": 60,
    "reset_time": "2025-07-08T15:30:00Z",
    "reset_timestamp": 1736349000
  },
  "last_sync_times": {
    "web-development-standards.yaml": "2025-07-08T10:15:30Z",
    "api-design-standards.yaml": "2025-07-08T10:15:31Z",
    "testing-standards.yaml": "2025-07-08T10:15:32Z",
    "security-standards.yaml": "2025-07-08T10:15:33Z",
    "mcp-server-patterns.yaml": "2025-07-08T10:15:34Z"
  },
  "config": {
    "repository": {
      "owner": "williamzujkowski",
      "repo": "standards",
      "branch": "main",
      "path": "standards"
    },
    "sync": {
      "cache_ttl_hours": 24,
      "parallel_downloads": 5
    }
  },
  "cache_health": {
    "status": "healthy",
    "writable": true,
    "space_available_gb": 45.2
  }
}
```

### Detailed Status

```bash
mcp-standards status --detailed
```

Output:
```
MCP Standards Server - Detailed Status

=== Cache Information ===
Location: /home/user/.cache/mcp-standards
Total files: 25
Total size: 3.42 MB
Space available: 45.2 GB

=== Cached Files ===
web-development/
  ├── html5-standards.yaml (12.3 KB) - synced 2h ago
  ├── css3-standards.yaml (15.7 KB) - synced 2h ago
  └── javascript-es2025.yaml (23.1 KB) - synced 2h ago

api-design/
  ├── rest-api-standards.yaml (18.9 KB) - synced 2h ago
  └── graphql-standards.yaml (21.4 KB) - synced 2h ago

testing/
  ├── javascript-testing.yaml (16.2 KB) - synced 2h ago
  └── python-testing.yaml (14.8 KB) - synced 2h ago

[... more files ...]

=== Sync History ===
Last successful sync: 2025-07-08 10:15:30
Total syncs today: 3
Failed syncs today: 0
Average sync duration: 12.3 seconds

=== System Health ===
✓ Cache directory writable
✓ Network connectivity OK
✓ GitHub API accessible
✓ No rate limit issues
```

### Health Check

```bash
mcp-standards status --check-health
```

Output:
```
Performing health checks...

✓ Cache directory accessible
✓ Configuration valid
✓ Network connectivity OK
✓ GitHub API reachable
✓ Rate limit healthy (58/60)
⚠ 3 files outdated (older than 24h)
✓ Disk space adequate (45.2 GB free)

Overall health: GOOD (1 warning)

Recommendations:
- Run 'mcp-standards sync' to update outdated files
```

## Status Information

### Cache Metrics

- **Total Files**: Number of standards files in cache
- **Cache Size**: Total disk space used by cached files
- **Space Available**: Free disk space in cache directory

### Rate Limiting

- **Remaining/Limit**: Current API calls available
- **Reset Time**: When the rate limit resets
- **Authenticated**: Whether using authenticated requests

### Sync History

- **Recent Syncs**: Last 5 synchronized files
- **Last Sync**: Timestamp of most recent sync
- **Sync Statistics**: Success/failure counts

### Health Indicators

- **Cache Writable**: Can write to cache directory
- **Network OK**: Internet connectivity available
- **API Accessible**: Can reach GitHub API
- **Rate Limit OK**: Sufficient API calls remaining

## Monitoring Integration

### Prometheus Metrics

```bash
# Export metrics in Prometheus format
mcp-standards status --format prometheus
```

Output:
```
# HELP mcp_standards_cache_files_total Total number of cached files
# TYPE mcp_standards_cache_files_total gauge
mcp_standards_cache_files_total 25

# HELP mcp_standards_cache_size_bytes Total size of cache in bytes
# TYPE mcp_standards_cache_size_bytes gauge
mcp_standards_cache_size_bytes 3588096

# HELP mcp_standards_rate_limit_remaining GitHub API rate limit remaining
# TYPE mcp_standards_rate_limit_remaining gauge
mcp_standards_rate_limit_remaining 58
```

### Nagios/Icinga Check

```bash
#!/bin/bash
# check_mcp_standards.sh

STATUS=$(mcp-standards status --json)
OUTDATED=$(echo $STATUS | jq '.outdated_files | length')

if [ $OUTDATED -gt 10 ]; then
    echo "CRITICAL: $OUTDATED outdated files"
    exit 2
elif [ $OUTDATED -gt 5 ]; then
    echo "WARNING: $OUTDATED outdated files"
    exit 1
else
    echo "OK: All files up to date"
    exit 0
fi
```

## Scripting Examples

### Check if Sync Needed

```bash
#!/bin/bash
# Check if any files are outdated

if mcp-standards status --json | jq -e '.outdated_files | length > 0' > /dev/null; then
    echo "Sync needed"
    mcp-standards sync
else
    echo "All files up to date"
fi
```

### Monitor Cache Size

```bash
#!/bin/bash
# Alert if cache exceeds size limit

MAX_SIZE_MB=500
CURRENT_SIZE=$(mcp-standards status --json | jq '.total_size_mb')

if (( $(echo "$CURRENT_SIZE > $MAX_SIZE_MB" | bc -l) )); then
    echo "Warning: Cache size ($CURRENT_SIZE MB) exceeds limit ($MAX_SIZE_MB MB)"
    # Could trigger cache cleanup here
fi
```

## Related Commands

- [sync](./sync.md) - Synchronize standards
- [cache](./cache.md) - Manage cache
- [config](./config.md) - View configuration