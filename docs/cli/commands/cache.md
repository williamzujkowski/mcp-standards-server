# cache Command

Manage the local standards cache.

## Synopsis

```bash
mcp-standards cache [options]
```

## Description

The `cache` command provides tools for managing the local cache of standards files, including listing, clearing, exporting, and analyzing cached content.

## Options

### `--list`
List all cached files with details.

```bash
mcp-standards cache --list
```

### `--clear`
Clear all cached files.

```bash
mcp-standards cache --clear
```

### `--clear-outdated`
Clear only outdated files (based on TTL).

```bash
mcp-standards cache --clear-outdated
```

### `--export <path>`
Export cache to a directory or archive.

```bash
mcp-standards cache --export /path/to/export
mcp-standards cache --export backup.tar.gz
```

### `--import <path>`
Import cache from a directory or archive.

```bash
mcp-standards cache --import /path/to/import
mcp-standards cache --import backup.tar.gz
```

### `--analyze`
Analyze cache usage and provide statistics.

```bash
mcp-standards cache --analyze
```

### `--verify`
Verify cache integrity.

```bash
mcp-standards cache --verify
```

### `--size-limit <MB>`
Set or check cache size limit.

```bash
mcp-standards cache --size-limit 500
```

## Examples

### List Cached Files

```bash
mcp-standards cache --list
```

Output:
```
Cached files (25):

web-development/
  html5-standards.yaml (12.3 KB) - cached 2h ago, expires in 22h
  css3-standards.yaml (15.7 KB) - cached 2h ago, expires in 22h
  javascript-es2025.yaml (23.1 KB) - cached 2h ago, expires in 22h

api-design/
  rest-api-standards.yaml (18.9 KB) - cached 25h ago, OUTDATED
  graphql-standards.yaml (21.4 KB) - cached 2h ago, expires in 22h

testing/
  javascript-testing.yaml (16.2 KB) - cached 2h ago, expires in 22h
  python-testing.yaml (14.8 KB) - cached 2h ago, expires in 22h

Total: 25 files, 3.42 MB
Outdated: 1 file
Cache directory: /home/user/.cache/mcp-standards
```

### Clear Cache

```bash
# Clear all cached files
mcp-standards cache --clear
```

Output:
```
Clearing cache...
Removed 25 files (3.42 MB)
Cache cleared successfully!
```

### Clear Outdated Files

```bash
# Clear only outdated files
mcp-standards cache --clear-outdated
```

Output:
```
Clearing outdated files...
Removed 3 files:
  - rest-api-standards.yaml (25h old)
  - legacy-patterns.yaml (48h old)
  - deprecated-standards.yaml (72h old)

Freed 156 KB of disk space
```

### Export Cache

```bash
# Export to directory
mcp-standards cache --export ./standards-backup

# Export to archive
mcp-standards cache --export standards-backup.tar.gz
```

Output:
```
Exporting cache to standards-backup.tar.gz...
Exported 25 files (3.42 MB)
Archive created successfully
```

### Import Cache

```bash
# Import from another installation
mcp-standards cache --import /mnt/backup/standards-cache.tar.gz
```

Output:
```
Importing cache from standards-cache.tar.gz...
Extracted 25 files
Verified checksums: OK
Import completed successfully
```

### Analyze Cache Usage

```bash
mcp-standards cache --analyze
```

Output:
```
Cache Analysis Report
====================

Location: /home/user/.cache/mcp-standards
Total size: 3.42 MB (3,588,096 bytes)
File count: 25
Average file size: 139.7 KB

Size by category:
  Web Development: 892 KB (26.1%) - 8 files
  API Design: 654 KB (19.1%) - 6 files
  Testing: 543 KB (15.9%) - 5 files
  Security: 487 KB (14.2%) - 4 files
  Other: 844 KB (24.7%) - 2 files

Age distribution:
  < 1 hour: 5 files
  1-6 hours: 12 files
  6-24 hours: 5 files
  > 24 hours: 3 files (outdated)

Compression ratio: 2.3:1 (files are compressed)
Disk space available: 45.2 GB

Recommendations:
- 3 files are outdated and can be cleared
- Cache is within size limits (3.42 MB / 500 MB)
- Consider running 'sync' to update outdated files
```

### Verify Cache Integrity

```bash
mcp-standards cache --verify
```

Output:
```
Verifying cache integrity...

Checking file structure... OK
Checking metadata... OK
Verifying checksums... 

✓ web-development/html5-standards.yaml
✓ web-development/css3-standards.yaml
✓ web-development/javascript-es2025.yaml
✗ api-design/rest-api-standards.yaml - checksum mismatch!
✓ api-design/graphql-standards.yaml

Results:
- Files checked: 25
- Valid files: 24
- Corrupted files: 1

Recommendation: Run 'mcp-standards sync --force' to repair corrupted files
```

## Cache Management

### Setting Size Limits

```bash
# Set cache size limit to 100 MB
mcp-standards cache --size-limit 100

# Check current usage against limit
mcp-standards cache --size-limit
```

Output:
```
Current cache usage: 3.42 MB / 100 MB (3.4%)
Warning threshold: 80 MB (80%)
Action threshold: 95 MB (95%)
```

### Automatic Cleanup

Configure automatic cleanup in your configuration:

```yaml
# .mcp-standards.yaml
cache:
  directory: ~/.cache/mcp-standards
  max_size_mb: 500
  auto_cleanup:
    enabled: true
    threshold_percent: 80
    remove_oldest: true
    keep_recent_hours: 24
```

### Cache Warming

```bash
# Pre-populate cache with all standards
mcp-standards sync --force --all

# Warm cache with specific patterns
mcp-standards sync --include "web-*" --include "api-*"
```

## Advanced Usage

### Cache Statistics Script

```bash
#!/bin/bash
# cache-stats.sh - Monitor cache usage over time

while true; do
    STATS=$(mcp-standards cache --analyze --json)
    TIMESTAMP=$(date +%s)
    SIZE_MB=$(echo $STATS | jq '.total_size_mb')
    FILE_COUNT=$(echo $STATS | jq '.file_count')
    
    echo "$TIMESTAMP,$SIZE_MB,$FILE_COUNT" >> cache-stats.csv
    sleep 3600  # Check every hour
done
```

### Backup Script

```bash
#!/bin/bash
# backup-cache.sh - Regular cache backups

BACKUP_DIR="/var/backups/mcp-standards"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Create backup
mcp-standards cache --export "$BACKUP_DIR/cache_$TIMESTAMP.tar.gz"

# Keep only last 7 backups
ls -t "$BACKUP_DIR"/cache_*.tar.gz | tail -n +8 | xargs -r rm
```

### Cache Sync Between Machines

```bash
# On source machine
mcp-standards cache --export - | ssh target "mcp-standards cache --import -"

# Using rsync for incremental sync
rsync -av ~/.cache/mcp-standards/ remote:~/.cache/mcp-standards/
```

## Performance Tips

1. **Regular Cleanup**: Schedule regular cleanup of outdated files
2. **Size Monitoring**: Monitor cache size to prevent disk space issues
3. **Compression**: Cache files are automatically compressed
4. **Network Cache**: Consider sharing cache across team using network storage
5. **CI/CD Cache**: Export and import cache in CI/CD pipelines for faster builds

## Related Commands

- [sync](./sync.md) - Synchronize standards
- [status](./status.md) - Check cache status
- [config](./config.md) - Configure cache settings