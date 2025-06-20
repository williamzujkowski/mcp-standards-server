# Standards Versioning and Updates

## Overview

The MCP Standards Server includes a comprehensive versioning system for managing standards documents. This system supports multiple versioning strategies, automated updates, and rollback capabilities.

## Features

### Versioning Strategies

1. **Semantic Versioning** (default)
   - Format: `MAJOR.MINOR.PATCH` (e.g., `1.2.3`)
   - Automatic patch increment by default
   - Suitable for tracking breaking changes

2. **Date-Based Versioning**
   - Format: `YYYY.MM.DD` (e.g., `2025.06.20`)
   - Useful for time-based releases
   - Clear indication of when version was created

3. **Incremental Versioning**
   - Format: `v1`, `v2`, `v3`, etc.
   - Simple sequential numbering
   - Good for straightforward version tracking

4. **Hash-Based Versioning**
   - Format: 8-character hash (e.g., `a1b2c3d4`)
   - Content-based versioning
   - Ensures unique version identifiers

### Version Management

- **Create Versions**: Snapshot current state of standards
- **Compare Versions**: See what changed between versions
- **Rollback**: Revert to previous versions when needed
- **History Tracking**: Full audit trail of all changes

### Remote Updates

- **Automated Updates**: Pull latest standards from remote sources
- **Selective Updates**: Update specific standards only
- **Validation**: Ensure standards meet quality requirements
- **Backup**: Automatic backups before updates

## CLI Commands

### Check Version History

```bash
mcp-standards standards version <standard_id>
```

Example:
```bash
mcp-standards standards version coding_standards
```

### Create New Version

```bash
mcp-standards standards create-version <standard_id> \
  --changelog "Description of changes" \
  --author "Your Name" \
  --strategy semantic
```

### Compare Versions

```bash
mcp-standards standards compare <standard_id> <version1> <version2>
```

Example:
```bash
mcp-standards standards compare coding_standards 1.0.0 1.1.0
```

### Update from Remote

```bash
mcp-standards standards update \
  --source https://github.com/williamzujkowski/standards \
  --standard coding_standards \
  --standard testing_standards
```

### Rollback Version

```bash
mcp-standards standards rollback <standard_id> <target_version> \
  --reason "Reason for rollback"
```

### Configure Auto-Updates

```bash
mcp-standards standards schedule \
  --frequency weekly \
  --enable
```

## Configuration

### Update Configuration

Create `.mcp-standards/update-config.yaml`:

```yaml
source_url: https://github.com/williamzujkowski/standards
update_frequency: monthly
auto_update: true
backup_enabled: true
validation_required: true
allowed_sources:
  - https://github.com/williamzujkowski/standards
  - https://internal.company.com/standards
update_schedule:
  day: "Monday"
  time: "02:00"
```

### Version Registry

The version registry is stored in `.versions/version_registry.json`:

```json
{
  "versions": {
    "coding_standards": [
      {
        "version": "1.0.0",
        "created_at": "2025-06-20T10:00:00",
        "author": "user1",
        "changelog": "Initial version"
      }
    ]
  },
  "latest": {
    "coding_standards": "1.0.0"
  },
  "history": []
}
```

## API Usage

### Python API

```python
from pathlib import Path
from src.core.standards.versioning import (
    StandardsVersionManager,
    VersioningStrategy,
    UpdateConfiguration
)

# Initialize version manager
manager = StandardsVersionManager(Path("data/standards"))

# Create a new version
version = await manager.create_version(
    "coding_standards",
    content_dict,
    author="developer",
    changelog="Added new security guidelines",
    strategy=VersioningStrategy.SEMANTIC
)

# Compare versions
diff = await manager.compare_versions(
    "coding_standards", 
    "1.0.0", 
    "1.1.0"
)

# Update from remote
report = await manager.update_from_source(
    "https://github.com/williamzujkowski/standards",
    ["coding_standards", "testing_standards"]
)
```

### Integration with Standards Engine

The Standards Engine automatically supports versioned loading:

```python
from src.core.standards.engine import StandardsEngine
from src.core.standards.models import StandardQuery

engine = StandardsEngine(Path("data/standards"))

# Load specific version
query = StandardQuery(
    query="secure api",
    version="1.0.0"  # Specific version
)
result = await engine.load_standards(query)

# Load latest version (default)
query = StandardQuery(
    query="secure api",
    version="latest"
)
result = await engine.load_standards(query)
```

## Best Practices

### When to Create Versions

1. **Before Major Updates**: Create a version before making significant changes
2. **After Review Cycles**: Version after standards have been reviewed and approved
3. **Regular Intervals**: Consider monthly or quarterly versioning
4. **Before Deployment**: Version before rolling out to production

### Version Naming

- Use semantic versioning for code-related standards
- Use date-based for policy or compliance standards
- Be consistent within your organization

### Update Strategy

1. **Test First**: Always test updates in a staging environment
2. **Backup**: Ensure backups are enabled before updates
3. **Validate**: Use validation to catch issues early
4. **Gradual Rollout**: Update one standard at a time for critical systems

### Rollback Guidelines

- Document rollback reasons clearly
- Test the rollback in staging first
- Communicate changes to affected teams
- Create a new version after rollback to track the change

## NIST Compliance

The versioning system implements several NIST controls:

- **CM-2 (Baseline Configuration)**: Version tracking maintains configuration baselines
- **CM-3 (Configuration Change Control)**: Controlled updates with validation
- **CM-4 (Security Impact Analysis)**: Version comparison shows impact
- **CM-9 (Configuration Management Plan)**: Structured approach to changes

## Troubleshooting

### Common Issues

1. **Version Not Found**
   ```bash
   # Check available versions
   mcp-standards standards version <standard_id>
   ```

2. **Update Failures**
   - Check network connectivity
   - Verify source URL is in allowed sources
   - Check validation errors in logs

3. **Rollback Issues**
   - Ensure target version exists
   - Check file permissions
   - Verify standard file hasn't been manually modified

### Debug Mode

Enable debug logging:
```bash
export MCP_STANDARDS_DEBUG=true
mcp-standards standards update --source <url>
```

## Future Enhancements

- **Conflict Resolution**: Automatic merge of concurrent changes
- **Branch Support**: Git-like branching for standards
- **Approval Workflow**: Require approvals for version promotion
- **Diff Visualization**: Visual comparison of versions
- **Automated Testing**: Run compliance checks on new versions