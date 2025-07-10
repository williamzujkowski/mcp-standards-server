# Standards Cache Population Guide

This guide documents how to populate the standards cache for MCP tools to access standards data.

## Overview

The MCP Standards Server requires standards to be cached locally in JSON format for the tools to function properly. The cache population process involves:

1. Synchronizing standards from the GitHub repository (downloads as Markdown files)
2. Converting Markdown files to JSON format with proper metadata
3. Verifying MCP tools can access the cached data

## Prerequisites

- Python 3.8+ installed
- MCP Standards Server dependencies installed (`pip install -e .`)
- Internet connection for GitHub access

## Cache Population Process

### 1. Initial Synchronization

The synchronization process downloads standards from the configured GitHub repository:

```python
from src.mcp_server import MCPStandardsServer

# Initialize the server
server = MCPStandardsServer()

# Run synchronization
result = await server.synchronizer.sync(force=True)
print(f"Synced {len(result.synced_files)} files")
```

This downloads Markdown files to `data/standards/cache/`.

### 2. Convert to JSON Format

The MCP tools expect JSON format files with metadata. You need to convert the Markdown files:

```python
import json
import re
from pathlib import Path

def convert_md_to_json(md_file: Path):
    """Convert a markdown standard to JSON format."""
    content = md_file.read_text(encoding='utf-8')
    
    # Extract metadata
    title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else md_file.stem
    
    # Create JSON structure
    standard = {
        "id": md_file.stem.lower(),
        "name": title,
        "category": determine_category(md_file.name),
        "tags": extract_tags(content),
        "version": "1.0.0",
        "content": content
    }
    
    # Save as JSON
    json_file = md_file.with_suffix('.json')
    with open(json_file, 'w') as f:
        json.dump(standard, f, indent=2)
```

### 3. Verify Cache Population

After population, verify the cache contains both formats:

```bash
# Check cache directory
ls -la data/standards/cache/

# Should see:
# - sync_metadata.json (tracking file)
# - *.md files (raw standards)
# - *.json files (formatted for MCP tools)
```

## Current Cache State

As of the last sync:
- **Total Standards**: 21
- **Cache Location**: `/data/standards/cache/`
- **Formats Available**: Markdown (.md) and JSON (.json)

### Available Standards Categories:

- **Development**: coding_standards
- **Security**: modern_security_standards
- **Cloud**: cloud_native_standards
- **Testing**: testing_standards
- **Frontend**: frontend_mobile_standards, web_design_ux_standards
- **DevOps**: devops_platform_standards
- **Data**: data_engineering_standards
- **Architecture**: event_driven_standards
- **Compliance**: compliance_standards, legal_compliance_standards
- **Operations**: cost_optimization_standards, observability_standards
- **Management**: project_management_standards, knowledge_management_standards
- **Content**: content_standards, seo_web_marketing_standards
- **Tools**: github_platform_standards, toolchain_standards
- **MCP**: model_context_protocol_standards
- **General**: unified_standards

## Using MCP Tools

Once the cache is populated, MCP tools can access standards:

```python
# List all standards
standards = await server._list_available_standards()

# Get standards for a project
applicable = await server._get_applicable_standards({
    "project_type": "web_application",
    "language": "python"
})

# Search standards
results = await server._search_standards("security")
```

## Troubleshooting

### Issue: "Standards cache directory not found"
**Solution**: Run the synchronization process to create and populate the cache.

### Issue: Tools not finding standards by ID
**Solution**: Ensure JSON files use lowercase IDs matching the expected format.

### Issue: Search returns no results
**Solution**: The semantic search feature requires indexing. Check if search is enabled in configuration.

## Automation

To automate cache population on server startup:

```python
async def initialize_server():
    """Initialize server with populated cache."""
    server = MCPStandardsServer()
    
    # Check if cache needs population
    cache_dir = Path("data/standards/cache")
    json_files = list(cache_dir.glob("*.json"))
    
    if len(json_files) < 2:  # Only sync_metadata.json exists
        print("Cache empty, running sync...")
        await server.synchronizer.sync(force=True)
        # Convert to JSON format
        await convert_all_standards()
    
    return server
```

## Maintenance

- **Sync Frequency**: Standards are synced from GitHub every 6 hours by default
- **Cache TTL**: 24 hours (configurable)
- **Manual Sync**: Use `sync_standards` tool with `force=true` parameter

## Related Documentation

- [MCP Server Documentation](./MCP_SERVER_GUIDE.md)
- [Standards Synchronization](../src/core/standards/sync.py)
- [Creating Standards Guide](./CREATING_STANDARDS_GUIDE.md)