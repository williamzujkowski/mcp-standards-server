#!/usr/bin/env python3
"""
Test script to inspect available standards and their metadata structure.
"""

import asyncio
import json
from pathlib import Path

from src.core.mcp.handlers import StandardsHandler

# Import the MCP server components directly
from src.core.standards.engine import StandardsEngine


async def main():
    """Test available standards and their structure."""

    print("🔍 Inspecting Available Standards and Metadata")
    print("=" * 60)

    # Initialize the standards engine
    data_dir = "/home/william/git/mcp-standards-server/data/standards"
    standards_engine = StandardsEngine(data_dir=data_dir)
    await standards_engine.initialize()

    # Initialize the handler
    handler = StandardsHandler(standards_engine)
    await handler.initialize()

    # Get all available standards
    print("📚 Getting list of available standards...")
    standards_result = await handler.handle_tool("list_available_standards", {})

    if standards_result and "result" in standards_result:
        all_standards = standards_result["result"]
        print(f"Found {len(all_standards)} standards")

        # Show first few standards and their structure
        print("\n📋 Sample standards structure:")
        for i, std in enumerate(all_standards[:5]):
            print(f"\n{i+1}. Standard: {std}")

        # Test getting specific standards to check metadata
        print("\n🔬 Testing specific standards for NIST metadata...")

        test_standard_ids = [
            "react-18-patterns",
            "typescript-5-guidelines",
            "security-review-audit-process",
            "data-privacy-compliance"
        ]

        for std_id in test_standard_ids:
            print(f"\n📖 Testing standard: {std_id}")

            # Get the standard
            std_result = await handler.handle_tool("get_standard", {"standard_id": std_id})

            if std_result and "result" in std_result and std_result["result"]:
                standard = std_result["result"]
                print(f"   ✅ Found standard: {standard.title if hasattr(standard, 'title') else 'Unknown'}")

                # Check if it has metadata
                if hasattr(standard, 'metadata') and standard.metadata:
                    print(f"   📊 Metadata exists: {type(standard.metadata)}")

                    # Check for NIST controls
                    if hasattr(standard.metadata, 'nist_controls'):
                        controls = standard.metadata.nist_controls
                        print(f"   🛡️  NIST Controls: {controls} (count: {len(controls)})")
                    else:
                        print("   ❌ No nist_controls attribute in metadata")
                        print(f"   📋 Available metadata attributes: {dir(standard.metadata)}")

                    # Show metadata structure
                    if hasattr(standard.metadata, '__dict__'):
                        print(f"   📋 Metadata content: {standard.metadata.__dict__}")
                else:
                    print("   ❌ No metadata found")
                    print(f"   📋 Standard attributes: {dir(standard)}")
            else:
                print(f"   ❌ Standard not found or error: {std_result}")
    else:
        print(f"❌ Failed to get standards list: {standards_result}")

    # Also check the actual file structure
    print(f"\n📁 Checking file structure in {data_dir}")
    data_path = Path(data_dir)

    # Look for markdown files with NIST references
    md_files = list(data_path.glob("*.md"))
    print(f"Found {len(md_files)} markdown files")

    for md_file in md_files[:3]:  # Check first 3
        print(f"\n📄 Checking {md_file.name}:")
        content = md_file.read_text()
        if "NIST" in content:
            # Find NIST references
            lines_with_nist = [line.strip() for line in content.split('\n') if 'NIST' in line]
            print(f"   🛡️  NIST references found: {len(lines_with_nist)}")
            for line in lines_with_nist[:2]:  # Show first 2
                print(f"     - {line}")
        else:
            print("   ❌ No NIST references found")

    # Check cache directory structure
    cache_dir = data_path / "cache"
    if cache_dir.exists():
        cache_files = list(cache_dir.glob("*.json"))
        print(f"\n💾 Found {len(cache_files)} cached standards")

        # Check a few cache files for NIST data
        for cache_file in cache_files[:3]:
            print(f"\n🗂️  Checking cached file: {cache_file.name}")
            try:
                with open(cache_file) as f:
                    cache_data = json.load(f)

                print(f"   📋 Cache structure keys: {list(cache_data.keys())}")

                # Look for metadata or NIST references
                if 'metadata' in cache_data:
                    metadata = cache_data['metadata']
                    print(f"   📊 Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")

                    if isinstance(metadata, dict) and 'nist_controls' in metadata:
                        print(f"   🛡️  NIST controls in cache: {metadata['nist_controls']}")
                    else:
                        print("   ❌ No nist_controls in cached metadata")
                else:
                    print("   ❌ No metadata in cache file")

            except Exception as e:
                print(f"   ❌ Error reading cache file: {e}")

if __name__ == "__main__":
    asyncio.run(main())
