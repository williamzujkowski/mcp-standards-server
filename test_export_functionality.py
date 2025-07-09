#!/usr/bin/env python3
"""
Test script for the export functionality of the MCP Standards Server Web UI.
"""

import asyncio
import aiohttp
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))


async def test_export_endpoints():
    """Test the export endpoints."""
    base_url = "http://localhost:8000"
    
    async with aiohttp.ClientSession() as session:
        print("Testing Export Functionality")
        print("="*50)
        
        # Test 1: Get all standards first
        print("\n1. Getting all standards...")
        async with session.get(f"{base_url}/api/standards") as resp:
            if resp.status == 200:
                data = await resp.json()
                print(f"✓ Found {data['total']} standards")
                
                # Get first standard ID for testing
                first_standard_id = None
                for category, standards in data['standards'].items():
                    if standards:
                        first_standard_id = standards[0]['id']
                        break
            else:
                print(f"✗ Failed to get standards: {resp.status}")
                return
        
        # Test 2: Export single standard as Markdown
        if first_standard_id:
            print(f"\n2. Testing single standard export (Markdown)...")
            print(f"   Standard ID: {first_standard_id}")
            
            async with session.get(
                f"{base_url}/api/export/{first_standard_id}",
                params={"format": "markdown"}
            ) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    print(f"✓ Exported standard as Markdown ({len(content)} bytes)")
                    print(f"   Content preview: {content[:100].decode('utf-8')}...")
                else:
                    print(f"✗ Failed to export as Markdown: {resp.status}")
        
        # Test 3: Export single standard as JSON
        if first_standard_id:
            print(f"\n3. Testing single standard export (JSON)...")
            
            async with session.get(
                f"{base_url}/api/export/{first_standard_id}",
                params={"format": "json"}
            ) as resp:
                if resp.status == 200:
                    content = await resp.read()
                    print(f"✓ Exported standard as JSON ({len(content)} bytes)")
                    
                    # Verify it's valid JSON
                    try:
                        json_data = json.loads(content)
                        print(f"   Title: {json_data.get('title', 'N/A')}")
                        print(f"   Category: {json_data.get('category', 'N/A')}")
                    except json.JSONDecodeError:
                        print("✗ Invalid JSON format")
                else:
                    print(f"✗ Failed to export as JSON: {resp.status}")
        
        # Test 4: Bulk export
        print(f"\n4. Testing bulk export...")
        
        # Test with specific standards
        standard_ids = []
        for category, standards in data['standards'].items():
            for std in standards[:2]:  # Take first 2 from each category
                standard_ids.append(std['id'])
            if len(standard_ids) >= 5:  # Limit to 5 standards for testing
                break
        
        print(f"   Exporting {len(standard_ids)} standards...")
        
        async with session.post(
            f"{base_url}/api/export/bulk",
            json={"standards": standard_ids, "format": "json"}
        ) as resp:
            if resp.status == 200:
                content = await resp.read()
                print(f"✓ Bulk exported {len(standard_ids)} standards ({len(content)} bytes)")
                
                # Verify it's valid JSON
                try:
                    json_data = json.loads(content)
                    print(f"   Export date: {json_data.get('exportDate', 'N/A')}")
                    print(f"   Total standards: {json_data.get('totalStandards', 0)}")
                except json.JSONDecodeError:
                    print("✗ Invalid JSON format")
            else:
                print(f"✗ Failed bulk export: {resp.status}")
                error_text = await resp.text()
                print(f"   Error: {error_text}")
        
        # Test 5: Bulk export all standards
        print(f"\n5. Testing bulk export all standards...")
        
        async with session.post(
            f"{base_url}/api/export/bulk",
            json={"standards": [], "format": "json"}  # Empty list exports all
        ) as resp:
            if resp.status == 200:
                content = await resp.read()
                print(f"✓ Bulk exported all standards ({len(content)} bytes)")
                
                # Verify it's valid JSON
                try:
                    json_data = json.loads(content)
                    print(f"   Export date: {json_data.get('exportDate', 'N/A')}")
                    print(f"   Total standards: {json_data.get('totalStandards', 0)}")
                except json.JSONDecodeError:
                    print("✗ Invalid JSON format")
            else:
                print(f"✗ Failed bulk export all: {resp.status}")
        
        print("\n" + "="*50)
        print("Export functionality test complete!")


async def main():
    """Main function."""
    print("Starting export functionality tests...")
    print("Make sure the backend server is running on http://localhost:8000")
    
    try:
        await test_export_endpoints()
    except aiohttp.ClientError as e:
        print(f"\nError: Could not connect to server - {e}")
        print("Make sure the backend server is running with:")
        print("  cd web/backend && python main.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())