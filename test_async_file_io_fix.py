#!/usr/bin/env python3
"""
Test async file I/O fixes for MCP Standards Server.

This test verifies that the async file I/O changes work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_standards_engine():
    """Test that StandardsEngine still works with async file I/O."""
    print("🔧 Testing StandardsEngine with async file I/O...")
    
    try:
        from src.core.standards.engine import StandardsEngine
        
        # Initialize engine (this will trigger the async file loading)
        engine = StandardsEngine(data_dir=Path("data/standards"))
        
        print("   ✅ StandardsEngine initialized successfully")
        
        # Test listing standards
        standards = await engine.list_standards()
        print(f"   ✅ Loaded {len(standards)} standards")
        
        if len(standards) > 0:
            print(f"   📊 Example standard: {standards[0].id}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_server_functions():
    """Test that MCP server functions still work with async file I/O."""
    print("\n🔧 Testing MCP Server functions with async file I/O...")
    
    try:
        from src.mcp_server import MCPStandardsServer
        
        # Initialize server with minimal config
        config = {
            "auth": {"enabled": False},
            "search": {"enabled": True},
            "rate_limit_max_requests": 1000,
        }
        
        server = MCPStandardsServer(config)
        print("   ✅ MCP Server initialized successfully")
        
        # Test list available standards (this uses the async file I/O we fixed)
        standards_result = await server._list_available_standards()
        standards_count = len(standards_result.get("standards", []))
        print(f"   ✅ Listed {standards_count} standards via MCP")
        
        # Test get standard details (this also uses async file I/O we fixed)
        if standards_count > 0:
            # Try to get a known standard
            try:
                details = await server._get_standard_details("react-18-patterns")
                print("   ✅ Retrieved standard details via MCP")
            except Exception as e:
                # If that specific standard doesn't exist, try any available one
                standards = standards_result.get("standards", [])
                if standards:
                    standard_id = standards[0]["id"]
                    details = await server._get_standard_details(standard_id)
                    print(f"   ✅ Retrieved standard details for {standard_id} via MCP")
                else:
                    print("   ⚠️  No standards available for detailed testing")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run async file I/O fix tests."""
    print("🚀 Testing Async File I/O Fixes")
    print("=" * 50)
    
    results = []
    
    # Test StandardsEngine
    result1 = await test_standards_engine()
    results.append(result1)
    
    # Test MCP Server functions
    result2 = await test_mcp_server_functions()
    results.append(result2)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 ASYNC FILE I/O FIX TEST RESULTS")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All async file I/O fixes working correctly!")
        print("\nExpected improvements:")
        print("   - Non-blocking file operations during concurrent requests")
        print("   - Better response times under concurrent load")
        print("   - No event loop blocking during standards loading")
    else:
        print("❌ Some async file I/O fixes have issues")
        print("   Check the error messages above for details")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)