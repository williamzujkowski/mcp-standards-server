#!/usr/bin/env python3
"""
Functional test script for MCP functions.

Tests all 7 core MCP functions with basic scenarios to verify functionality.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.mcp_server import MCPStandardsServer


async def test_mcp_functions():
    """Test all MCP functions with basic scenarios."""
    
    print("🔧 Initializing MCP Standards Server...")
    
    # Initialize server
    config = {
        "auth": {"enabled": False},  # Disable auth for testing
        "search": {"enabled": True},
        "rate_limit_max_requests": 1000,
    }
    
    try:
        server = MCPStandardsServer(config)
        print("✅ Server initialized successfully")
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")
        return False
    
    # Test scenarios
    test_results = {}
    
    print("\n" + "="*60)
    print("TESTING MCP FUNCTIONS")
    print("="*60)
    
    # Test 1: list_available_standards
    print("\n1. Testing list_available_standards...")
    try:
        result = await server._list_available_standards()
        standards_count = len(result.get("standards", []))
        if standards_count > 0:
            print(f"   ✅ Found {standards_count} standards")
            test_results["list_available_standards"] = "PASS"
        else:
            print(f"   ⚠️  No standards found")
            test_results["list_available_standards"] = "WARN"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["list_available_standards"] = "FAIL"
    
    # Test 2: get_applicable_standards
    print("\n2. Testing get_applicable_standards...")
    try:
        context = {
            "project_type": "web_application",
            "framework": "react",
            "language": "javascript"
        }
        result = await server._get_applicable_standards(context)
        applicable_standards = result.get("standards", [])
        if isinstance(applicable_standards, list):
            print(f"   ✅ Found {len(applicable_standards)} applicable standards")
            test_results["get_applicable_standards"] = "PASS"
        else:
            print(f"   ❌ Unexpected result format")
            test_results["get_applicable_standards"] = "FAIL"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["get_applicable_standards"] = "FAIL"
    
    # Test 3: search_standards (if search is available)
    print("\n3. Testing search_standards...")
    try:
        result = await server._search_standards("react javascript", limit=5)
        search_results = result.get("results", [])
        if isinstance(search_results, list):
            print(f"   ✅ Search returned {len(search_results)} results")
            test_results["search_standards"] = "PASS"
        else:
            print(f"   ⚠️  Search disabled or no results: {result}")
            test_results["search_standards"] = "WARN"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["search_standards"] = "FAIL"
    
    # Test 4: get_standard_details
    print("\n4. Testing get_standard_details...")
    try:
        # Try to get the first available standard
        standards_list = await server._list_available_standards()
        standards = standards_list.get("standards", [])
        
        if standards:
            # Use a known working standard ID
            standard_id = "react-18-patterns"  # This exists in cache
            result = await server._get_standard_details(standard_id)
            if result and isinstance(result, dict):
                print(f"   ✅ Retrieved standard details for '{standard_id}'")
                test_results["get_standard_details"] = "PASS"
            else:
                print(f"   ❌ Invalid result format")
                test_results["get_standard_details"] = "FAIL"
        else:
            print(f"   ⚠️  No standards available for testing")
            test_results["get_standard_details"] = "WARN"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["get_standard_details"] = "FAIL"
    
    # Test 5: validate_against_standard  
    print("\n5. Testing validate_against_standard...")
    try:
        test_code = '''
function HelloWorld() {
    var message = "Hello, World!";
    return message;
}
'''
        result = await server._validate_against_standard(
            test_code, 
            "react-18-patterns", 
            "javascript"
        )
        if isinstance(result, dict) and "passed" in result:
            status = "passed" if result["passed"] else "failed with violations"
            print(f"   ✅ Validation completed: {status}")
            test_results["validate_against_standard"] = "PASS"
        else:
            print(f"   ❌ Invalid result format")
            test_results["validate_against_standard"] = "FAIL"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["validate_against_standard"] = "FAIL"
    
    # Test 6: get_optimized_standard
    print("\n6. Testing get_optimized_standard...")
    try:
        # Use a standard that should exist
        standards_list = await server._list_available_standards()
        standards = standards_list.get("standards", [])
        
        if standards:
            # Use a known working standard ID
            standard_id = "react-18-patterns"  # This exists in cache
            result = await server._get_optimized_standard(
                standard_id=standard_id,
                format_type="condensed",
                token_budget=2000
            )
            if isinstance(result, dict) and "content" in result:
                original_tokens = result.get("original_tokens", 0)
                compressed_tokens = result.get("compressed_tokens", 0)
                print(f"   ✅ Optimized standard: {original_tokens} → {compressed_tokens} tokens")
                test_results["get_optimized_standard"] = "PASS"
            else:
                print(f"   ❌ Invalid result format")
                test_results["get_optimized_standard"] = "FAIL"
        else:
            print(f"   ⚠️  No standards available for testing")
            test_results["get_optimized_standard"] = "WARN"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["get_optimized_standard"] = "FAIL"
    
    # Test 7: sync_standards
    print("\n7. Testing sync_standards...")
    try:
        result = await server._sync_standards(force=False)
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            synced_count = len(result.get("synced_files", []))
            print(f"   ✅ Sync completed with status '{status}', {synced_count} files")
            test_results["sync_standards"] = "PASS"
        else:
            print(f"   ❌ Invalid result format")
            test_results["sync_standards"] = "FAIL"
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        test_results["sync_standards"] = "FAIL"
    
    # Test summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result == "PASS")
    warned_tests = sum(1 for result in test_results.values() if result == "WARN")
    failed_tests = sum(1 for result in test_results.values() if result == "FAIL")
    
    for function, result in test_results.items():
        status_icon = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}[result]
        print(f"{status_icon} {function}: {result}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} PASSED, {warned_tests} WARNINGS, {failed_tests} FAILED")
    
    # Write results to file
    results_file = Path("test_results_mcp_functions.json")
    with open(results_file, 'w') as f:
        json.dump({
            "timestamp": asyncio.get_event_loop().time(),
            "total_tests": total_tests,
            "passed": passed_tests,
            "warned": warned_tests,
            "failed": failed_tests,
            "details": test_results
        }, f, indent=2)
    
    print(f"\n📄 Results saved to {results_file}")
    
    # Return overall success
    return failed_tests == 0


if __name__ == "__main__":
    try:
        success = asyncio.run(test_mcp_functions())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        sys.exit(1)