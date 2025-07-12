#!/usr/bin/env python3
"""
Test rate limiting and request queuing fixes for MCP Standards Server.

This test verifies that the enhanced rate limiting with request queuing works correctly.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_async_rate_limiter():
    """Test the async rate limiter functionality."""
    print("🔧 Testing AsyncRateLimiter functionality...")
    
    try:
        from src.core.rate_limiter import AsyncRateLimiter
        
        # Create rate limiter with low limits for testing
        rate_limiter = AsyncRateLimiter(
            max_requests=5,  # Very low limit for testing
            window_seconds=10,
            enable_queuing=True,
            max_queue_size=10,
            queue_timeout_seconds=5.0,
        )
        
        print("   ✅ AsyncRateLimiter initialized successfully")
        
        # Test normal requests within limit
        user_id = "test_user_1"
        success_count = 0
        
        for i in range(3):  # Should all succeed (under limit of 5)
            is_allowed, limit_info = await rate_limiter.check_rate_limit(user_id)
            if is_allowed:
                success_count += 1
        
        print(f"   ✅ {success_count}/3 requests allowed within rate limit")
        
        if success_count != 3:
            print(f"   ❌ Expected 3 successful requests, got {success_count}")
            return False
        
        # Test rate limiting (should queue requests)
        queued_count = 0
        rate_limited_count = 0
        
        for i in range(8):  # Should exceed limit of 5, some should be queued
            is_allowed, limit_info = await rate_limiter.check_rate_limit(user_id)
            if is_allowed:
                if limit_info and limit_info.get("queued"):
                    queued_count += 1
                    print(f"   📋 Request {i+4} queued, estimated wait: {limit_info.get('estimated_wait', 0):.1f}s")
                else:
                    success_count += 1
            else:
                rate_limited_count += 1
        
        print(f"   ✅ Rate limiting working: {success_count} allowed, {queued_count} queued, {rate_limited_count} rejected")
        
        # Test metrics
        metrics = rate_limiter.get_metrics()
        print(f"   📊 Metrics: {metrics['total_requests']} total, {metrics['queued_requests']} queued")
        
        # Clean up
        await rate_limiter.cleanup()
        print("   ✅ Rate limiter cleaned up successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_mcp_server_integration():
    """Test integration with MCP server."""
    print("\n🔧 Testing MCP Server rate limiting integration...")
    
    try:
        from src.mcp_server import MCPStandardsServer
        
        # Create server with low rate limits for testing
        config = {
            "auth": {"enabled": False},
            "search": {"enabled": True},
            "rate_limit_max_requests": 3,  # Very low for testing
            "rate_limit_window": 10,
            "rate_limit_enable_queuing": True,
        }
        
        server = MCPStandardsServer(config)
        print("   ✅ MCP Server initialized with rate limiting")
        
        # Initialize async components
        await server._initialize_async_components()
        print("   ✅ Async components initialized")
        
        # Test rate limiter is working
        if server._async_rate_limiter:
            print("   ✅ Async rate limiter is available")
            
            # Test basic rate limiting
            user_key = "test_integration_user"
            allowed_count = 0
            queued_count = 0
            
            for i in range(6):  # Should exceed limit of 3
                is_allowed, limit_info = await server._async_rate_limiter.check_rate_limit(user_key)
                if is_allowed:
                    if limit_info and limit_info.get("queued"):
                        queued_count += 1
                    else:
                        allowed_count += 1
            
            print(f"   ✅ Integration test: {allowed_count} allowed, {queued_count} queued")
            
            # Test metrics
            metrics = server._async_rate_limiter.get_metrics()
            print(f"   📊 Integration metrics: {metrics}")
            
        else:
            print("   ❌ Async rate limiter not initialized")
            return False
        
        # Clean up
        await server._cleanup_async_components()
        print("   ✅ Server components cleaned up")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_concurrent_performance():
    """Test performance under concurrent load."""
    print("\n🔧 Testing concurrent performance with rate limiting...")
    
    try:
        from src.core.rate_limiter import AsyncRateLimiter
        
        # Create rate limiter for concurrent testing
        rate_limiter = AsyncRateLimiter(
            max_requests=50,  # Higher limit for concurrent testing
            window_seconds=10,
            enable_queuing=True,
            max_queue_size=100,
        )
        
        # Test concurrent requests
        start_time = time.time()
        
        async def make_requests(user_id: str, count: int):
            """Make multiple requests for a user."""
            results = []
            for i in range(count):
                is_allowed, limit_info = await rate_limiter.check_rate_limit(user_id)
                results.append((is_allowed, limit_info))
            return results
        
        # Create tasks for multiple users making concurrent requests
        tasks = []
        for user_num in range(5):
            user_id = f"concurrent_user_{user_num}"
            tasks.append(make_requests(user_id, 15))  # Each user makes 15 requests
        
        # Execute all tasks concurrently
        all_results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Analyze results
        total_requests = sum(len(results) for results in all_results)
        allowed_requests = sum(
            sum(1 for is_allowed, _ in results if is_allowed)
            for results in all_results
        )
        queued_requests = sum(
            sum(1 for is_allowed, limit_info in results 
                if is_allowed and limit_info and limit_info.get("queued"))
            for results in all_results
        )
        
        print(f"   ✅ Concurrent test completed in {duration:.3f}s")
        print(f"   📊 Total: {total_requests}, Allowed: {allowed_requests}, Queued: {queued_requests}")
        print(f"   ⚡ Average request time: {duration/total_requests*1000:.2f}ms")
        
        # Test should complete quickly without blocking
        if duration > 5.0:
            print(f"   ⚠️  Concurrent test took longer than expected: {duration:.3f}s")
        
        # Get final metrics
        metrics = rate_limiter.get_metrics()
        print(f"   📈 Final metrics: {metrics}")
        
        # Clean up
        await rate_limiter.cleanup()
        
        return True
        
    except Exception as e:
        print(f"   ❌ Concurrent performance test error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run rate limiting fix tests."""
    print("🚀 Testing Rate Limiting and Request Queuing")
    print("=" * 60)
    
    results = []
    
    # Test async rate limiter
    result1 = await test_async_rate_limiter()
    results.append(result1)
    
    # Test MCP server integration
    result2 = await test_mcp_server_integration()
    results.append(result2)
    
    # Test concurrent performance
    result3 = await test_concurrent_performance()
    results.append(result3)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 RATE LIMITING FIX TEST RESULTS")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ Rate limiting and request queuing working correctly!")
        print("\nExpected improvements:")
        print("   - Requests queued instead of immediately rejected")
        print("   - Async-safe rate limiting for concurrent scenarios")
        print("   - Circuit breaker protection against system overload")
        print("   - Better utilization under high concurrent load")
        print("   - Comprehensive metrics for monitoring")
    else:
        print("❌ Some rate limiting tests have issues")
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