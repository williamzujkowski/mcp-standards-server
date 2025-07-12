#!/usr/bin/env python3
"""
Simple performance test for MCP Standards Server HTTP API.

Tests the actual available endpoints with response time measurements.
"""

import asyncio
import aiohttp
import time
import json
from statistics import mean, stdev
from typing import List, Dict, Any

async def test_endpoint(session: aiohttp.ClientSession, url: str, label: str, iterations: int = 10) -> Dict[str, Any]:
    """Test a single endpoint multiple times and collect metrics."""
    response_times = []
    errors = 0
    
    print(f"  Testing {label}...")
    
    for i in range(iterations):
        start_time = time.time()
        try:
            async with session.get(url) as response:
                await response.read()  # Consume response
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds
                response_times.append(response_time)
                
                if response.status != 200:
                    errors += 1
                    
        except Exception as e:
            errors += 1
            end_time = time.time()
            response_time = (end_time - start_time) * 1000
            response_times.append(response_time)
    
    # Calculate statistics
    avg_response = mean(response_times) if response_times else 0
    min_response = min(response_times) if response_times else 0
    max_response = max(response_times) if response_times else 0
    std_response = stdev(response_times) if len(response_times) > 1 else 0
    success_rate = ((iterations - errors) / iterations) * 100
    
    return {
        "label": label,
        "url": url,
        "iterations": iterations,
        "avg_response_ms": round(avg_response, 2),
        "min_response_ms": round(min_response, 2),
        "max_response_ms": round(max_response, 2),
        "std_response_ms": round(std_response, 2),
        "success_rate": round(success_rate, 2),
        "errors": errors
    }

async def test_concurrent_users(session: aiohttp.ClientSession, url: str, concurrent_users: int) -> Dict[str, Any]:
    """Test endpoint with concurrent users."""
    print(f"  Testing {concurrent_users} concurrent users...")
    
    async def single_request():
        start_time = time.time()
        try:
            async with session.get(url) as response:
                await response.read()
                end_time = time.time()
                return {
                    "response_time": (end_time - start_time) * 1000,
                    "status": response.status,
                    "success": response.status == 200
                }
        except Exception as e:
            end_time = time.time()
            return {
                "response_time": (end_time - start_time) * 1000,
                "status": 0,
                "success": False,
                "error": str(e)
            }
    
    # Run concurrent requests
    start_time = time.time()
    tasks = [single_request() for _ in range(concurrent_users)]
    results = await asyncio.gather(*tasks)
    total_time = (time.time() - start_time) * 1000
    
    # Calculate metrics
    response_times = [r["response_time"] for r in results]
    successful_requests = sum(1 for r in results if r["success"])
    
    return {
        "concurrent_users": concurrent_users,
        "total_requests": len(results),
        "successful_requests": successful_requests,
        "success_rate": round((successful_requests / len(results)) * 100, 2),
        "total_time_ms": round(total_time, 2),
        "avg_response_ms": round(mean(response_times), 2),
        "min_response_ms": round(min(response_times), 2),
        "max_response_ms": round(max(response_times), 2),
        "requests_per_second": round(len(results) / (total_time / 1000), 2)
    }

async def main():
    """Run performance tests."""
    base_url = "http://localhost:8000"
    
    print("🚀 Starting Simple Performance Test")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        
        # Test basic endpoints
        print("\n📊 Testing Basic Endpoints:")
        basic_tests = [
            (f"{base_url}/health", "Health Check"),
            (f"{base_url}/api/standards", "List All Standards"),
            (f"{base_url}/info", "Server Info"),
            (f"{base_url}/metrics", "Metrics Endpoint")
        ]
        
        basic_results = []
        for url, label in basic_tests:
            result = await test_endpoint(session, url, label)
            basic_results.append(result)
        
        # Test concurrent users
        print("\n🔄 Testing Concurrent Users:")
        concurrent_tests = []
        test_url = f"{base_url}/api/standards"
        
        for users in [1, 5, 10, 20]:
            result = await test_concurrent_users(session, test_url, users)
            concurrent_tests.append(result)
        
        # Generate report
        print("\n" + "=" * 70)
        print("📊 PERFORMANCE TEST RESULTS")
        print("=" * 70)
        
        print("\n🎯 Basic Endpoint Performance:")
        print(f"{'Endpoint':<25} {'Avg (ms)':<10} {'Min (ms)':<10} {'Max (ms)':<10} {'Success %':<10}")
        print("-" * 70)
        
        for result in basic_results:
            print(f"{result['label']:<25} {result['avg_response_ms']:<10} "
                  f"{result['min_response_ms']:<10} {result['max_response_ms']:<10} "
                  f"{result['success_rate']:<10}")
        
        print("\n🚀 Concurrent User Performance:")
        print(f"{'Users':<8} {'Total Req':<12} {'Success %':<12} {'Avg (ms)':<10} {'RPS':<8}")
        print("-" * 50)
        
        for result in concurrent_tests:
            print(f"{result['concurrent_users']:<8} {result['total_requests']:<12} "
                  f"{result['success_rate']:<12} {result['avg_response_ms']:<10} "
                  f"{result['requests_per_second']:<8}")
        
        # Performance targets check
        print("\n🎯 Performance vs Targets:")
        print(f"{'Metric':<30} {'Target':<12} {'Actual':<12} {'Status':<10}")
        print("-" * 64)
        
        targets = {
            "Health Check": 50,
            "List All Standards": 100,
            "Server Info": 50,
            "Metrics Endpoint": 100
        }
        
        for result in basic_results:
            target = targets.get(result['label'], 100)
            actual = result['avg_response_ms']
            status = "✅ PASS" if actual <= target else "❌ FAIL"
            print(f"{result['label']:<30} {target}ms{'':<7} {actual}ms{'':<7} {status}")
        
        # Save results
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "basic_endpoints": basic_results,
            "concurrent_tests": concurrent_tests,
            "summary": {
                "total_tests": len(basic_results) + len(concurrent_tests),
                "avg_response_time": round(mean([r['avg_response_ms'] for r in basic_results]), 2),
                "max_concurrent_users_tested": max([r['concurrent_users'] for r in concurrent_tests]),
                "overall_success_rate": round(mean([r['success_rate'] for r in basic_results + concurrent_tests]), 2)
            }
        }
        
        with open("simple_performance_results.json", "w") as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Results saved to: simple_performance_results.json")
        print(f"\n✅ Performance test completed!")
        
        # Summary
        avg_response = report_data['summary']['avg_response_time']
        max_users = report_data['summary']['max_concurrent_users_tested'] 
        success_rate = report_data['summary']['overall_success_rate']
        
        print(f"\n📈 Summary:")
        print(f"   Average Response Time: {avg_response}ms")
        print(f"   Max Concurrent Users Tested: {max_users}")
        print(f"   Overall Success Rate: {success_rate}%")

if __name__ == "__main__":
    asyncio.run(main())