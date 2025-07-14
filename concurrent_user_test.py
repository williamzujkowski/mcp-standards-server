#!/usr/bin/env python3
"""
Concurrent User Testing for MCP Standards Server.

Tests specific concurrent user loads: 10, 50, 100 users as required by evaluation plan.
"""

import asyncio
import json
import sys
import time
from statistics import mean, median, stdev
from typing import Any

import aiohttp


async def single_request(
    session: aiohttp.ClientSession, url: str, user_id: int
) -> dict[str, Any]:
    """Make a single request and record metrics."""
    start_time = time.time()
    try:
        async with session.get(url) as response:
            content = await response.read()
            end_time = time.time()

            return {
                "user_id": user_id,
                "response_time": (end_time - start_time) * 1000,  # ms
                "status": response.status,
                "success": response.status == 200,
                "content_size": len(content),
                "timestamp": start_time,
            }
    except Exception as e:
        end_time = time.time()
        return {
            "user_id": user_id,
            "response_time": (end_time - start_time) * 1000,
            "status": 0,
            "success": False,
            "error": str(e),
            "content_size": 0,
            "timestamp": start_time,
        }


async def test_concurrent_users(
    base_url: str, concurrent_users: int, requests_per_user: int = 3
) -> dict[str, Any]:
    """Test with specified number of concurrent users."""
    print(f"\n🚀 Testing {concurrent_users} Concurrent Users")
    print("=" * 60)

    test_url = f"{base_url}/api/standards"

    # Create connector with appropriate limits
    connector = aiohttp.TCPConnector(
        limit=concurrent_users * 2,  # Total connection pool size
        limit_per_host=concurrent_users * 2,  # Per-host limit
        ttl_dns_cache=300,
        use_dns_cache=True,
    )

    async with aiohttp.ClientSession(
        connector=connector,
        timeout=aiohttp.ClientTimeout(total=60),  # 60 second timeout
    ) as session:

        # Prepare all tasks
        tasks = []
        for user in range(concurrent_users):
            for _request in range(requests_per_user):
                task = single_request(session, test_url, user)
                tasks.append(task)

        print(
            f"  Launching {len(tasks)} total requests ({concurrent_users} users × {requests_per_user} requests each)..."
        )

        # Execute all requests concurrently
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_duration = time.time() - start_time

        # Process results
        valid_results = []
        errors = []

        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
            else:
                valid_results.append(result)

        # Calculate metrics
        if valid_results:
            response_times = [r["response_time"] for r in valid_results]
            successful_requests = [r for r in valid_results if r["success"]]
            failed_requests = [r for r in valid_results if not r["success"]]

            # Response time statistics
            avg_response = mean(response_times)
            min_response = min(response_times)
            max_response = max(response_times)
            median_response = median(response_times)
            p95_response = (
                sorted(response_times)[int(len(response_times) * 0.95)]
                if len(response_times) > 20
                else max_response
            )
            p99_response = (
                sorted(response_times)[int(len(response_times) * 0.99)]
                if len(response_times) > 100
                else max_response
            )

            # Throughput metrics
            total_requests = len(valid_results)
            successful_count = len(successful_requests)
            failed_count = len(failed_requests)
            success_rate = (
                (successful_count / total_requests) * 100 if total_requests > 0 else 0
            )
            requests_per_second = total_requests / total_duration

            metrics = {
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "total_duration_seconds": round(total_duration, 2),
                "total_requests": total_requests,
                "successful_requests": successful_count,
                "failed_requests": failed_count,
                "exception_count": len(errors),
                "success_rate_percent": round(success_rate, 2),
                "requests_per_second": round(requests_per_second, 2),
                "response_times": {
                    "avg_ms": round(avg_response, 2),
                    "min_ms": round(min_response, 2),
                    "max_ms": round(max_response, 2),
                    "median_ms": round(median_response, 2),
                    "p95_ms": round(p95_response, 2),
                    "p99_ms": round(p99_response, 2),
                    "std_dev": round(
                        stdev(response_times) if len(response_times) > 1 else 0, 2
                    ),
                },
            }
        else:
            metrics = {
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "total_duration_seconds": round(total_duration, 2),
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "exception_count": len(errors),
                "success_rate_percent": 0,
                "requests_per_second": 0,
                "response_times": {
                    "avg_ms": 0,
                    "min_ms": 0,
                    "max_ms": 0,
                    "median_ms": 0,
                    "p95_ms": 0,
                    "p99_ms": 0,
                    "std_dev": 0,
                },
                "errors": errors[:10],  # Include first 10 errors for debugging
            }

        # Print results
        print(f"  ⏱️  Total Duration: {metrics['total_duration_seconds']}s")
        print(f"  📊 Total Requests: {metrics['total_requests']}")
        print(
            f"  ✅ Successful: {metrics['successful_requests']} ({metrics['success_rate_percent']}%)"
        )
        print(f"  ❌ Failed: {metrics['failed_requests']}")
        if metrics["exception_count"] > 0:
            print(f"  🚨 Exceptions: {metrics['exception_count']}")
        print(f"  🚀 RPS: {metrics['requests_per_second']}")
        print("  📈 Response Times:")
        print(f"     Avg: {metrics['response_times']['avg_ms']}ms")
        print(f"     Min: {metrics['response_times']['min_ms']}ms")
        print(f"     Max: {metrics['response_times']['max_ms']}ms")
        print(f"     P95: {metrics['response_times']['p95_ms']}ms")
        print(f"     P99: {metrics['response_times']['p99_ms']}ms")

        return metrics


async def main():
    """Run concurrent user tests."""
    base_url = "http://localhost:8000"

    print("🔄 CONCURRENT USER TESTING")
    print("=" * 70)
    print("Testing MCP Standards Server under concurrent load...")

    # Test different concurrent user levels
    test_levels = [10, 50, 100]
    all_results = []

    for users in test_levels:
        try:
            result = await test_concurrent_users(base_url, users)
            all_results.append(result)

            # Brief pause between tests
            print("\n💤 Cooling down for 3 seconds...")
            await asyncio.sleep(3)

        except Exception as e:
            print(f"❌ Test with {users} users failed: {e}")
            all_results.append(
                {"concurrent_users": users, "error": str(e), "success_rate_percent": 0}
            )

    # Generate summary report
    print("\n" + "=" * 80)
    print("📊 CONCURRENT USER TEST SUMMARY")
    print("=" * 80)

    print(
        f"\n{'Users':<8} {'Success %':<12} {'Avg (ms)':<12} {'P95 (ms)':<12} {'RPS':<8} {'Status'}"
    )
    print("-" * 72)

    for result in all_results:
        if "error" in result:
            print(
                f"{result['concurrent_users']:<8} {'0.0':<12} {'N/A':<12} {'N/A':<12} {'0.0':<8} ❌ ERROR"
            )
        else:
            users = result["concurrent_users"]
            success_rate = result["success_rate_percent"]
            avg_ms = result["response_times"]["avg_ms"]
            p95_ms = result["response_times"]["p95_ms"]
            rps = result["requests_per_second"]

            # Determine status
            if success_rate >= 95 and avg_ms <= 1000:
                status = "✅ GOOD"
            elif success_rate >= 80 and avg_ms <= 5000:
                status = "⚠️  WARN"
            else:
                status = "❌ POOR"

            print(
                f"{users:<8} {success_rate:<12} {avg_ms:<12} {p95_ms:<12} {rps:<8} {status}"
            )

    # Performance analysis
    print("\n🔍 Performance Analysis:")

    successful_tests = [
        r for r in all_results if "error" not in r and r["success_rate_percent"] >= 80
    ]

    if successful_tests:
        max_successful_users = max(r["concurrent_users"] for r in successful_tests)
        print(
            f"   Maximum concurrent users handled successfully: {max_successful_users}"
        )

        # Find performance degradation point
        if len(successful_tests) >= 2:
            degradation_factor = 2.0  # 2x response time increase
            baseline = successful_tests[0]["response_times"]["avg_ms"]
            degradation_point = None

            for result in successful_tests[1:]:
                if result["response_times"]["avg_ms"] > baseline * degradation_factor:
                    degradation_point = result["concurrent_users"]
                    break

            if degradation_point:
                print(
                    f"   Performance degradation starts at: {degradation_point} users"
                )
            else:
                print("   No significant performance degradation observed")

        # Throughput analysis
        max_rps = max(r["requests_per_second"] for r in successful_tests)
        max_rps_users = next(
            r["concurrent_users"]
            for r in successful_tests
            if r["requests_per_second"] == max_rps
        )
        print(f"   Peak throughput: {max_rps} RPS at {max_rps_users} users")

    else:
        print("   ❌ Server cannot handle concurrent users effectively")

    # Recommendations
    print("\n💡 Recommendations:")
    if len(successful_tests) == len(all_results):
        print("   ✅ Server handles all tested concurrent loads")
        print("   📈 Consider testing higher user counts")
    elif successful_tests:
        max_users = max(r["concurrent_users"] for r in successful_tests)
        print(f"   ⚠️  Server degrades significantly above {max_users} users")
        print("   🔧 Consider optimizing for higher concurrent loads")
    else:
        print("   🚨 Critical: Server cannot handle concurrent users")
        print("   🛠️  Immediate optimization required")

    # Save results
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "concurrent_users",
        "test_levels": test_levels,
        "results": all_results,
        "summary": {
            "total_tests": len(all_results),
            "successful_tests": len(successful_tests),
            "max_concurrent_users": (
                max([r["concurrent_users"] for r in successful_tests])
                if successful_tests
                else 0
            ),
            "peak_rps": (
                max([r["requests_per_second"] for r in successful_tests])
                if successful_tests
                else 0
            ),
        },
    }

    with open("concurrent_user_test_results.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n📄 Results saved to: concurrent_user_test_results.json")
    print("\n✅ Concurrent user testing completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
