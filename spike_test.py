#!/usr/bin/env python3
"""
Spike Testing for MCP Standards Server.

Tests incremental user loads to find the exact breaking point.
Given that 10 concurrent users already fail, we'll test smaller increments.
"""

import asyncio
import json
import time
from statistics import mean
from typing import Any

import aiohttp


async def spike_test_single_level(
    users: int, timeout_seconds: int = 30
) -> dict[str, Any]:
    """Test a single user level with timeout protection."""
    print(f"  📊 Testing {users} users...")

    async def single_request(session, user_id):
        start = time.time()
        try:
            async with session.get("http://localhost:8000/api/standards") as response:
                await response.read()
                return {
                    "user": user_id,
                    "response_time": (time.time() - start) * 1000,
                    "status": response.status,
                    "success": response.status == 200,
                }
        except Exception as e:
            return {
                "user": user_id,
                "response_time": (time.time() - start) * 1000,
                "status": 0,
                "success": False,
                "error": str(e),
            }

    try:
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=users + 5),
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as session:

            start_time = time.time()

            # Create tasks
            tasks = [single_request(session, i) for i in range(users)]

            # Execute with timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_seconds
            )

            total_time = time.time() - start_time

            # Process results
            valid_results = [r for r in results if isinstance(r, dict)]
            successful = [r for r in valid_results if r.get("success", False)]

            if valid_results:
                response_times = [r["response_time"] for r in valid_results]
                avg_response = mean(response_times)
                max_response = max(response_times)
                min_response = min(response_times)
                success_rate = len(successful) / len(valid_results) * 100
                rps = len(valid_results) / total_time if total_time > 0 else 0

                result = {
                    "users": users,
                    "total_time": round(total_time, 2),
                    "total_requests": len(valid_results),
                    "successful_requests": len(successful),
                    "success_rate": round(success_rate, 1),
                    "avg_response_ms": round(avg_response, 1),
                    "min_response_ms": round(min_response, 1),
                    "max_response_ms": round(max_response, 1),
                    "requests_per_second": round(rps, 2),
                    "status": "completed",
                }
            else:
                result = {
                    "users": users,
                    "total_time": round(total_time, 2),
                    "total_requests": 0,
                    "successful_requests": 0,
                    "success_rate": 0,
                    "avg_response_ms": 0,
                    "requests_per_second": 0,
                    "status": "failed",
                    "error": "No valid results",
                }

    except asyncio.TimeoutError:
        result = {
            "users": users,
            "status": "timeout",
            "success_rate": 0,
            "error": f"Test timed out after {timeout_seconds}s",
        }
    except Exception as e:
        result = {"users": users, "status": "error", "success_rate": 0, "error": str(e)}

    # Print result
    if result["status"] == "completed":
        print(
            f"     ✅ {result['success_rate']}% success, {result['avg_response_ms']}ms avg, {result['requests_per_second']} RPS"
        )
    elif result["status"] == "timeout":
        print(f"     ⏰ TIMEOUT after {timeout_seconds}s")
    else:
        print(f"     ❌ FAILED: {result.get('error', 'Unknown error')}")

    return result


async def main():
    """Run spike testing."""
    print("⚡ SPIKE TESTING - Finding Performance Breaking Point")
    print("=" * 60)
    print("Testing incremental user loads to identify server limits...")

    # Spike test levels - start small since we know 10 users fail
    spike_levels = [1, 2, 3, 5, 8, 10, 15, 20]

    results = []
    breaking_point = None
    last_successful_level = None

    for users in spike_levels:
        print(f"\n🎯 Spike Level: {users} concurrent users")

        result = await spike_test_single_level(users, timeout_seconds=45)
        results.append(result)

        # Check if this level is successful
        if result.get("success_rate", 0) >= 80:
            last_successful_level = users
            print(f"     🟢 Level {users} passed")
        else:
            if breaking_point is None:
                breaking_point = users
            print(f"     🔴 Level {users} failed - performance degraded")

            # If we hit 3 consecutive failures or very low success rate, stop testing higher levels
            recent_failures = sum(
                1 for r in results[-3:] if r.get("success_rate", 0) < 50
            )
            if recent_failures >= 2 or result.get("success_rate", 0) < 20:
                print("     🛑 Stopping spike test - consistent failures detected")
                break

        # Cool down between tests
        if users < max(spike_levels):
            print("     💤 Cooling down for 3 seconds...")
            await asyncio.sleep(3)

    # Analysis
    print("\n" + "=" * 60)
    print("📊 SPIKE TEST ANALYSIS")
    print("=" * 60)

    print("\n🎯 Test Summary:")
    for result in results:
        status_icon = "✅" if result.get("success_rate", 0) >= 80 else "❌"
        if result["status"] == "completed":
            print(
                f"   {status_icon} {result['users']} users: {result['success_rate']}% success, {result['avg_response_ms']}ms avg"
            )
        else:
            print(f"   ❌ {result['users']} users: {result['status'].upper()}")

    # Determine server characteristics
    print("\n🔍 Server Characteristics:")

    if last_successful_level:
        print(f"   📈 Maximum successful concurrent users: {last_successful_level}")
    else:
        print("   🚨 Server cannot handle ANY concurrent load effectively")

    if breaking_point:
        print(f"   📉 Performance breaking point: {breaking_point} users")
    else:
        print("   ⚠️  Breaking point not clearly identified")

    # Performance pattern analysis
    successful_results = [
        r
        for r in results
        if r.get("success_rate", 0) >= 80 and r.get("avg_response_ms", 0) > 0
    ]

    if len(successful_results) >= 2:
        baseline_response = successful_results[0]["avg_response_ms"]
        degradation_detected = False

        for result in successful_results[1:]:
            if result["avg_response_ms"] > baseline_response * 2:  # 2x increase
                print(
                    f"   📊 Significant response time degradation at {result['users']} users"
                )
                degradation_detected = True
                break

        if not degradation_detected:
            print("   📊 Response times remain stable across successful test levels")

    # Recommendations
    print("\n💡 Spike Test Recommendations:")

    if last_successful_level and last_successful_level >= 5:
        print(
            f"   ✅ Server can handle light concurrent loads ({last_successful_level} users)"
        )
        print("   📈 Consider optimizing for higher concurrent loads")
    elif last_successful_level and last_successful_level >= 2:
        print(
            f"   ⚠️  Server has limited concurrency ({last_successful_level} users max)"
        )
        print("   🔧 Significant optimization needed for production use")
    else:
        print("   🚨 Critical: Server cannot handle concurrent users effectively")
        print("   🛠️  Major architectural changes required")
        print(
            "   📋 Consider: async processing, connection pooling, caching, load balancing"
        )

    # Overall spike test verdict
    print("\n🏁 Spike Test Verdict:")
    if last_successful_level and last_successful_level >= 10:
        verdict = "✅ ACCEPTABLE - Server handles moderate concurrent loads"
    elif last_successful_level and last_successful_level >= 5:
        verdict = "⚠️  CONCERNING - Limited concurrency support"
    else:
        verdict = "❌ CRITICAL - Poor concurrency performance"

    print(f"   {verdict}")

    # Save results
    report_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_type": "spike_test",
        "spike_levels": spike_levels,
        "results": results,
        "analysis": {
            "last_successful_level": last_successful_level,
            "breaking_point": breaking_point,
            "total_levels_tested": len(results),
            "successful_levels": len(
                [r for r in results if r.get("success_rate", 0) >= 80]
            ),
            "verdict": verdict,
        },
    }

    with open("spike_test_results.json", "w") as f:
        json.dump(report_data, f, indent=2)

    print("\n📄 Results saved to: spike_test_results.json")
    print("✅ Spike testing completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n❌ Spike test interrupted by user")
    except Exception as e:
        print(f"\n❌ Spike test failed: {e}")
        import traceback

        traceback.print_exc()
