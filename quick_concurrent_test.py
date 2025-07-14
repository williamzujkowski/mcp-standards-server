#!/usr/bin/env python3
"""
Quick concurrent user test - simpler version to avoid timeouts.
"""

import asyncio
import json
import time
from statistics import mean

import aiohttp


async def test_users(users: int, requests_each: int = 1):
    """Test specific number of concurrent users."""
    print(f"\n🔄 Testing {users} concurrent users...")

    async def single_request(session, user_id):
        start = time.time()
        try:
            async with session.get("http://localhost:8000/api/standards") as response:
                await response.read()
                return {
                    "user": user_id,
                    "time": (time.time() - start) * 1000,
                    "status": response.status,
                    "success": response.status == 200,
                }
        except Exception as e:
            return {
                "user": user_id,
                "time": (time.time() - start) * 1000,
                "status": 0,
                "success": False,
                "error": str(e),
            }

    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=users + 10),
        timeout=aiohttp.ClientTimeout(total=30),
    ) as session:

        start_time = time.time()

        # Create tasks for concurrent users
        tasks = []
        for user_id in range(users):
            for _ in range(requests_each):
                tasks.append(single_request(session, user_id))

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Process results
        valid_results = [r for r in results if isinstance(r, dict)]
        successful = [r for r in valid_results if r.get("success", False)]

        if valid_results:
            avg_response = mean([r["time"] for r in valid_results])
            success_rate = len(successful) / len(valid_results) * 100
            rps = len(valid_results) / total_time

            print(f"   ⏱️  Total time: {total_time:.2f}s")
            print(f"   📊 Requests: {len(valid_results)}")
            print(f"   ✅ Success rate: {success_rate:.1f}%")
            print(f"   📈 Avg response: {avg_response:.1f}ms")
            print(f"   🚀 RPS: {rps:.2f}")

            return {
                "users": users,
                "total_time": total_time,
                "requests": len(valid_results),
                "successful": len(successful),
                "success_rate": success_rate,
                "avg_response_ms": avg_response,
                "rps": rps,
            }
        else:
            print("   ❌ All requests failed")
            return {"users": users, "success_rate": 0}


async def main():
    """Run quick concurrent tests."""
    print("🚀 Quick Concurrent User Test")
    print("=" * 40)

    results = []

    # Test progressively
    for users in [10, 50]:  # Start with smaller numbers
        try:
            result = await test_users(users, 1)  # 1 request per user
            results.append(result)

            # If success rate drops below 80%, stop testing higher loads
            if result.get("success_rate", 0) < 80:
                print(
                    f"\n⚠️  Success rate dropped to {result['success_rate']:.1f}% at {users} users"
                )
                break

            await asyncio.sleep(2)  # Cool down

        except Exception as e:
            print(f"❌ Test with {users} users failed: {e}")
            break

    # If we successfully tested 50 users, try 100
    if len(results) >= 2 and results[-1].get("success_rate", 0) >= 80:
        try:
            print("\n📈 Testing 100 users (this may take longer)...")
            result = await test_users(100, 1)
            results.append(result)
        except Exception as e:
            print(f"❌ Test with 100 users failed: {e}")

    # Summary
    print("\n📊 Summary:")
    for result in results:
        if "success_rate" in result and result["success_rate"] > 0:
            print(
                f"   {result['users']} users: {result['success_rate']:.1f}% success, {result['avg_response_ms']:.1f}ms avg"
            )
        else:
            print(f"   {result['users']} users: FAILED")

    # Save results
    with open("quick_concurrent_results.json", "w") as f:
        json.dump(
            {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results},
            f,
            indent=2,
        )

    print("\n💾 Results saved to quick_concurrent_results.json")


if __name__ == "__main__":
    asyncio.run(main())
