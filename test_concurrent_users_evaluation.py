#!/usr/bin/env python3
"""
Comprehensive concurrent user test for MCP Standards Server.

This test simulates the original evaluation scenarios to validate that
the critical fixes have resolved the 0% concurrent user success rate issue.
"""

import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class TestResult:
    """Result of a single test operation."""
    operation: str
    user_id: str
    success: bool
    duration_ms: float
    error: Optional[str] = None
    response_size: int = 0


@dataclass
class TestScenario:
    """Configuration for a test scenario."""
    name: str
    concurrent_users: int
    operations_per_user: int
    operation_mix: Dict[str, float]  # operation -> probability
    duration_seconds: int = 60
    

class ConcurrentUserTester:
    """Comprehensive concurrent user testing for MCP Standards Server."""
    
    def __init__(self):
        self.server = None
        self.results: List[TestResult] = []
        self.start_time = 0
        
    async def setup(self):
        """Initialize the MCP server for testing."""
        from src.mcp_server import MCPStandardsServer
        
        # Configure server with appropriate settings for testing
        config = {
            "auth": {"enabled": False},
            "search": {"enabled": True},
            "rate_limit_max_requests": 1000,  # High limit for testing
            "rate_limit_window": 60,
            "rate_limit_enable_queuing": True,
        }
        
        self.server = MCPStandardsServer(config)
        await self.server._initialize_async_components()
        print("✅ Server initialized for testing")
    
    async def cleanup(self):
        """Clean up server resources."""
        if self.server:
            await self.server._cleanup_async_components()
    
    async def simulate_user(self, user_id: str, scenario: TestScenario):
        """Simulate a single user performing operations."""
        operations_completed = 0
        user_start_time = time.time()
        
        while operations_completed < scenario.operations_per_user:
            # Check if we've exceeded the scenario duration
            if time.time() - self.start_time > scenario.duration_seconds:
                break
            
            # Select operation based on probability distribution
            operation = self._select_operation(scenario.operation_mix)
            
            # Execute the operation
            result = await self._execute_operation(user_id, operation)
            self.results.append(result)
            
            operations_completed += 1
            
            # Small delay between operations (simulate think time)
            await asyncio.sleep(0.1)
        
        duration = time.time() - user_start_time
        print(f"   👤 User {user_id} completed {operations_completed} operations in {duration:.2f}s")
    
    def _select_operation(self, operation_mix: Dict[str, float]) -> str:
        """Select an operation based on probability distribution."""
        import random
        rand = random.random()
        cumulative = 0.0
        
        for operation, probability in operation_mix.items():
            cumulative += probability
            if rand <= cumulative:
                return operation
        
        # Fallback to first operation
        return list(operation_mix.keys())[0]
    
    async def _execute_operation(self, user_id: str, operation: str) -> TestResult:
        """Execute a single operation and measure performance."""
        start_time = time.time()
        success = False
        error = None
        response_size = 0
        
        try:
            if operation == "list_standards":
                response = await self.server._list_available_standards()
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "get_standard_details":
                # Get a known standard ID (or use a test standard)
                response = await self.server._get_standard_details("react-18-patterns")
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "search_standards":
                response = await self.server._search_standards(
                    query="security best practices",
                    limit=10,
                    min_relevance=0.5
                )
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "get_applicable_standards":
                context = {
                    "project_type": "web_application",
                    "framework": "react",
                    "requirements": ["security", "accessibility"]
                }
                response = await self.server._get_applicable_standards(context)
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "suggest_improvements":
                response = await self.server._suggest_improvements(
                    code="const password = '12345';",
                    context={"language": "javascript"}
                )
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "get_compliance_mapping":
                response = await self.server._get_compliance_mapping(
                    standard_id="security-best-practices",
                    framework="NIST-800-53"
                )
                response_size = len(json.dumps(response))
                success = True
                
            elif operation == "validate_against_standard":
                response = await self.server._validate_against_standard(
                    code="function getData() { return fetch('/api/data'); }",
                    standard="javascript-best-practices",
                    language="javascript"
                )
                response_size = len(json.dumps(response))
                success = True
                
            else:
                error = f"Unknown operation: {operation}"
                
        except Exception as e:
            error = str(e)
            success = False
        
        duration_ms = (time.time() - start_time) * 1000
        
        return TestResult(
            operation=operation,
            user_id=user_id,
            success=success,
            duration_ms=duration_ms,
            error=error,
            response_size=response_size
        )
    
    async def run_scenario(self, scenario: TestScenario) -> Dict[str, Any]:
        """Run a complete test scenario with multiple concurrent users."""
        print(f"\n🎯 Running scenario: {scenario.name}")
        print(f"   • Concurrent users: {scenario.concurrent_users}")
        print(f"   • Operations per user: {scenario.operations_per_user}")
        print(f"   • Duration: {scenario.duration_seconds}s")
        print(f"   • Operation mix: {scenario.operation_mix}")
        
        self.results = []
        self.start_time = time.time()
        
        # Create tasks for all concurrent users
        tasks = []
        for i in range(scenario.concurrent_users):
            user_id = f"user_{i+1}"
            task = asyncio.create_task(self.simulate_user(user_id, scenario))
            tasks.append(task)
        
        # Wait for all users to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results
        return self._analyze_results(scenario)
    
    def _analyze_results(self, scenario: TestScenario) -> Dict[str, Any]:
        """Analyze test results and generate metrics."""
        total_operations = len(self.results)
        successful_operations = sum(1 for r in self.results if r.success)
        failed_operations = total_operations - successful_operations
        
        # Calculate success rate
        success_rate = (successful_operations / total_operations * 100) if total_operations > 0 else 0
        
        # Calculate response time statistics
        response_times = [r.duration_ms for r in self.results if r.success]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = median_response_time = p95_response_time = p99_response_time = 0
            min_response_time = max_response_time = 0
        
        # Group results by operation
        operation_stats = defaultdict(lambda: {"success": 0, "failed": 0, "total": 0, "avg_ms": 0})
        
        for result in self.results:
            op = result.operation
            operation_stats[op]["total"] += 1
            if result.success:
                operation_stats[op]["success"] += 1
            else:
                operation_stats[op]["failed"] += 1
        
        # Calculate average response time per operation
        for op in operation_stats:
            op_times = [r.duration_ms for r in self.results if r.operation == op and r.success]
            if op_times:
                operation_stats[op]["avg_ms"] = statistics.mean(op_times)
        
        # Collect error messages
        errors = defaultdict(int)
        for result in self.results:
            if not result.success and result.error:
                errors[result.error] += 1
        
        # Calculate throughput
        total_duration = time.time() - self.start_time
        throughput_ops_per_sec = total_operations / total_duration if total_duration > 0 else 0
        
        return {
            "scenario": scenario.name,
            "concurrent_users": scenario.concurrent_users,
            "total_operations": total_operations,
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "success_rate": success_rate,
            "response_times": {
                "average_ms": avg_response_time,
                "median_ms": median_response_time,
                "p95_ms": p95_response_time,
                "p99_ms": p99_response_time,
                "min_ms": min_response_time,
                "max_ms": max_response_time,
            },
            "operation_stats": dict(operation_stats),
            "errors": dict(errors),
            "throughput_ops_per_sec": throughput_ops_per_sec,
            "total_duration_seconds": total_duration,
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted test results."""
        print(f"\n📊 Results for: {results['scenario']}")
        print("=" * 80)
        
        # Overall metrics
        print(f"\n🎯 Overall Performance:")
        print(f"   • Total operations: {results['total_operations']}")
        print(f"   • Successful: {results['successful_operations']}")
        print(f"   • Failed: {results['failed_operations']}")
        print(f"   • Success rate: {results['success_rate']:.2f}%")
        print(f"   • Throughput: {results['throughput_ops_per_sec']:.2f} ops/sec")
        
        # Response time metrics
        rt = results['response_times']
        print(f"\n⏱️  Response Times:")
        print(f"   • Average: {rt['average_ms']:.2f}ms")
        print(f"   • Median: {rt['median_ms']:.2f}ms")
        print(f"   • 95th percentile: {rt['p95_ms']:.2f}ms")
        print(f"   • 99th percentile: {rt['p99_ms']:.2f}ms")
        print(f"   • Min: {rt['min_ms']:.2f}ms")
        print(f"   • Max: {rt['max_ms']:.2f}ms")
        
        # Per-operation statistics
        print(f"\n📈 Per-Operation Statistics:")
        for op, stats in results['operation_stats'].items():
            success_rate = (stats['success'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {op}:")
            print(f"      • Success rate: {success_rate:.2f}% ({stats['success']}/{stats['total']})")
            print(f"      • Avg response: {stats['avg_ms']:.2f}ms")
        
        # Errors
        if results['errors']:
            print(f"\n❌ Errors:")
            for error, count in results['errors'].items():
                print(f"   • {error}: {count} occurrences")
        
        # Success/Failure determination
        print(f"\n🏁 Test Result: ", end="")
        if results['success_rate'] >= 95:
            print("✅ PASSED (≥95% success rate)")
        else:
            print(f"❌ FAILED ({results['success_rate']:.2f}% < 95% required)")


async def main():
    """Run comprehensive concurrent user tests."""
    print("🚀 MCP Standards Server - Concurrent User Evaluation")
    print("=" * 80)
    print("Testing if critical fixes have resolved the 0% concurrent user success rate")
    
    tester = ConcurrentUserTester()
    
    try:
        # Setup
        await tester.setup()
        
        # Define test scenarios matching the original evaluation
        scenarios = [
            TestScenario(
                name="Light Load (5 concurrent users)",
                concurrent_users=5,
                operations_per_user=20,
                operation_mix={
                    "list_standards": 0.3,
                    "get_standard_details": 0.2,
                    "search_standards": 0.2,
                    "get_applicable_standards": 0.15,
                    "suggest_improvements": 0.1,
                    "validate_against_standard": 0.05,
                },
                duration_seconds=30
            ),
            TestScenario(
                name="Moderate Load (10 concurrent users)",
                concurrent_users=10,
                operations_per_user=20,
                operation_mix={
                    "list_standards": 0.25,
                    "search_standards": 0.25,
                    "get_standard_details": 0.2,
                    "get_applicable_standards": 0.15,
                    "suggest_improvements": 0.1,
                    "validate_against_standard": 0.05,
                },
                duration_seconds=30
            ),
            TestScenario(
                name="Heavy Load (20 concurrent users)",
                concurrent_users=20,
                operations_per_user=15,
                operation_mix={
                    "list_standards": 0.3,
                    "search_standards": 0.3,
                    "get_standard_details": 0.2,
                    "get_applicable_standards": 0.1,
                    "suggest_improvements": 0.05,
                    "validate_against_standard": 0.05,
                },
                duration_seconds=30
            ),
            TestScenario(
                name="Stress Test (50 concurrent users)",
                concurrent_users=50,
                operations_per_user=10,
                operation_mix={
                    "list_standards": 0.4,
                    "search_standards": 0.3,
                    "get_standard_details": 0.2,
                    "get_applicable_standards": 0.1,
                },
                duration_seconds=30
            ),
        ]
        
        # Run all scenarios
        all_results = []
        for scenario in scenarios:
            results = await tester.run_scenario(scenario)
            tester.print_results(results)
            all_results.append(results)
            
            # Brief pause between scenarios
            await asyncio.sleep(2)
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 OVERALL SUMMARY")
        print("=" * 80)
        
        all_passed = True
        for result in all_results:
            status = "✅ PASSED" if result['success_rate'] >= 95 else "❌ FAILED"
            print(f"{result['scenario']}: {result['success_rate']:.2f}% success rate - {status}")
            if result['success_rate'] < 95:
                all_passed = False
        
        print(f"\n🏁 Overall Result: ", end="")
        if all_passed:
            print("✅ ALL SCENARIOS PASSED!")
            print("\n🎉 The critical fixes have successfully resolved the concurrent user issues!")
            print("   The system now handles concurrent load effectively.")
        else:
            print("❌ SOME SCENARIOS FAILED")
            print("\n⚠️  Additional investigation needed for failed scenarios.")
        
        # Performance comparison with targets
        print("\n📈 Performance vs Targets:")
        for result in all_results:
            avg_ms = result['response_times']['average_ms']
            print(f"\n{result['scenario']}:")
            print(f"   • Average response: {avg_ms:.0f}ms")
            
            # Check specific operation targets
            ops = result['operation_stats']
            if 'list_standards' in ops:
                list_ms = ops['list_standards']['avg_ms']
                target_met = "✅" if list_ms < 100 else "❌"
                print(f"   • List standards: {list_ms:.0f}ms (target <100ms) {target_met}")
            
            if 'search_standards' in ops:
                search_ms = ops['search_standards']['avg_ms']
                target_met = "✅" if search_ms < 500 else "❌"
                print(f"   • Search standards: {search_ms:.0f}ms (target <500ms) {target_met}")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        await tester.cleanup()


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