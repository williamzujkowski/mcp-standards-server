#!/usr/bin/env python3
"""
Test script for NIST compliance mapping functionality.
Tests the get_compliance_mapping MCP tool with various scenarios.
"""

import asyncio
import json
import time
from typing import Any

from src.core.mcp.handlers import StandardsHandler

# Import the MCP server components directly
from src.core.standards.engine import StandardsEngine


class ComplianceMappingTester:
    """Test the compliance mapping functionality."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []
        self.test_cases = [
            {
                "name": "Test Case 1 - Security Standards",
                "args": {
                    "standard_ids": ["security-review-audit-process"],
                    "framework": "nist-800-53"
                }
            },
            {
                "name": "Test Case 2 - Data Privacy Standards",
                "args": {
                    "standard_ids": ["data-privacy-compliance"],
                    "framework": "nist-800-53"
                }
            },
            {
                "name": "Test Case 3 - Multiple Standards",
                "args": {
                    "standard_ids": ["security-review-audit-process", "data-privacy-compliance", "authentication-authorization"],
                    "framework": "nist-800-53"
                }
            },
            {
                "name": "Test Case 4 - Technology Standards",
                "args": {
                    "standard_ids": ["react-18-patterns", "typescript-5-guidelines"],
                    "framework": "nist-800-53"
                }
            },
            {
                "name": "Test Case 5 - Invalid Framework Test",
                "args": {
                    "standard_ids": ["security-review-audit-process"],
                    "framework": "iso-27001"
                }
            },
            {
                "name": "Test Case 6 - Empty Standards List",
                "args": {
                    "standard_ids": [],
                    "framework": "nist-800-53"
                }
            }
        ]

    async def setup(self):
        """Initialize the standards engine and handler."""
        print("Setting up standards engine...")

        # Initialize the standards engine with data directory
        data_dir = "/home/william/git/mcp-standards-server/data/standards"
        self.standards_engine = StandardsEngine(data_dir=data_dir)
        await self.standards_engine.initialize()

        # Initialize the handler
        self.handler = StandardsHandler(self.standards_engine)
        await self.handler.initialize()

        print("✅ Setup complete")

    async def test_case(self, test_case: dict[str, Any]) -> dict[str, Any]:
        """Run a single test case."""
        print(f"\n🧪 Running {test_case['name']}")

        start_time = time.time()

        try:
            # Call the get_compliance_mapping handler directly
            result = await self.handler.handle_tool("get_compliance_mapping", test_case["args"])

            end_time = time.time()
            response_time = end_time - start_time

            test_result = {
                "test_case": test_case["name"],
                "args": test_case["args"],
                "success": True,
                "response_time_ms": round(response_time * 1000, 2),
                "result": result,
                "error": None
            }

            # Analyze the result
            if result and "result" in result:
                mappings = result["result"]
                test_result["mapping_count"] = len(mappings)
                test_result["control_families"] = list({m.get("control_id", "").split("-")[0] for m in mappings if m.get("control_id")})
                test_result["standards_covered"] = list({m.get("standard_id") for m in mappings})

                print(f"   📊 Found {len(mappings)} NIST control mappings")
                print(f"   ⏱️  Response time: {response_time*1000:.2f}ms")

                # Show some sample mappings
                for i, mapping in enumerate(mappings[:3]):
                    print(f"   📋 Mapping {i+1}: {mapping.get('standard_id')} → {mapping.get('control_id')}")

                if len(mappings) > 3:
                    print(f"   📋 ... and {len(mappings) - 3} more mappings")

            elif result and "error" in result:
                test_result["success"] = False
                test_result["error"] = result["error"]
                print(f"   ❌ Error: {result['error']}")
            else:
                test_result["success"] = False
                test_result["error"] = "No result returned"
                print("   ❌ No result returned")

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time

            test_result = {
                "test_case": test_case["name"],
                "args": test_case["args"],
                "success": False,
                "response_time_ms": round(response_time * 1000, 2),
                "result": None,
                "error": str(e)
            }

            print(f"   ❌ Exception: {e}")

        return test_result

    async def test_all_available_standards(self):
        """Test getting mappings for all available standards."""
        print("\n🧪 Running Test Case: All Available Standards")

        try:
            # First get all available standards
            standards_result = await self.handler.handle_tool("list_available_standards", {})

            if standards_result and "result" in standards_result:
                all_standards = standards_result["result"]
                standard_ids = [s.get("id") for s in all_standards if s.get("id")]

                print(f"   📚 Testing {len(standard_ids)} standards for NIST mappings")

                # Test compliance mapping for all standards
                start_time = time.time()
                result = await self.handler.handle_tool("get_compliance_mapping", {
                    "standard_ids": standard_ids,
                    "framework": "nist-800-53"
                })
                end_time = time.time()

                test_result = {
                    "test_case": "All Available Standards",
                    "args": {"standard_ids": "all", "framework": "nist-800-53"},
                    "success": True,
                    "response_time_ms": round((end_time - start_time) * 1000, 2),
                    "result": result,
                    "standards_tested": len(standard_ids),
                    "standards_list": standard_ids[:10]  # First 10 for brevity
                }

                if result and "result" in result:
                    mappings = result["result"]
                    test_result["mapping_count"] = len(mappings)
                    test_result["control_families"] = list({m.get("control_id", "").split("-")[0] for m in mappings if m.get("control_id")})
                    test_result["standards_with_mappings"] = list({m.get("standard_id") for m in mappings})

                    print(f"   📊 Found {len(mappings)} total NIST control mappings")
                    print(f"   📋 Standards with mappings: {len(test_result['standards_with_mappings'])}")
                    print(f"   🏷️  Control families covered: {test_result['control_families']}")
                    print(f"   ⏱️  Response time: {(end_time - start_time)*1000:.2f}ms")

                return test_result

        except Exception as e:
            return {
                "test_case": "All Available Standards",
                "success": False,
                "error": str(e)
            }

    async def run_all_tests(self):
        """Run all test cases."""
        print("🚀 Starting NIST Compliance Mapping Tests")
        print("=" * 60)

        await self.setup()

        # Run individual test cases
        for test_case in self.test_cases:
            result = await self.test_case(test_case)
            self.results.append(result)

        # Run test for all available standards
        all_standards_result = await self.test_all_available_standards()
        if all_standards_result:
            self.results.append(all_standards_result)

        await self.generate_report()

    async def generate_report(self):
        """Generate a comprehensive test report."""
        print("\n" + "=" * 60)
        print("📊 COMPLIANCE MAPPING TEST REPORT")
        print("=" * 60)

        successful_tests = [r for r in self.results if r.get("success", False)]
        failed_tests = [r for r in self.results if not r.get("success", False)]

        print(f"✅ Successful tests: {len(successful_tests)}")
        print(f"❌ Failed tests: {len(failed_tests)}")
        print(f"📈 Success rate: {len(successful_tests)/len(self.results)*100:.1f}%")

        # Response time analysis
        response_times = [r.get("response_time_ms", 0) for r in successful_tests]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            min_response_time = min(response_times)

            print("\n⏱️  Response Time Analysis:")
            print(f"   • Average: {avg_response_time:.2f}ms")
            print(f"   • Minimum: {min_response_time:.2f}ms")
            print(f"   • Maximum: {max_response_time:.2f}ms")

        # Mapping coverage analysis
        total_mappings = 0
        all_control_families = set()
        all_standards_with_mappings = set()

        for result in successful_tests:
            if result.get("mapping_count"):
                total_mappings += result["mapping_count"]
            if result.get("control_families"):
                all_control_families.update(result["control_families"])
            if result.get("standards_with_mappings"):
                all_standards_with_mappings.update(result["standards_with_mappings"])

        print("\n📋 Mapping Coverage Analysis:")
        print(f"   • Total mappings found: {total_mappings}")
        print(f"   • NIST control families covered: {len(all_control_families)}")
        print(f"   • Control families: {sorted(all_control_families)}")
        print(f"   • Standards with mappings: {len(all_standards_with_mappings)}")

        # Detailed results
        print("\n📝 Detailed Test Results:")
        for result in self.results:
            status = "✅" if result.get("success") else "❌"
            print(f"{status} {result['test_case']}")
            if result.get("error"):
                print(f"     Error: {result['error']}")
            elif result.get("mapping_count") is not None:
                print(f"     Mappings: {result['mapping_count']}, Time: {result.get('response_time_ms', 0):.2f}ms")

        # Recommendations
        print("\n💡 Recommendations:")

        if failed_tests:
            print("   • Investigate failed test cases for potential data or implementation issues")

        if total_mappings == 0:
            print("   • No NIST control mappings found - verify standards have compliance metadata")

        if len(all_control_families) < 5:
            print("   • Limited NIST control family coverage - consider expanding compliance mappings")

        if response_times and max(response_times) > 1000:
            print("   • Some queries are slow (>1s) - consider performance optimization")

        # Save detailed report to file
        report_data = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests)/len(self.results)*100,
                "total_mappings": total_mappings,
                "control_families_covered": len(all_control_families),
                "standards_with_mappings": len(all_standards_with_mappings)
            },
            "performance": {
                "average_response_time_ms": avg_response_time if response_times else 0,
                "min_response_time_ms": min_response_time if response_times else 0,
                "max_response_time_ms": max_response_time if response_times else 0
            },
            "coverage": {
                "control_families": sorted(all_control_families),
                "standards_with_mappings": sorted(all_standards_with_mappings)
            },
            "detailed_results": self.results
        }

        with open("/home/william/git/mcp-standards-server/compliance_mapping_test_report.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print("\n📄 Detailed report saved to: compliance_mapping_test_report.json")

async def main():
    """Main test function."""
    tester = ComplianceMappingTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
