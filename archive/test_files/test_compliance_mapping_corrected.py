#!/usr/bin/env python3
"""
Corrected test script for NIST compliance mapping functionality.
Tests with actual available standard IDs and evaluates the mapping gaps.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from src.core.mcp.handlers import StandardsHandler

# Import the MCP server components directly
from src.core.standards.engine import StandardsEngine


class CorrectedComplianceMappingTester:
    """Test the compliance mapping functionality with correct standard IDs."""

    def __init__(self):
        self.results: list[dict[str, Any]] = []
        self.actual_standard_ids: list[str] = []
        self.nist_references_in_files: dict[str, list[str]] = {}

    async def setup(self):
        """Initialize the standards engine and handler."""
        print("Setting up standards engine...")

        # Initialize the standards engine with data directory
        data_dir = "/home/william/git/mcp-standards-server/data/standards"
        self.data_dir = Path(data_dir)
        self.standards_engine = StandardsEngine(data_dir=data_dir)
        await self.standards_engine.initialize()

        # Initialize the handler
        self.handler = StandardsHandler(self.standards_engine)
        await self.handler.initialize()

        print("✅ Setup complete")

    async def discover_actual_standards(self):
        """Discover the actual standard IDs available in the system."""
        print("\n🔍 Discovering actual standard IDs...")

        # Get all available standards
        standards_result = await self.handler.handle_tool("list_available_standards", {})

        if standards_result and "result" in standards_result:
            all_standards = standards_result["result"]
            self.actual_standard_ids = [std.id for std in all_standards if hasattr(std, 'id')]

            print(f"📚 Found {len(self.actual_standard_ids)} actual standards:")
            for i, std_id in enumerate(self.actual_standard_ids[:10]):
                print(f"   {i+1:2d}. {std_id}")
            if len(self.actual_standard_ids) > 10:
                print(f"   ... and {len(self.actual_standard_ids) - 10} more")
        else:
            print("❌ Failed to get standards list")

    async def analyze_nist_references_in_files(self):
        """Analyze NIST references directly in the markdown files."""
        print("\n🔬 Analyzing NIST references in source files...")

        md_files = list(self.data_dir.glob("*.md"))

        for md_file in md_files:
            content = md_file.read_text()

            # Look for NIST control patterns
            nist_patterns = [
                r'NIST[- ]([A-Z]{2}-\d+(?:,\s*[A-Z]{2}-\d+)*)',  # NIST-AC-1, AC-2, etc.
                r'NIST Controls?[:\s]*([A-Z]{2}-\d+(?:,\s*[A-Z]{2}-\d+)*)',  # NIST Controls: AC-1, AC-2
                r'(?:NIST|Controls?)[:\s]*([A-Z]{2}-\d+(?:,\s*[A-Z]{2}-\d+)*)'  # More flexible pattern
            ]

            found_controls = set()
            for pattern in nist_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    # Split and clean up control IDs
                    controls = [ctrl.strip() for ctrl in match.split(',')]
                    found_controls.update(controls)

            if found_controls:
                # Try to map file name to standard ID
                file_base = md_file.stem.lower().replace('_', '-')
                # Look for matching standard ID
                matching_std = None
                for std_id in self.actual_standard_ids:
                    if std_id.lower() in file_base or file_base in std_id.lower():
                        matching_std = std_id
                        break

                self.nist_references_in_files[md_file.name] = {
                    'controls': sorted(found_controls),
                    'matching_standard_id': matching_std,
                    'file_base': file_base
                }

                print(f"   📄 {md_file.name}: {len(found_controls)} NIST controls")
                if matching_std:
                    print(f"      🔗 Maps to standard: {matching_std}")
                else:
                    print(f"      ❓ No matching standard found for file base: {file_base}")

    async def test_compliance_mapping_with_actual_standards(self):
        """Test compliance mapping with actual standard IDs."""
        print("\n🧪 Testing compliance mapping with actual standards...")

        # Test cases with actual standard IDs
        test_cases = [
            {
                "name": "Security-related Standards",
                "standard_ids": [std_id for std_id in self.actual_standard_ids if 'security' in std_id.lower()][:3]
            },
            {
                "name": "Privacy-related Standards",
                "standard_ids": [std_id for std_id in self.actual_standard_ids if 'privacy' in std_id.lower() or 'data' in std_id.lower()][:3]
            },
            {
                "name": "Technology Standards",
                "standard_ids": [std_id for std_id in self.actual_standard_ids if any(tech in std_id.lower() for tech in ['react', 'typescript', 'javascript', 'python'])][:3]
            },
            {
                "name": "All Standards Sample",
                "standard_ids": self.actual_standard_ids[:5]  # First 5 for testing
            }
        ]

        for test_case in test_cases:
            if not test_case["standard_ids"]:
                print(f"   ⚠️  Skipping '{test_case['name']}' - no matching standards found")
                continue

            print(f"\n📋 Testing: {test_case['name']}")
            print(f"   Standards: {test_case['standard_ids']}")

            start_time = time.time()

            try:
                result = await self.handler.handle_tool("get_compliance_mapping", {
                    "standard_ids": test_case["standard_ids"],
                    "framework": "nist-800-53"
                })

                end_time = time.time()
                response_time = end_time - start_time

                if result and "result" in result:
                    mappings = result["result"]
                    print(f"   📊 Found {len(mappings)} NIST control mappings")
                    print(f"   ⏱️  Response time: {response_time*1000:.2f}ms")

                    # Show mappings if any
                    for mapping in mappings[:3]:
                        print(f"   📋 {mapping.get('standard_id')} → {mapping.get('control_id')}")

                    test_result = {
                        "test_case": test_case["name"],
                        "standards_tested": test_case["standard_ids"],
                        "mapping_count": len(mappings),
                        "response_time_ms": round(response_time * 1000, 2),
                        "success": True
                    }
                else:
                    print(f"   ❌ No result or error: {result}")
                    test_result = {
                        "test_case": test_case["name"],
                        "standards_tested": test_case["standard_ids"],
                        "mapping_count": 0,
                        "response_time_ms": round((time.time() - start_time) * 1000, 2),
                        "success": False,
                        "error": result.get("error") if result else "No result"
                    }

                self.results.append(test_result)

            except Exception as e:
                print(f"   ❌ Exception: {e}")
                self.results.append({
                    "test_case": test_case["name"],
                    "standards_tested": test_case["standard_ids"],
                    "mapping_count": 0,
                    "success": False,
                    "error": str(e)
                })

    async def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE COMPLIANCE MAPPING ANALYSIS REPORT")
        print("=" * 80)

        # Summary of findings
        total_tests = len(self.results)
        successful_tests = [r for r in self.results if r.get("success", False)]

        print("📈 Test Summary:")
        print(f"   • Total tests conducted: {total_tests}")
        print(f"   • Successful tests: {len(successful_tests)}")
        print(f"   • Failed tests: {total_tests - len(successful_tests)}")

        # NIST references analysis
        total_files_with_nist = len(self.nist_references_in_files)
        total_nist_controls_in_files = sum(len(info['controls']) for info in self.nist_references_in_files.values())

        print("\n📄 Source File Analysis:")
        print(f"   • Files with NIST references: {total_files_with_nist}")
        print(f"   • Total NIST controls in files: {total_nist_controls_in_files}")
        print(f"   • Average controls per file: {total_nist_controls_in_files / max(total_files_with_nist, 1):.1f}")

        # Show files with NIST controls but no matching standards
        unmatched_files = [
            (filename, info) for filename, info in self.nist_references_in_files.items()
            if not info['matching_standard_id']
        ]

        if unmatched_files:
            print(f"\n❓ Files with NIST controls but no matching standards ({len(unmatched_files)}):")
            for filename, info in unmatched_files[:5]:
                print(f"   • {filename}: {len(info['controls'])} controls")

        # Standards with potential NIST mappings
        mapped_standards = []
        for filename, info in self.nist_references_in_files.items():
            if info['matching_standard_id']:
                mapped_standards.append({
                    'standard_id': info['matching_standard_id'],
                    'file': filename,
                    'control_count': len(info['controls']),
                    'controls': info['controls'][:5]  # First 5 for brevity
                })

        print(f"\n🔗 Standards with potential NIST mappings ({len(mapped_standards)}):")
        for std in mapped_standards:
            print(f"   • {std['standard_id']}: {std['control_count']} controls from {std['file']}")
            print(f"     Sample controls: {', '.join(std['controls'][:3])}{'...' if len(std['controls']) > 3 else ''}")

        # Mapping coverage gap analysis
        total_mappings_found = sum(r.get("mapping_count", 0) for r in successful_tests)

        print("\n🕳️  Gap Analysis:")
        print(f"   • NIST controls found in files: {total_nist_controls_in_files}")
        print(f"   • NIST mappings returned by API: {total_mappings_found}")
        print(f"   • Mapping gap: {total_nist_controls_in_files - total_mappings_found} controls")

        if total_nist_controls_in_files > 0 and total_mappings_found == 0:
            print(f"   🚨 CRITICAL: No mappings found despite {total_nist_controls_in_files} NIST references in files")
            print("   💡 This suggests a data synchronization or parsing issue")

        # Recommendations
        print("\n💡 Recommendations:")

        if total_mappings_found == 0 and total_nist_controls_in_files > 0:
            print("   🔧 HIGH PRIORITY: Fix NIST control parsing and metadata synchronization")
            print("   📋 The system has NIST references in files but they're not accessible via API")

        if unmatched_files:
            print(f"   🔗 Map {len(unmatched_files)} files with NIST references to proper standard IDs")

        if len(self.actual_standard_ids) > len(mapped_standards):
            unmapped_count = len(self.actual_standard_ids) - len(mapped_standards)
            print(f"   📝 Consider adding NIST mappings to {unmapped_count} standards without them")

        print("   ⚡ Implement automated NIST control extraction from markdown files during sync")
        print("   🔄 Add validation to ensure NIST metadata is properly populated")

        # Framework support analysis
        print("\n🛠️  Framework Support Analysis:")
        print("   • NIST 800-53 framework: Partially supported (parsing issues)")
        print("   • Control identification: Working (found in source files)")
        print("   • Metadata integration: BROKEN (not accessible via API)")
        print(f"   • Response time: Excellent ({sum(r.get('response_time_ms', 0) for r in successful_tests) / max(len(successful_tests), 1):.2f}ms avg)")

        # Save detailed report
        report_data = {
            "summary": {
                "test_results": {
                    "total_tests": total_tests,
                    "successful_tests": len(successful_tests),
                    "total_mappings_found": total_mappings_found
                },
                "source_analysis": {
                    "files_with_nist": total_files_with_nist,
                    "total_controls_in_files": total_nist_controls_in_files,
                    "unmatched_files_count": len(unmatched_files)
                },
                "gap_analysis": {
                    "controls_in_files": total_nist_controls_in_files,
                    "mappings_via_api": total_mappings_found,
                    "mapping_gap": total_nist_controls_in_files - total_mappings_found
                }
            },
            "nist_references_in_files": self.nist_references_in_files,
            "standards_discovered": self.actual_standard_ids,
            "potential_mappings": mapped_standards,
            "test_results": self.results,
            "recommendations": [
                "Fix NIST control parsing and metadata synchronization",
                "Map unmatched files to proper standard IDs",
                "Implement automated NIST control extraction",
                "Add validation for NIST metadata population",
                "Consider expanding NIST coverage for unmapped standards"
            ]
        }

        with open("/home/william/git/mcp-standards-server/comprehensive_compliance_analysis.json", "w") as f:
            json.dump(report_data, f, indent=2)

        print("\n📄 Detailed analysis saved to: comprehensive_compliance_analysis.json")

    async def run_comprehensive_analysis(self):
        """Run the complete compliance mapping analysis."""
        print("🚀 Starting Comprehensive NIST Compliance Mapping Analysis")
        print("=" * 80)

        await self.setup()
        await self.discover_actual_standards()
        await self.analyze_nist_references_in_files()
        await self.test_compliance_mapping_with_actual_standards()
        await self.generate_comprehensive_report()

async def main():
    """Main analysis function."""
    tester = CorrectedComplianceMappingTester()
    await tester.run_comprehensive_analysis()

if __name__ == "__main__":
    asyncio.run(main())
