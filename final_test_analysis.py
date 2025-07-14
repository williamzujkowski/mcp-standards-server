#!/usr/bin/env python3
"""
Comprehensive Analysis of get_applicable_standards Function
Testing the intelligent standard selection functionality of the MCP server.
"""

import asyncio
from typing import Any

from src.core.standards.engine import StandardsEngine
from src.core.standards.models import StandardMetadata


async def test_get_applicable_standards():
    """Run comprehensive test analysis of get_applicable_standards."""

    print("🔬 COMPREHENSIVE ANALYSIS: get_applicable_standards Function")
    print("=" * 80)

    # Setup engine with compatibility patches
    original_init = StandardMetadata.__init__

    def patched_init(self, **kwargs):
        if "author" in kwargs and "authors" not in kwargs:
            kwargs["authors"] = [kwargs["author"]]
            del kwargs["author"]
        known_fields = {
            "version",
            "last_updated",
            "authors",
            "source",
            "compliance_frameworks",
            "nist_controls",
            "tags",
            "dependencies",
            "language",
            "scope",
            "applicability",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        original_init(self, **filtered_kwargs)

    StandardMetadata.__init__ = patched_init

    # Initialize engine
    engine = StandardsEngine(
        data_dir="./data/standards",
        enable_semantic_search=True,
        enable_rule_engine=True,
        enable_token_optimization=True,
        enable_caching=True,
    )

    await engine.initialize()

    # Load rules
    from pathlib import Path

    rules_path = Path("./data/standards/meta/enhanced-selection-rules.json")
    if rules_path.exists() and engine.rule_engine:
        engine.rule_engine.load_rules(rules_path)

    print(
        f"✅ System initialized: {len(engine._standards_cache)} standards, {len(engine.rule_engine.rules) if engine.rule_engine else 0} rules"
    )

    # Define test cases
    test_cases = [
        {
            "name": "Test Case 1 - React Web App",
            "context": {
                "project_type": "web_application",
                "technologies": ["react", "javascript", "npm"],
                "requirements": ["accessibility", "security", "performance"],
                "framework": "react",
                "language": "javascript",
            },
            "expected_relevance": 8.0,
        },
        {
            "name": "Test Case 2 - Python API",
            "context": {
                "project_type": "api",
                "technologies": ["python", "fastapi", "postgresql"],
                "requirements": ["security", "database", "authentication"],
                "language": "python",
                "framework": "fastapi",
            },
            "expected_relevance": 7.0,
        },
        {
            "name": "Test Case 3 - Mobile IoT",
            "context": {
                "project_type": "mobile_app",
                "technologies": ["react-native", "iot", "bluetooth"],
                "requirements": ["privacy", "iot", "edge_computing"],
                "framework": "react-native",
                "language": "javascript",
            },
            "expected_relevance": 6.0,
        },
        {
            "name": "Test Case 4 - AI/ML Project",
            "context": {
                "project_type": "machine_learning",
                "technologies": ["python", "tensorflow", "docker"],
                "requirements": ["mlops", "ethics", "monitoring"],
                "language": "python",
                "domain": "ai_ml",
            },
            "expected_relevance": 7.0,
        },
    ]

    # Test each case
    results = []
    for _i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-' * 60}")
        print(f"🧪 {test_case['name']}")
        print(f"{'-' * 60}")

        # Display context
        print("📋 Input Context:")
        for key, value in test_case["context"].items():
            print(f"  • {key}: {value}")

        # Run test
        import time

        start_time = time.time()
        try:
            applicable_standards = await engine.get_applicable_standards(
                test_case["context"]
            )
            response_time = time.time() - start_time

            print(f"\n⏱️ Response Time: {response_time:.3f}s")
            print(f"📦 Standards Returned: {len(applicable_standards)}")

            # Analyze returned standards
            if applicable_standards:
                print("\n🔍 Returned Standards:")
                for j, result in enumerate(applicable_standards, 1):
                    standard = result.get("standard")
                    confidence = result.get("confidence", 0.0)
                    reasoning = result.get("reasoning", "No reasoning provided")
                    priority = result.get("priority", 99)

                    print(f"  {j}. {standard.id}")
                    print(f"     Title: {standard.title}")
                    print(f"     Category: {standard.category}")
                    print(f"     Tags: {list(standard.tags)}")
                    print(f"     Confidence: {confidence}")
                    print(f"     Priority: {priority}")
                    print(f"     Reasoning: {reasoning}")
                    print()
            else:
                print("❌ No standards returned")

            # Calculate relevance score
            relevance_score = calculate_relevance_score(applicable_standards, test_case)
            print(
                f"🎯 Relevance Score: {relevance_score:.1f}/10 (Expected: {test_case['expected_relevance']:.1f})"
            )

            # Check requirements coverage
            requirements_coverage = analyze_requirements_coverage(
                applicable_standards, test_case["context"].get("requirements", [])
            )
            print("📊 Requirements Coverage:")
            for req, coverage in requirements_coverage.items():
                status = "✅" if coverage["covered"] else "❌"
                covering_standards = (
                    ", ".join(coverage["standards"])
                    if coverage["standards"]
                    else "None"
                )
                print(f"  {status} {req}: {covering_standards}")

            # Check for missing critical standards
            missing_standards = identify_missing_standards(
                test_case["context"], applicable_standards
            )
            if missing_standards:
                print(f"⚠️ Missing Critical Standards: {', '.join(missing_standards)}")

            # Store results
            results.append(
                {
                    "test_case": test_case["name"],
                    "standards_count": len(applicable_standards),
                    "response_time": response_time,
                    "relevance_score": relevance_score,
                    "expected_relevance": test_case["expected_relevance"],
                    "requirements_coverage": requirements_coverage,
                    "missing_standards": missing_standards,
                    "success": len(applicable_standards) > 0,
                }
            )

        except Exception as e:
            print(f"❌ Error: {e}")
            results.append(
                {"test_case": test_case["name"], "error": str(e), "success": False}
            )

    # Generate final summary
    print(f"\n{'=' * 80}")
    print("📊 FINAL ANALYSIS SUMMARY")
    print(f"{'=' * 80}")

    successful_tests = [r for r in results if r.get("success", False)]
    avg_response_time = sum(r.get("response_time", 0) for r in results) / len(results)
    avg_relevance = (
        sum(r.get("relevance_score", 0) for r in successful_tests)
        / len(successful_tests)
        if successful_tests
        else 0
    )

    print(
        f"✅ Test Success Rate: {len(successful_tests)}/{len(results)} ({len(successful_tests)/len(results)*100:.1f}%)"
    )
    print(f"⏱️ Average Response Time: {avg_response_time:.3f}s")
    print(f"🎯 Average Relevance Score: {avg_relevance:.1f}/10")

    # Individual scores
    print("\n📋 Individual Test Scores:")
    for result in results:
        if result.get("success"):
            expected = result.get("expected_relevance", 0)
            actual = result.get("relevance_score", 0)
            variance = actual - expected
            variance_str = f"({variance:+.1f})" if variance != 0 else ""
            print(f"  • {result['test_case']}: {actual:.1f}/10 {variance_str}")
        else:
            print(f"  • {result['test_case']}: FAILED")

    # System performance assessment
    print("\n⚡ Performance Assessment:")
    if avg_response_time < 0.1:
        print("  ✅ Excellent response time (<100ms)")
    elif avg_response_time < 0.5:
        print("  ✅ Good response time (<500ms)")
    elif avg_response_time < 1.0:
        print("  ⚠️ Acceptable response time (<1s)")
    else:
        print("  ❌ Slow response time (>1s)")

    # Accuracy assessment
    print("\n🔍 Accuracy Assessment:")
    if avg_relevance >= 8.0:
        print("  ✅ Excellent relevance matching")
    elif avg_relevance >= 6.0:
        print("  ✅ Good relevance matching")
    elif avg_relevance >= 4.0:
        print("  ⚠️ Fair relevance matching - needs improvement")
    else:
        print("  ❌ Poor relevance matching - major improvements needed")

    # Key recommendations
    print("\n💡 Recommendations for Improvement:")

    if len(successful_tests) < len(results):
        print("  1. Fix rule engine to handle AI/ML and IoT domains")
        print("     - Add rules for machine_learning project_type")
        print("     - Create IoT-specific standard mappings")

    if avg_relevance < 6.0:
        print("  2. Improve relevance scoring algorithm")
        print("     - Better tag matching between context and standards")
        print("     - Weight standards by how well they match requirements")

    # Coverage analysis
    all_requirements = set()
    covered_requirements = set()
    for result in successful_tests:
        coverage = result.get("requirements_coverage", {})
        for req, cov in coverage.items():
            all_requirements.add(req)
            if cov.get("covered"):
                covered_requirements.add(req)

    if all_requirements:
        coverage_pct = len(covered_requirements) / len(all_requirements) * 100
        print(f"  3. Requirements coverage is {coverage_pct:.1f}%")
        if coverage_pct < 50:
            print("     - Add more standards that address specific requirements")
            print("     - Improve requirement-to-standard matching logic")

    print("\n🎖️ Overall System Rating: ", end="")
    overall_score = (
        len(successful_tests) / len(results) * 40
        + min(avg_relevance / 10 * 40, 40)
        + (100 - min(avg_response_time * 20, 20))
    )

    if overall_score >= 85:
        print(f"EXCELLENT ({overall_score:.0f}/100)")
    elif overall_score >= 70:
        print(f"GOOD ({overall_score:.0f}/100)")
    elif overall_score >= 50:
        print(f"FAIR ({overall_score:.0f}/100)")
    else:
        print(f"NEEDS IMPROVEMENT ({overall_score:.0f}/100)")

    await engine.close()


def calculate_relevance_score(
    standards: list[dict[str, Any]], test_case: dict[str, Any]
) -> float:
    """Calculate relevance score based on how well standards match the context."""
    if not standards:
        return 0.0

    context = test_case["context"]
    project_type = context.get("project_type", "")
    technologies = context.get("technologies", [])
    requirements = context.get("requirements", [])
    framework = context.get("framework", "")
    language = context.get("language", "")

    total_score = 0.0

    for result in standards:
        standard = result.get("standard")
        if not standard:
            continue

        score = 0.0

        # Check project type alignment
        if project_type:
            if project_type in standard.category.lower():
                score += 2.0
            if any(tech.lower() in standard.category.lower() for tech in technologies):
                score += 1.5

        # Check technology alignment
        std_tags = [tag.lower() for tag in standard.tags]
        std_title_lower = standard.title.lower()

        if framework and framework.lower() in std_tags:
            score += 2.0
        if language and language.lower() in std_tags:
            score += 1.5

        # Check if technologies mentioned
        for tech in technologies:
            if tech.lower() in std_tags or tech.lower() in std_title_lower:
                score += 1.0

        # Check requirements coverage (basic keyword matching)
        for req in requirements:
            if (
                req.lower() in std_title_lower
                or req.lower() in standard.description.lower()
                or req.lower() in std_tags
            ):
                score += 0.5

        # Use confidence from system
        confidence = result.get("confidence", 0.5)
        score *= confidence

        total_score += min(score, 10.0)

    return min(total_score / len(standards), 10.0)


def analyze_requirements_coverage(
    standards: list[dict[str, Any]], requirements: list[str]
) -> dict[str, dict]:
    """Analyze how well standards cover the stated requirements."""
    coverage = {}

    for requirement in requirements:
        covered = False
        covering_standards = []

        for result in standards:
            standard = result.get("standard")
            if not standard:
                continue

            # Check if requirement appears in standard
            search_text = f"{standard.title} {standard.description} {' '.join(standard.tags)}".lower()
            if requirement.lower() in search_text:
                covered = True
                covering_standards.append(standard.id)

        coverage[requirement] = {"covered": covered, "standards": covering_standards}

    return coverage


def identify_missing_standards(
    context: dict[str, Any], returned_standards: list[dict[str, Any]]
) -> list[str]:
    """Identify standards that should probably be included but weren't."""
    missing = []

    # Define expected standards for common contexts
    context.get("project_type", "")
    technologies = context.get("technologies", [])
    requirements = context.get("requirements", [])

    # Security requirements should always include security standards
    if "security" in requirements:
        returned_ids = [r.get("standard", {}).get("id", "") for r in returned_standards]
        if not any("security" in r_id.lower() for r_id in returned_ids):
            missing.append("security-standards")

    # Performance requirements should include performance standards
    if "performance" in requirements:
        returned_ids = [r.get("standard", {}).get("id", "") for r in returned_standards]
        if not any("performance" in r_id.lower() for r_id in returned_ids):
            missing.append("performance-standards")

    # Database requirements should include database standards
    if "database" in requirements or any(
        db in technologies for db in ["postgresql", "mysql", "mongodb"]
    ):
        returned_ids = [r.get("standard", {}).get("id", "") for r in returned_standards]
        if not any(
            "database" in r_id.lower() or "data" in r_id.lower()
            for r_id in returned_ids
        ):
            missing.append("database-standards")

    return missing


if __name__ == "__main__":
    asyncio.run(test_get_applicable_standards())
