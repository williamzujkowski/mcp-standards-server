#!/usr/bin/env python3
"""
Analyze the available standards catalog to understand search limitations.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.standards.engine import StandardsEngine


async def analyze_standards_catalog():
    """Analyze what standards are available and their content."""

    # Initialize the standards engine
    data_dir = Path(__file__).parent / "data" / "standards"
    engine = StandardsEngine(
        data_dir=data_dir,
        enable_semantic_search=True,
        enable_rule_engine=True,
        enable_token_optimization=True,
        enable_caching=True,
    )

    print("🔍 Analyzing Standards Catalog...")
    await engine.initialize()

    # Get all standards
    all_standards = await engine.list_standards(limit=1000)

    print(f"\n📚 TOTAL STANDARDS LOADED: {len(all_standards)}")
    print("=" * 80)

    # Group by category
    by_category = {}
    for standard in all_standards:
        category = standard.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(standard)

    # Show categories and counts
    print("\n📁 STANDARDS BY CATEGORY:")
    print("-" * 40)
    for category in sorted(by_category.keys()):
        count = len(by_category[category])
        print(f"  {category}: {count} standards")

    # Detailed analysis for each category
    for category in sorted(by_category.keys()):
        standards = by_category[category]
        print(f"\n📂 {category.upper()} CATEGORY ({len(standards)} standards):")
        print("-" * 60)

        for standard in standards:
            print(f"  🏷️  {standard.id}")
            print(f"     Title: {standard.title}")
            print(f"     Tags: {', '.join(standard.tags)}")
            print(
                f"     Description: {standard.description[:100]}{'...' if len(standard.description) > 100 else ''}"
            )
            print()

    # Analyze search relevance for test queries
    print("\n🔍 SEARCH RELEVANCE ANALYSIS:")
    print("=" * 80)

    test_queries = [
        (
            "Security/Authentication",
            "authentication security best practices",
            ["security", "auth", "authentication"],
        ),
        (
            "Database Performance",
            "database optimization performance tuning",
            ["database", "performance", "optimization"],
        ),
        (
            "Accessibility",
            "WCAG accessibility guidelines screen readers",
            ["accessibility", "wcag", "a11y"],
        ),
        (
            "AI/ML",
            "machine learning model deployment mlops",
            ["ai", "ml", "machine learning", "mlops"],
        ),
        (
            "React",
            "reactjs component patterns hooks",
            ["react", "javascript", "frontend"],
        ),
    ]

    for query_name, query, keywords in test_queries:
        print(f"\n🎯 {query_name}")
        print(f'   Query: "{query}"')
        print(f"   Expected Keywords: {keywords}")

        # Find potentially relevant standards
        relevant_standards = []
        for standard in all_standards:
            content = f"{standard.title} {standard.description} {' '.join(standard.tags)}".lower()
            matches = sum(1 for keyword in keywords if keyword in content)
            if matches > 0:
                relevant_standards.append((standard, matches))

        relevant_standards.sort(key=lambda x: x[1], reverse=True)

        if relevant_standards:
            print(
                f"   📈 Found {len(relevant_standards)} potentially relevant standards:"
            )
            for standard, match_count in relevant_standards[:3]:  # Show top 3
                print(
                    f"     • {standard.title} ({standard.category}) - {match_count} keyword matches"
                )
        else:
            print("   ❌ No standards found with matching keywords")

    # Check for specific domains that might be missing
    print("\n🔍 DOMAIN COVERAGE ANALYSIS:")
    print("-" * 40)

    key_domains = {
        "Security": ["security", "auth", "authentication", "authorization"],
        "Database": ["database", "db", "sql", "nosql"],
        "Performance": ["performance", "optimization", "tuning"],
        "Accessibility": ["accessibility", "wcag", "a11y", "aria"],
        "AI/ML": ["ai", "ml", "machine learning", "neural", "model"],
        "Frontend": ["react", "vue", "angular", "javascript"],
        "Testing": ["test", "testing", "unit", "integration"],
        "DevOps": ["devops", "ci", "cd", "deployment"],
    }

    for domain, keywords in key_domains.items():
        matching_standards = []
        for standard in all_standards:
            content = f"{standard.title} {standard.description} {' '.join(standard.tags)}".lower()
            if any(keyword in content for keyword in keywords):
                matching_standards.append(standard)

        print(f"  {domain}: {len(matching_standards)} standards")
        if len(matching_standards) > 0:
            print(f"    Example: {matching_standards[0].title}")

    # Check if WCAG/accessibility standards exist
    print("\n🔍 SPECIFIC STANDARD CHECKS:")
    print("-" * 40)

    accessibility_standards = [
        s
        for s in all_standards
        if "accessibility" in s.title.lower()
        or "wcag" in s.title.lower()
        or "a11y" in " ".join(s.tags)
    ]
    print(f"Accessibility standards: {len(accessibility_standards)}")

    ai_ml_standards = [
        s
        for s in all_standards
        if any(
            term in f"{s.title} {s.description}".lower()
            for term in ["ai", "ml", "machine learning", "neural", "model training"]
        )
    ]
    print(f"AI/ML standards: {len(ai_ml_standards)}")

    db_standards = [
        s
        for s in all_standards
        if any(
            term in f"{s.title} {s.description}".lower()
            for term in ["database", "sql", "nosql", "db"]
        )
    ]
    print(f"Database standards: {len(db_standards)}")

    await engine.close()


if __name__ == "__main__":
    asyncio.run(analyze_standards_catalog())
