#!/usr/bin/env python3
"""
Test script for search_standards function.
Tests semantic search functionality of the MCP server.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.standards.engine import StandardsEngine


async def test_search_standards():
    """Test the search_standards functionality."""

    # Initialize the standards engine
    data_dir = Path(__file__).parent / "data" / "standards"
    engine = StandardsEngine(
        data_dir=data_dir,
        enable_semantic_search=True,
        enable_rule_engine=True,
        enable_token_optimization=True,
        enable_caching=True,
    )

    print("🚀 Initializing Standards Engine...")
    await engine.initialize()

    # Get total number of standards loaded
    all_standards = await engine.list_standards(limit=1000)
    print(f"📚 Loaded {len(all_standards)} standards")

    # Test cases from the task
    test_cases = [
        {
            "name": "Security Search",
            "query": "authentication security best practices",
            "max_results": 5,
            "expected_domains": ["security", "authentication", "auth"],
        },
        {
            "name": "Performance Search",
            "query": "database optimization performance tuning",
            "max_results": 3,
            "expected_domains": ["database", "performance", "optimization"],
        },
        {
            "name": "Accessibility Search",
            "query": "WCAG accessibility guidelines screen readers",
            "max_results": 5,
            "expected_domains": ["accessibility", "wcag", "a11y"],
        },
        {
            "name": "AI/ML Search",
            "query": "machine learning model deployment mlops",
            "max_results": 5,
            "expected_domains": ["ml", "ai", "mlops", "machine learning"],
        },
        {
            "name": "Fuzzy Search",
            "query": "reactjs component patterns hooks",
            "max_results": 3,
            "expected_domains": ["react", "javascript", "frontend"],
        },
    ]

    print("\n" + "=" * 80)
    print("🔍 SEARCH STANDARDS TEST RESULTS")
    print("=" * 80)

    overall_results = []

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"🔍 Query: \"{test_case['query']}\"")
        print(f"📊 Max Results: {test_case['max_results']}")

        # Measure response time
        start_time = time.time()

        try:
            # Perform search using the engine's search_standards method
            results = await engine.search_standards(
                query=test_case["query"], limit=test_case["max_results"]
            )

            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000

            print(f"⏱️  Response Time: {response_time_ms:.2f}ms")
            print(f"📈 Results Found: {len(results)}")

            if results:
                print("\n📑 Search Results:")
                print("-" * 60)

                for j, result in enumerate(results, 1):
                    standard = result.get("standard")
                    score = result.get("score", 0.0)
                    highlights = result.get("highlights", [])

                    if standard:
                        print(f"  {j}. {standard.title}")
                        print(f"     📁 Category: {standard.category}")
                        print(f"     🏷️  Tags: {', '.join(standard.tags)}")
                        print(f"     ⭐ Score: {score:.3f}")
                        if highlights:
                            print(f"     💡 Highlights: {len(highlights)} snippets")
                        print()

                # Evaluate semantic understanding
                semantic_score = evaluate_semantic_understanding(
                    test_case["query"], results, test_case["expected_domains"]
                )

                print(f"🧠 Semantic Understanding Score: {semantic_score}/10")

                # Check diversity
                diversity_score = evaluate_result_diversity(results)
                print(f"🎯 Result Diversity Score: {diversity_score}/10")

                overall_results.append(
                    {
                        "test_case": test_case["name"],
                        "query": test_case["query"],
                        "results_count": len(results),
                        "response_time_ms": response_time_ms,
                        "semantic_score": semantic_score,
                        "diversity_score": diversity_score,
                        "results": results,
                    }
                )

            else:
                print("❌ No results found")
                overall_results.append(
                    {
                        "test_case": test_case["name"],
                        "query": test_case["query"],
                        "results_count": 0,
                        "response_time_ms": response_time_ms,
                        "semantic_score": 0,
                        "diversity_score": 0,
                        "results": [],
                    }
                )

        except Exception as e:
            print(f"❌ Error during search: {e}")
            overall_results.append(
                {
                    "test_case": test_case["name"],
                    "query": test_case["query"],
                    "error": str(e),
                    "results_count": 0,
                    "response_time_ms": 0,
                    "semantic_score": 0,
                    "diversity_score": 0,
                }
            )

    # Generate summary report
    print("\n" + "=" * 80)
    print("📊 OVERALL ANALYSIS & RECOMMENDATIONS")
    print("=" * 80)

    generate_analysis_report(overall_results)

    # Cleanup
    await engine.close()


def evaluate_semantic_understanding(query, results, expected_domains):
    """Evaluate how well the search understood the semantic intent."""
    if not results:
        return 0

    query_lower = query.lower()
    score = 0

    # Check if results contain expected domain concepts
    for result in results:
        standard = result.get("standard")
        if not standard:
            continue

        content_text = (
            f"{standard.title} {standard.description} {' '.join(standard.tags)}".lower()
        )

        # Check for direct query terms
        query_terms = query_lower.split()
        matching_terms = sum(1 for term in query_terms if term in content_text)
        term_score = (matching_terms / len(query_terms)) * 3

        # Check for expected domain concepts
        domain_matches = sum(1 for domain in expected_domains if domain in content_text)
        domain_score = (domain_matches / len(expected_domains)) * 4

        # Category relevance
        category_score = (
            2
            if any(domain in standard.category.lower() for domain in expected_domains)
            else 0
        )

        # Combine scores for this result
        result_score = min(term_score + domain_score + category_score, 10)
        score = max(score, result_score)  # Take the best result as the overall score

    return min(score, 10)


def evaluate_result_diversity(results):
    """Evaluate diversity of search results."""
    if not results:
        return 0

    categories = set()
    subcategories = set()

    for result in results:
        standard = result.get("standard")
        if standard:
            categories.add(standard.category)
            if standard.subcategory:
                subcategories.add(standard.subcategory)

    # Diversity based on unique categories and subcategories
    category_diversity = min(len(categories) * 3, 6)  # Up to 6 points for categories
    subcategory_diversity = min(
        len(subcategories) * 2, 4
    )  # Up to 4 points for subcategories

    return min(category_diversity + subcategory_diversity, 10)


def generate_analysis_report(results):
    """Generate analysis and recommendations."""

    successful_tests = [r for r in results if r.get("results_count", 0) > 0]
    failed_tests = [
        r for r in results if r.get("results_count", 0) == 0 or "error" in r
    ]

    if successful_tests:
        avg_response_time = sum(r["response_time_ms"] for r in successful_tests) / len(
            successful_tests
        )
        avg_semantic_score = sum(r["semantic_score"] for r in successful_tests) / len(
            successful_tests
        )
        avg_diversity_score = sum(r["diversity_score"] for r in successful_tests) / len(
            successful_tests
        )
        avg_results_count = sum(r["results_count"] for r in successful_tests) / len(
            successful_tests
        )

        print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
        print(f"⏱️  Average Response Time: {avg_response_time:.2f}ms")
        print(f"🧠 Average Semantic Score: {avg_semantic_score:.1f}/10")
        print(f"🎯 Average Diversity Score: {avg_diversity_score:.1f}/10")
        print(f"📊 Average Results per Query: {avg_results_count:.1f}")

        # Performance assessment
        if avg_response_time < 100:
            print("🚀 Performance: Excellent (< 100ms)")
        elif avg_response_time < 500:
            print("⚡ Performance: Good (< 500ms)")
        else:
            print("🐌 Performance: Needs improvement (> 500ms)")

        # Semantic understanding assessment
        if avg_semantic_score >= 8:
            print("🎯 Semantic Understanding: Excellent")
        elif avg_semantic_score >= 6:
            print("👍 Semantic Understanding: Good")
        elif avg_semantic_score >= 4:
            print("⚠️  Semantic Understanding: Fair")
        else:
            print("❌ Semantic Understanding: Poor")

    if failed_tests:
        print(f"\n❌ Failed Tests: {len(failed_tests)}")
        for test in failed_tests:
            if "error" in test:
                print(f"   - {test['test_case']}: {test['error']}")
            else:
                print(f"   - {test['test_case']}: No results found")

    print("\n🔧 RECOMMENDATIONS:")

    if not successful_tests:
        print("• Critical: No searches returned results. Check:")
        print("  - Standards loading and indexing")
        print("  - Semantic search initialization")
        print("  - Query preprocessing")
        return

    if avg_response_time > 500:
        print("• Performance: Consider optimizing:")
        print("  - Vector indexing (use Faiss or similar)")
        print("  - Caching layer improvements")
        print("  - Embedding batch processing")

    if avg_semantic_score < 6:
        print("• Semantic Search: Consider improvements:")
        print("  - Better embedding model (e.g., all-mpnet-base-v2)")
        print("  - Query expansion with domain synonyms")
        print("  - Fine-tuning embeddings on standards domain")
        print("  - Hybrid search (semantic + keyword)")

    if avg_diversity_score < 6:
        print("• Result Diversity: Consider:")
        print("  - Diversification algorithms")
        print("  - Category-aware ranking")
        print("  - Maximum margin relevance (MMR)")

    print("• Search Engine Type Detection:")
    # Determine if semantic or keyword search was used
    has_high_scores = any(r["semantic_score"] > 7 for r in successful_tests)
    if has_high_scores:
        print("  ✅ Semantic search appears to be working effectively")
    else:
        print("  ⚠️  May be falling back to keyword search - check embedding quality")


if __name__ == "__main__":
    print("🔍 Testing MCP Standards Server Search Functionality")
    print("=" * 60)
    asyncio.run(test_search_standards())
