"""
Demo script showcasing enhanced semantic search capabilities.
"""

import time

from src.core.standards.semantic_search import create_search_engine


def demo_basic_search():
    """Demonstrate basic semantic search."""
    print("\n=== Basic Semantic Search Demo ===")

    # Create search engine
    search = create_search_engine(enable_analytics=True)

    # Index sample standards documents
    documents = [
        (
            "react-hooks",
            """
        React Hooks Best Practices

        Use useState for local component state management.
        Use useEffect for side effects and lifecycle events.
        Custom hooks should start with 'use' prefix.
        Avoid using hooks inside conditionals or loops.
        """,
            {"category": "frontend", "framework": "react", "type": "hooks"},
        ),
        (
            "api-security",
            """
        API Security Standards

        Always use HTTPS for API endpoints.
        Implement proper authentication using JWT or OAuth.
        Validate all input data and sanitize outputs.
        Use rate limiting to prevent abuse.
        """,
            {"category": "security", "type": "api", "protocol": "rest"},
        ),
        (
            "python-testing",
            """
        Python Testing Guidelines

        Use pytest as the primary testing framework.
        Aim for at least 80% code coverage.
        Mock external dependencies in unit tests.
        Write integration tests for API endpoints.
        """,
            {"category": "testing", "language": "python", "framework": "pytest"},
        ),
        (
            "vue-components",
            """
        Vue.js Component Standards

        Use single-file components (.vue files).
        Follow the Vue style guide for naming.
        Use Composition API for complex components.
        Implement proper prop validation.
        """,
            {"category": "frontend", "framework": "vue", "version": "3"},
        ),
        (
            "database-design",
            """
        Database Design Best Practices

        Normalize data to reduce redundancy.
        Use appropriate indexes for query performance.
        Implement proper foreign key constraints.
        Plan for scalability from the beginning.
        """,
            {"category": "database", "type": "design", "applies_to": "sql"},
        ),
    ]

    print(f"Indexing {len(documents)} documents...")
    search.index_documents_batch(documents)

    # Perform various searches
    queries = [
        "React hooks useState",
        "API security authentication",
        "testing coverage pytest",
        "frontend component guidelines",
        "database performance",
    ]

    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = search.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.id} (score: {result.score:.3f})")
            if result.highlights:
                print(f"     Highlight: {result.highlights[0][:100]}...")

    # Show analytics
    print("\n=== Search Analytics ===")
    report = search.get_analytics_report()
    print(f"Total queries: {report['total_queries']}")
    print(f"Average latency: {report['average_latency_ms']:.2f} ms")
    print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")

    search.close()


def demo_advanced_features():
    """Demonstrate advanced search features."""
    print("\n=== Advanced Search Features Demo ===")

    search = create_search_engine()

    # Index more detailed documents
    documents = [
        (
            "mcp-server-guide",
            """
        MCP Server Development Guide

        Model Context Protocol (MCP) servers provide tools and resources to LLMs.
        Implement proper error handling and validation.
        Use TypeScript for better type safety.
        Follow the MCP specification for tool definitions.
        """,
            {"type": "guide", "technology": "mcp", "language": "typescript"},
        ),
        (
            "react-testing",
            """
        React Component Testing Standards

        Use React Testing Library for component tests.
        Test user interactions, not implementation details.
        Mock API calls and external dependencies.
        Write tests that resemble how users interact with your app.
        """,
            {"framework": "react", "type": "testing", "library": "rtl"},
        ),
        (
            "angular-security",
            """
        Angular Security Best Practices

        Sanitize user inputs to prevent XSS attacks.
        Use Angular's built-in security features.
        Implement Content Security Policy headers.
        Avoid using ElementRef for DOM manipulation.
        """,
            {"framework": "angular", "type": "security", "version": "15+"},
        ),
    ]

    search.index_documents_batch(documents)

    # 1. Fuzzy search (with typos)
    print("\n1. Fuzzy Search (handling typos)")
    results = search.search("Reakt tesing componets", use_fuzzy=True, top_k=2)
    print(f"   Query with typos found {len(results)} results")
    if results:
        print(f"   Top result: {results[0].id}")

    # 2. Boolean operators
    print("\n2. Boolean Operator Search")
    results = search.search("security AND angular NOT react", top_k=3)
    print(f"   Boolean query found {len(results)} results")
    for result in results:
        print(f"   - {result.id}: {result.content[:50]}...")

    # 3. Synonym expansion
    print("\n3. Synonym Expansion")
    results = search.search("web application safety", top_k=3)  # 'safety' -> 'security'
    print(f"   Synonym search found {len(results)} results")
    security_results = [r for r in results if "security" in r.content.lower()]
    print(f"   Found {len(security_results)} security-related results")

    # 4. Metadata filtering
    print("\n4. Metadata Filtering")
    results = search.search("best practices", filters={"type": "testing"}, top_k=3)
    print(f"   Filtered search found {len(results)} testing-related results")

    # 5. Re-ranking demonstration
    print("\n5. Re-ranking Results")
    results = search.search("testing", top_k=5, rerank=True)
    print("   Results with re-ranking explanations:")
    for result in results[:3]:
        print(f"   - {result.id}: {result.explanation}")

    search.close()


def demo_performance_optimization():
    """Demonstrate performance optimization features."""
    print("\n=== Performance Optimization Demo ===")

    search = create_search_engine()

    # Generate a larger dataset
    print("Generating 500 documents...")
    documents = []
    topics = [
        "python",
        "javascript",
        "react",
        "angular",
        "vue",
        "testing",
        "security",
        "api",
        "database",
        "performance",
    ]
    categories = ["frontend", "backend", "testing", "security", "database"]

    for i in range(500):
        topic = topics[i % len(topics)]
        category = categories[i % len(categories)]
        content = f"""
        Document {i}: {topic.capitalize()} Best Practices

        This document covers {topic} development standards and guidelines.
        Key topics include {category} considerations and implementation details.
        Follow these practices for better code quality and maintainability.
        """

        documents.append(
            (
                f"doc-{i:03d}",
                content,
                {"topic": topic, "category": category, "index": i},
            )
        )

    # Time batch indexing
    start = time.time()
    search.index_documents_batch(documents)
    index_time = time.time() - start
    print(f"Indexed 500 documents in {index_time:.2f} seconds")

    # Test search performance
    print("\nTesting search performance...")
    queries = [
        "python testing best practices",
        "javascript security guidelines",
        "frontend performance optimization",
        "database design patterns",
        "api development standards",
    ]

    # First pass - no cache
    print("\nFirst pass (no cache):")
    first_times = []
    for query in queries:
        start = time.time()
        results = search.search(query, top_k=10)
        elapsed = time.time() - start
        first_times.append(elapsed)
        print(f"  '{query[:30]}...' -> {elapsed*1000:.2f} ms ({len(results)} results)")

    # Second pass - with cache
    print("\nSecond pass (with cache):")
    second_times = []
    for query in queries:
        start = time.time()
        results = search.search(query, top_k=10)
        elapsed = time.time() - start
        second_times.append(elapsed)
        print(f"  '{query[:30]}...' -> {elapsed*1000:.2f} ms ({len(results)} results)")

    # Calculate speedup
    avg_first = sum(first_times) / len(first_times)
    avg_second = sum(second_times) / len(second_times)
    speedup = avg_first / avg_second

    print(f"\nAverage first pass: {avg_first*1000:.2f} ms")
    print(f"Average second pass: {avg_second*1000:.2f} ms")
    print(f"Cache speedup: {speedup:.2f}x faster")

    # Show final analytics
    report = search.get_analytics_report()
    print(f"\nTotal queries executed: {report['total_queries']}")
    print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")

    search.close()


def demo_async_search():
    """Demonstrate async search capabilities."""
    print("\n=== Async Search Demo ===")

    import asyncio

    async def async_demo():
        # Create async search engine
        search = create_search_engine(async_mode=True)

        # Index documents
        documents = [
            (
                "async-1",
                "Asynchronous programming in Python with asyncio",
                {"lang": "python"},
            ),
            (
                "async-2",
                "JavaScript promises and async/await patterns",
                {"lang": "javascript"},
            ),
            ("async-3", "Concurrent processing with Go routines", {"lang": "go"}),
            ("async-4", "Reactive programming with RxJS", {"lang": "javascript"}),
            ("async-5", "Async database queries with SQLAlchemy", {"lang": "python"}),
        ]

        print("Indexing documents asynchronously...")
        await search.index_documents_batch_async(documents)

        # Perform concurrent searches
        print("\nPerforming 3 concurrent searches...")
        queries = [
            "async programming Python",
            "JavaScript promises",
            "concurrent processing",
        ]

        start = time.time()
        # Run searches concurrently
        results_list = await asyncio.gather(
            *[search.search_async(query, top_k=3) for query in queries]
        )
        elapsed = time.time() - start

        print(f"All searches completed in {elapsed*1000:.2f} ms")

        for query, results in zip(queries, results_list, strict=False):
            print(f"\nResults for '{query}':")
            for i, result in enumerate(results[:2], 1):
                print(f"  {i}. {result.id} (score: {result.score:.3f})")

        search.close()

    # Run async demo
    asyncio.run(async_demo())


if __name__ == "__main__":
    print("MCP Standards Server - Enhanced Semantic Search Demo")
    print("=" * 50)

    # Run all demos
    demo_basic_search()
    demo_advanced_features()
    demo_performance_optimization()
    demo_async_search()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")
