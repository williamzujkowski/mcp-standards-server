"""
Accuracy tests for semantic search functionality.

Tests focused on search quality, relevance ranking, and semantic understanding.
"""

import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from src.core.standards.semantic_search import SearchResult, create_search_engine
from tests.mocks.semantic_search_mocks import TestDataGenerator, patch_ml_dependencies


@dataclass
class RelevanceTestCase:
    """Test case for relevance evaluation."""

    query: str
    relevant_docs: list[str]  # Document IDs that should be retrieved
    irrelevant_docs: list[str]  # Document IDs that should NOT be retrieved
    description: str


@dataclass
class SemanticTestCase:
    """Test case for semantic understanding."""

    query: str
    semantic_equivalents: list[str]  # Queries that should return similar results
    semantic_opposites: list[str]  # Queries that should return different results
    description: str


class RelevanceMetrics:
    """Calculate relevance metrics for search results."""

    @staticmethod
    def precision_at_k(
        results: list[SearchResult], relevant_ids: list[str], k: int
    ) -> float:
        """Calculate precision at k."""
        if not results or k == 0:
            return 0.0

        top_k_ids = [r.id for r in results[:k]]
        relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant_ids)

        return relevant_in_top_k / k

    @staticmethod
    def recall_at_k(
        results: list[SearchResult], relevant_ids: list[str], k: int
    ) -> float:
        """Calculate recall at k."""
        if not relevant_ids:
            return 0.0

        top_k_ids = [r.id for r in results[:k]]
        relevant_in_top_k = sum(1 for doc_id in top_k_ids if doc_id in relevant_ids)

        return relevant_in_top_k / len(relevant_ids)

    @staticmethod
    def f1_at_k(results: list[SearchResult], relevant_ids: list[str], k: int) -> float:
        """Calculate F1 score at k."""
        precision = RelevanceMetrics.precision_at_k(results, relevant_ids, k)
        recall = RelevanceMetrics.recall_at_k(results, relevant_ids, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def ndcg_at_k(
        results: list[SearchResult], relevant_ids: list[str], k: int
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        if not results or k == 0:
            return 0.0

        # Binary relevance (1 if relevant, 0 if not)
        dcg = 0.0
        for i, result in enumerate(results[:k]):
            if result.id in relevant_ids:
                # Relevance score / log2(position + 1)
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i starts at 0

        # Ideal DCG (all relevant docs at top)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_ids), k)))

        if idcg == 0:
            return 0.0

        return cast(float, dcg / idcg)

    @staticmethod
    def mean_reciprocal_rank(
        results: list[SearchResult], relevant_ids: list[str]
    ) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, result in enumerate(results):
            if result.id in relevant_ids:
                return 1.0 / (i + 1)
        return 0.0


class TestSemanticSearchRelevance:
    """Test relevance and ranking quality."""

    @pytest.fixture
    def search_engine_with_corpus(self):
        """Create search engine with curated test corpus."""
        engine = create_search_engine()

        # Create curated test corpus
        test_corpus = [
            # Security documents
            (
                "sec-api-001",
                """
            API Security Best Practices

            Implement OAuth 2.0 or JWT for API authentication. Use HTTPS for all
            communications. Validate all input parameters. Implement rate limiting
            to prevent abuse. Use parameterized queries to prevent SQL injection.
            """,
                {
                    "category": "security",
                    "subcategory": "api",
                    "tags": ["oauth", "jwt", "https"],
                },
            ),
            (
                "sec-web-001",
                """
            Web Application Security Guidelines

            Protect against XSS attacks by escaping user input. Implement CSRF tokens
            for state-changing operations. Use Content Security Policy headers.
            Enable HSTS for HTTPS enforcement. Regular security audits are essential.
            """,
                {
                    "category": "security",
                    "subcategory": "web",
                    "tags": ["xss", "csrf", "csp"],
                },
            ),
            # Testing documents
            (
                "test-react-001",
                """
            React Component Testing Standards

            Use Jest and React Testing Library for unit tests. Test component behavior
            rather than implementation details. Achieve minimum 80% code coverage.
            Mock external dependencies appropriately. Write integration tests for
            component interactions.
            """,
                {
                    "category": "testing",
                    "subcategory": "frontend",
                    "tags": ["jest", "react", "coverage"],
                },
            ),
            (
                "test-api-001",
                """
            API Testing Best Practices

            Test all API endpoints with various input scenarios. Include negative test
            cases and error handling. Validate response schemas. Test rate limiting
            and authentication. Use contract testing for microservices.
            """,
                {
                    "category": "testing",
                    "subcategory": "api",
                    "tags": ["endpoints", "contract", "validation"],
                },
            ),
            # Performance documents
            (
                "perf-db-001",
                """
            Database Performance Optimization

            Use appropriate indexes for frequent queries. Implement query result caching.
            Optimize database schema design. Monitor slow queries. Use connection pooling
            for better resource utilization.
            """,
                {
                    "category": "performance",
                    "subcategory": "database",
                    "tags": ["indexing", "caching", "optimization"],
                },
            ),
            (
                "perf-web-001",
                """
            Web Performance Best Practices

            Minimize JavaScript bundle sizes. Implement code splitting and lazy loading.
            Optimize images and use WebP format. Enable HTTP/2 and compression.
            Use CDN for static assets. Monitor Core Web Vitals.
            """,
                {
                    "category": "performance",
                    "subcategory": "web",
                    "tags": ["optimization", "cdn", "bundling"],
                },
            ),
        ]

        # Index corpus
        for doc_id, content, metadata in test_corpus:
            engine.index_document(doc_id, content, metadata)

        yield engine
        engine.close()

    @patch_ml_dependencies()
    def test_relevance_basic_queries(self, search_engine_with_corpus):
        """Test relevance for basic queries."""
        test_cases = [
            RelevanceTestCase(
                query="API security authentication",
                relevant_docs=["sec-api-001", "test-api-001"],
                irrelevant_docs=["perf-db-001"],
                description="Should find API security documents",
            ),
            RelevanceTestCase(
                query="React testing Jest",
                relevant_docs=["test-react-001"],
                irrelevant_docs=["sec-web-001", "perf-db-001"],
                description="Should find React testing documents",
            ),
            RelevanceTestCase(
                query="database performance optimization indexing",
                relevant_docs=["perf-db-001"],
                irrelevant_docs=["test-react-001"],
                description="Should find database performance documents",
            ),
        ]

        for test_case in test_cases:
            results = search_engine_with_corpus.search(test_case.query, top_k=10)
            result_ids = [r.id for r in results]

            # Calculate metrics
            precision = RelevanceMetrics.precision_at_k(
                results, test_case.relevant_docs, 5
            )
            recall = RelevanceMetrics.recall_at_k(results, test_case.relevant_docs, 5)
            ndcg = RelevanceMetrics.ndcg_at_k(results, test_case.relevant_docs, 5)

            print(f"\n{test_case.description}")
            print(f"Query: {test_case.query}")
            print(f"Results: {result_ids[:5]}")
            print(
                f"Precision@5: {precision:.2f}, Recall@5: {recall:.2f}, NDCG@5: {ndcg:.2f}"
            )

            # Assertions
            assert any(
                doc_id in result_ids[:5] for doc_id in test_case.relevant_docs
            ), f"None of {test_case.relevant_docs} found in top 5 results"

            # Irrelevant docs should not be in top results
            for irrelevant_id in test_case.irrelevant_docs:
                if irrelevant_id in result_ids:
                    rank = result_ids.index(irrelevant_id)
                    assert (
                        rank > 3
                    ), f"Irrelevant doc {irrelevant_id} ranked too high at position {rank}"

    def test_relevance_with_synonyms(self, search_engine_with_corpus):
        """Test relevance with synonym expansion."""
        synonym_queries = [
            ("web application safety", ["sec-web-001"]),  # safety -> security
            ("frontend unit tests", ["test-react-001"]),  # frontend -> react context
            ("API endpoints validation", ["test-api-001"]),  # Related terms
            ("database speed improvements", ["perf-db-001"]),  # speed -> performance
        ]

        for query, expected_docs in synonym_queries:
            results = search_engine_with_corpus.search(query, top_k=5)
            result_ids = [r.id for r in results]

            # Should find documents even with synonyms
            found = any(doc_id in result_ids for doc_id in expected_docs)
            assert found, f"Query '{query}' did not find expected docs {expected_docs}"

    def test_ranking_quality_metrics(self, search_engine_with_corpus):
        """Test overall ranking quality across multiple queries."""
        # Define relevance judgments
        relevance_data = [
            {
                "query": "security best practices",
                "relevant": ["sec-api-001", "sec-web-001"],
                "highly_relevant": ["sec-api-001"],  # More comprehensive security doc
            },
            {
                "query": "testing standards coverage",
                "relevant": ["test-react-001", "test-api-001"],
                "highly_relevant": ["test-react-001"],  # Mentions coverage explicitly
            },
            {
                "query": "performance optimization caching",
                "relevant": ["perf-db-001", "perf-web-001"],
                "highly_relevant": ["perf-db-001"],  # Mentions caching explicitly
            },
        ]

        aggregate_metrics = defaultdict(list)

        for relevance_case in relevance_data:
            query = relevance_case["query"]
            relevant_docs = relevance_case["relevant"]

            results = search_engine_with_corpus.search(query, top_k=10)

            # Calculate various metrics
            metrics = {
                "precision@1": RelevanceMetrics.precision_at_k(
                    results, relevant_docs, 1
                ),
                "precision@3": RelevanceMetrics.precision_at_k(
                    results, relevant_docs, 3
                ),
                "precision@5": RelevanceMetrics.precision_at_k(
                    results, relevant_docs, 5
                ),
                "recall@5": RelevanceMetrics.recall_at_k(results, relevant_docs, 5),
                "f1@5": RelevanceMetrics.f1_at_k(results, relevant_docs, 5),
                "ndcg@5": RelevanceMetrics.ndcg_at_k(results, relevant_docs, 5),
                "mrr": RelevanceMetrics.mean_reciprocal_rank(results, relevant_docs),
            }

            # Store for aggregation
            for metric, value in metrics.items():
                aggregate_metrics[metric].append(value)

            print(f"\nQuery: {query}")
            print(f"Metrics: {json.dumps(metrics, indent=2)}")

        # Calculate aggregate metrics
        avg_metrics = {
            metric: np.mean(values) for metric, values in aggregate_metrics.items()
        }

        print(f"\nAggregate metrics across {len(relevance_data)} queries:")
        print(json.dumps(avg_metrics, indent=2))

        # Quality assertions
        # Note: With only 2 relevant documents per query, max precision@5 is 0.4
        assert avg_metrics["precision@5"] >= 0.4, "Average precision@5 too low"
        assert avg_metrics["recall@5"] >= 0.7, "Average recall@5 too low"
        assert avg_metrics["ndcg@5"] >= 0.6, "Average NDCG@5 too low"
        assert avg_metrics["mrr"] >= 0.7, "Average MRR too low"


class TestSemanticUnderstanding:
    """Test semantic understanding capabilities."""

    @pytest.fixture
    def search_engine(self):
        """Create search engine with diverse corpus."""
        engine = create_search_engine()

        # Generate diverse corpus
        docs = TestDataGenerator.generate_standards_corpus(200)
        engine.index_documents_batch(docs)

        yield engine
        engine.close()

    def test_semantic_similarity(self, search_engine):
        """Test semantic similarity understanding."""
        semantic_test_cases = [
            SemanticTestCase(
                query="implement user authentication",
                semantic_equivalents=[
                    "user login implementation",
                    "authentication system setup",
                    "implement auth for users",
                ],
                semantic_opposites=[
                    "remove authentication",
                    "disable user login",
                    "public access without auth",
                ],
                description="Authentication implementation queries",
            ),
            SemanticTestCase(
                query="optimize database performance",
                semantic_equivalents=[
                    "improve database speed",
                    "database optimization techniques",
                    "make database faster",
                ],
                semantic_opposites=[
                    "remove database entirely",
                    "frontend UI styling",
                    "network routing protocols",
                ],
                description="Database optimization queries",
            ),
        ]

        for test_case in semantic_test_cases:
            print(f"\n{test_case.description}")

            # Get baseline results
            baseline_results = search_engine.search(test_case.query, top_k=10)
            baseline_ids = {r.id for r in baseline_results[:5]}

            # Test semantic equivalents
            for equivalent in test_case.semantic_equivalents:
                equiv_results = search_engine.search(equivalent, top_k=10)
                equiv_ids = {r.id for r in equiv_results[:5]}

                # Calculate overlap
                overlap = (
                    len(baseline_ids & equiv_ids) / len(baseline_ids)
                    if baseline_ids
                    else 0
                )

                print(f"Query: '{equivalent}' - Overlap: {overlap:.2f}")
                assert overlap >= 0.4, "Low overlap for semantically equivalent query"

            # Test semantic opposites
            for opposite in test_case.semantic_opposites:
                opp_results = search_engine.search(opposite, top_k=10)
                opp_ids = {r.id for r in opp_results[:5]}

                # Calculate overlap (should be low)
                overlap = (
                    len(baseline_ids & opp_ids) / len(baseline_ids)
                    if baseline_ids
                    else 0
                )

                print(f"Opposite: '{opposite}' - Overlap: {overlap:.2f}")
                assert overlap <= 0.3, "High overlap for semantically opposite query"

    def test_context_understanding(self, search_engine):
        """Test understanding of context and domain-specific terms."""
        # Index documents with specific contexts
        context_docs = [
            (
                "ctx-001",
                "React hooks like useState and useEffect for state management",
                {"context": "react", "topic": "hooks"},
            ),
            (
                "ctx-002",
                "Git hooks for pre-commit and post-commit automation",
                {"context": "git", "topic": "hooks"},
            ),
            (
                "ctx-003",
                "Database triggers and stored procedures for data integrity",
                {"context": "database", "topic": "automation"},
            ),
        ]

        for doc_id, content, metadata in context_docs:
            search_engine.index_document(doc_id, content, metadata)

        # Test context-aware searches
        context_queries = [
            ("React hooks", "ctx-001"),  # Should find React hooks, not Git hooks
            ("Git hooks", "ctx-002"),  # Should find Git hooks, not React hooks
            ("database automation", "ctx-003"),  # Should find database triggers
        ]

        for query, expected_top in context_queries:
            results = search_engine.search(query, top_k=3)

            if results:
                top_result = results[0].id
                print(f"\nQuery: '{query}' - Top result: {top_result}")
                assert (
                    top_result == expected_top
                ), f"Expected {expected_top} as top result for '{query}'"

    def test_query_intent_understanding(self, search_engine):
        """Test understanding of different query intents."""
        # Index documents for different intents
        intent_docs = [
            (
                "how-001",
                "How to implement REST API authentication step by step guide",
                {"intent": "tutorial", "topic": "api-auth"},
            ),
            (
                "what-001",
                "What is OAuth 2.0 and why use it for API authentication",
                {"intent": "explanation", "topic": "api-auth"},
            ),
            (
                "best-001",
                "Best practices for secure API authentication methods",
                {"intent": "best-practices", "topic": "api-auth"},
            ),
            (
                "debug-001",
                "Debugging common API authentication errors and issues",
                {"intent": "troubleshooting", "topic": "api-auth"},
            ),
        ]

        for doc_id, content, metadata in intent_docs:
            search_engine.index_document(doc_id, content, metadata)

        # Test intent-based queries
        intent_queries = [
            ("how to implement API authentication", "how-001"),
            ("what is oauth authentication", "what-001"),
            ("API authentication best practices", "best-001"),
            ("debug API auth errors", "debug-001"),
        ]

        for query, expected_top in intent_queries:
            results = search_engine.search(query, top_k=3)

            if results:
                top_result = results[0].id
                print(f"\nIntent query: '{query}' - Top result: {top_result}")

                # The expected document should be in top 2
                top_2_ids = [r.id for r in results[:2]]
                assert (
                    expected_top in top_2_ids
                ), f"Expected {expected_top} in top 2 for intent query '{query}'"


class TestQueryOperatorAccuracy:
    """Test accuracy of boolean operators and filters."""

    @pytest.fixture
    def search_engine_with_tagged_corpus(self):
        """Create search engine with well-tagged corpus."""
        engine = create_search_engine()

        # Create corpus with clear distinctions
        tagged_docs = [
            # JavaScript docs
            (
                "js-001",
                "JavaScript async/await patterns for modern web development",
                {"language": "javascript", "topic": "async", "level": "intermediate"},
            ),
            (
                "js-002",
                "JavaScript testing with Jest and React Testing Library",
                {"language": "javascript", "topic": "testing", "level": "beginner"},
            ),
            (
                "js-003",
                "Advanced JavaScript performance optimization techniques",
                {"language": "javascript", "topic": "performance", "level": "advanced"},
            ),
            # Python docs
            (
                "py-001",
                "Python async programming with asyncio and aiohttp",
                {"language": "python", "topic": "async", "level": "intermediate"},
            ),
            (
                "py-002",
                "Python testing with pytest and test-driven development",
                {"language": "python", "topic": "testing", "level": "beginner"},
            ),
            (
                "py-003",
                "Python performance profiling and optimization strategies",
                {"language": "python", "topic": "performance", "level": "advanced"},
            ),
            # Java docs
            (
                "java-001",
                "Java CompletableFuture for asynchronous programming",
                {"language": "java", "topic": "async", "level": "intermediate"},
            ),
            (
                "java-002",
                "Java unit testing with JUnit 5 and Mockito",
                {"language": "java", "topic": "testing", "level": "beginner"},
            ),
        ]

        for doc_id, content, metadata in tagged_docs:
            engine.index_document(doc_id, content, metadata)

        yield engine
        engine.close()

    def test_and_operator_accuracy(self, search_engine_with_tagged_corpus):
        """Test AND operator accuracy."""
        test_cases = [
            {
                "query": "javascript AND testing",
                "expected_ids": ["js-002"],
                "excluded_ids": ["py-002", "java-002"],  # Other language testing docs
            },
            {
                "query": "async AND python",
                "expected_ids": ["py-001"],
                "excluded_ids": ["js-001", "java-001"],  # Other language async docs
            },
            {
                "query": "performance AND optimization",
                "expected_ids": ["js-003", "py-003"],  # Both mention optimization
                "excluded_ids": ["js-002", "py-002"],  # Testing docs
            },
        ]

        for test in test_cases:
            results = search_engine_with_tagged_corpus.search(test["query"], top_k=10)
            result_ids = [r.id for r in results]

            print(f"\nAND Query: {test['query']}")
            print(f"Results: {result_ids[:5]}")

            # Check expected documents are found
            for expected_id in test["expected_ids"]:
                assert (
                    expected_id in result_ids
                ), f"Expected {expected_id} not found for query '{test['query']}'"

            # Check excluded documents are not in top results
            top_3_ids = result_ids[:3]
            for excluded_id in test["excluded_ids"]:
                assert (
                    excluded_id not in top_3_ids
                ), f"Excluded {excluded_id} found in top 3 for query '{test['query']}'"

    def test_not_operator_accuracy(self, search_engine_with_tagged_corpus):
        """Test NOT operator accuracy."""
        test_cases = [
            {
                "query": "testing NOT javascript",
                "expected_present": ["py-002", "java-002"],
                "expected_absent": ["js-002"],
            },
            {
                "query": "async NOT java",
                "expected_present": ["js-001", "py-001"],
                "expected_absent": ["java-001"],
            },
        ]

        for test in test_cases:
            results = search_engine_with_tagged_corpus.search(test["query"], top_k=10)
            result_ids = [r.id for r in results]

            print(f"\nNOT Query: {test['query']}")
            print(f"Results: {result_ids[:5]}")

            # Check excluded documents are not present
            for excluded_id in test["expected_absent"]:
                assert (
                    excluded_id not in result_ids
                ), f"Document {excluded_id} should be excluded by NOT operator"

            # Check expected documents are present
            for expected_id in test["expected_present"]:
                assert (
                    expected_id in result_ids
                ), f"Document {expected_id} should be present"

    def test_filter_accuracy(self, search_engine_with_tagged_corpus):
        """Test metadata filter accuracy."""
        test_cases = [
            {
                "query": "testing",
                "filters": {"language": "python"},
                "expected_ids": ["py-002"],
                "excluded_ids": ["js-002", "java-002"],
            },
            {
                "query": "programming",
                "filters": {"level": "advanced"},
                "expected_ids": ["js-003", "py-003"],
                "excluded_ids": ["js-002", "py-002"],  # Beginner level
            },
            {
                "query": "development",
                "filters": {"language": ["javascript", "python"], "topic": "async"},
                "expected_ids": ["js-001", "py-001"],
                "excluded_ids": ["java-001", "js-002", "py-002"],
            },
        ]

        for test in test_cases:
            results = search_engine_with_tagged_corpus.search(
                test["query"], filters=test["filters"], top_k=10
            )
            result_ids = [r.id for r in results]

            print(f"\nFiltered Query: {test['query']} with filters {test['filters']}")
            print(f"Results: {result_ids}")

            # All results should match filters
            for result in results:
                for filter_key, filter_value in test["filters"].items():
                    doc_value = result.metadata.get(filter_key)

                    if isinstance(filter_value, list):
                        assert (
                            doc_value in filter_value
                        ), f"Document {result.id} doesn't match filter {filter_key}"
                    else:
                        assert (
                            doc_value == filter_value
                        ), f"Document {result.id} doesn't match filter {filter_key}"

            # Check expected documents are present
            for expected_id in test["expected_ids"]:
                assert (
                    expected_id in result_ids
                ), f"Expected {expected_id} not found with filters"


class TestSearchAccuracyEdgeCases:
    """Test accuracy in edge cases and challenging scenarios."""

    @pytest.fixture
    def search_engine(self):
        """Create search engine."""
        engine = create_search_engine()
        yield engine
        engine.close()

    def test_ambiguous_queries(self, search_engine):
        """Test handling of ambiguous queries."""
        # Index documents with ambiguous terms
        ambiguous_docs = [
            (
                "bank-001",
                "River bank erosion prevention techniques for environmental protection",
                {"domain": "environment", "topic": "erosion"},
            ),
            (
                "bank-002",
                "Bank account security best practices for online banking",
                {"domain": "finance", "topic": "security"},
            ),
            (
                "spring-001",
                "Spring framework dependency injection patterns in Java",
                {"domain": "programming", "topic": "framework"},
            ),
            (
                "spring-002",
                "Spring season gardening tips and plant care guide",
                {"domain": "gardening", "topic": "seasons"},
            ),
        ]

        for doc_id, content, metadata in ambiguous_docs:
            search_engine.index_document(doc_id, content, metadata)

        # Test disambiguation through context
        disambiguation_tests = [
            ("bank security", "bank-002"),  # Financial context
            ("river bank", "bank-001"),  # Environmental context
            ("Spring Java", "spring-001"),  # Programming context
            ("spring planting", "spring-002"),  # Gardening context
        ]

        for query, expected_top in disambiguation_tests:
            results = search_engine.search(query, top_k=2)

            if results:
                assert (
                    results[0].id == expected_top
                ), f"Failed to disambiguate '{query}' - expected {expected_top}"

    def test_acronym_handling(self, search_engine):
        """Test handling of acronyms and abbreviations."""
        # Index documents with acronyms
        acronym_docs = [
            (
                "api-001",
                "API (Application Programming Interface) design guidelines",
                {"topic": "api"},
            ),
            (
                "rest-001",
                "REST (Representational State Transfer) API best practices",
                {"topic": "rest"},
            ),
            (
                "crud-001",
                "CRUD operations (Create, Read, Update, Delete) implementation",
                {"topic": "crud"},
            ),
            (
                "ci-001",
                "CI/CD (Continuous Integration/Continuous Deployment) pipeline setup",
                {"topic": "cicd"},
            ),
        ]

        for doc_id, content, metadata in acronym_docs:
            search_engine.index_document(doc_id, content, metadata)

        # Test acronym searches
        acronym_tests = [
            ("API", "api-001"),
            ("Application Programming Interface", "api-001"),
            ("REST", "rest-001"),
            ("CRUD operations", "crud-001"),
            ("continuous integration", "ci-001"),
        ]

        for query, expected_doc in acronym_tests:
            results = search_engine.search(query, top_k=3)
            result_ids = [r.id for r in results]

            assert (
                expected_doc in result_ids[:2]
            ), f"Failed to find {expected_doc} for acronym query '{query}'"

    def test_multilingual_terms(self, search_engine):
        """Test handling of multilingual technical terms."""
        # Index documents with multilingual terms
        multilingual_docs = [
            (
                "color-001",
                "CSS color and colour styling properties guide",
                {"variant": "both"},
            ),
            (
                "optimize-001",
                "Optimize and optimise database performance",
                {"variant": "both"},
            ),
            (
                "center-001",
                "Center and centre alignment in web design",
                {"variant": "both"},
            ),
        ]

        for doc_id, content, metadata in multilingual_docs:
            search_engine.index_document(doc_id, content, metadata)

        # Test variant searches
        variant_tests = [
            ("color properties", "color-001"),
            ("colour properties", "color-001"),  # British spelling
            ("optimize database", "optimize-001"),
            ("optimise database", "optimize-001"),  # British spelling
        ]

        for query, expected_doc in variant_tests:
            results = search_engine.search(query, top_k=3)
            result_ids = [r.id for r in results]

            assert (
                expected_doc in result_ids[:2]
            ), f"Failed to find {expected_doc} for variant query '{query}'"


def test_accuracy_benchmarks():
    """
    Run comprehensive accuracy benchmarks and save results.

    This can be used to track accuracy improvements over time.
    """
    with patch_ml_dependencies():
        engine = create_search_engine()

        # Create evaluation dataset
        eval_docs = [
            # Create diverse test corpus
            (
                "eval-001",
                "Python web development with Django and Flask frameworks",
                {"language": "python", "category": "web", "level": "intermediate"},
            ),
            (
                "eval-002",
                "JavaScript modern frontend development with React and Vue",
                {
                    "language": "javascript",
                    "category": "frontend",
                    "level": "intermediate",
                },
            ),
            (
                "eval-003",
                "Database design patterns and SQL optimization techniques",
                {"language": "sql", "category": "database", "level": "advanced"},
            ),
            (
                "eval-004",
                "RESTful API design best practices and standards",
                {"language": "agnostic", "category": "api", "level": "intermediate"},
            ),
            (
                "eval-005",
                "Microservices architecture patterns and deployment",
                {
                    "language": "agnostic",
                    "category": "architecture",
                    "level": "advanced",
                },
            ),
            (
                "eval-006",
                "Security best practices for web applications",
                {
                    "language": "agnostic",
                    "category": "security",
                    "level": "intermediate",
                },
            ),
            (
                "eval-007",
                "Test-driven development and unit testing strategies",
                {"language": "agnostic", "category": "testing", "level": "beginner"},
            ),
            (
                "eval-008",
                "DevOps practices and CI/CD pipeline configuration",
                {"language": "agnostic", "category": "devops", "level": "intermediate"},
            ),
        ]

        # Index evaluation documents
        for doc_id, content, metadata in eval_docs:
            engine.index_document(doc_id, content, metadata)

        # Define evaluation queries with relevance judgments
        eval_queries = [
            {
                "query": "Python web development",
                "relevant": ["eval-001"],
                "somewhat_relevant": ["eval-002", "eval-006"],
            },
            {
                "query": "frontend frameworks JavaScript",
                "relevant": ["eval-002"],
                "somewhat_relevant": ["eval-001"],
            },
            {
                "query": "API design security",
                "relevant": ["eval-004", "eval-006"],
                "somewhat_relevant": ["eval-005"],
            },
            {
                "query": "testing strategies TDD",
                "relevant": ["eval-007"],
                "somewhat_relevant": ["eval-008"],
            },
            {
                "query": "database optimization SQL",
                "relevant": ["eval-003"],
                "somewhat_relevant": ["eval-005"],
            },
        ]

        # Run evaluation
        results: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "queries": len(eval_queries),
            "metrics": defaultdict(list),
        }

        for eval_case in eval_queries:
            query = eval_case["query"]
            relevant = eval_case["relevant"]

            # Search
            search_results = engine.search(query, top_k=10)

            # Calculate metrics
            metrics = {
                "precision@1": RelevanceMetrics.precision_at_k(
                    search_results, relevant, 1
                ),
                "precision@3": RelevanceMetrics.precision_at_k(
                    search_results, relevant, 3
                ),
                "precision@5": RelevanceMetrics.precision_at_k(
                    search_results, relevant, 5
                ),
                "recall@5": RelevanceMetrics.recall_at_k(search_results, relevant, 5),
                "ndcg@5": RelevanceMetrics.ndcg_at_k(search_results, relevant, 5),
                "mrr": RelevanceMetrics.mean_reciprocal_rank(search_results, relevant),
            }

            # Store metrics
            for metric, value in metrics.items():
                results["metrics"][metric].append(value)

        # Calculate averages
        results["average_metrics"] = {
            metric: np.mean(values) for metric, values in results["metrics"].items()
        }

        # Save results
        output_path = Path("accuracy_benchmark.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nAccuracy benchmark results saved to {output_path}")
        print("\nAverage metrics:")
        for metric, value in results["average_metrics"].items():
            print(f"{metric}: {value:.3f}")

        engine.close()

        # Assert minimum accuracy requirements
        assert results["average_metrics"]["precision@5"] >= 0.5
        assert results["average_metrics"]["recall@5"] >= 0.6
        assert results["average_metrics"]["ndcg@5"] >= 0.5
        assert results["average_metrics"]["mrr"] >= 0.6
