"""
Performance benchmark for enhanced semantic search.
"""

import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.standards.semantic_search import create_search_engine


class SemanticSearchBenchmark:
    """Benchmark suite for semantic search performance."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.results = {}

    def cleanup(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)

    def generate_test_documents(self, count: int) -> list:
        """Generate test documents for benchmarking."""
        topics = [
            "React", "Angular", "Vue", "Python", "JavaScript", "TypeScript",
            "API", "REST", "GraphQL", "Testing", "Security", "Performance",
            "Database", "SQL", "NoSQL", "Docker", "Kubernetes", "Cloud"
        ]

        categories = ["frontend", "backend", "testing", "security", "devops", "database"]

        documents = []
        for i in range(count):
            # Generate realistic content
            main_topic = random.choice(topics)
            secondary_topic = random.choice(topics)
            category = random.choice(categories)

            content = f"""
            {main_topic} Best Practices and Standards

            This document covers {main_topic} development guidelines and {secondary_topic} integration.

            Key Topics:
            - {main_topic} architecture patterns
            - {secondary_topic} compatibility
            - Performance optimization for {category} applications
            - Security considerations in {main_topic}
            - Testing strategies for {main_topic} and {secondary_topic}

            Implementation Guidelines:
            Follow these practices when working with {main_topic}:
            1. Use proper error handling
            2. Implement comprehensive testing
            3. Follow {category} best practices
            4. Optimize for performance
            5. Ensure security compliance

            Related Standards: {secondary_topic}, {category}, {random.choice(topics)}
            """

            metadata = {
                "id": f"doc-{i:05d}",
                "category": category,
                "main_topic": main_topic,
                "secondary_topic": secondary_topic,
                "index": i
            }

            documents.append((metadata["id"], content, metadata))

        return documents

    def benchmark_indexing(self):
        """Benchmark document indexing performance."""
        print("\n=== Indexing Performance Benchmark ===")

        search = create_search_engine(cache_dir=Path(self.temp_dir))

        # Test different batch sizes
        batch_sizes = [10, 50, 100, 500, 1000]
        indexing_times = []

        for size in batch_sizes:
            documents = self.generate_test_documents(size)

            start = time.time()
            search.index_documents_batch(documents)
            elapsed = time.time() - start

            indexing_times.append(elapsed)
            docs_per_second = size / elapsed

            print(f"Indexed {size:4d} documents in {elapsed:6.2f}s "
                  f"({docs_per_second:6.1f} docs/sec)")

        self.results['indexing'] = {
            'batch_sizes': batch_sizes,
            'times': indexing_times
        }

        search.close()
        return batch_sizes, indexing_times

    def benchmark_search_latency(self):
        """Benchmark search latency with various configurations."""
        print("\n=== Search Latency Benchmark ===")

        # Create and populate search engine
        search = create_search_engine(cache_dir=Path(self.temp_dir))
        documents = self.generate_test_documents(1000)
        search.index_documents_batch(documents)

        # Generate test queries
        test_queries = [
            "React testing best practices",
            "Python API security",
            "JavaScript performance optimization",
            "Docker Kubernetes deployment",
            "Database SQL optimization",
            "Frontend Vue components",
            "Backend REST GraphQL",
            "Testing strategies patterns",
            "Security compliance standards",
            "Cloud architecture patterns"
        ]

        configurations = [
            ("Basic", {"use_fuzzy": False, "rerank": False, "use_cache": False}),
            ("With Fuzzy", {"use_fuzzy": True, "rerank": False, "use_cache": False}),
            ("With Rerank", {"use_fuzzy": False, "rerank": True, "use_cache": False}),
            ("Full Features", {"use_fuzzy": True, "rerank": True, "use_cache": False}),
            ("With Cache", {"use_fuzzy": True, "rerank": True, "use_cache": True}),
        ]

        results = {}

        for config_name, config in configurations:
            latencies = []

            # Run each query multiple times
            for query in test_queries:
                for _ in range(3):  # 3 runs per query
                    start = time.time()
                    search.search(query, top_k=10, **config)
                    latency = (time.time() - start) * 1000  # Convert to ms
                    latencies.append(latency)

            avg_latency = statistics.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            results[config_name] = {
                'avg': avg_latency,
                'p95': p95_latency,
                'all': latencies
            }

            print(f"{config_name:15s}: avg={avg_latency:6.2f}ms, p95={p95_latency:6.2f}ms")

        self.results['latency'] = results
        search.close()
        return results

    def benchmark_cache_effectiveness(self):
        """Benchmark cache effectiveness."""
        print("\n=== Cache Effectiveness Benchmark ===")

        search = create_search_engine(cache_dir=Path(self.temp_dir))
        documents = self.generate_test_documents(500)
        search.index_documents_batch(documents)

        # Test queries
        queries = [
            "React component testing",
            "Python API development",
            "JavaScript security"
        ]

        # First pass - no cache
        first_pass_times = []
        for query in queries:
            start = time.time()
            search.search(query, top_k=10, use_cache=True)
            first_pass_times.append(time.time() - start)

        # Second pass - with cache
        second_pass_times = []
        for query in queries:
            start = time.time()
            search.search(query, top_k=10, use_cache=True)
            second_pass_times.append(time.time() - start)

        # Calculate speedup
        speedups = []
        for first, second in zip(first_pass_times, second_pass_times, strict=False):
            speedup = first / second if second > 0 else 1.0
            speedups.append(speedup)

        avg_speedup = statistics.mean(speedups)

        print(f"Average cache speedup: {avg_speedup:.2f}x")
        print(f"First pass avg: {statistics.mean(first_pass_times)*1000:.2f}ms")
        print(f"Second pass avg: {statistics.mean(second_pass_times)*1000:.2f}ms")

        # Get cache statistics
        report = search.get_analytics_report()
        print(f"Cache hit rate: {report['cache_hit_rate']:.2%}")

        self.results['cache'] = {
            'first_pass': first_pass_times,
            'second_pass': second_pass_times,
            'speedups': speedups,
            'hit_rate': report['cache_hit_rate']
        }

        search.close()
        return speedups

    def benchmark_fuzzy_matching(self):
        """Benchmark fuzzy matching accuracy and performance."""
        print("\n=== Fuzzy Matching Benchmark ===")

        search = create_search_engine(cache_dir=Path(self.temp_dir))
        documents = self.generate_test_documents(500)
        search.index_documents_batch(documents)

        # Test cases with typos
        test_cases = [
            ("React component testing", "Reakt componet testng"),  # Multiple typos
            ("Python API security", "Pyton API securty"),  # Minor typos
            ("JavaScript performance", "JavaScrpt performence"),  # Common mistakes
            ("Database optimization", "Databse optimizaton"),  # Missing letters
            ("Frontend development", "Frntend developmnt"),  # Vowel errors
        ]

        accuracies = []
        time_differences = []

        for correct_query, typo_query in test_cases:
            # Search with correct query
            correct_results = search.search(correct_query, top_k=5, use_fuzzy=False)
            correct_ids = [r.id for r in correct_results]

            # Search with typo query (no fuzzy)
            start = time.time()
            search.search(typo_query, top_k=5, use_fuzzy=False)
            no_fuzzy_time = time.time() - start

            # Search with typo query (with fuzzy)
            start = time.time()
            typo_results_fuzzy = search.search(typo_query, top_k=5, use_fuzzy=True)
            fuzzy_time = time.time() - start

            fuzzy_ids = [r.id for r in typo_results_fuzzy]

            # Calculate accuracy (how many correct results were found)
            matches = len(set(correct_ids[:3]) & set(fuzzy_ids[:3]))
            accuracy = matches / 3.0
            accuracies.append(accuracy)

            time_diff = (fuzzy_time - no_fuzzy_time) * 1000
            time_differences.append(time_diff)

            print(f"Query: '{typo_query[:30]}...' - Accuracy: {accuracy:.2%}, "
                  f"Time diff: {time_diff:.2f}ms")

        avg_accuracy = statistics.mean(accuracies)
        avg_time_diff = statistics.mean(time_differences)

        print(f"\nAverage fuzzy matching accuracy: {avg_accuracy:.2%}")
        print(f"Average additional latency: {avg_time_diff:.2f}ms")

        self.results['fuzzy'] = {
            'accuracies': accuracies,
            'time_differences': time_differences
        }

        search.close()
        return accuracies, time_differences

    def benchmark_scalability(self):
        """Benchmark scalability with increasing document counts."""
        print("\n=== Scalability Benchmark ===")

        document_counts = [100, 500, 1000, 2000, 5000]
        search_times = []
        memory_usage = []

        for count in document_counts:
            # Create fresh search engine
            search = create_search_engine(cache_dir=Path(self.temp_dir))

            # Index documents
            documents = self.generate_test_documents(count)
            search.index_documents_batch(documents)

            # Measure search performance
            queries = ["testing best practices", "security guidelines", "performance optimization"]
            times = []

            for query in queries:
                start = time.time()
                search.search(query, top_k=10)
                times.append(time.time() - start)

            avg_time = statistics.mean(times)
            search_times.append(avg_time)

            # Estimate memory usage (simplified)
            import sys
            mem_usage = sys.getsizeof(search.documents) + sys.getsizeof(search.document_embeddings)
            memory_usage.append(mem_usage / 1024 / 1024)  # Convert to MB

            print(f"{count:5d} documents: {avg_time*1000:6.2f}ms avg search, "
                  f"{memory_usage[-1]:6.1f}MB memory")

            search.close()

        self.results['scalability'] = {
            'document_counts': document_counts,
            'search_times': search_times,
            'memory_usage': memory_usage
        }

        return document_counts, search_times

    def generate_report(self):
        """Generate a visual benchmark report."""
        print("\n=== Generating Benchmark Report ===")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Semantic Search Performance Benchmarks', fontsize=16)

        # 1. Indexing Performance
        if 'indexing' in self.results:
            ax = axes[0, 0]
            data = self.results['indexing']
            ax.plot(data['batch_sizes'], data['times'], 'b-o')
            ax.set_xlabel('Batch Size')
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Indexing Performance')
            ax.grid(True)

        # 2. Search Latency Comparison
        if 'latency' in self.results:
            ax = axes[0, 1]
            data = self.results['latency']
            configs = list(data.keys())
            avg_latencies = [data[c]['avg'] for c in configs]
            ax.bar(configs, avg_latencies)
            ax.set_ylabel('Latency (ms)')
            ax.set_title('Search Latency by Configuration')
            ax.tick_params(axis='x', rotation=45)

        # 3. Cache Speedup
        if 'cache' in self.results:
            ax = axes[0, 2]
            data = self.results['cache']
            speedups = data['speedups']
            ax.bar(range(len(speedups)), speedups)
            ax.axhline(y=1, color='r', linestyle='--', label='No speedup')
            ax.set_xlabel('Query')
            ax.set_ylabel('Speedup Factor')
            ax.set_title('Cache Speedup')
            ax.legend()

        # 4. Fuzzy Matching Accuracy
        if 'fuzzy' in self.results:
            ax = axes[1, 0]
            data = self.results['fuzzy']
            ax.bar(range(len(data['accuracies'])), data['accuracies'])
            ax.set_xlabel('Test Case')
            ax.set_ylabel('Accuracy')
            ax.set_title('Fuzzy Matching Accuracy')
            ax.set_ylim(0, 1)

        # 5. Scalability
        if 'scalability' in self.results:
            ax = axes[1, 1]
            data = self.results['scalability']
            ax.plot(data['document_counts'],
                   [t*1000 for t in data['search_times']], 'g-o')
            ax.set_xlabel('Number of Documents')
            ax.set_ylabel('Search Time (ms)')
            ax.set_title('Search Scalability')
            ax.grid(True)

        # 6. Memory Usage
        if 'scalability' in self.results:
            ax = axes[1, 2]
            data = self.results['scalability']
            ax.plot(data['document_counts'], data['memory_usage'], 'r-o')
            ax.set_xlabel('Number of Documents')
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_title('Memory Scalability')
            ax.grid(True)

        plt.tight_layout()
        report_path = Path(self.temp_dir) / 'benchmark_report.png'
        plt.savefig(report_path, dpi=300, bbox_inches='tight')
        print(f"Report saved to: {report_path}")

        # Also save raw results
        import json
        results_path = Path(self.temp_dir) / 'benchmark_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    serializable_results[key] = {}
                    for k, v in value.items():
                        if isinstance(v, list | tuple):
                            serializable_results[key][k] = list(v)
                        else:
                            serializable_results[key][k] = v
                else:
                    serializable_results[key] = value

            json.dump(serializable_results, f, indent=2)
        print(f"Raw results saved to: {results_path}")

    def run_all_benchmarks(self):
        """Run all benchmarks."""
        print("Starting Semantic Search Benchmarks...")
        print("=" * 50)

        self.benchmark_indexing()
        self.benchmark_search_latency()
        self.benchmark_cache_effectiveness()
        self.benchmark_fuzzy_matching()
        self.benchmark_scalability()

        self.generate_report()

        print("\n" + "=" * 50)
        print("Benchmarks completed!")
        print(f"Results saved in: {self.temp_dir}")


if __name__ == "__main__":
    benchmark = SemanticSearchBenchmark()
    try:
        benchmark.run_all_benchmarks()
    finally:
        # Uncomment to clean up temp files
        # benchmark.cleanup()
        pass
