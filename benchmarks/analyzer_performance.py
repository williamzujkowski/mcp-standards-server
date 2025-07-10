"""Benchmark script for language analyzers."""

import statistics
import time
from pathlib import Path

from src.analyzers.base import AnalyzerPlugin
from src.analyzers.go_analyzer import GoAnalyzer
from src.analyzers.java_analyzer import JavaAnalyzer
from src.analyzers.rust_analyzer import RustAnalyzer
from src.analyzers.typescript_analyzer import TypeScriptAnalyzer


class AnalyzerBenchmark:
    """Benchmark analyzer performance."""

    def __init__(self):
        self.examples_dir = Path(__file__).parent.parent / "examples" / "analyzer-test-samples"
        self.results = {}

    def run_benchmarks(self, iterations: int = 10):
        """Run benchmarks for all analyzers."""
        print(f"Running analyzer benchmarks ({iterations} iterations each)...\n")

        test_files = {
            "go": self.examples_dir / "go-example.go",
            "java": self.examples_dir / "java-example.java",
            "rust": self.examples_dir / "rust-example.rs",
            "typescript": self.examples_dir / "typescript-example.tsx"
        }

        for language, file_path in test_files.items():
            if not file_path.exists():
                print(f"Warning: {file_path} not found, skipping {language}")
                continue

            analyzer = AnalyzerPlugin.get_analyzer(language)
            if not analyzer:
                print(f"Warning: No analyzer for {language}")
                continue

            print(f"Benchmarking {language} analyzer...")
            times = []

            for i in range(iterations):
                start_time = time.time()
                result = analyzer.analyze_file(file_path)
                end_time = time.time()

                elapsed = end_time - start_time
                times.append(elapsed)

                if i == 0:  # First run - collect stats
                    self.results[language] = {
                        "file_size": file_path.stat().st_size,
                        "lines": len(file_path.read_text().splitlines()),
                        "issues_found": len(result.issues),
                        "issue_breakdown": self._get_issue_breakdown(result)
                    }

            # Calculate statistics
            self.results[language].update({
                "min_time": min(times),
                "max_time": max(times),
                "avg_time": statistics.mean(times),
                "median_time": statistics.median(times),
                "std_dev": statistics.stdev(times) if len(times) > 1 else 0
            })

        self._print_results()

    def _get_issue_breakdown(self, result) -> dict[str, int]:
        """Get breakdown of issues by type."""
        breakdown = {}
        for issue in result.issues:
            issue_type = issue.type.value
            breakdown[issue_type] = breakdown.get(issue_type, 0) + 1
        return breakdown

    def _print_results(self):
        """Print benchmark results."""
        print("\n" + "=" * 80)
        print("ANALYZER BENCHMARK RESULTS")
        print("=" * 80 + "\n")

        for language, data in self.results.items():
            print(f"{language.upper()} Analyzer")
            print("-" * 40)
            print(f"File size: {data['file_size']:,} bytes")
            print(f"Lines of code: {data['lines']}")
            print(f"Total issues found: {data['issues_found']}")

            if data['issue_breakdown']:
                print("\nIssue breakdown:")
                for issue_type, count in sorted(data['issue_breakdown'].items()):
                    print(f"  - {issue_type}: {count}")

            print("\nPerformance (seconds):")
            print(f"  - Min: {data['min_time']:.4f}")
            print(f"  - Max: {data['max_time']:.4f}")
            print(f"  - Average: {data['avg_time']:.4f}")
            print(f"  - Median: {data['median_time']:.4f}")
            print(f"  - Std Dev: {data['std_dev']:.4f}")

            # Calculate throughput
            throughput = data['lines'] / data['avg_time']
            print(f"  - Throughput: {throughput:.0f} lines/second")
            print()

    def compare_analyzers(self):
        """Compare analyzer performance."""
        if not self.results:
            print("No results to compare. Run benchmarks first.")
            return

        print("\nCOMPARATIVE ANALYSIS")
        print("=" * 80 + "\n")

        # Speed comparison
        print("Speed Ranking (fastest to slowest):")
        speed_ranking = sorted(
            [(lang, data['avg_time']) for lang, data in self.results.items()],
            key=lambda x: x[1]
        )

        for i, (lang, avg_time) in enumerate(speed_ranking, 1):
            print(f"{i}. {lang}: {avg_time:.4f}s average")

        # Issue detection comparison
        print("\nIssue Detection:")
        for lang, data in self.results.items():
            issues_per_line = data['issues_found'] / data['lines']
            print(f"- {lang}: {issues_per_line:.2f} issues per line")

        # Efficiency score (issues found per second)
        print("\nEfficiency (issues detected per second):")
        efficiency = []
        for lang, data in self.results.items():
            eff = data['issues_found'] / data['avg_time']
            efficiency.append((lang, eff))

        efficiency.sort(key=lambda x: x[1], reverse=True)
        for lang, eff in efficiency:
            print(f"- {lang}: {eff:.1f} issues/second")


def main():
    """Run the benchmark."""
    benchmark = AnalyzerBenchmark()

    # Ensure analyzers are registered
    _ = GoAnalyzer()
    _ = JavaAnalyzer()
    _ = RustAnalyzer()
    _ = TypeScriptAnalyzer()

    print("Language Analyzer Performance Benchmark")
    print("======================================\n")

    # Run benchmarks
    benchmark.run_benchmarks(iterations=10)

    # Compare results
    benchmark.compare_analyzers()

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
