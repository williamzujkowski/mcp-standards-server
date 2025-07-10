#!/usr/bin/env python3
"""
Detect performance regressions by comparing benchmark results.

Compares current benchmark results against baseline and alerts on regressions.
"""

import argparse
import json
import sys


class PerformanceRegressionDetector:
    """Detect performance regressions in benchmark results."""

    def __init__(self, threshold_percent: float = 10.0):
        self.threshold_percent = threshold_percent
        self.regressions = []
        self.improvements = []

    def compare_results(
        self,
        current: dict,
        baseline: dict
    ) -> dict[str, any]:
        """Compare current results against baseline."""
        comparison = {
            'total_benchmarks': 0,
            'regressions': [],
            'improvements': [],
            'unchanged': [],
            'new_benchmarks': [],
            'removed_benchmarks': []
        }

        # Get all benchmark names
        current_benchmarks = self._extract_benchmarks(current)
        baseline_benchmarks = self._extract_benchmarks(baseline)

        all_names = set(current_benchmarks.keys()) | set(baseline_benchmarks.keys())
        comparison['total_benchmarks'] = len(all_names)

        for name in all_names:
            if name not in baseline_benchmarks:
                comparison['new_benchmarks'].append(name)
                continue

            if name not in current_benchmarks:
                comparison['removed_benchmarks'].append(name)
                continue

            # Compare metrics
            current_metric = current_benchmarks[name]
            baseline_metric = baseline_benchmarks[name]

            change_percent = self._calculate_change_percent(
                current_metric,
                baseline_metric
            )

            result = {
                'name': name,
                'current': current_metric,
                'baseline': baseline_metric,
                'change_percent': change_percent
            }

            if abs(change_percent) < 5.0:  # Within 5% is considered unchanged
                comparison['unchanged'].append(result)
            elif change_percent > self.threshold_percent:
                comparison['regressions'].append(result)
            elif change_percent < -self.threshold_percent:
                comparison['improvements'].append(result)
            else:
                comparison['unchanged'].append(result)

        return comparison

    def _extract_benchmarks(self, results: dict) -> dict[str, float]:
        """Extract benchmark metrics from results."""
        benchmarks = {}

        # Handle different benchmark formats
        if 'benchmarks' in results:
            for bench in results['benchmarks']:
                name = bench.get('name', 'unknown')
                # Use mean time as primary metric
                if 'stats' in bench and 'mean' in bench['stats']:
                    benchmarks[name] = bench['stats']['mean']
                elif 'time' in bench:
                    benchmarks[name] = bench['time']

        elif 'tests' in results:
            # Alternative format
            for test in results['tests']:
                name = test.get('name', 'unknown')
                if 'duration' in test:
                    benchmarks[name] = test['duration']

        return benchmarks

    def _calculate_change_percent(
        self,
        current: float,
        baseline: float
    ) -> float:
        """Calculate percentage change from baseline."""
        if baseline == 0:
            return 100.0 if current > 0 else 0.0

        return ((current - baseline) / baseline) * 100

    def generate_report(self, comparison: dict) -> str:
        """Generate human-readable report."""
        lines = []
        lines.append("Performance Comparison Report")
        lines.append("=" * 40)
        lines.append(f"Total benchmarks: {comparison['total_benchmarks']}")
        lines.append("")

        if comparison['regressions']:
            lines.append(f"⚠️  REGRESSIONS ({len(comparison['regressions'])})")
            lines.append("-" * 20)
            for reg in comparison['regressions']:
                lines.append(
                    f"  {reg['name']}: "
                    f"{reg['baseline']:.3f}s → {reg['current']:.3f}s "
                    f"({reg['change_percent']:+.1f}%)"
                )
            lines.append("")

        if comparison['improvements']:
            lines.append(f"✅ IMPROVEMENTS ({len(comparison['improvements'])})")
            lines.append("-" * 20)
            for imp in comparison['improvements']:
                lines.append(
                    f"  {imp['name']}: "
                    f"{imp['baseline']:.3f}s → {imp['current']:.3f}s "
                    f"({imp['change_percent']:+.1f}%)"
                )
            lines.append("")

        if comparison['new_benchmarks']:
            lines.append(f"🆕 NEW BENCHMARKS ({len(comparison['new_benchmarks'])})")
            lines.append("-" * 20)
            for name in comparison['new_benchmarks']:
                lines.append(f"  {name}")
            lines.append("")

        if comparison['removed_benchmarks']:
            lines.append(f"🗑️  REMOVED BENCHMARKS ({len(comparison['removed_benchmarks'])})")
            lines.append("-" * 20)
            for name in comparison['removed_benchmarks']:
                lines.append(f"  {name}")
            lines.append("")

        # Summary
        lines.append("SUMMARY")
        lines.append("-" * 20)
        lines.append(f"Regressions: {len(comparison['regressions'])}")
        lines.append(f"Improvements: {len(comparison['improvements'])}")
        lines.append(f"Unchanged: {len(comparison['unchanged'])}")

        return "\n".join(lines)

    def should_fail(self, comparison: dict) -> bool:
        """Determine if the build should fail based on regressions."""
        # Fail if there are any significant regressions
        return len(comparison['regressions']) > 0


def load_benchmark_results(filepath: str) -> dict:
    """Load benchmark results from file."""
    with open(filepath) as f:
        return json.load(f)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Detect performance regressions in benchmark results'
    )
    parser.add_argument(
        'current',
        help='Path to current benchmark results'
    )
    parser.add_argument(
        'baseline',
        help='Path to baseline benchmark results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=10.0,
        help='Regression threshold percentage (default: 10.0)'
    )
    parser.add_argument(
        '--output',
        choices=['text', 'json', 'markdown'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with error code if regressions detected'
    )

    args = parser.parse_args()

    # Load results
    try:
        current_results = load_benchmark_results(args.current)
        baseline_results = load_benchmark_results(args.baseline)
    except Exception as e:
        print(f"Error loading benchmark results: {e}", file=sys.stderr)
        sys.exit(2)

    # Detect regressions
    detector = PerformanceRegressionDetector(threshold_percent=args.threshold)
    comparison = detector.compare_results(current_results, baseline_results)

    # Output results
    if args.output == 'json':
        print(json.dumps(comparison, indent=2))
    elif args.output == 'markdown':
        # Markdown format for GitHub comments
        report = detector.generate_report(comparison)
        print(report.replace('⚠️', ':warning:').replace('✅', ':white_check_mark:')
              .replace('🆕', ':new:').replace('🗑️', ':wastebasket:'))
    else:
        print(detector.generate_report(comparison))

    # Exit code
    if args.fail_on_regression and detector.should_fail(comparison):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
