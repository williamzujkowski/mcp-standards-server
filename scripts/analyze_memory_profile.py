#!/usr/bin/env python3
"""
Analyze memory profile output from tests.

Detects potential memory leaks and provides recommendations.
"""

import argparse
import re
import sys


class MemoryProfileAnalyzer:
    """Analyze memory profile data for potential issues."""

    def __init__(self, threshold_mb: float = 10.0):
        self.threshold_mb = threshold_mb
        self.issues = []

    def analyze_file(self, filepath: str) -> dict[str, any]:
        """Analyze memory profile file."""
        with open(filepath) as f:
            content = f.read()

        # Parse memory allocations
        allocations = self._parse_allocations(content)

        # Detect potential leaks
        leaks = self._detect_leaks(allocations)

        # Generate report
        report = {
            "total_allocations": len(allocations),
            "potential_leaks": len(leaks),
            "largest_allocation": (
                max(allocations, key=lambda x: x[1]) if allocations else None
            ),
            "total_memory_mb": sum(size for _, size, _ in allocations),
            "leak_details": leaks,
        }

        return report

    def _parse_allocations(self, content: str) -> list[tuple[str, float, str]]:
        """Parse memory allocations from profile content."""
        allocations = []

        # Pattern to match memory allocation lines
        pattern = r"(\S+:\d+)\s+(\d+\.?\d*)\s+MiB\s+(.*)"

        for match in re.finditer(pattern, content):
            location = match.group(1)
            size_mb = float(match.group(2))
            description = match.group(3).strip()
            allocations.append((location, size_mb, description))

        return allocations

    def _detect_leaks(self, allocations: list[tuple[str, float, str]]) -> list[dict]:
        """Detect potential memory leaks."""
        leaks = []

        # Group allocations by location
        location_totals = {}
        for location, size, desc in allocations:
            if location not in location_totals:
                location_totals[location] = {"size": 0, "count": 0, "descriptions": []}
            location_totals[location]["size"] += size
            location_totals[location]["count"] += 1
            location_totals[location]["descriptions"].append(desc)

        # Identify suspicious allocations
        for location, data in location_totals.items():
            if data["size"] > self.threshold_mb:
                leaks.append(
                    {
                        "location": location,
                        "total_size_mb": data["size"],
                        "allocation_count": data["count"],
                        "average_size_mb": data["size"] / data["count"],
                        "descriptions": list(set(data["descriptions"]))[
                            :5
                        ],  # Top 5 unique
                    }
                )

        return sorted(leaks, key=lambda x: x["total_size_mb"], reverse=True)

    def generate_recommendations(self, report: dict) -> list[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        if report["potential_leaks"]:
            recommendations.append(
                f"Found {report['potential_leaks']} potential memory leaks. "
                "Review the following locations:"
            )

            for leak in report["leak_details"][:5]:  # Top 5 leaks
                recommendations.append(
                    f"  - {leak['location']}: {leak['total_size_mb']:.2f} MB "
                    f"({leak['allocation_count']} allocations)"
                )

        if report["total_memory_mb"] > 500:
            recommendations.append(
                f"High total memory usage: {report['total_memory_mb']:.2f} MB. "
                "Consider optimizing data structures or implementing pagination."
            )

        if report["largest_allocation"] and report["largest_allocation"][1] > 50:
            recommendations.append(
                f"Large single allocation detected: {report['largest_allocation'][1]:.2f} MB "
                f"at {report['largest_allocation'][0]}"
            )

        if not recommendations:
            recommendations.append("No significant memory issues detected.")

        return recommendations


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze memory profile for potential issues"
    )
    parser.add_argument("profile_file", help="Path to memory profile file")
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Memory threshold in MB for leak detection (default: 10.0)",
    )
    parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    args = parser.parse_args()

    # Analyze profile
    analyzer = MemoryProfileAnalyzer(threshold_mb=args.threshold)

    try:
        report = analyzer.analyze_file(args.profile_file)
        recommendations = analyzer.generate_recommendations(report)

        if args.output == "json":
            import json

            output = {"report": report, "recommendations": recommendations}
            print(json.dumps(output, indent=2))
        else:
            print("Memory Profile Analysis Report")
            print("=" * 40)
            print(f"Total allocations: {report['total_allocations']}")
            print(f"Total memory used: {report['total_memory_mb']:.2f} MB")
            print(f"Potential leaks detected: {report['potential_leaks']}")

            if report["largest_allocation"]:
                print(f"Largest allocation: {report['largest_allocation'][1]:.2f} MB")

            print("\nRecommendations:")
            for rec in recommendations:
                print(rec)

        # Exit with error code if leaks detected
        sys.exit(1 if report["potential_leaks"] > 0 else 0)

    except Exception as e:
        print(f"Error analyzing profile: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
