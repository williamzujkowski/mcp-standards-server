"""Historical performance trend analysis."""

import json
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from scipy import stats as scipy_stats

from ..framework import BenchmarkResult


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be safe for use as a filename."""
    import re
    
    # Replace spaces with underscores
    sanitized = name.replace(' ', '_')
    
    # Replace forward slashes with dashes
    sanitized = sanitized.replace('/', '-')
    
    # Replace other problematic characters with underscores
    sanitized = re.sub(r'[<>:"|?*\\]', '_', sanitized)
    
    # Remove any leading/trailing dots or spaces
    sanitized = sanitized.strip('. ')
    
    # Ensure it's not empty
    if not sanitized:
        sanitized = "unnamed_benchmark"
        
    return sanitized


class HistoricalAnalyzer:
    """Analyze historical performance trends."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.benchmark_history: dict[str, list[BenchmarkResult]] = {}

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result to history."""
        if result.name not in self.benchmark_history:
            self.benchmark_history[result.name] = []

        self.benchmark_history[result.name].append(result)

        # Save to disk
        self._save_result(result)

    def _save_result(self, result: BenchmarkResult):
        """Save result to disk."""
        # Create directory for benchmark
        safe_name = sanitize_filename(result.name)
        benchmark_dir = self.data_dir / safe_name
        benchmark_dir.mkdir(exist_ok=True)

        # Save with timestamp
        filename = f"{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = benchmark_dir / filename

        with open(filepath, "w") as f:
            json.dump(result.to_dict(), f, indent=2)

    def load_history(self, benchmark_name: str | None = None):
        """Load historical data from disk."""
        self.benchmark_history.clear()

        # Load all benchmarks or specific one
        if benchmark_name:
            safe_name = sanitize_filename(benchmark_name)
            benchmark_dirs = [self.data_dir / safe_name]
        else:
            benchmark_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]

        for benchmark_dir in benchmark_dirs:
            if not benchmark_dir.exists():
                continue

            # Try to recover original benchmark name
            # This is a best-effort reverse mapping
            benchmark_name_from_dir = benchmark_dir.name.replace("_", " ").replace("-", "/")
            if benchmark_name:
                # Use provided name if loading specific benchmark
                actual_benchmark_name = benchmark_name
            else:
                # Use directory-derived name for discovery
                actual_benchmark_name = benchmark_name_from_dir
            results = []

            # Load all result files
            for filepath in sorted(benchmark_dir.glob("*.json")):
                try:
                    with open(filepath) as f:
                        data = json.load(f)
                    results.append(BenchmarkResult.from_dict(data))
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")

            if results:
                self.benchmark_history[actual_benchmark_name] = results

    def analyze_trends(
        self, benchmark_name: str, metric: str = "mean_time", days: int = 7
    ) -> dict[str, Any]:
        """Analyze performance trends for a benchmark."""
        if benchmark_name not in self.benchmark_history:
            return {"error": "No history for benchmark"}

        results = self.benchmark_history[benchmark_name]

        # Filter by date range
        cutoff = datetime.now() - timedelta(days=days)
        recent_results = [r for r in results if r.timestamp >= cutoff]

        if len(recent_results) < 2:
            return {"error": "Insufficient data for trend analysis"}

        # Extract metric values
        values = []
        timestamps = []

        for result in sorted(recent_results, key=lambda r: r.timestamp):
            value = self._extract_metric(result, metric)
            if value is not None:
                values.append(value)
                timestamps.append(result.timestamp)

        if len(values) < 2:
            return {"error": "Insufficient metric data"}

        # Perform trend analysis
        analysis = {
            "benchmark": benchmark_name,
            "metric": metric,
            "period_days": days,
            "data_points": len(values),
            "statistics": self._calculate_statistics(values),
            "trend": self._detect_trend(timestamps, values),
            "anomalies": self._detect_anomalies(values),
            "forecast": self._forecast_trend(timestamps, values),
            "change_points": self._detect_change_points(values),
        }

        return analysis

    def _extract_metric(self, result: BenchmarkResult, metric: str) -> float | None:
        """Extract metric value from result."""
        # Standard metrics
        metric_map = {
            "mean_time": result.mean_time,
            "median_time": result.median_time,
            "min_time": result.min_time,
            "max_time": result.max_time,
            "std_dev": result.std_dev,
            "throughput": result.throughput,
            "peak_memory_mb": result.peak_memory_mb,
            "avg_memory_mb": result.avg_memory_mb,
        }

        if metric in metric_map:
            return metric_map[metric]

        # Check percentiles
        if metric.startswith("p") and metric[1:].isdigit():
            percentile = int(metric[1:])
            return result.percentiles.get(percentile)

        # Check custom metrics
        if metric in result.custom_metrics:
            value = result.custom_metrics[metric]
            if isinstance(value, dict) and "mean" in value:
                return value["mean"]
            elif isinstance(value, int | float):
                return value

        return None

    def _calculate_statistics(self, values: list[float]) -> dict[str, float]:
        """Calculate statistical summary."""
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "cv": (
                statistics.stdev(values) / statistics.mean(values)
                if len(values) > 1 and statistics.mean(values) > 0
                else 0
            ),
            "latest": values[-1],
            "first": values[0],
            "change_pct": (
                ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            ),
        }

    def _detect_trend(
        self, timestamps: list[datetime], values: list[float]
    ) -> dict[str, Any]:
        """Detect trend using linear regression."""
        if len(values) < 3:
            return {"type": "insufficient_data"}

        # Convert timestamps to numeric values (hours since first)
        first_ts = timestamps[0]
        x = [(ts - first_ts).total_seconds() / 3600 for ts in timestamps]

        # Perform linear regression
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, values)

        # Determine trend type
        if abs(r_value) < 0.3:
            trend_type = "no_trend"
        elif slope > 0 and p_value < 0.05:
            trend_type = "increasing"
        elif slope < 0 and p_value < 0.05:
            trend_type = "decreasing"
        else:
            trend_type = "stable"

        # Calculate change rate
        hours_total = x[-1] if x[-1] > 0 else 1
        total_change = slope * hours_total
        change_per_day = slope * 24

        return {
            "type": trend_type,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "change_per_hour": slope,
            "change_per_day": change_per_day,
            "total_change": total_change,
            "confidence": (
                "high" if p_value < 0.01 else "medium" if p_value < 0.05 else "low"
            ),
        }

    def _detect_anomalies(self, values: list[float]) -> list[dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(values) < 10:
            return []

        anomalies = []

        # Method 1: Z-score
        mean = statistics.mean(values)
        std = statistics.stdev(values)

        for i, value in enumerate(values):
            z_score = (value - mean) / std if std > 0 else 0
            if abs(z_score) > 3:
                anomalies.append(
                    {
                        "index": i,
                        "value": value,
                        "z_score": z_score,
                        "method": "z_score",
                    }
                )

        # Method 2: Isolation Forest (simplified version)
        # Check for values that are far from neighbors
        for i in range(1, len(values) - 1):
            prev_val = values[i - 1]
            curr_val = values[i]
            next_val = values[i + 1]

            # Check if current value is significantly different from neighbors
            neighbor_avg = (prev_val + next_val) / 2
            diff_pct = (
                abs(curr_val - neighbor_avg) / neighbor_avg if neighbor_avg > 0 else 0
            )

            if diff_pct > 0.5:  # 50% difference
                anomalies.append(
                    {
                        "index": i,
                        "value": curr_val,
                        "neighbor_diff_pct": diff_pct * 100,
                        "method": "neighbor_comparison",
                    }
                )

        return anomalies

    def _forecast_trend(
        self, timestamps: list[datetime], values: list[float], forecast_days: int = 3
    ) -> dict[str, Any]:
        """Forecast future values based on trend."""
        if len(values) < 5:
            return {"error": "Insufficient data for forecasting"}

        # Simple linear extrapolation
        x = list(range(len(values)))
        slope, intercept, _, _, _ = scipy_stats.linregress(x, values)

        # Generate forecast
        forecast_points = []
        last_timestamp = timestamps[-1]

        for day in range(1, forecast_days + 1):
            future_x = len(values) + (day * 24)  # Assuming hourly data
            forecast_value = slope * future_x + intercept
            forecast_timestamp = last_timestamp + timedelta(days=day)

            forecast_points.append(
                {
                    "timestamp": forecast_timestamp.isoformat(),
                    "value": max(0, forecast_value),  # Ensure non-negative
                    "confidence_interval": self._calculate_forecast_ci(
                        values, forecast_value, day
                    ),
                }
            )

        return {
            "method": "linear_extrapolation",
            "forecast": forecast_points,
            "trend_strength": abs(slope),
            "warning": self._generate_forecast_warning(slope, values),
        }

    def _calculate_forecast_ci(
        self, historical: list[float], forecast_value: float, days_ahead: int
    ) -> tuple[float, float]:
        """Calculate confidence interval for forecast."""
        # Simple approach: CI widens with time
        std = statistics.stdev(historical) if len(historical) > 1 else 0
        margin = std * (1 + 0.1 * days_ahead)  # 10% wider per day

        return (max(0, forecast_value - 2 * margin), forecast_value + 2 * margin)

    def _generate_forecast_warning(
        self, slope: float, values: list[float]
    ) -> str | None:
        """Generate warning based on forecast."""
        if slope > 0:
            # Calculate when metric might double
            current = values[-1]
            if current > 0 and slope > 0:
                hours_to_double = current / slope
                if hours_to_double < 168:  # Less than a week
                    return f"Metric may double in {hours_to_double:.0f} hours"

        return None

    def _detect_change_points(self, values: list[float]) -> list[dict[str, Any]]:
        """Detect significant change points in the data."""
        if len(values) < 20:
            return []

        change_points = []
        window = 5

        for i in range(window, len(values) - window):
            # Compare before and after windows
            before = values[i - window : i]
            after = values[i : i + window]

            before_mean = statistics.mean(before)
            after_mean = statistics.mean(after)

            # Check for significant change
            if before_mean > 0:
                change_pct = abs(after_mean - before_mean) / before_mean * 100

                if change_pct > 20:  # 20% change threshold
                    change_points.append(
                        {
                            "index": i,
                            "before_mean": before_mean,
                            "after_mean": after_mean,
                            "change_pct": change_pct,
                            "direction": (
                                "increase" if after_mean > before_mean else "decrease"
                            ),
                        }
                    )

        return change_points

    def generate_report(self, output_path: Path, days: int = 7) -> str:
        """Generate comprehensive historical analysis report."""
        lines = [
            "# Historical Performance Analysis Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Analysis Period: Last {days} days\n",
        ]

        for benchmark_name in sorted(self.benchmark_history.keys()):
            lines.append(f"\n## {benchmark_name}")

            # Analyze key metrics
            for metric in ["mean_time", "throughput", "peak_memory_mb"]:
                analysis = self.analyze_trends(benchmark_name, metric, days)

                if "error" not in analysis:
                    lines.append(f"\n### {metric}")

                    # Statistics
                    stats = analysis["statistics"]
                    lines.append(f"- Latest: {stats['latest']:.3f}")
                    lines.append(f"- Change: {stats['change_pct']:+.1f}%")
                    lines.append(f"- Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

                    # Trend
                    trend = analysis["trend"]
                    lines.append(
                        f"- Trend: {trend['type']} ({trend['confidence']} confidence)"
                    )

                    if trend["type"] != "no_trend":
                        lines.append(
                            f"- Change rate: {trend['change_per_day']:.3f} per day"
                        )

                    # Anomalies
                    if analysis["anomalies"]:
                        lines.append(
                            f"- Anomalies detected: {len(analysis['anomalies'])}"
                        )

                    # Forecast warning
                    if "forecast" in analysis and analysis["forecast"].get("warning"):
                        lines.append(f"- ⚠️  {analysis['forecast']['warning']}")

        report = "\n".join(lines)

        # Save report
        with open(output_path, "w") as f:
            f.write(report)

        return report
