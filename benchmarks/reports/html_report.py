"""Generate comprehensive HTML performance reports."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..framework import BenchmarkResult


class HTMLReportGenerator:
    """Generate detailed HTML performance reports."""

    def __init__(self):
        self.template = self._get_html_template()

    def generate_report(
        self,
        results: dict[str, list[BenchmarkResult]],
        output_path: Path,
        title: str = "MCP Performance Report",
    ):
        """Generate comprehensive HTML report."""
        # Prepare data
        report_data = self._prepare_report_data(results)

        # Generate HTML sections
        sections = []

        # Executive summary
        sections.append(self._generate_executive_summary(report_data))

        # Detailed results by benchmark
        for benchmark_name, benchmark_data in report_data["benchmarks"].items():
            sections.append(
                self._generate_benchmark_section(benchmark_name, benchmark_data)
            )

        # Performance trends
        if len(results) > 1:
            sections.append(self._generate_trends_section(report_data))

        # Recommendations
        sections.append(self._generate_recommendations_section(report_data))

        # Combine sections
        content = "\n".join(sections)

        # Generate final HTML
        html = self.template.format(
            title=title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            content=content,
        )

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(html)

        # Also save raw data as JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

    def _prepare_report_data(
        self, results: dict[str, list[BenchmarkResult]]
    ) -> dict[str, Any]:
        """Prepare data for report generation."""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_benchmarks": len(results),
                "total_runs": sum(len(runs) for runs in results.values()),
            },
            "benchmarks": {},
        }

        for benchmark_name, runs in results.items():
            if not runs:
                continue

            latest = runs[-1]

            benchmark_data = {
                "latest_run": {
                    "timestamp": latest.timestamp.isoformat(),
                    "duration": latest.duration,
                    "iterations": latest.iterations,
                    "mean_time": latest.mean_time,
                    "median_time": latest.median_time,
                    "p95_time": latest.percentiles.get(95, 0),
                    "p99_time": latest.percentiles.get(99, 0),
                    "throughput": latest.throughput,
                    "peak_memory_mb": latest.peak_memory_mb,
                    "errors": len(latest.errors),
                },
                "history": [],
            }

            # Add historical data
            for run in runs:
                benchmark_data["history"].append(
                    {
                        "timestamp": run.timestamp.isoformat(),
                        "mean_time": run.mean_time,
                        "throughput": run.throughput,
                        "peak_memory_mb": run.peak_memory_mb,
                    }
                )

            report_data["benchmarks"][benchmark_name] = benchmark_data

        return report_data

    def _generate_executive_summary(self, data: dict[str, Any]) -> str:
        """Generate executive summary section."""
        summary_html = """
        <section class="executive-summary">
            <h2>Executive Summary</h2>
            <div class="summary-grid">
                <div class="summary-card">
                    <h3>Total Benchmarks</h3>
                    <div class="metric-value">{total_benchmarks}</div>
                </div>
                <div class="summary-card">
                    <h3>Total Test Runs</h3>
                    <div class="metric-value">{total_runs}</div>
                </div>
                <div class="summary-card">
                    <h3>Report Generated</h3>
                    <div class="metric-value">{generated_at}</div>
                </div>
            </div>

            <h3>Key Findings</h3>
            <ul class="findings">
                {findings}
            </ul>
        </section>
        """

        # Generate findings
        findings = []

        # Check for performance issues
        for name, benchmark in data["benchmarks"].items():
            latest = benchmark["latest_run"]

            if latest["mean_time"] > 0.5:  # 500ms
                findings.append(
                    f'<li class="warning">⚠️ {name}: High response time '
                    f'({latest["mean_time"]*1000:.0f}ms average)</li>'
                )

            if latest["peak_memory_mb"] > 500:
                findings.append(
                    f'<li class="warning">⚠️ {name}: High memory usage '
                    f'({latest["peak_memory_mb"]:.0f}MB peak)</li>'
                )

            if latest["errors"] > 0:
                findings.append(
                    f'<li class="error">❌ {name}: {latest["errors"]} errors detected</li>'
                )

        if not findings:
            findings.append(
                '<li class="success">✅ All benchmarks completed successfully</li>'
            )

        return summary_html.format(
            total_benchmarks=data["summary"]["total_benchmarks"],
            total_runs=data["summary"]["total_runs"],
            generated_at=data["generated_at"],
            findings="\n".join(findings),
        )

    def _generate_benchmark_section(self, name: str, data: dict[str, Any]) -> str:
        """Generate section for individual benchmark."""
        latest = data["latest_run"]

        section_html = f"""
        <section class="benchmark-section">
            <h2>{name}</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>Mean Response Time</h4>
                    <div class="metric-value">{latest['mean_time']*1000:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <h4>P95 Response Time</h4>
                    <div class="metric-value">{latest['p95_time']*1000:.1f}ms</div>
                </div>
                <div class="metric-card">
                    <h4>Throughput</h4>
                    <div class="metric-value">{latest['throughput']:.1f} ops/s</div>
                </div>
                <div class="metric-card">
                    <h4>Peak Memory</h4>
                    <div class="metric-value">{latest['peak_memory_mb']:.1f} MB</div>
                </div>
            </div>

            <div class="charts">
                <canvas id="chart-{name.replace(' ', '-')}" width="800" height="400"></canvas>
            </div>
        </section>
        """

        return section_html

    def _generate_trends_section(self, data: dict[str, Any]) -> str:
        """Generate trends analysis section."""
        trends_html = """
        <section class="trends-section">
            <h2>Performance Trends</h2>
            <div class="trends-analysis">
                {trend_items}
            </div>
        </section>
        """

        trend_items = []

        for name, benchmark in data["benchmarks"].items():
            history = benchmark["history"]
            if len(history) < 2:
                continue

            # Calculate trend
            first_time = history[0]["mean_time"]
            last_time = history[-1]["mean_time"]
            change_pct = (
                ((last_time - first_time) / first_time * 100) if first_time > 0 else 0
            )

            trend_class = (
                "improving"
                if change_pct < -5
                else "degrading" if change_pct > 5 else "stable"
            )
            trend_icon = "📈" if change_pct > 5 else "📉" if change_pct < -5 else "➡️"

            trend_items.append(
                f'<div class="trend-item {trend_class}">'
                f"{trend_icon} {name}: {change_pct:+.1f}% "
                f"({first_time*1000:.1f}ms → {last_time*1000:.1f}ms)"
                f"</div>"
            )

        return trends_html.format(trend_items="\n".join(trend_items))

    def _generate_recommendations_section(self, data: dict[str, Any]) -> str:
        """Generate recommendations section."""
        recommendations = []

        # Analyze and generate recommendations
        for name, benchmark in data["benchmarks"].items():
            latest = benchmark["latest_run"]

            if latest["mean_time"] > 0.5:
                recommendations.append(
                    {
                        "type": "performance",
                        "severity": "high",
                        "message": f"Optimize {name} - response time exceeds 500ms threshold",
                    }
                )

            if latest["peak_memory_mb"] > 500:
                recommendations.append(
                    {
                        "type": "memory",
                        "severity": "medium",
                        "message": f"Review memory usage in {name} - consider optimization",
                    }
                )

            if latest["throughput"] and latest["throughput"] < 10:
                recommendations.append(
                    {
                        "type": "throughput",
                        "severity": "medium",
                        "message": f"Low throughput in {name} - investigate bottlenecks",
                    }
                )

        # Generate HTML
        if not recommendations:
            recommendations_html = (
                '<p class="success">✅ No critical issues detected</p>'
            )
        else:
            items = []
            for rec in recommendations:
                severity_class = f"severity-{rec['severity']}"
                items.append(f'<li class="{severity_class}">{rec["message"]}</li>')
            recommendations_html = (
                f'<ul class="recommendations">{chr(10).join(items)}</ul>'
            )

        return f"""
        <section class="recommendations-section">
            <h2>Recommendations</h2>
            {recommendations_html}
        </section>
        """

    def _get_html_template(self) -> str:
        """Get HTML report template."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}

        header {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}

        h1, h2, h3, h4 {{
            margin-top: 0;
        }}

        section {{
            background: white;
            padding: 30px;
            margin-bottom: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .summary-grid, .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .summary-card, .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}

        .metric-value {{
            font-size: 36px;
            font-weight: bold;
            color: #2c3e50;
        }}

        .findings {{
            list-style: none;
            padding: 0;
        }}

        .findings li {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }}

        .success {{
            background: #d4edda;
            color: #155724;
        }}

        .warning {{
            background: #fff3cd;
            color: #856404;
        }}

        .error {{
            background: #f8d7da;
            color: #721c24;
        }}

        .trend-item {{
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
        }}

        .improving {{
            background: #d4edda;
        }}

        .degrading {{
            background: #f8d7da;
        }}

        .stable {{
            background: #e2e3e5;
        }}

        .recommendations {{
            list-style: none;
            padding: 0;
        }}

        .recommendations li {{
            padding: 10px;
            margin: 5px 0;
            border-left: 4px solid;
            background: #f8f9fa;
        }}

        .severity-high {{
            border-color: #dc3545;
        }}

        .severity-medium {{
            border-color: #ffc107;
        }}

        .severity-low {{
            border-color: #28a745;
        }}

        .charts {{
            margin-top: 30px;
        }}

        footer {{
            text-align: center;
            color: #666;
            margin-top: 50px;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <header>
        <h1>{title}</h1>
        <p>Generated at: {generated_at}</p>
    </header>

    {content}

    <footer>
        <p>MCP Standards Server Performance Report</p>
    </footer>

    <script>
        // Add any JavaScript for interactive charts here
    </script>
</body>
</html>"""
