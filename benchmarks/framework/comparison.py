"""Tools for comparing benchmark results and detecting regressions."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .base import BenchmarkResult
from .stats import StatisticalAnalyzer


@dataclass
class RegressionReport:
    """Report of detected performance regressions."""
    
    benchmark_name: str
    baseline: BenchmarkResult
    current: BenchmarkResult
    
    # Regression flags
    time_regression: bool
    memory_regression: bool
    throughput_regression: bool
    
    # Detailed metrics
    time_change_pct: float
    memory_change_pct: float
    throughput_change_pct: float
    
    # Statistical significance
    is_significant: bool
    confidence_level: float
    
    # Thresholds used
    time_threshold: float
    memory_threshold: float
    
    # Additional analysis
    likely_cause: Optional[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []
    
    @property
    def has_regression(self) -> bool:
        """Check if any regression was detected."""
        return self.time_regression or self.memory_regression or self.throughput_regression
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "baseline_timestamp": self.baseline.timestamp.isoformat(),
            "current_timestamp": self.current.timestamp.isoformat(),
            "regressions": {
                "time": self.time_regression,
                "memory": self.memory_regression,
                "throughput": self.throughput_regression,
            },
            "changes": {
                "time_pct": self.time_change_pct,
                "memory_pct": self.memory_change_pct,
                "throughput_pct": self.throughput_change_pct,
            },
            "significance": {
                "is_significant": self.is_significant,
                "confidence": self.confidence_level,
            },
            "analysis": {
                "likely_cause": self.likely_cause,
                "recommendations": self.recommendations,
            }
        }


class RegressionDetector:
    """Detect performance regressions between benchmark runs."""
    
    def __init__(
        self,
        time_threshold: float = 0.10,  # 10% slower
        memory_threshold: float = 0.20,  # 20% more memory
        throughput_threshold: float = 0.10,  # 10% less throughput
        confidence_level: float = 0.95
    ):
        self.time_threshold = time_threshold
        self.memory_threshold = memory_threshold
        self.throughput_threshold = throughput_threshold
        self.confidence_level = confidence_level
        self.stats = StatisticalAnalyzer()
    
    def compare(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult
    ) -> RegressionReport:
        """Compare two benchmark results and detect regressions."""
        # Calculate percentage changes
        time_change_pct = self._calculate_change_pct(
            baseline.mean_time, current.mean_time
        )
        memory_change_pct = self._calculate_change_pct(
            baseline.peak_memory_mb, current.peak_memory_mb
        )
        throughput_change_pct = self._calculate_change_pct(
            baseline.throughput or 1, current.throughput or 1
        )
        
        # Detect regressions
        time_regression = time_change_pct > self.time_threshold
        memory_regression = memory_change_pct > self.memory_threshold
        throughput_regression = -throughput_change_pct > self.throughput_threshold
        
        # Check statistical significance
        is_significant = self._check_significance(baseline, current)
        
        # Create report
        report = RegressionReport(
            benchmark_name=current.name,
            baseline=baseline,
            current=current,
            time_regression=time_regression,
            memory_regression=memory_regression,
            throughput_regression=throughput_regression,
            time_change_pct=time_change_pct,
            memory_change_pct=memory_change_pct,
            throughput_change_pct=throughput_change_pct,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            time_threshold=self.time_threshold,
            memory_threshold=self.memory_threshold
        )
        
        # Analyze likely causes
        self._analyze_regression_causes(report)
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        return report
    
    def compare_multiple(
        self,
        results: List[BenchmarkResult],
        baseline_index: int = 0
    ) -> List[RegressionReport]:
        """Compare multiple results against a baseline."""
        if len(results) < 2:
            return []
        
        baseline = results[baseline_index]
        reports = []
        
        for i, result in enumerate(results):
            if i != baseline_index:
                report = self.compare(baseline, result)
                reports.append(report)
        
        return reports
    
    def detect_trend_regression(
        self,
        results: List[BenchmarkResult],
        window_size: int = 5
    ) -> Dict[str, any]:
        """Detect regression trends over multiple runs."""
        if len(results) < window_size:
            return {"error": "Not enough data for trend analysis"}
        
        # Sort by timestamp
        sorted_results = sorted(results, key=lambda r: r.timestamp)
        
        # Extract time series
        times = [r.mean_time for r in sorted_results]
        memories = [r.peak_memory_mb for r in sorted_results]
        throughputs = [r.throughput or 0 for r in sorted_results]
        
        # Detect trends
        time_trend = self.stats.trend_direction(times, window_size)
        memory_trend = self.stats.trend_direction(memories, window_size)
        throughput_trend = self.stats.trend_direction(throughputs, window_size)
        
        # Calculate regression scores
        recent_window = sorted_results[-window_size:]
        older_window = sorted_results[-window_size*2:-window_size] if len(sorted_results) >= window_size*2 else sorted_results[:window_size]
        
        recent_avg_time = self.stats.mean([r.mean_time for r in recent_window])
        older_avg_time = self.stats.mean([r.mean_time for r in older_window])
        
        time_regression_score = (recent_avg_time - older_avg_time) / older_avg_time if older_avg_time > 0 else 0
        
        return {
            "trends": {
                "time": time_trend,
                "memory": memory_trend,
                "throughput": throughput_trend,
            },
            "regression_scores": {
                "time": time_regression_score,
            },
            "analysis_window": window_size,
            "total_samples": len(results),
            "warnings": self._generate_trend_warnings(
                time_trend, memory_trend, throughput_trend, time_regression_score
            )
        }
    
    def _calculate_change_pct(self, baseline: float, current: float) -> float:
        """Calculate percentage change."""
        if baseline == 0:
            return 0.0
        return ((current - baseline) / baseline) * 100
    
    def _check_significance(
        self,
        baseline: BenchmarkResult,
        current: BenchmarkResult
    ) -> bool:
        """Check if the difference is statistically significant."""
        # Simplified significance check based on standard deviations
        # For proper analysis, would use t-test or Mann-Whitney U test
        
        baseline_cv = baseline.std_dev / baseline.mean_time if baseline.mean_time > 0 else 0
        current_cv = current.std_dev / current.mean_time if current.mean_time > 0 else 0
        
        # High variability reduces confidence in regression detection
        avg_cv = (baseline_cv + current_cv) / 2
        
        # Simple heuristic: significant if change > 2 * average CV
        change_ratio = abs(current.mean_time - baseline.mean_time) / baseline.mean_time
        
        return change_ratio > 2 * avg_cv
    
    def _analyze_regression_causes(self, report: RegressionReport):
        """Analyze likely causes of regression."""
        causes = []
        
        if report.time_regression:
            # Check for increased variability
            baseline_cv = report.baseline.std_dev / report.baseline.mean_time
            current_cv = report.current.std_dev / report.current.mean_time
            
            if current_cv > baseline_cv * 1.5:
                causes.append("Increased performance variability")
            
            # Check for memory pressure
            if report.memory_regression:
                causes.append("Possible memory pressure affecting performance")
        
        if report.memory_regression:
            # Check for memory leak pattern
            if report.current.memory_samples:
                # Simple check: is memory constantly increasing?
                samples = report.current.memory_samples
                increasing = all(samples[i] <= samples[i+1] for i in range(len(samples)-1))
                if increasing:
                    causes.append("Possible memory leak detected")
        
        if report.throughput_regression:
            if report.time_regression:
                causes.append("Overall performance degradation")
        
        report.likely_cause = "; ".join(causes) if causes else "Unknown"
    
    def _generate_recommendations(self, report: RegressionReport):
        """Generate recommendations based on regression analysis."""
        recommendations = []
        
        if report.time_regression:
            recommendations.append("Profile code to identify performance bottlenecks")
            recommendations.append("Check for recent code changes that might impact performance")
            
            if report.current.std_dev > report.baseline.std_dev * 1.5:
                recommendations.append("Investigate sources of performance variability")
        
        if report.memory_regression:
            recommendations.append("Run memory profiler to identify allocation hotspots")
            recommendations.append("Check for memory leaks or excessive allocations")
            
            if "leak" in report.likely_cause.lower():
                recommendations.append("Use memory leak detection tools")
        
        if report.throughput_regression:
            recommendations.append("Analyze request processing pipeline")
            recommendations.append("Check for resource contention or blocking operations")
        
        if not report.is_significant:
            recommendations.append("Results may not be statistically significant - run more iterations")
        
        report.recommendations = recommendations
    
    def _generate_trend_warnings(
        self,
        time_trend: str,
        memory_trend: str,
        throughput_trend: str,
        time_regression_score: float
    ) -> List[str]:
        """Generate warnings based on trend analysis."""
        warnings = []
        
        if time_trend == "increasing":
            warnings.append("Performance is degrading over time")
        
        if memory_trend == "increasing":
            warnings.append("Memory usage is increasing - possible leak")
        
        if throughput_trend == "decreasing":
            warnings.append("Throughput is declining")
        
        if time_regression_score > 0.05:  # 5% regression
            warnings.append(f"Significant performance regression detected: {time_regression_score*100:.1f}% slower")
        
        return warnings
    
    def generate_report(
        self,
        reports: List[RegressionReport],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate a comprehensive regression report."""
        # Build report content
        content = ["# Performance Regression Report", ""]
        content.append(f"Generated: {datetime.now().isoformat()}")
        content.append("")
        
        # Summary
        total_regressions = sum(1 for r in reports if r.has_regression)
        content.append("## Summary")
        content.append(f"- Total benchmarks analyzed: {len(reports)}")
        content.append(f"- Regressions detected: {total_regressions}")
        content.append("")
        
        # Detailed results
        content.append("## Detailed Results")
        
        for report in reports:
            content.append(f"\n### {report.benchmark_name}")
            
            if not report.has_regression:
                content.append("✅ No regression detected")
            else:
                if report.time_regression:
                    content.append(f"❌ Time regression: {report.time_change_pct:+.1f}%")
                if report.memory_regression:
                    content.append(f"❌ Memory regression: {report.memory_change_pct:+.1f}%")
                if report.throughput_regression:
                    content.append(f"❌ Throughput regression: {report.throughput_change_pct:+.1f}%")
            
            content.append(f"\n**Statistical Significance**: {'Yes' if report.is_significant else 'No'}")
            content.append(f"**Likely Cause**: {report.likely_cause}")
            
            if report.recommendations:
                content.append("\n**Recommendations**:")
                for rec in report.recommendations:
                    content.append(f"- {rec}")
            
            content.append("")
        
        # Join content
        report_text = "\n".join(content)
        
        # Save if path provided
        if output_path:
            output_path.write_text(report_text)
        
        return report_text