"""
NIST Control Coverage Reporting
@nist-controls: CA-7, PM-31, AU-6
@evidence: Automated control coverage analysis and reporting
"""
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, CodeAnnotation
from .enhanced_patterns import EnhancedNISTPatterns


@dataclass
class ControlCoverageMetrics:
    """Metrics for control coverage analysis"""
    total_controls_detected: int
    unique_controls: set[str]
    control_families: dict[str, int]
    family_coverage_percentage: dict[str, float]
    high_confidence_controls: set[str]
    suggested_missing_controls: dict[str, list[str]]
    files_analyzed: int
    files_with_controls: int


class ControlCoverageReporter:
    """
    Generate comprehensive NIST control coverage reports
    @nist-controls: CA-7, AU-6, PM-31
    @evidence: Continuous monitoring and reporting of control implementation
    """

    def __init__(self):
        self.patterns = EnhancedNISTPatterns()
        self.annotations_by_file: dict[str, list[CodeAnnotation]] = {}
        self.all_controls: set[str] = set()

    async def analyze_project(self, project_path: Path, analyzers: dict[str, BaseAnalyzer]) -> ControlCoverageMetrics:
        """Analyze entire project for control coverage"""
        files_analyzed = 0
        files_with_controls = 0

        # Analyze all files
        for _analyzer_name, analyzer in analyzers.items():
            results = await analyzer.analyze_project(project_path)

            # Handle different result formats
            if isinstance(results, dict) and 'files' in results:
                # New format from compliance scanner
                for file_info in results.get('files', []):
                    file_path = file_info.get('file', '')
                    annotations = file_info.get('annotations', [])
                files_analyzed += 1
                if annotations:
                    files_with_controls += 1
                    self.annotations_by_file[file_path] = annotations

                    # Collect all controls
                    for ann in annotations:
                        self.all_controls.update(ann.control_ids)

        # Calculate metrics
        control_families = self._group_by_family(self.all_controls)
        family_coverage = self.patterns.get_control_family_coverage(self.all_controls)
        high_confidence = self._get_high_confidence_controls()
        suggestions = self.patterns.suggest_missing_controls(self.all_controls)

        return ControlCoverageMetrics(
            total_controls_detected=len(self.all_controls),
            unique_controls=self.all_controls,
            control_families=control_families,
            family_coverage_percentage=family_coverage,
            high_confidence_controls=high_confidence,
            suggested_missing_controls=suggestions,
            files_analyzed=files_analyzed,
            files_with_controls=files_with_controls
        )

    def generate_report(self, metrics: ControlCoverageMetrics, output_format: str = "markdown") -> str:
        """Generate formatted coverage report"""
        if output_format == "markdown":
            return self._generate_markdown_report(metrics)
        elif output_format == "json":
            return self._generate_json_report(metrics)
        elif output_format == "html":
            return self._generate_html_report(metrics)
        else:
            raise ValueError(f"Unsupported format: {output_format}")

    def _generate_markdown_report(self, metrics: ControlCoverageMetrics) -> str:
        """Generate Markdown formatted report"""
        report = ["# NIST 800-53 Control Coverage Report\n"]

        # Executive Summary
        report.append("## Executive Summary\n")
        report.append(f"- **Total Unique Controls Implemented**: {metrics.total_controls_detected}")
        report.append(f"- **Files Analyzed**: {metrics.files_analyzed}")
        report.append(f"- **Files with Controls**: {metrics.files_with_controls}")
        coverage_pct = (metrics.files_with_controls / metrics.files_analyzed * 100) if metrics.files_analyzed > 0 else 0
        report.append(f"- **File Coverage**: {coverage_pct:.1f}%\n")

        # Control Family Summary
        report.append("## Control Family Coverage\n")
        report.append("| Family | Controls | Coverage % | Status |")
        report.append("|--------|----------|------------|--------|")

        for family in sorted(metrics.family_coverage_percentage.keys()):
            count = metrics.control_families.get(family, 0)
            coverage = metrics.family_coverage_percentage[family]
            status = self._get_coverage_status(coverage)
            report.append(f"| {family} | {count} | {coverage:.1f}% | {status} |")

        # High Confidence Controls
        report.append("\n## High Confidence Controls\n")
        report.append("Controls with explicit implementation or high-confidence pattern matches:\n")
        for control in sorted(metrics.high_confidence_controls):
            report.append(f"- {control}")

        # Suggested Missing Controls
        if metrics.suggested_missing_controls:
            report.append("\n## Suggested Additional Controls\n")
            report.append("Based on implemented controls, consider adding:\n")
            for control, suggestions in sorted(metrics.suggested_missing_controls.items()):
                report.append(f"- **{control}** suggests: {', '.join(suggestions)}")

        # Detailed Control List
        report.append("\n## All Implemented Controls\n")
        families = defaultdict(list)
        for control in sorted(metrics.unique_controls):
            family = control.split('-')[0]
            families[family].append(control)

        for family in sorted(families.keys()):
            report.append(f"\n### {family} - {self._get_family_name(family)}")
            for control in families[family]:
                description = self._get_control_description(control)
                report.append(f"- **{control}**: {description}")

        # File-Level Details
        report.append("\n## File-Level Control Implementation\n")
        for file_path in sorted(self.annotations_by_file.keys()):
            annotations = self.annotations_by_file[file_path]
            file_controls = set()
            for ann in annotations:
                file_controls.update(ann.control_ids)

            if file_controls:
                report.append(f"\n### {Path(file_path).name}")
                report.append(f"Controls: {', '.join(sorted(file_controls))}")

        return "\n".join(report)

    def _generate_json_report(self, metrics: ControlCoverageMetrics) -> str:
        """Generate JSON formatted report"""
        report = {
            "summary": {
                "total_controls": metrics.total_controls_detected,
                "files_analyzed": metrics.files_analyzed,
                "files_with_controls": metrics.files_with_controls,
                "coverage_percentage": (metrics.files_with_controls / metrics.files_analyzed * 100)
                                     if metrics.files_analyzed > 0 else 0
            },
            "control_families": metrics.control_families,
            "family_coverage": metrics.family_coverage_percentage,
            "unique_controls": sorted(metrics.unique_controls),
            "high_confidence_controls": sorted(metrics.high_confidence_controls),
            "suggested_controls": metrics.suggested_missing_controls,
            "file_annotations": {
                file_path: [
                    {
                        "line": ann.line_number,
                        "controls": ann.control_ids,
                        "evidence": ann.evidence,
                        "confidence": ann.confidence
                    }
                    for ann in annotations
                ]
                for file_path, annotations in self.annotations_by_file.items()
            }
        }

        return json.dumps(report, indent=2)

    def _generate_html_report(self, metrics: ControlCoverageMetrics) -> str:
        """Generate HTML formatted report"""
        html = ["""
<!DOCTYPE html>
<html>
<head>
    <title>NIST 800-53 Control Coverage Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f0f0f0; padding: 15px; border-radius: 5px; }
        .metric { margin: 10px 0; }
        .high { color: green; }
        .medium { color: orange; }
        .low { color: red; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .control-family { margin: 20px 0; }
        .control-list { margin-left: 20px; }
    </style>
</head>
<body>
    <h1>NIST 800-53 Control Coverage Report</h1>
"""]

        # Summary section
        html.append('<div class="summary">')
        html.append('<h2>Executive Summary</h2>')
        html.append(f'<div class="metric"><strong>Total Controls:</strong> {metrics.total_controls_detected}</div>')
        html.append(f'<div class="metric"><strong>Files Analyzed:</strong> {metrics.files_analyzed}</div>')
        html.append(f'<div class="metric"><strong>Files with Controls:</strong> {metrics.files_with_controls}</div>')
        html.append('</div>')

        # Family coverage table
        html.append('<h2>Control Family Coverage</h2>')
        html.append('<table>')
        html.append('<tr><th>Family</th><th>Controls</th><th>Coverage %</th><th>Status</th></tr>')

        for family in sorted(metrics.family_coverage_percentage.keys()):
            count = metrics.control_families.get(family, 0)
            coverage = metrics.family_coverage_percentage[family]
            status_class = 'high' if coverage > 70 else 'medium' if coverage > 30 else 'low'
            html.append(f'<tr><td>{family}</td><td>{count}</td><td>{coverage:.1f}%</td>')
            html.append(f'<td class="{status_class}">{self._get_coverage_status(coverage)}</td></tr>')

        html.append('</table>')

        # All controls by family
        html.append('<h2>Implemented Controls by Family</h2>')
        families = defaultdict(list)
        for control in sorted(metrics.unique_controls):
            family = control.split('-')[0]
            families[family].append(control)

        for family in sorted(families.keys()):
            html.append('<div class="control-family">')
            html.append(f'<h3>{family} - {self._get_family_name(family)}</h3>')
            html.append('<div class="control-list">')
            for control in families[family]:
                html.append(f'<div>• <strong>{control}</strong>: {self._get_control_description(control)}</div>')
            html.append('</div></div>')

        html.append('</body></html>')

        return '\n'.join(html)

    def _group_by_family(self, controls: set[str]) -> dict[str, int]:
        """Group controls by family"""
        families: dict[str, int] = defaultdict(int)
        for control in controls:
            family = control.split('-')[0]
            families[family] += 1
        return dict(families)

    def _get_high_confidence_controls(self) -> set[str]:
        """Get controls with high confidence scores"""
        high_confidence = set()

        for annotations in self.annotations_by_file.values():
            for ann in annotations:
                if ann.confidence >= 0.9:
                    high_confidence.update(ann.control_ids)

        return high_confidence

    def _get_coverage_status(self, percentage: float) -> str:
        """Get coverage status based on percentage"""
        if percentage >= 70:
            return "✅ Excellent"
        elif percentage >= 50:
            return "👍 Good"
        elif percentage >= 30:
            return "⚠️ Fair"
        else:
            return "❌ Needs Work"

    def _get_family_name(self, family: str) -> str:
        """Get human-readable family name"""
        family_names = {
            "AC": "Access Control",
            "AU": "Audit and Accountability",
            "CM": "Configuration Management",
            "CP": "Contingency Planning",
            "IA": "Identification and Authentication",
            "IR": "Incident Response",
            "MA": "Maintenance",
            "MP": "Media Protection",
            "PE": "Physical and Environmental Protection",
            "PL": "Planning",
            "PM": "Program Management",
            "PS": "Personnel Security",
            "PT": "Privacy Authorization",
            "RA": "Risk Assessment",
            "SC": "System and Communications Protection",
            "SI": "System and Information Integrity",
            "SR": "Supply Chain Risk Management"
        }
        return family_names.get(family, family)

    def _get_control_description(self, control: str) -> str:
        """Get brief control description"""
        # This would ideally load from NIST catalog
        # For now, common controls
        descriptions = {
            "AC-2": "Account Management",
            "AC-3": "Access Enforcement",
            "AC-6": "Least Privilege",
            "AU-2": "Auditable Events",
            "AU-3": "Content of Audit Records",
            "CM-2": "Baseline Configuration",
            "CP-9": "Information System Backup",
            "IA-2": "Identification and Authentication",
            "SC-8": "Transmission Confidentiality",
            "SC-13": "Cryptographic Protection",
            "SI-10": "Information Input Validation"
        }
        return descriptions.get(control, "Security control implementation")

    def _generate_control_summary(self) -> dict[str, dict[str, Any]]:
        """Generate control summary from annotations"""
        summary: dict[str, dict[str, Any]] = {}

        for file_path, annotations in self.annotations_by_file.items():
            for ann in annotations:
                for control_id in ann.control_ids:
                    if control_id not in summary:
                        summary[control_id] = {
                            "count": 0,
                            "files": [],
                            "confidence": 0.0
                        }

                    summary[control_id]["count"] += 1
                    if file_path not in summary[control_id]["files"]:
                        summary[control_id]["files"].append(file_path)
                    # Update confidence as max
                    summary[control_id]["confidence"] = max(
                        summary[control_id]["confidence"],
                        ann.confidence
                    )

        return summary

    def _generate_family_coverage(self, control_summary: dict[str, dict[str, Any]]) -> dict[str, int]:
        """Generate family coverage from control summary"""
        families: defaultdict[str, int] = defaultdict(int)

        for control_id in control_summary:
            family = control_id.split('-')[0]
            families[family] += 1

        return dict(families)

    def _suggest_missing_controls(self, implemented_controls: set[str]) -> dict[str, list[str]]:
        """Suggest missing controls based on implemented ones"""
        return self.patterns.suggest_missing_controls(implemented_controls)

    def _calculate_confidence_scores(self, control_summary: dict[str, dict[str, Any]]) -> set[str]:
        """Calculate high confidence controls"""
        high_confidence = set()

        for control_id, info in control_summary.items():
            # High confidence if: high score AND multiple instances
            if info["confidence"] >= 0.8 and info["count"] >= 2:
                high_confidence.add(control_id)

        return high_confidence

    def _get_family_statistics(self, control_families: dict[str, int]) -> dict[str, Any]:
        """Get statistics about control families"""
        if not control_families:
            return {
                "total_families": 0,
                "total_controls": 0,
                "average_controls_per_family": 0,
                "most_common_family": None
            }

        total_controls = sum(control_families.values())
        most_common = max(control_families.items(), key=lambda x: x[1])

        return {
            "total_families": len(control_families),
            "total_controls": total_controls,
            "average_controls_per_family": total_controls / len(control_families),
            "most_common_family": most_common[0]
        }

    def _generate_recommendations(self, metrics: ControlCoverageMetrics) -> list[str]:
        """Generate recommendations based on coverage metrics"""
        recommendations = []

        # Check file coverage
        if metrics.files_analyzed > 0:
            coverage_pct = (metrics.files_with_controls / metrics.files_analyzed) * 100
            if coverage_pct < 50:
                recommendations.append(
                    f"Only {coverage_pct:.0f}% of files have NIST control annotations. "
                    "Consider adding annotations to more files."
                )

        # Check for missing critical controls
        critical_controls = {"AC-2", "AC-3", "AU-2", "IA-2", "SC-8"}
        missing_critical = critical_controls - metrics.unique_controls
        if missing_critical:
            recommendations.append(
                f"Missing critical controls: {', '.join(sorted(missing_critical))}"
            )

        # Suggest controls based on what's implemented
        for family, suggestions in metrics.suggested_missing_controls.items():
            if suggestions:
                recommendations.append(
                    f"Based on {family} controls, consider implementing: {', '.join(suggestions[:2])}"
                )

        # Check confidence scores
        if metrics.high_confidence_controls:
            low_confidence_pct = (
                (len(metrics.unique_controls) - len(metrics.high_confidence_controls))
                / len(metrics.unique_controls) * 100
            )
            if low_confidence_pct > 50:
                recommendations.append(
                    f"{low_confidence_pct:.0f}% of controls have low confidence scores. "
                    "Consider adding more evidence or implementation details."
                )

        return recommendations

    def _format_percentage(self, value: float) -> str:
        """Format percentage for display"""
        return f"{value * 100:.1f}%"

    def _get_control_family(self, control_id: str) -> str:
        """Extract control family from control ID"""
        return control_id.split('-')[0]

    async def export_report(self, metrics: ControlCoverageMetrics, file_path: str, format: str = "markdown") -> None:
        """Export report to file"""
        report = self.generate_report(metrics, format)

        with open(file_path, 'w') as f:
            f.write(report)
