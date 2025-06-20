"""
NIST Control Coverage Reporting
@nist-controls: CA-7, PM-31, AU-6
@evidence: Automated control coverage analysis and reporting
"""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple
import json

from .enhanced_patterns import EnhancedNISTPatterns
from .base import BaseAnalyzer, CodeAnnotation


@dataclass
class ControlCoverageMetrics:
    """Metrics for control coverage analysis"""
    total_controls_detected: int
    unique_controls: Set[str]
    control_families: Dict[str, int]
    family_coverage_percentage: Dict[str, float]
    high_confidence_controls: Set[str]
    suggested_missing_controls: Dict[str, List[str]]
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
        self.annotations_by_file: Dict[str, List[CodeAnnotation]] = {}
        self.all_controls: Set[str] = set()
        
    def analyze_project(self, project_path: Path, analyzers: Dict[str, BaseAnalyzer]) -> ControlCoverageMetrics:
        """Analyze entire project for control coverage"""
        files_analyzed = 0
        files_with_controls = 0
        
        # Analyze all files
        for analyzer_name, analyzer in analyzers.items():
            results = analyzer.analyze_project(project_path)
            
            for file_path, annotations in results.items():
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
            "unique_controls": sorted(list(metrics.unique_controls)),
            "high_confidence_controls": sorted(list(metrics.high_confidence_controls)),
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
            html.append(f'<div class="control-family">')
            html.append(f'<h3>{family} - {self._get_family_name(family)}</h3>')
            html.append('<div class="control-list">')
            for control in families[family]:
                html.append(f'<div>• <strong>{control}</strong>: {self._get_control_description(control)}</div>')
            html.append('</div></div>')
        
        html.append('</body></html>')
        
        return '\n'.join(html)
    
    def _group_by_family(self, controls: Set[str]) -> Dict[str, int]:
        """Group controls by family"""
        families = defaultdict(int)
        for control in controls:
            family = control.split('-')[0]
            families[family] += 1
        return dict(families)
    
    def _get_high_confidence_controls(self) -> Set[str]:
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