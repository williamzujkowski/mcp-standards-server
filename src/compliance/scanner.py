"""
Compliance Scanner Module
@nist-controls: CA-7, RA-5, SA-11
@evidence: Automated compliance scanning
"""
import ast
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging import get_logger

logger = get_logger(__name__)


class ComplianceScanner:
    """
    Scanner for NIST compliance checking
    @nist-controls: CA-7, RA-5
    @evidence: Continuous monitoring implementation
    """

    def __init__(self) -> None:
        self.scan_results: list[dict[str, Any]] = []
        self.total_files = 0
        self.files_with_controls = 0
        self.control_stats: dict[str, int] = defaultdict(int)
        self.control_definitions = self._load_control_definitions()

    def _load_control_definitions(self) -> dict[str, dict[str, str]]:
        """Load NIST control definitions"""
        # Basic definitions for common controls
        return {
            "AC-2": {"name": "Account Management", "category": "Access Control"},
            "AC-3": {"name": "Access Enforcement", "category": "Access Control"},
            "AC-4": {"name": "Information Flow Enforcement", "category": "Access Control"},
            "AC-6": {"name": "Least Privilege", "category": "Access Control"},
            "AU-2": {"name": "Audit Events", "category": "Audit and Accountability"},
            "AU-3": {"name": "Content of Audit Records", "category": "Audit and Accountability"},
            "AU-12": {"name": "Audit Generation", "category": "Audit and Accountability"},
            "CA-7": {"name": "Continuous Monitoring", "category": "Security Assessment"},
            "CM-2": {"name": "Baseline Configuration", "category": "Configuration Management"},
            "IA-2": {"name": "Identification and Authentication", "category": "Identification and Authentication"},
            "RA-5": {"name": "Vulnerability Scanning", "category": "Risk Assessment"},
            "SA-11": {"name": "Developer Testing", "category": "System and Services Acquisition"},
            "SC-8": {"name": "Transmission Confidentiality", "category": "System and Communications Protection"},
            "SC-13": {"name": "Cryptographic Protection", "category": "System and Communications Protection"},
            "SI-10": {"name": "Information Input Validation", "category": "System and Information Integrity"},
            "SI-11": {"name": "Error Handling", "category": "System and Information Integrity"},
        }

    async def scan_file(self, file_path: Path) -> dict[str, Any]:
        """Scan a single file for compliance"""
        self.total_files += 1
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Extract NIST controls from comments
            controls_found = self._extract_nist_controls(content)
            evidence_found = self._extract_evidence(content)
            
            # Analyze code for security patterns
            security_analysis = self._analyze_security_patterns(content, file_path)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                controls_found, 
                security_analysis,
                file_path
            )
            
            if controls_found:
                self.files_with_controls += 1
                for control in controls_found:
                    self.control_stats[control] += 1
            
            return {
                "file": str(file_path),
                "controls_found": controls_found,
                "evidence": evidence_found,
                "security_patterns": security_analysis,
                "issues": security_analysis.get("issues", []),
                "recommendations": recommendations,
                "has_controls": bool(controls_found)
            }
            
        except Exception as e:
            logger.error(f"Error scanning {file_path}: {e}")
            return {
                "file": str(file_path),
                "controls_found": [],
                "evidence": [],
                "security_patterns": {},
                "issues": [{"type": "scan_error", "message": str(e)}],
                "recommendations": [],
                "has_controls": False
            }

    def _extract_nist_controls(self, content: str) -> list[str]:
        """Extract NIST control annotations from code"""
        # Pattern to match @nist-controls: AC-3, AU-2, etc.
        pattern = r'@nist-controls?:\s*([A-Z]{2}-\d+(?:\s*,\s*[A-Z]{2}-\d+)*)'
        
        controls = set()
        for match in re.finditer(pattern, content, re.MULTILINE):
            control_list = match.group(1)
            # Split by comma and clean up
            for control in control_list.split(','):
                control = control.strip()
                if re.match(r'^[A-Z]{2}-\d+$', control):
                    controls.add(control)
        
        return sorted(list(controls))

    def _extract_evidence(self, content: str) -> list[dict[str, Any]]:
        """Extract evidence annotations"""
        pattern = r'@evidence:\s*(.+?)(?=\n|$)'
        
        evidence = []
        for match in re.finditer(pattern, content, re.MULTILINE):
            evidence.append({
                "description": match.group(1).strip(),
                "line": content[:match.start()].count('\n') + 1
            })
        
        return evidence

    def _analyze_security_patterns(self, content: str, file_path: Path) -> dict[str, Any]:
        """Analyze code for security patterns and anti-patterns"""
        analysis = {
            "patterns_found": [],
            "issues": [],
            "suggestions": []
        }
        
        # Skip security analysis for the scanner itself
        if "scanner.py" in str(file_path) and "compliance" in str(file_path):
            return analysis
        
        # Check for authentication patterns
        if re.search(r'(authenticate|login|verify_password)', content, re.IGNORECASE):
            analysis["patterns_found"].append("authentication")
            if not re.search(r'@nist-controls?:.*IA-2', content):
                analysis["suggestions"].append({
                    "control": "IA-2",
                    "reason": "Authentication code detected without IA-2 annotation"
                })
        
        # Check for authorization patterns
        if re.search(r'(check_permission|authorize|has_role|access_control)', content, re.IGNORECASE):
            analysis["patterns_found"].append("authorization")
            if not re.search(r'@nist-controls?:.*AC-3', content):
                analysis["suggestions"].append({
                    "control": "AC-3",
                    "reason": "Authorization code detected without AC-3 annotation"
                })
        
        # Check for logging patterns
        if re.search(r'(logger\.|logging\.|audit_log|log_event)', content):
            analysis["patterns_found"].append("logging")
            if not re.search(r'@nist-controls?:.*AU-2', content):
                analysis["suggestions"].append({
                    "control": "AU-2",
                    "reason": "Logging code detected without AU-2 annotation"
                })
        
        # Check for encryption patterns
        if re.search(r'(encrypt|decrypt|hash|crypto|tls|ssl)', content, re.IGNORECASE):
            analysis["patterns_found"].append("encryption")
            if not re.search(r'@nist-controls?:.*SC-13', content):
                analysis["suggestions"].append({
                    "control": "SC-13",
                    "reason": "Cryptographic code detected without SC-13 annotation"
                })
        
        # Check for input validation
        if re.search(r'(validate|sanitize|escape|clean.*input)', content, re.IGNORECASE):
            analysis["patterns_found"].append("input_validation")
            if not re.search(r'@nist-controls?:.*SI-10', content):
                analysis["suggestions"].append({
                    "control": "SI-10",
                    "reason": "Input validation detected without SI-10 annotation"
                })
        
        # Check for potential security issues
        if re.search(r'\beval\s*\(|\bexec\s*\(', content):
            analysis["issues"].append({
                "type": "dangerous_function",
                "severity": "high",
                "message": "Use of eval() or exec() detected - potential code injection risk"
            })
        
        if re.search(r'password\s*=\s*["\'][\w\d]+["\']', content, re.IGNORECASE):
            analysis["issues"].append({
                "type": "hardcoded_secret",
                "severity": "critical",
                "message": "Potential hardcoded password detected"
            })
        
        return analysis

    def _generate_recommendations(
        self, 
        controls_found: list[str], 
        security_analysis: dict[str, Any],
        file_path: Path
    ) -> list[dict[str, str]]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Add suggestions from security analysis
        for suggestion in security_analysis.get("suggestions", []):
            recommendations.append({
                "type": "add_control",
                "control": suggestion["control"],
                "reason": suggestion["reason"],
                "priority": "medium"
            })
        
        # Check for missing evidence
        if controls_found and not security_analysis.get("evidence"):
            recommendations.append({
                "type": "add_evidence",
                "reason": "Controls are annotated but no @evidence tags found",
                "priority": "low"
            })
        
        # Suggest control grouping
        if len(controls_found) > 5:
            recommendations.append({
                "type": "refactor",
                "reason": f"File has {len(controls_found)} controls - consider splitting functionality",
                "priority": "low"
            })
        
        return recommendations

    async def scan_directory(self, directory: Path) -> list[dict[str, Any]]:
        """Scan directory for compliance"""
        results = []
        
        # Scan Python files
        for file_path in directory.rglob("*.py"):
            # Skip test files and hidden directories
            if any(part.startswith('.') for part in file_path.parts):
                continue
            if 'test' in file_path.parts or '__pycache__' in str(file_path):
                continue
                
            result = await self.scan_file(file_path)
            results.append(result)
        
        return results

    def generate_report(self, scan_results: list[dict[str, Any]], output_format: str = "json") -> dict[str, Any]:
        """Generate compliance report"""
        # Calculate statistics
        total_controls = sum(len(r["controls_found"]) for r in scan_results)
        unique_controls = set()
        for result in scan_results:
            unique_controls.update(result["controls_found"])
        
        total_issues = sum(len(r["issues"]) for r in scan_results)
        critical_issues = sum(
            1 for r in scan_results 
            for issue in r["issues"] 
            if issue.get("severity") == "critical"
        )
        
        report = {
            "scan_date": datetime.now().isoformat(),
            "summary": {
                "total_files": self.total_files,
                "files_with_controls": self.files_with_controls,
                "coverage_percentage": round(
                    (self.files_with_controls / self.total_files * 100) if self.total_files > 0 else 0, 
                    2
                ),
                "total_controls_found": total_controls,
                "unique_controls": len(unique_controls),
                "total_issues": total_issues,
                "critical_issues": critical_issues
            },
            "control_statistics": dict(self.control_stats),
            "controls_by_category": self._group_controls_by_category(unique_controls),
            "files": [
                {
                    "path": r["file"],
                    "controls": r["controls_found"],
                    "has_evidence": bool(r.get("evidence")),
                    "issues": r["issues"],
                    "recommendations": r["recommendations"]
                }
                for r in scan_results
            ],
            "recommendations": self._generate_overall_recommendations(scan_results)
        }
        
        # For compatibility with existing compliance check
        report["total_controls"] = len(self.control_definitions)
        report["implemented"] = len(unique_controls)
        report["coverage_percentage"] = round(
            (len(unique_controls) / len(self.control_definitions) * 100) if self.control_definitions else 0,
            2
        )
        report["controls"] = {
            control: {
                "status": "implemented" if control in unique_controls else "not_implemented",
                "evidence": [
                    r["file"] for r in scan_results 
                    if control in r["controls_found"]
                ]
            }
            for control in self.control_definitions
        }
        
        return report

    def _group_controls_by_category(self, controls: set[str]) -> dict[str, list[str]]:
        """Group controls by category"""
        categories = defaultdict(list)
        for control in controls:
            if control in self.control_definitions:
                category = self.control_definitions[control]["category"]
                categories[category].append(control)
        return dict(categories)

    def _generate_overall_recommendations(self, scan_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Generate overall recommendations"""
        recommendations = []
        
        # Check coverage
        if self.files_with_controls < self.total_files * 0.5:
            recommendations.append({
                "type": "coverage",
                "priority": "high",
                "message": f"Only {self.files_with_controls}/{self.total_files} files have NIST controls annotated"
            })
        
        # Check for missing common controls
        implemented = set()
        for result in scan_results:
            implemented.update(result["controls_found"])
        
        essential_controls = {"AC-3", "AU-2", "SI-10", "SC-13"}
        missing_essential = essential_controls - implemented
        
        if missing_essential:
            recommendations.append({
                "type": "missing_controls",
                "priority": "high",
                "message": f"Essential controls not implemented: {', '.join(sorted(missing_essential))}"
            })
        
        return recommendations

    def format_output(self, report: dict[str, Any], output_format: str) -> str:
        """Format report for output"""
        if output_format == "json":
            return json.dumps(report, indent=2)
        
        elif output_format == "yaml":
            import yaml
            return yaml.dump(report, default_flow_style=False)
        
        elif output_format == "table":
            from rich.table import Table
            from rich.console import Console
            
            console = Console()
            
            # Summary table
            table = Table(title="NIST Compliance Scan Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            summary = report["summary"]
            table.add_row("Total Files", str(summary["total_files"]))
            table.add_row("Files with Controls", str(summary["files_with_controls"]))
            table.add_row("Coverage %", f"{summary['coverage_percentage']}%")
            table.add_row("Unique Controls", str(summary["unique_controls"]))
            table.add_row("Total Issues", str(summary["total_issues"]))
            table.add_row("Critical Issues", str(summary["critical_issues"]))
            
            # Control distribution
            control_table = Table(title="Control Distribution")
            control_table.add_column("Control", style="cyan")
            control_table.add_column("Count", style="green")
            control_table.add_column("Description", style="yellow")
            
            for control, count in sorted(report["control_statistics"].items(), key=lambda x: x[1], reverse=True):
                desc = self.control_definitions.get(control, {}).get("name", "Unknown")
                control_table.add_row(control, str(count), desc)
            
            # Issues table
            if summary["total_issues"] > 0:
                issues_table = Table(title="Security Issues Found")
                issues_table.add_column("File", style="cyan")
                issues_table.add_column("Type", style="yellow")
                issues_table.add_column("Severity", style="red")
                issues_table.add_column("Message", style="white")
                
                for file_result in report["files"]:
                    for issue in file_result["issues"]:
                        issues_table.add_row(
                            Path(file_result["path"]).name,
                            issue.get("type", "unknown"),
                            issue.get("severity", "medium"),
                            issue.get("message", "")
                        )
            
            # Print tables
            console.print(table)
            console.print()
            console.print(control_table)
            if summary["total_issues"] > 0:
                console.print()
                console.print(issues_table)
            
            # Print recommendations
            if report.get("recommendations"):
                console.print("\n[bold yellow]Overall Recommendations:[/bold yellow]")
                for rec in report["recommendations"]:
                    console.print(f"  • [{rec['priority']}] {rec['message']}")
            
            return ""  # Rich console handles printing
        
        else:
            # Default to JSON
            return json.dumps(report, indent=2)