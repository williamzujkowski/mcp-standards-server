"""
Unit tests for control coverage report analyzer
@nist-controls: SA-11, CA-7
@evidence: Control coverage report testing
"""

from unittest.mock import MagicMock, patch

import pytest

from src.analyzers.control_coverage_report import (
    ControlCoverageReport,
    CoverageLevel,
    CoverageStats,
    ImplementationStatus,
)
from src.core.nist_controls import NISTControl


class TestCoverageLevel:
    """Test CoverageLevel enum"""

    def test_coverage_levels(self):
        """Test coverage level values"""
        assert CoverageLevel.NONE.value == "none"
        assert CoverageLevel.PARTIAL.value == "partial"
        assert CoverageLevel.COMPLETE.value == "complete"


class TestImplementationStatus:
    """Test ImplementationStatus enum"""

    def test_implementation_status_values(self):
        """Test implementation status values"""
        assert ImplementationStatus.NOT_IMPLEMENTED.value == "not_implemented"
        assert ImplementationStatus.PLANNED.value == "planned"
        assert ImplementationStatus.IN_PROGRESS.value == "in_progress"
        assert ImplementationStatus.IMPLEMENTED.value == "implemented"
        assert ImplementationStatus.NOT_APPLICABLE.value == "not_applicable"


class TestCoverageStats:
    """Test CoverageStats model"""

    def test_coverage_stats_creation(self):
        """Test creating coverage stats"""
        stats = CoverageStats(
            total_controls=100,
            implemented_controls=75,
            partial_controls=15,
            missing_controls=10,
            coverage_percentage=75.0,
            by_family={"AC": 10, "AU": 5},
            by_priority={"high": 20, "medium": 10}
        )

        assert stats.total_controls == 100
        assert stats.implemented_controls == 75
        assert stats.partial_controls == 15
        assert stats.missing_controls == 10
        assert stats.coverage_percentage == 75.0
        assert stats.by_family["AC"] == 10
        assert stats.by_priority["high"] == 20

    def test_coverage_stats_defaults(self):
        """Test coverage stats with defaults"""
        stats = CoverageStats()

        assert stats.total_controls == 0
        assert stats.implemented_controls == 0
        assert stats.partial_controls == 0
        assert stats.missing_controls == 0
        assert stats.coverage_percentage == 0.0
        assert stats.by_family == {}
        assert stats.by_priority == {}


class TestControlCoverageReport:
    """Test ControlCoverageReport class"""

    @pytest.fixture
    def mock_scanner(self):
        """Create mock compliance scanner"""
        scanner = MagicMock()
        scanner.scan_directory.return_value = {
            "controls_found": [
                {
                    "control_id": "AC-3",
                    "control_name": "Access Enforcement",
                    "evidence": ["RBAC implementation"],
                    "files": ["auth.py"]
                },
                {
                    "control_id": "AU-2",
                    "control_name": "Audit Events",
                    "evidence": ["Logging implementation"],
                    "files": ["logger.py"]
                }
            ],
            "scan_summary": {
                "total_files": 10,
                "files_with_controls": 2
            }
        }
        return scanner

    @pytest.fixture
    def report(self, mock_scanner):
        """Create report instance with mock scanner"""
        with patch('src.analyzers.control_coverage_report.ComplianceScanner', return_value=mock_scanner):
            return ControlCoverageReport()

    def test_initialization(self, report):
        """Test report initialization"""
        assert report.scanner is not None
        assert hasattr(report, 'required_controls')
        assert hasattr(report, 'priority_weights')

    def test_analyze_coverage_basic(self, report, tmp_path):
        """Test basic coverage analysis"""
        # Create test files
        test_file = tmp_path / "test.py"
        test_file.write_text("""
        # @nist-controls: AC-3, AU-2
        # @evidence: RBAC implementation
        def check_access():
            pass
        """)

        results = report.analyze_coverage(str(tmp_path))

        assert isinstance(results, dict)
        assert "coverage_stats" in results
        assert "implemented_controls" in results
        assert "missing_controls" in results
        assert "recommendations" in results

    def test_calculate_coverage_stats(self, report):
        """Test coverage statistics calculation"""
        implemented = [
            NISTControl(id="AC-3", name="Access Enforcement", description="", family="AC"),
            NISTControl(id="AU-2", name="Audit Events", description="", family="AU")
        ]

        partial = [
            NISTControl(id="IA-2", name="Authentication", description="", family="IA")
        ]

        missing = [
            NISTControl(id="SC-8", name="Transmission Confidentiality", description="", family="SC")
        ]

        stats = report._calculate_coverage_stats(implemented, partial, missing)

        assert stats.total_controls == 4
        assert stats.implemented_controls == 2
        assert stats.partial_controls == 1
        assert stats.missing_controls == 1
        assert stats.coverage_percentage == 62.5  # (2 + 0.5) / 4 * 100
        assert stats.by_family["AC"] == 1
        assert stats.by_family["AU"] == 1

    def test_categorize_controls(self, report):
        """Test control categorization"""
        scan_results = {
            "controls_found": [
                {"control_id": "AC-3", "evidence": ["Full implementation"]},
                {"control_id": "AU-2", "evidence": ["Partial logging"]}
            ]
        }

        implemented, partial, missing = report._categorize_controls(scan_results)

        assert len(implemented) >= 1
        assert any(c.id == "AC-3" for c in implemented)
        # The actual categorization depends on evidence analysis

    def test_generate_recommendations(self, report):
        """Test recommendation generation"""
        missing_controls = [
            NISTControl(
                id="AC-3",
                name="Access Enforcement",
                description="Enforce access control",
                family="AC",
                priority="high"
            ),
            NISTControl(
                id="AU-2",
                name="Audit Events",
                description="Log audit events",
                family="AU",
                priority="medium"
            )
        ]

        recommendations = report._generate_recommendations(
            missing_controls,
            partial_controls=[]
        )

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("AC-3" in rec for rec in recommendations)
        # High priority should be mentioned first
        assert "high" in recommendations[0].lower() or "AC-3" in recommendations[0]

    def test_format_report_json(self, report):
        """Test JSON report formatting"""
        results = {
            "coverage_stats": CoverageStats(
                total_controls=10,
                implemented_controls=7,
                partial_controls=2,
                missing_controls=1,
                coverage_percentage=80.0
            ),
            "implemented_controls": [
                {"id": "AC-3", "name": "Access Enforcement"}
            ],
            "missing_controls": [
                {"id": "SC-8", "name": "Transmission Confidentiality"}
            ],
            "recommendations": ["Implement SC-8"]
        }

        json_report = report.format_report(results, "json")

        assert isinstance(json_report, str)
        import json
        parsed = json.loads(json_report)
        assert parsed["coverage_stats"]["total_controls"] == 10
        assert parsed["coverage_stats"]["coverage_percentage"] == 80.0

    def test_format_report_markdown(self, report):
        """Test Markdown report formatting"""
        results = {
            "coverage_stats": CoverageStats(
                total_controls=10,
                implemented_controls=7,
                partial_controls=2,
                missing_controls=1,
                coverage_percentage=80.0,
                by_family={"AC": 3, "AU": 2}
            ),
            "implemented_controls": [
                {"id": "AC-3", "name": "Access Enforcement", "evidence": ["RBAC"]}
            ],
            "missing_controls": [
                {"id": "SC-8", "name": "Transmission Confidentiality"}
            ],
            "recommendations": ["Implement SC-8 for data in transit"]
        }

        md_report = report.format_report(results, "markdown")

        assert isinstance(md_report, str)
        assert "# NIST Control Coverage Report" in md_report
        assert "## Coverage Summary" in md_report
        assert "80.0%" in md_report
        assert "AC-3" in md_report
        assert "SC-8" in md_report

    def test_format_report_html(self, report):
        """Test HTML report formatting"""
        results = {
            "coverage_stats": CoverageStats(
                total_controls=10,
                implemented_controls=7,
                partial_controls=2,
                missing_controls=1,
                coverage_percentage=80.0
            ),
            "implemented_controls": [],
            "missing_controls": [],
            "recommendations": []
        }

        html_report = report.format_report(results, "html")

        assert isinstance(html_report, str)
        assert "<html>" in html_report
        assert "NIST Control Coverage Report" in html_report
        assert "80.0%" in html_report

    def test_format_report_text(self, report):
        """Test text report formatting"""
        results = {
            "coverage_stats": CoverageStats(coverage_percentage=75.0),
            "implemented_controls": [],
            "missing_controls": [],
            "recommendations": ["Test recommendation"]
        }

        text_report = report.format_report(results, "text")

        assert isinstance(text_report, str)
        assert "NIST Control Coverage Report" in text_report
        assert "75.0%" in text_report
        assert "Test recommendation" in text_report

    def test_generate_gap_analysis(self, report):
        """Test gap analysis generation"""
        missing_controls = [
            NISTControl(
                id="AC-3",
                name="Access Enforcement",
                description="Test description",
                family="AC",
                implementation_guidance="Use RBAC"
            )
        ]

        gaps = report._generate_gap_analysis(missing_controls)

        assert isinstance(gaps, list)
        assert len(gaps) > 0
        gap = gaps[0]
        assert gap["control_id"] == "AC-3"
        assert gap["control_name"] == "Access Enforcement"
        assert gap["implementation_status"] == ImplementationStatus.NOT_IMPLEMENTED.value
        assert "suggested_implementation" in gap

    def test_prioritize_controls(self, report):
        """Test control prioritization"""
        controls = [
            NISTControl(id="AC-3", name="Access", description="", family="AC", priority="low"),
            NISTControl(id="AU-2", name="Audit", description="", family="AU", priority="high"),
            NISTControl(id="IA-2", name="Auth", description="", family="IA", priority="medium")
        ]

        prioritized = report._prioritize_controls(controls)

        assert len(prioritized) == 3
        # High priority should come first
        assert prioritized[0].id == "AU-2"
        assert prioritized[0].priority == "high"
        # Low priority should come last
        assert prioritized[2].id == "AC-3"
        assert prioritized[2].priority == "low"

    def test_get_implementation_examples(self, report):
        """Test getting implementation examples"""
        control = NISTControl(
            id="AC-3",
            name="Access Enforcement",
            description="Enforce access control",
            family="AC"
        )

        examples = report._get_implementation_examples(control)

        assert isinstance(examples, list)
        assert len(examples) > 0
        # Should provide language-specific examples
        assert any("python" in ex.lower() or "code" in ex.lower() for ex in examples)

    def test_analyze_coverage_with_filters(self, report, tmp_path):
        """Test coverage analysis with filters"""
        results = report.analyze_coverage(
            str(tmp_path),
            control_families=["AC", "AU"],
            min_priority="medium"
        )

        assert isinstance(results, dict)
        assert "coverage_stats" in results
        # Should only include specified families
        if results["implemented_controls"]:
            for control in results["implemented_controls"]:
                assert control.get("family") in ["AC", "AU", None]

    def test_export_results(self, report, tmp_path):
        """Test exporting results to file"""
        results = {
            "coverage_stats": CoverageStats(coverage_percentage=80.0),
            "implemented_controls": [],
            "missing_controls": [],
            "recommendations": []
        }

        # Export as JSON
        json_file = tmp_path / "coverage.json"
        report.export_results(results, str(json_file), "json")
        assert json_file.exists()

        # Export as Markdown
        md_file = tmp_path / "coverage.md"
        report.export_results(results, str(md_file), "markdown")
        assert md_file.exists()

    def test_compare_coverage(self, report):
        """Test comparing coverage between scans"""
        previous = {
            "coverage_stats": CoverageStats(
                implemented_controls=5,
                coverage_percentage=50.0
            ),
            "implemented_controls": [{"id": "AC-3"}]
        }

        current = {
            "coverage_stats": CoverageStats(
                implemented_controls=7,
                coverage_percentage=70.0
            ),
            "implemented_controls": [{"id": "AC-3"}, {"id": "AU-2"}]
        }

        comparison = report.compare_coverage(previous, current)

        assert comparison["coverage_change"] == 20.0
        assert comparison["new_controls"] == 2
        assert len(comparison["newly_implemented"]) >= 1

    def test_generate_executive_summary(self, report):
        """Test executive summary generation"""
        results = {
            "coverage_stats": CoverageStats(
                total_controls=100,
                implemented_controls=75,
                coverage_percentage=75.0,
                by_family={"AC": 20, "AU": 15, "IA": 10}
            ),
            "recommendations": [
                "Implement remaining AC controls",
                "Enhance audit logging"
            ]
        }

        summary = report.generate_executive_summary(results)

        assert isinstance(summary, str)
        assert "75.0%" in summary
        assert "75 of 100" in summary
        # Should mention top families
        assert "AC" in summary

    def test_error_handling(self, report):
        """Test error handling"""
        # Invalid directory
        results = report.analyze_coverage("/nonexistent/path")
        assert results is not None
        assert "error" in results or "coverage_stats" in results

        # Invalid format
        formatted = report.format_report({}, "invalid_format")
        assert isinstance(formatted, str)  # Should fallback to text

    def test_custom_control_requirements(self, report):
        """Test with custom control requirements"""
        custom_controls = [
            "AC-3", "AC-4", "AU-2", "IA-2", "SC-8"
        ]

        report.set_required_controls(custom_controls)

        assert len(report.required_controls) == 5
        assert all(c in report.required_controls for c in custom_controls)

    def test_coverage_trends(self, report):
        """Test coverage trend analysis"""
        historical_data = [
            {"date": "2024-01-01", "coverage": 50.0},
            {"date": "2024-02-01", "coverage": 60.0},
            {"date": "2024-03-01", "coverage": 75.0}
        ]

        trends = report.analyze_trends(historical_data)

        assert "average_improvement" in trends
        assert "projected_full_coverage" in trends
        assert trends["average_improvement"] > 0

    def test_control_mapping_validation(self, report):
        """Test control mapping validation"""
        mappings = {
            "AC-3": ["auth.py", "rbac.py"],
            "AU-2": ["logger.py"],
            "INVALID-1": ["test.py"]  # Invalid control
        }

        valid_mappings = report.validate_control_mappings(mappings)

        assert "AC-3" in valid_mappings
        assert "AU-2" in valid_mappings
        assert "INVALID-1" not in valid_mappings

    def test_generate_remediation_plan(self, report):
        """Test remediation plan generation"""
        missing_controls = [
            NISTControl(
                id="AC-3",
                name="Access Enforcement",
                description="",
                family="AC",
                priority="high"
            ),
            NISTControl(
                id="SC-8",
                name="Transmission Confidentiality",
                description="",
                family="SC",
                priority="medium"
            )
        ]

        plan = report.generate_remediation_plan(missing_controls)

        assert isinstance(plan, dict)
        assert "phases" in plan
        assert len(plan["phases"]) > 0
        # High priority should be in early phases
        phase1 = plan["phases"][0]
        assert any("AC-3" in task["control_id"] for task in phase1["tasks"])

