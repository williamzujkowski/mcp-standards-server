"""
Unit tests for control coverage report analyzer
@nist-controls: SA-11, CA-7
@evidence: Control coverage report testing
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.analyzers.control_coverage_report import (
    ControlCoverageMetrics,
    ControlCoverageReporter,
)


class TestControlCoverageMetrics:
    """Test ControlCoverageMetrics dataclass"""

    def test_metrics_creation(self):
        """Test creating coverage metrics"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=100,
            unique_controls={"AC-3", "AU-2", "IA-2"},
            control_families={"AC": 1, "AU": 1, "IA": 1},
            family_coverage_percentage={"AC": 33.3, "AU": 33.3, "IA": 33.3},
            high_confidence_controls={"AC-3", "AU-2"},
            suggested_missing_controls={"SC": ["SC-8", "SC-12"]},
            files_analyzed=50,
            files_with_controls=25
        )

        assert metrics.total_controls_detected == 100
        assert len(metrics.unique_controls) == 3
        assert metrics.control_families["AC"] == 1
        assert metrics.family_coverage_percentage["AC"] == 33.3
        assert len(metrics.high_confidence_controls) == 2
        assert len(metrics.suggested_missing_controls["SC"]) == 2
        assert metrics.files_analyzed == 50
        assert metrics.files_with_controls == 25


class TestControlCoverageReporter:
    """Test ControlCoverageReporter class"""

    @pytest.fixture
    def reporter(self):
        """Create reporter instance"""
        return ControlCoverageReporter()

    def test_initialization(self, reporter):
        """Test reporter initialization"""
        assert reporter.patterns is not None
        assert isinstance(reporter.annotations_by_file, dict)
        assert isinstance(reporter.all_controls, set)

    @pytest.mark.asyncio
    async def test_analyze_project_with_mocked_analyzers(self, reporter, tmp_path):
        """Test analyzing project with mocked analyzers"""
        # Create test files
        test_file = tmp_path / "test.py"
        test_file.write_text("""
        # @nist-controls: AC-3, AU-2
        # @evidence: RBAC implementation
        def check_access():
            pass
        """)

        # Mock analyzer
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project = AsyncMock(return_value={
            'files': {
                str(test_file): {
                    'annotations': [
                        {
                            'control_id': 'AC-3',
                            'control_name': 'Access Enforcement',
                            'evidence': 'RBAC implementation',
                            'confidence': 0.9,
                            'file': str(test_file),
                            'line': 2
                        }
                    ]
                }
            }
        })

        analyzers = {"python": mock_analyzer}
        
        metrics = await reporter.analyze_project(tmp_path, analyzers)
        
        assert isinstance(metrics, ControlCoverageMetrics)
        assert metrics.files_analyzed > 0
        assert 'AC-3' in metrics.unique_controls

    @pytest.mark.asyncio
    async def test_analyze_project_different_result_format(self, reporter, tmp_path):
        """Test analyzing project with different analyzer result formats"""
        # Mock analyzer with different result format
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project = AsyncMock(return_value=[
            {
                'control_id': 'AU-2',
                'control_name': 'Audit Events',
                'evidence': 'Logging implementation',
                'file': 'test.py',
                'line': 10
            }
        ])

        analyzers = {"python": mock_analyzer}
        
        metrics = await reporter.analyze_project(tmp_path, analyzers)
        
        assert isinstance(metrics, ControlCoverageMetrics)

    def test_generate_control_summary(self, reporter):
        """Test generating control summary"""
        # Add some test data
        reporter.all_controls = {"AC-3", "AU-2", "IA-2", "SC-8"}
        reporter.annotations_by_file = {
            "file1.py": [
                MagicMock(control_id="AC-3", confidence=0.9),
                MagicMock(control_id="AU-2", confidence=0.8)
            ],
            "file2.py": [
                MagicMock(control_id="IA-2", confidence=0.7)
            ]
        }

        summary = reporter._generate_control_summary()
        
        assert isinstance(summary, dict)
        assert len(summary) == 4  # Should have all 4 controls
        assert "AC-3" in summary
        assert summary["AC-3"]["count"] == 1
        assert summary["AC-3"]["files"] == ["file1.py"]

    def test_generate_family_coverage(self, reporter):
        """Test generating family coverage statistics"""
        control_summary = {
            "AC-3": {"count": 2, "confidence": 0.9},
            "AC-4": {"count": 1, "confidence": 0.8},
            "AU-2": {"count": 3, "confidence": 0.85},
            "IA-2": {"count": 1, "confidence": 0.7}
        }

        families = reporter._generate_family_coverage(control_summary)
        
        assert isinstance(families, dict)
        assert "AC" in families
        assert "AU" in families
        assert "IA" in families
        assert families["AC"] == 2  # AC-3 and AC-4
        assert families["AU"] == 1  # AU-2
        assert families["IA"] == 1  # IA-2

    def test_suggest_missing_controls(self, reporter):
        """Test suggesting missing controls"""
        implemented_controls = {"AC-3", "AU-2"}
        
        suggestions = reporter._suggest_missing_controls(implemented_controls)
        
        assert isinstance(suggestions, dict)
        # Should suggest related controls
        if "AC" in suggestions:
            assert isinstance(suggestions["AC"], list)
            # Might suggest AC-2, AC-4, etc.

    def test_calculate_confidence_scores(self, reporter):
        """Test calculating confidence scores"""
        control_summary = {
            "AC-3": {"count": 5, "confidence": 0.9, "files": ["a.py", "b.py"]},
            "AU-2": {"count": 2, "confidence": 0.7, "files": ["c.py"]},
            "IA-2": {"count": 1, "confidence": 0.5, "files": ["d.py"]}
        }

        high_confidence = reporter._calculate_confidence_scores(control_summary)
        
        assert isinstance(high_confidence, set)
        assert "AC-3" in high_confidence  # High count and confidence
        # AU-2 and IA-2 might not be in high confidence due to lower scores

    def test_generate_html_report(self, reporter):
        """Test generating HTML report"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=50,
            unique_controls={"AC-3", "AU-2"},
            control_families={"AC": 1, "AU": 1},
            family_coverage_percentage={"AC": 50.0, "AU": 50.0},
            high_confidence_controls={"AC-3"},
            suggested_missing_controls={},
            files_analyzed=10,
            files_with_controls=5
        )

        html = reporter.generate_html_report(metrics)
        
        assert isinstance(html, str)
        assert "<html>" in html
        assert "Control Coverage Report" in html
        assert "AC-3" in html
        assert "50%" in html

    def test_generate_markdown_report(self, reporter):
        """Test generating Markdown report"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=50,
            unique_controls={"AC-3", "AU-2", "IA-2"},
            control_families={"AC": 1, "AU": 1, "IA": 1},
            family_coverage_percentage={"AC": 33.3, "AU": 33.3, "IA": 33.3},
            high_confidence_controls={"AC-3", "AU-2"},
            suggested_missing_controls={"SC": ["SC-8"]},
            files_analyzed=20,
            files_with_controls=10
        )

        markdown = reporter.generate_markdown_report(metrics)
        
        assert isinstance(markdown, str)
        assert "# NIST Control Coverage Report" in markdown
        assert "## Summary" in markdown
        assert "## Control Families" in markdown
        assert "AC-3" in markdown
        assert "33.3%" in markdown
        assert "SC-8" in markdown

    def test_generate_json_report(self, reporter):
        """Test generating JSON report"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=10,
            unique_controls={"AC-3"},
            control_families={"AC": 1},
            family_coverage_percentage={"AC": 100.0},
            high_confidence_controls={"AC-3"},
            suggested_missing_controls={},
            files_analyzed=5,
            files_with_controls=3
        )

        import json
        json_report = reporter.generate_json_report(metrics)
        
        assert isinstance(json_report, str)
        data = json.loads(json_report)
        assert data["metrics"]["total_controls_detected"] == 10
        assert data["metrics"]["unique_controls_count"] == 1
        assert data["metrics"]["files_analyzed"] == 5

    @pytest.mark.asyncio
    async def test_export_report(self, reporter, tmp_path):
        """Test exporting report to file"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=5,
            unique_controls={"AC-3"},
            control_families={"AC": 1},
            family_coverage_percentage={"AC": 100.0},
            high_confidence_controls=set(),
            suggested_missing_controls={},
            files_analyzed=2,
            files_with_controls=1
        )

        # Test HTML export
        html_file = tmp_path / "report.html"
        await reporter.export_report(metrics, str(html_file), "html")
        assert html_file.exists()
        assert "<html>" in html_file.read_text()

        # Test Markdown export
        md_file = tmp_path / "report.md"
        await reporter.export_report(metrics, str(md_file), "markdown")
        assert md_file.exists()
        assert "# NIST Control Coverage Report" in md_file.read_text()

        # Test JSON export
        json_file = tmp_path / "report.json"
        await reporter.export_report(metrics, str(json_file), "json")
        assert json_file.exists()
        import json
        data = json.loads(json_file.read_text())
        assert "metrics" in data

    def test_get_family_statistics(self, reporter):
        """Test getting family statistics"""
        control_families = {"AC": 5, "AU": 3, "IA": 2, "SC": 1}
        
        stats = reporter._get_family_statistics(control_families)
        
        assert isinstance(stats, dict)
        assert stats["total_families"] == 4
        assert stats["total_controls"] == 11
        assert stats["average_controls_per_family"] == 2.75
        assert stats["most_common_family"] == "AC"

    @pytest.mark.asyncio
    async def test_analyze_empty_project(self, reporter, tmp_path):
        """Test analyzing empty project"""
        analyzers = {"python": MagicMock()}
        analyzers["python"].analyze_project = AsyncMock(return_value={'files': {}})
        
        metrics = await reporter.analyze_project(tmp_path, analyzers)
        
        assert metrics.total_controls_detected == 0
        assert len(metrics.unique_controls) == 0
        assert metrics.files_analyzed == 0
        assert metrics.files_with_controls == 0

    def test_format_percentage(self, reporter):
        """Test percentage formatting"""
        # Test internal percentage calculation
        assert reporter._format_percentage(0.5) == "50.0%"
        assert reporter._format_percentage(0.333) == "33.3%"
        assert reporter._format_percentage(1.0) == "100.0%"
        assert reporter._format_percentage(0.0) == "0.0%"

    def test_get_control_family(self, reporter):
        """Test extracting control family from control ID"""
        assert reporter._get_control_family("AC-3") == "AC"
        assert reporter._get_control_family("AU-2(1)") == "AU"
        assert reporter._get_control_family("IA-2") == "IA"
        assert reporter._get_control_family("SC-8(1)(2)") == "SC"

    @pytest.mark.asyncio
    async def test_analyze_with_errors(self, reporter, tmp_path):
        """Test handling analyzer errors"""
        # Mock analyzer that raises exception
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_project = AsyncMock(side_effect=Exception("Analyzer error"))
        
        analyzers = {"python": mock_analyzer}
        
        # Should handle error gracefully
        metrics = await reporter.analyze_project(tmp_path, analyzers)
        
        assert isinstance(metrics, ControlCoverageMetrics)
        assert metrics.files_analyzed == 0

    def test_generate_recommendations(self, reporter):
        """Test generating recommendations based on coverage"""
        metrics = ControlCoverageMetrics(
            total_controls_detected=10,
            unique_controls={"AC-3", "AU-2"},
            control_families={"AC": 1, "AU": 1},
            family_coverage_percentage={"AC": 20.0, "AU": 20.0},
            high_confidence_controls=set(),
            suggested_missing_controls={"AC": ["AC-2", "AC-4"], "SC": ["SC-8"]},
            files_analyzed=100,
            files_with_controls=10
        )

        recommendations = reporter._generate_recommendations(metrics)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        # Should recommend implementing missing controls
        assert any("AC-2" in rec or "AC-4" in rec for rec in recommendations)
        # Should note low file coverage
        assert any("10%" in rec or "coverage" in rec.lower() for rec in recommendations)

    def test_enhanced_patterns_integration(self, reporter):
        """Test integration with EnhancedNISTPatterns"""
        assert reporter.patterns is not None
        # Should be able to use pattern matching
        assert hasattr(reporter.patterns, 'CONTROL_PATTERNS')
        assert hasattr(reporter.patterns, 'EVIDENCE_PATTERNS')

    # Helper methods for testing
    def _format_percentage(self, value: float) -> str:
        """Format percentage for display"""
        return f"{value * 100:.1f}%"

    def _get_control_family(self, control_id: str) -> str:
        """Extract control family from control ID"""
        return control_id.split('-')[0]