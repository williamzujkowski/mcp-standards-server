"""
Unit tests for the Rule Engine module.
"""

import json
from pathlib import Path
from typing import cast

import pytest

from src.core.standards.rule_engine import (
    ConditionLogic,
    RuleCondition,
    RuleEngine,
    RuleGroup,
    RuleOperator,
    StandardRule,
)


class TestRuleCondition:
    """Test cases for RuleCondition class."""

    def test_equals_operator(self):
        """Test equals operator evaluation."""
        condition = RuleCondition(
            field="language", operator=RuleOperator.EQUALS, value="python"
        )

        assert condition.evaluate({"language": "python"}) is True
        assert condition.evaluate({"language": "javascript"}) is False
        assert condition.evaluate({"other": "python"}) is False

    def test_not_equals_operator(self):
        """Test not equals operator evaluation."""
        condition = RuleCondition(
            field="language", operator=RuleOperator.NOT_EQUALS, value="python"
        )

        assert condition.evaluate({"language": "javascript"}) is True
        assert condition.evaluate({"language": "python"}) is False

    def test_contains_operator(self):
        """Test contains operator for lists and strings."""
        # List contains
        condition = RuleCondition(
            field="requirements", operator=RuleOperator.CONTAINS, value="security"
        )

        assert condition.evaluate({"requirements": ["security", "performance"]}) is True
        assert condition.evaluate({"requirements": ["performance"]}) is False

        # String contains
        condition_str = RuleCondition(
            field="description", operator=RuleOperator.CONTAINS, value="API"
        )

        assert condition_str.evaluate({"description": "REST API Service"}) is True
        assert condition_str.evaluate({"description": "Web Application"}) is False

    def test_in_operator(self):
        """Test in operator evaluation."""
        condition = RuleCondition(
            field="framework",
            operator=RuleOperator.IN,
            value=["react", "vue", "angular"],
        )

        assert condition.evaluate({"framework": "react"}) is True
        assert condition.evaluate({"framework": "svelte"}) is False

    def test_exists_operator(self):
        """Test exists and not exists operators."""
        exists_condition = RuleCondition(
            field="database", operator=RuleOperator.EXISTS, value=None
        )

        not_exists_condition = RuleCondition(
            field="database", operator=RuleOperator.NOT_EXISTS, value=None
        )

        assert exists_condition.evaluate({"database": "postgresql"}) is True
        assert exists_condition.evaluate({"other": "value"}) is False

        assert not_exists_condition.evaluate({"other": "value"}) is True
        assert not_exists_condition.evaluate({"database": "postgresql"}) is False

    def test_case_insensitive_comparison(self):
        """Test case-insensitive string comparison."""
        condition = RuleCondition(
            field="language",
            operator=RuleOperator.EQUALS,
            value="Python",
            case_sensitive=False,
        )

        assert condition.evaluate({"language": "python"}) is True
        assert condition.evaluate({"language": "PYTHON"}) is True
        assert condition.evaluate({"language": "Python"}) is True

    def test_nested_field_access(self):
        """Test accessing nested fields with dot notation."""
        condition = RuleCondition(
            field="config.database.type",
            operator=RuleOperator.EQUALS,
            value="postgresql",
        )

        context = {"config": {"database": {"type": "postgresql", "host": "localhost"}}}

        assert condition.evaluate(context) is True
        assert condition.evaluate({"config": {"database": {"type": "mysql"}}}) is False
        assert condition.evaluate({"config": {}}) is False

    def test_numeric_comparisons(self):
        """Test numeric comparison operators."""
        gt_condition = RuleCondition(
            field="team_size", operator=RuleOperator.GREATER_THAN, value=10
        )

        lt_condition = RuleCondition(
            field="response_time", operator=RuleOperator.LESS_THAN, value=100
        )

        assert gt_condition.evaluate({"team_size": 15}) is True
        assert gt_condition.evaluate({"team_size": 5}) is False
        assert gt_condition.evaluate({"team_size": "not_a_number"}) is False

        assert lt_condition.evaluate({"response_time": 50}) is True
        assert lt_condition.evaluate({"response_time": 150}) is False

    def test_regex_matching(self):
        """Test regex pattern matching."""
        condition = RuleCondition(
            field="version",
            operator=RuleOperator.MATCHES_REGEX,
            value=r"^\d+\.\d+\.\d+$",
        )

        assert condition.evaluate({"version": "1.2.3"}) is True
        assert condition.evaluate({"version": "1.2"}) is False
        assert condition.evaluate({"version": "v1.2.3"}) is False


class TestRuleGroup:
    """Test cases for RuleGroup class."""

    def test_and_logic(self):
        """Test AND logic for combining conditions."""
        group = RuleGroup(
            logic=ConditionLogic.AND,
            conditions=[
                RuleCondition("language", RuleOperator.EQUALS, "python"),
                RuleCondition("framework", RuleOperator.EQUALS, "fastapi"),
            ],
        )

        assert group.evaluate({"language": "python", "framework": "fastapi"}) is True
        assert group.evaluate({"language": "python", "framework": "django"}) is False
        assert (
            group.evaluate({"language": "javascript", "framework": "fastapi"}) is False
        )

    def test_or_logic(self):
        """Test OR logic for combining conditions."""
        group = RuleGroup(
            logic=ConditionLogic.OR,
            conditions=[
                RuleCondition("language", RuleOperator.EQUALS, "python"),
                RuleCondition("language", RuleOperator.EQUALS, "javascript"),
            ],
        )

        assert group.evaluate({"language": "python"}) is True
        assert group.evaluate({"language": "javascript"}) is True
        assert group.evaluate({"language": "rust"}) is False

    def test_not_logic(self):
        """Test NOT logic for negating conditions."""
        group = RuleGroup(
            logic=ConditionLogic.NOT,
            conditions=[RuleCondition("production", RuleOperator.EQUALS, True)],
        )

        assert group.evaluate({"production": False}) is True
        assert group.evaluate({"production": True}) is False
        assert group.evaluate({}) is True

    def test_nested_groups(self):
        """Test nested rule groups."""
        # (language = python AND framework = fastapi) OR (language = javascript AND runtime = node)
        group = RuleGroup(
            logic=ConditionLogic.OR,
            conditions=[
                RuleGroup(
                    logic=ConditionLogic.AND,
                    conditions=[
                        RuleCondition("language", RuleOperator.EQUALS, "python"),
                        RuleCondition("framework", RuleOperator.EQUALS, "fastapi"),
                    ],
                ),
                RuleGroup(
                    logic=ConditionLogic.AND,
                    conditions=[
                        RuleCondition("language", RuleOperator.EQUALS, "javascript"),
                        RuleCondition("runtime", RuleOperator.EQUALS, "node"),
                    ],
                ),
            ],
        )

        assert group.evaluate({"language": "python", "framework": "fastapi"}) is True
        assert group.evaluate({"language": "javascript", "runtime": "node"}) is True
        assert group.evaluate({"language": "python", "framework": "django"}) is False
        assert group.evaluate({"language": "javascript", "runtime": "browser"}) is False


class TestStandardRule:
    """Test cases for StandardRule class."""

    def test_rule_evaluation(self):
        """Test basic rule evaluation."""
        rule = StandardRule(
            id="python-api",
            name="Python API Standards",
            description="Standards for Python APIs",
            priority=10,
            conditions=RuleCondition("language", RuleOperator.EQUALS, "python"),
            standards=["python-pep8", "rest-api-design"],
            tags=["python", "api"],
        )

        matches, standards = rule.evaluate({"language": "python"})
        assert matches is True
        assert standards == ["python-pep8", "rest-api-design"]

        matches, standards = rule.evaluate({"language": "javascript"})
        assert matches is False
        assert standards == []

    def test_rule_with_complex_conditions(self):
        """Test rule with complex conditions."""
        rule = StandardRule(
            id="react-accessible",
            name="React with Accessibility",
            description="React apps requiring accessibility",
            priority=5,
            conditions=RuleGroup(
                logic=ConditionLogic.AND,
                conditions=[
                    RuleCondition("framework", RuleOperator.EQUALS, "react"),
                    RuleCondition(
                        "requirements", RuleOperator.CONTAINS, "accessibility"
                    ),
                ],
            ),
            standards=["react-patterns", "wcag-2.2"],
            tags=["react", "accessibility"],
        )

        context = {
            "framework": "react",
            "requirements": ["performance", "accessibility"],
        }

        matches, standards = rule.evaluate(context)
        assert matches is True
        assert "wcag-2.2" in standards


class TestRuleEngine:
    """Test cases for RuleEngine class."""

    @pytest.fixture
    def sample_rules_file(self, tmp_path) -> Path:
        """Create a sample rules file for testing."""
        rules_data = {
            "rules": [
                {
                    "id": "rule1",
                    "name": "Python Rule",
                    "description": "Test rule for Python",
                    "priority": 10,
                    "conditions": {
                        "field": "language",
                        "operator": "equals",
                        "value": "python",
                    },
                    "standards": ["python-pep8"],
                    "tags": ["python"],
                },
                {
                    "id": "rule2",
                    "name": "Security Rule",
                    "description": "Test rule for security",
                    "priority": 5,
                    "conditions": {
                        "field": "requirements",
                        "operator": "contains",
                        "value": "security",
                    },
                    "standards": ["security-best-practices"],
                    "tags": ["security"],
                },
            ]
        }

        rules_file = tmp_path / "test_rules.json"
        with open(rules_file, "w") as f:
            json.dump(rules_data, f)

        return cast(Path, rules_file)

    def test_load_rules_from_file(self, sample_rules_file):
        """Test loading rules from a JSON file."""
        engine = RuleEngine(sample_rules_file)

        assert len(engine.rules) == 2
        assert engine.rules[0].id == "rule1"
        assert engine.rules[1].id == "rule2"

    def test_add_rule(self):
        """Test adding rules programmatically."""
        engine = RuleEngine()

        rule = StandardRule(
            id="test-rule",
            name="Test Rule",
            description="A test rule",
            priority=10,
            conditions=RuleCondition("test", RuleOperator.EQUALS, True),
            standards=["test-standard"],
        )

        engine.add_rule(rule)

        assert len(engine.rules) == 1
        assert engine._rule_index["test-rule"] == rule

    def test_evaluate_simple_context(self, sample_rules_file):
        """Test evaluating a simple context."""
        engine = RuleEngine(sample_rules_file)

        result = engine.evaluate({"language": "python"})

        assert len(result["matched_rules"]) == 1
        assert result["matched_rules"][0]["rule_id"] == "rule1"
        assert "python-pep8" in result["resolved_standards"]

    def test_evaluate_multiple_matches(self, sample_rules_file):
        """Test evaluating context that matches multiple rules."""
        engine = RuleEngine(sample_rules_file)

        result = engine.evaluate(
            {"language": "python", "requirements": ["performance", "security"]}
        )

        assert len(result["matched_rules"]) == 2
        assert len(result["resolved_standards"]) == 2
        assert "python-pep8" in result["resolved_standards"]
        assert "security-best-practices" in result["resolved_standards"]

    def test_priority_resolution(self):
        """Test conflict resolution based on priority."""
        engine = RuleEngine()

        # Add two rules that provide the same standard with different priorities
        rule1 = StandardRule(
            id="rule1",
            name="Lower Priority",
            description="",
            priority=20,
            conditions=RuleCondition("type", RuleOperator.EQUALS, "api"),
            standards=["api-standard", "common-standard"],
        )

        rule2 = StandardRule(
            id="rule2",
            name="Higher Priority",
            description="",
            priority=10,
            conditions=RuleCondition("type", RuleOperator.EQUALS, "api"),
            standards=["specific-standard", "common-standard"],
        )

        engine.add_rule(rule1)
        engine.add_rule(rule2)

        result = engine.evaluate({"type": "api"})

        # Both rules should match
        assert len(result["matched_rules"]) == 2

        # All unique standards should be included
        assert set(result["resolved_standards"]) == {
            "api-standard",
            "common-standard",
            "specific-standard",
        }

        # Conflicts should be detected
        assert len(result["conflicts"]) == 1
        assert result["conflicts"][0]["standard"] == "common-standard"

    def test_tag_filtering(self, sample_rules_file):
        """Test filtering rules by tags."""
        engine = RuleEngine(sample_rules_file)

        # Evaluate with tag filter
        result = engine.evaluate(
            {"language": "python", "requirements": ["security"]}, tags=["python"]
        )

        # Only the Python rule should be evaluated
        assert len(result["matched_rules"]) == 1
        assert result["matched_rules"][0]["rule_id"] == "rule1"

    def test_max_priority_filtering(self, sample_rules_file):
        """Test filtering rules by maximum priority."""
        engine = RuleEngine(sample_rules_file)

        # Only evaluate rules with priority <= 8
        result = engine.evaluate(
            {"language": "python", "requirements": ["security"]}, max_priority=8
        )

        # Only the security rule (priority 5) should match
        assert len(result["matched_rules"]) == 1
        assert result["matched_rules"][0]["rule_id"] == "rule2"

    def test_decision_tree_generation(self, sample_rules_file):
        """Test generating a decision tree from rules."""
        engine = RuleEngine(sample_rules_file)

        tree = engine.get_decision_tree()

        assert tree["type"] == "root"
        assert tree["total_rules"] == 2
        assert len(tree["branches"]) > 0

        # Check that rules are grouped by decision fields
        for branch in tree["branches"]:
            assert "decision_fields" in branch
            assert "rules" in branch

    def test_export_rules(self, tmp_path):
        """Test exporting rules to a file."""
        engine = RuleEngine()

        rule = StandardRule(
            id="export-test",
            name="Export Test",
            description="Test rule for export",
            priority=10,
            conditions=RuleCondition("test", RuleOperator.EQUALS, True),
            standards=["test-standard"],
            tags=["test"],
        )

        engine.add_rule(rule)

        # Export to JSON
        json_path = tmp_path / "exported_rules.json"
        engine.export_rules(json_path, format="json")

        # Verify the exported file
        with open(json_path) as f:
            data = json.load(f)

        assert len(data["rules"]) == 1
        assert data["rules"][0]["id"] == "export-test"

    def test_complex_rule_parsing(self, tmp_path):
        """Test parsing complex rules with nested conditions."""
        complex_rules = {
            "rules": [
                {
                    "id": "complex-rule",
                    "name": "Complex Rule",
                    "description": "Rule with nested conditions",
                    "priority": 10,
                    "conditions": {
                        "logic": "AND",
                        "conditions": [
                            {
                                "field": "project_type",
                                "operator": "equals",
                                "value": "web_app",
                            },
                            {
                                "logic": "OR",
                                "conditions": [
                                    {
                                        "field": "framework",
                                        "operator": "equals",
                                        "value": "react",
                                    },
                                    {
                                        "field": "framework",
                                        "operator": "equals",
                                        "value": "vue",
                                    },
                                ],
                            },
                        ],
                    },
                    "standards": ["web-standards"],
                    "tags": ["web"],
                }
            ]
        }

        rules_file = tmp_path / "complex_rules.json"
        with open(rules_file, "w") as f:
            json.dump(complex_rules, f)

        engine = RuleEngine(rules_file)

        # Test with matching context
        result = engine.evaluate({"project_type": "web_app", "framework": "react"})

        assert len(result["matched_rules"]) == 1
        assert "web-standards" in result["resolved_standards"]

        # Test with non-matching framework
        result2 = engine.evaluate({"project_type": "web_app", "framework": "angular"})

        assert len(result2["matched_rules"]) == 0

    def test_rule_engine_statistics(self, sample_rules_file):
        """Test that evaluation statistics are correctly generated."""
        engine = RuleEngine(sample_rules_file)

        result = engine.evaluate({"language": "python", "requirements": ["security"]})

        stats = result["statistics"]
        assert stats["total_rules_evaluated"] == 2
        assert stats["rules_matched"] == 2
        assert stats["unique_standards"] == 2
        assert stats["conflicts_found"] == 0


class TestRuleEngineIntegration:
    """Integration tests using the actual standard-selection-rules.json file."""

    @pytest.fixture
    def rules_file_path(self) -> Path:
        """Get the path to the actual rules file."""
        return (
            Path(__file__).parent.parent.parent.parent.parent.parent
            / "data"
            / "standards"
            / "meta"
            / "standard-selection-rules.json"
        )

    def test_react_web_app_selection(self, rules_file_path):
        """Test selecting standards for a React web application."""
        if not rules_file_path.exists():
            pytest.skip("Rules file not found")

        engine = RuleEngine(rules_file_path)

        result = engine.evaluate(
            {
                "project_type": "web_application",
                "framework": "react",
                "language": "javascript",
            }
        )

        assert len(result["matched_rules"]) > 0
        assert "react-18-patterns" in result["resolved_standards"]
        assert "javascript-es2025" in result["resolved_standards"]

    def test_python_fastapi_selection(self, rules_file_path):
        """Test selecting standards for a Python FastAPI project."""
        if not rules_file_path.exists():
            pytest.skip("Rules file not found")

        engine = RuleEngine(rules_file_path)

        result = engine.evaluate(
            {"project_type": "api", "language": "python", "framework": "fastapi"}
        )

        # Should match both general Python API and specific FastAPI rules
        matched_ids = [r["rule_id"] for r in result["matched_rules"]]
        assert "python-api" in matched_ids
        assert "python-fastapi" in matched_ids

        # FastAPI should have higher priority (8 < 10)
        assert "fastapi-patterns" in result["resolved_standards"]
        assert "async-python-patterns" in result["resolved_standards"]

    def test_security_critical_application(self, rules_file_path):
        """Test security standards selection."""
        if not rules_file_path.exists():
            pytest.skip("Rules file not found")

        engine = RuleEngine(rules_file_path)

        result = engine.evaluate(
            {
                "project_type": "api",
                "language": "python",
                "requirements": ["security", "performance"],
                "security_level": "high",
            }
        )

        # Security rule should have highest priority (1)
        security_rule = next(
            (r for r in result["matched_rules"] if r["rule_id"] == "security-critical"),
            None,
        )
        assert security_rule is not None
        assert security_rule["priority"] == 1

        # Security standards should be included
        assert "security-owasp-top10" in result["resolved_standards"]
        assert "nist-800-53-controls" in result["resolved_standards"]

    def test_microservices_cloud_native(self, rules_file_path):
        """Test microservices and cloud-native standards."""
        if not rules_file_path.exists():
            pytest.skip("Rules file not found")

        engine = RuleEngine(rules_file_path)

        result = engine.evaluate(
            {
                "architecture": "microservices",
                "deployment_target": "kubernetes",
                "containerized": True,
            }
        )

        # Should match both microservices and cloud-native rules
        matched_ids = [r["rule_id"] for r in result["matched_rules"]]
        assert "microservices" in matched_ids
        assert "cloud-native" in matched_ids

        # Check for expected standards
        standards = result["resolved_standards"]
        assert "microservices-patterns" in standards
        assert "docker-best-practices" in standards
        assert "kubernetes-patterns" in standards
