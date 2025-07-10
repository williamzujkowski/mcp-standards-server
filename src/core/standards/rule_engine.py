"""
Rule Engine for Automatic Standard Selection

This module implements a flexible rule engine that evaluates project contexts
against defined rules to automatically select applicable standards.
"""

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Union

import yaml

logger = logging.getLogger(__name__)


class RuleOperator(Enum):
    """Logical operators for rule conditions."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES_REGEX = "matches_regex"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"


class ConditionLogic(Enum):
    """Logic for combining multiple conditions."""

    AND = "AND"
    OR = "OR"
    NOT = "NOT"


@dataclass
class RuleCondition:
    """Represents a single condition in a rule."""

    field: str
    operator: RuleOperator
    value: Any
    case_sensitive: bool = True

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate this condition against the given context."""
        try:
            field_value = self._get_nested_value(context, self.field)

            if self.operator == RuleOperator.EXISTS:
                return field_value is not None
            elif self.operator == RuleOperator.NOT_EXISTS:
                return field_value is None

            # Handle None values
            if field_value is None:
                return False

            # Normalize strings for case-insensitive comparison
            compare_value: Any
            if isinstance(field_value, str) and not self.case_sensitive:
                field_value = field_value.lower()
                if isinstance(self.value, str):
                    compare_value = self.value.lower()
                elif isinstance(self.value, list):
                    compare_value = [
                        v.lower() if isinstance(v, str) else v for v in self.value
                    ]
                else:
                    compare_value = self.value
            else:
                compare_value = self.value

            # Evaluate based on operator
            if self.operator == RuleOperator.EQUALS:
                return bool(field_value == compare_value)
            elif self.operator == RuleOperator.NOT_EQUALS:
                return bool(field_value != compare_value)
            elif self.operator == RuleOperator.CONTAINS:
                if isinstance(field_value, list | set | str):
                    return compare_value in field_value
                return False
            elif self.operator == RuleOperator.NOT_CONTAINS:
                if isinstance(field_value, list | set | str):
                    return compare_value not in field_value
                return True
            elif self.operator == RuleOperator.IN:
                return field_value in compare_value
            elif self.operator == RuleOperator.NOT_IN:
                return field_value not in compare_value
            elif self.operator == RuleOperator.MATCHES_REGEX:
                import re

                pattern = re.compile(
                    compare_value, re.IGNORECASE if not self.case_sensitive else 0
                )
                return bool(pattern.match(str(field_value)))
            elif self.operator in (
                RuleOperator.GREATER_THAN,
                RuleOperator.LESS_THAN,
                RuleOperator.GREATER_EQUAL,
                RuleOperator.LESS_EQUAL,
            ):
                return self._compare_numeric(field_value, compare_value)

            return False

        except Exception as e:
            logger.warning(
                f"Error evaluating condition {self.field} {self.operator}: {e}"
            )
            return False

    def _get_nested_value(self, context: dict[str, Any], field_path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = field_path.split(".")
        value = context

        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return None

        return value

    def _compare_numeric(self, field_value: Any, compare_value: Any) -> bool:
        """Compare numeric values based on operator."""
        try:
            field_num = float(field_value)
            compare_num = float(compare_value)

            if self.operator == RuleOperator.GREATER_THAN:
                return field_num > compare_num
            elif self.operator == RuleOperator.LESS_THAN:
                return field_num < compare_num
            elif self.operator == RuleOperator.GREATER_EQUAL:
                return field_num >= compare_num
            elif self.operator == RuleOperator.LESS_EQUAL:
                return field_num <= compare_num
        except (ValueError, TypeError):
            return False

        return False


@dataclass
class RuleGroup:
    """Groups multiple conditions with a logic operator."""

    logic: ConditionLogic
    conditions: list[Union[RuleCondition, "RuleGroup"]] = field(default_factory=list)

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate this rule group against the given context."""
        if self.logic == ConditionLogic.NOT:
            # NOT logic expects exactly one condition
            if self.conditions:
                return not self.conditions[0].evaluate(context)
            return True

        results = [cond.evaluate(context) for cond in self.conditions]

        if self.logic == ConditionLogic.AND:
            return all(results)
        elif self.logic == ConditionLogic.OR:
            return any(results)

        return False


@dataclass
class StandardRule:
    """Represents a rule for standard selection."""

    id: str
    name: str
    description: str
    priority: int
    conditions: RuleCondition | RuleGroup
    standards: list[str]
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, context: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Evaluate this rule against the context.

        Returns:
            Tuple of (matches, standards) where matches is bool and standards is list of standard IDs
        """
        matches = self.conditions.evaluate(context)
        return matches, self.standards if matches else []


class RuleEngine:
    """Main rule engine for automatic standard selection."""

    def __init__(self, rules_path: Path | None = None) -> None:
        """
        Initialize the rule engine.

        Args:
            rules_path: Path to rules configuration file (JSON or YAML)
        """
        self.rules: list[StandardRule] = []
        self._rule_index: dict[str, StandardRule] = {}

        if rules_path:
            self.load_rules(rules_path)

    def load_rules(self, rules_path: Path) -> None:
        """Load rules from a configuration file."""
        if not rules_path.exists():
            raise FileNotFoundError(f"Rules file not found: {rules_path}")

        # Load file based on extension
        if rules_path.suffix.lower() == ".json":
            with open(rules_path) as f:
                data = json.load(f)
        elif rules_path.suffix.lower() in (".yaml", ".yml"):
            with open(rules_path) as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {rules_path.suffix}")

        # Parse rules
        self.rules = []
        for rule_data in data.get("rules", []):
            rule = self._parse_rule(rule_data)
            self.add_rule(rule)

        logger.info(f"Loaded {len(self.rules)} rules from {rules_path}")

    def add_rule(self, rule: StandardRule) -> None:
        """Add a rule to the engine."""
        self.rules.append(rule)
        self._rule_index[rule.id] = rule

    def evaluate(
        self,
        context: dict[str, Any],
        max_priority: int | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Evaluate all rules against the given context.

        Args:
            context: Project context dictionary
            max_priority: Only consider rules with priority <= this value
            tags: Only consider rules with at least one matching tag

        Returns:
            Dictionary with evaluation results including selected standards
        """
        matched_rules = []
        all_standards = set()
        conflicts = []

        # Sort rules by priority (lower number = higher priority)
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)

        for rule in sorted_rules:
            # Apply filters
            if max_priority is not None and rule.priority > max_priority:
                continue

            if tags and not any(tag in rule.tags for tag in tags):
                continue

            # Evaluate rule
            matches, standards = rule.evaluate(context)

            if matches:
                matched_rules.append(
                    {
                        "rule_id": rule.id,
                        "rule_name": rule.name,
                        "priority": rule.priority,
                        "standards": standards,
                    }
                )

                # Check for conflicts
                for standard in standards:
                    if standard in all_standards:
                        conflicts.append(
                            {
                                "standard": standard,
                                "rules": [
                                    r["rule_id"]
                                    for r in matched_rules
                                    if "standards" in r
                                    and isinstance(r["standards"], list)
                                    and standard in r["standards"]
                                ],
                            }
                        )

                all_standards.update(standards)

        # Resolve conflicts based on priority
        resolved_standards = self._resolve_conflicts(matched_rules, conflicts)

        return {
            "context": context,
            "matched_rules": matched_rules,
            "all_standards": list(all_standards),
            "resolved_standards": resolved_standards,
            "conflicts": conflicts,
            "statistics": {
                "total_rules_evaluated": len(sorted_rules),
                "rules_matched": len(matched_rules),
                "unique_standards": len(resolved_standards),
                "conflicts_found": len(conflicts),
            },
        }

    def _resolve_conflicts(
        self, matched_rules: list[dict], conflicts: list[dict]
    ) -> list[str]:
        """Resolve conflicts based on rule priority."""
        # Create priority map for standards
        standard_priority: dict[str, int] = {}

        for rule in matched_rules:
            for standard in rule["standards"]:
                if (
                    standard not in standard_priority
                    or rule["priority"] < standard_priority[standard]
                ):
                    standard_priority[standard] = rule["priority"]

        # Return unique standards
        return list(standard_priority.keys())

    def _parse_rule(self, rule_data: dict) -> StandardRule:
        """Parse rule data into StandardRule object."""
        # Parse conditions
        conditions = self._parse_conditions(rule_data["conditions"])

        return StandardRule(
            id=rule_data["id"],
            name=rule_data["name"],
            description=rule_data.get("description", ""),
            priority=rule_data.get("priority", 100),
            conditions=conditions,
            standards=rule_data["standards"],
            tags=rule_data.get("tags", []),
            metadata=rule_data.get("metadata", {}),
        )

    def _parse_conditions(
        self, conditions_data: dict | list
    ) -> RuleCondition | RuleGroup:
        """Parse conditions data into condition objects."""
        if isinstance(conditions_data, dict):
            # Check if it's a group
            if "logic" in conditions_data:
                logic = ConditionLogic(conditions_data["logic"])
                conditions = []

                for cond_data in conditions_data.get("conditions", []):
                    conditions.append(self._parse_conditions(cond_data))

                return RuleGroup(logic=logic, conditions=conditions)
            else:
                # Single condition
                return RuleCondition(
                    field=conditions_data["field"],
                    operator=RuleOperator(conditions_data["operator"]),
                    value=conditions_data.get("value"),
                    case_sensitive=conditions_data.get("case_sensitive", True),
                )
        elif isinstance(conditions_data, list):
            # List implies AND logic
            conditions = [self._parse_conditions(cond) for cond in conditions_data]
            return RuleGroup(logic=ConditionLogic.AND, conditions=conditions)

        raise ValueError(f"Invalid conditions format: {type(conditions_data)}")

    def get_decision_tree(self) -> dict[str, Any]:
        """
        Generate a decision tree representation of the rules.

        Returns:
            Dictionary representing the decision tree structure
        """
        tree: dict[str, Any] = {
            "type": "root",
            "total_rules": len(self.rules),
            "branches": [],
        }

        # Group rules by common patterns
        grouped: dict[tuple[str, ...], list[StandardRule]] = {}
        for rule in self.rules:
            # Extract key fields from conditions
            key_fields = self._extract_key_fields(rule.conditions)
            key = tuple(sorted(key_fields))

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(rule)

        # Build tree branches
        for fields, rules in grouped.items():
            branch = {
                "decision_fields": list(fields),
                "rules": [
                    {
                        "id": rule.id,
                        "name": rule.name,
                        "priority": rule.priority,
                        "standards": rule.standards,
                    }
                    for rule in sorted(rules, key=lambda r: r.priority)
                ],
            }
            branches = tree["branches"]
            if isinstance(branches, list):
                branches.append(branch)

        return tree

    def _extract_key_fields(
        self, conditions: RuleCondition | RuleGroup, fields: set[str] | None = None
    ) -> set[str]:
        """Extract key fields from conditions recursively."""
        if fields is None:
            fields = set()

        if isinstance(conditions, RuleCondition):
            fields.add(conditions.field)
        elif isinstance(conditions, RuleGroup):
            for cond in conditions.conditions:
                self._extract_key_fields(cond, fields)

        return fields

    def export_rules(self, output_path: Path, format: str = "json") -> None:
        """Export rules to a file."""
        rules_data = {"rules": [self._rule_to_dict(rule) for rule in self.rules]}

        if format == "json":
            with open(output_path, "w") as f:
                json.dump(rules_data, f, indent=2)
        elif format in ("yaml", "yml"):
            with open(output_path, "w") as f:
                yaml.dump(rules_data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _rule_to_dict(self, rule: StandardRule) -> dict:
        """Convert rule to dictionary format."""
        return {
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "priority": rule.priority,
            "conditions": self._conditions_to_dict(rule.conditions),
            "standards": rule.standards,
            "tags": rule.tags,
            "metadata": rule.metadata,
        }

    def _conditions_to_dict(self, conditions: RuleCondition | RuleGroup) -> dict:
        """Convert conditions to dictionary format."""
        if isinstance(conditions, RuleCondition):
            return {
                "field": conditions.field,
                "operator": conditions.operator.value,
                "value": conditions.value,
                "case_sensitive": conditions.case_sensitive,
            }
        elif isinstance(conditions, RuleGroup):
            return {
                "logic": conditions.logic.value,
                "conditions": [
                    self._conditions_to_dict(cond) for cond in conditions.conditions
                ],
            }

        return {}
