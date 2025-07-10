#!/usr/bin/env python3
"""
Demonstration of the Rule Engine for automatic standard selection.

This script shows how to use the rule engine to automatically select
appropriate development standards based on project context.
"""

import json
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.standards.rule_engine import RuleEngine


def print_results(context: dict, result: dict):
    """Pretty print the evaluation results."""
    print("\n" + "=" * 80)
    print(f"Context: {json.dumps(context, indent=2)}")
    print("\n" + "-" * 40)

    print(f"\nMatched Rules ({len(result['matched_rules'])}):")
    for rule in result["matched_rules"]:
        print(f"  - {rule['rule_name']} (priority: {rule['priority']})")
        print(f"    Standards: {', '.join(rule['standards'])}")

    print(f"\nResolved Standards ({len(result['resolved_standards'])}):")
    for standard in sorted(result["resolved_standards"]):
        print(f"  - {standard}")

    if result["conflicts"]:
        print(f"\nConflicts Detected ({len(result['conflicts'])}):")
        for conflict in result["conflicts"]:
            print(
                f"  - Standard '{conflict['standard']}' provided by rules: {', '.join(conflict['rules'])}"
            )

    print("\nStatistics:")
    for key, value in result["statistics"].items():
        print(f"  - {key}: {value}")


def main():
    """Run demonstration scenarios."""
    # Load the rule engine with the standard selection rules
    rules_path = (
        Path(__file__).parent.parent
        / "data"
        / "standards"
        / "meta"
        / "standard-selection-rules.json"
    )

    if not rules_path.exists():
        print(f"Error: Rules file not found at {rules_path}")
        return

    engine = RuleEngine(rules_path)
    print(f"Loaded {len(engine.rules)} rules from {rules_path.name}")

    # Scenario 1: React Web Application with Accessibility
    print("\n\nScenario 1: React Web Application with Accessibility Requirements")
    context1 = {
        "project_type": "web_application",
        "framework": "react",
        "language": "javascript",
        "requirements": ["accessibility", "performance"],
    }
    result1 = engine.evaluate(context1)
    print_results(context1, result1)

    # Scenario 2: Python FastAPI Microservice
    print("\n\nScenario 2: Python FastAPI Microservice")
    context2 = {
        "project_type": "api",
        "language": "python",
        "framework": "fastapi",
        "architecture": "microservices",
        "deployment_target": "kubernetes",
        "database": "postgresql",
    }
    result2 = engine.evaluate(context2)
    print_results(context2, result2)

    # Scenario 3: Security-Critical Financial Application
    print("\n\nScenario 3: Security-Critical Financial Application")
    context3 = {
        "project_type": "api",
        "language": "python",
        "requirements": ["security", "compliance", "performance"],
        "security_level": "critical",
        "compliance": ["PCI-DSS", "SOC2"],
        "team_size": 15,
    }
    result3 = engine.evaluate(context3)
    print_results(context3, result3)

    # Scenario 4: Mobile Application with React Native
    print("\n\nScenario 4: Mobile Application with React Native")
    context4 = {
        "project_type": "mobile_app",
        "framework": "react-native",
        "language": "javascript",
        "requirements": ["offline-first", "performance"],
    }
    result4 = engine.evaluate(context4)
    print_results(context4, result4)

    # Scenario 5: Data Pipeline with Machine Learning
    print("\n\nScenario 5: Data Pipeline with Machine Learning")
    context5 = {
        "project_type": "data_pipeline",
        "language": "python",
        "components": ["data_pipeline", "ml_model"],
        "requirements": ["data-quality", "monitoring"],
    }
    result5 = engine.evaluate(context5)
    print_results(context5, result5)

    # Generate and display decision tree
    print("\n\n" + "=" * 80)
    print("Decision Tree Structure:")
    print("=" * 80)
    tree = engine.get_decision_tree()
    print(json.dumps(tree, indent=2))


if __name__ == "__main__":
    main()
