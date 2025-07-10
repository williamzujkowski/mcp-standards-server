"""Set up test data for E2E tests."""

import json
from pathlib import Path

import yaml


def setup_test_data(data_dir: Path):
    """Set up test standards and rules data."""

    # Create directories
    standards_dir = data_dir / "standards"
    meta_dir = standards_dir / "meta"
    cache_dir = standards_dir / "cache"

    standards_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    # Create test standards
    test_standards = {
        "react-18-patterns": {
            "id": "react-18-patterns",
            "name": "React 18 Best Practices",
            "version": "1.0.0",
            "category": "frontend",
            "tags": ["react", "javascript", "frontend", "components"],
            "content": "React 18 best practices for component development...",
            "metadata": {"author": "Test Author", "last_updated": "2024-01-01"},
        },
        "python-testing": {
            "id": "python-testing",
            "name": "Python Testing Standards",
            "version": "2.1.0",
            "category": "testing",
            "tags": ["python", "testing", "pytest", "quality"],
            "content": "Python testing standards using pytest...",
        },
        "javascript-es6-standards": {
            "id": "javascript-es6-standards",
            "name": "JavaScript ES6+ Standards",
            "version": "1.2.0",
            "category": "frontend",
            "tags": ["javascript", "es6", "frontend", "web"],
            "content": "Modern JavaScript standards for ES6 and beyond...",
        },
    }

    # Write standards to cache
    for std_id, std_data in test_standards.items():
        std_file = cache_dir / f"{std_id}.json"
        std_file.write_text(json.dumps(std_data, indent=2))

    # Create test rules
    test_rules = {
        "rules": [
            {
                "id": "react-web-rule",
                "name": "React Web Application",
                "priority": 10,
                "conditions": {
                    "logic": "AND",
                    "conditions": [
                        {
                            "field": "project_type",
                            "operator": "equals",
                            "value": "web_application",
                        },
                        {"field": "framework", "operator": "equals", "value": "react"},
                    ],
                },
                "standards": ["react-18-patterns", "javascript-es6-standards"],
                "tags": ["frontend", "react"],
            },
            {
                "id": "javascript-web-rule",
                "name": "JavaScript Web Application",
                "priority": 5,
                "conditions": {
                    "logic": "AND",
                    "conditions": [
                        {
                            "field": "project_type",
                            "operator": "equals",
                            "value": "web_application",
                        },
                        {
                            "field": "language",
                            "operator": "equals",
                            "value": "javascript",
                        },
                    ],
                },
                "standards": ["javascript-es6-standards"],
                "tags": ["frontend", "javascript"],
            },
            {
                "id": "general-javascript-rule",
                "name": "General JavaScript Projects",
                "priority": 3,
                "conditions": {
                    "logic": "OR",
                    "conditions": [
                        {
                            "field": "languages",
                            "operator": "contains",
                            "value": "javascript",
                        },
                        {
                            "field": "language",
                            "operator": "equals",
                            "value": "javascript",
                        },
                        {
                            "field": "project_type",
                            "operator": "in",
                            "value": ["web", "frontend", "fullstack"],
                        },
                    ],
                },
                "standards": ["javascript-es6-standards", "react-18-patterns"],
                "tags": ["javascript", "web", "frontend"],
            },
            {
                "id": "python-api-rule",
                "name": "Python API Application",
                "priority": 8,
                "conditions": {
                    "logic": "AND",
                    "conditions": [
                        {"field": "project_type", "operator": "equals", "value": "api"},
                        {"field": "language", "operator": "equals", "value": "python"},
                    ],
                },
                "standards": ["python-testing"],
                "tags": ["python", "api", "backend"],
            },
            {
                "id": "mobile-app-rule",
                "name": "Mobile Application",
                "priority": 7,
                "conditions": {
                    "logic": "AND",
                    "conditions": [
                        {
                            "field": "project_type",
                            "operator": "equals",
                            "value": "mobile_app",
                        },
                        {
                            "field": "framework",
                            "operator": "equals",
                            "value": "react-native",
                        },
                    ],
                },
                "standards": ["react-18-patterns", "javascript-es6-standards"],
                "tags": ["mobile", "react-native", "javascript"],
            },
        ]
    }

    # Write rules file
    rules_file = meta_dir / "enhanced-selection-rules.json"
    rules_file.write_text(json.dumps(test_rules, indent=2))

    # Create sync config that points to local test data
    sync_config = {
        "repository": {
            "owner": "test",
            "repo": "standards",
            "branch": "main",
            "path": "standards/",
        },
        "paths": {"standards": "standards/", "metadata": "metadata/"},
        "sync": {"enabled": False},  # Disable actual GitHub sync in tests
    }

    sync_file = standards_dir / "sync_config.yaml"
    sync_file.write_text(yaml.dump(sync_config, default_flow_style=False))
