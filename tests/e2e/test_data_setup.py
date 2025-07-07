"""Set up test data for E2E tests."""

import json
from pathlib import Path


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
            "metadata": {
                "author": "Test Author",
                "last_updated": "2024-01-01"
            }
        },
        "python-testing": {
            "id": "python-testing", 
            "name": "Python Testing Standards",
            "version": "2.1.0",
            "category": "testing",
            "tags": ["python", "testing", "pytest", "quality"],
            "content": "Python testing standards using pytest..."
        },
        "javascript-es6-standards": {
            "id": "javascript-es6-standards",
            "name": "JavaScript ES6+ Standards",
            "version": "1.2.0", 
            "category": "frontend",
            "tags": ["javascript", "es6", "frontend", "web"],
            "content": "Modern JavaScript standards for ES6 and beyond..."
        }
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
                            "value": "web_application"
                        },
                        {
                            "field": "framework",
                            "operator": "equals",
                            "value": "react"
                        }
                    ]
                },
                "standards": ["react-18-patterns", "javascript-es6-standards"],
                "tags": ["frontend", "react"]
            }
        ]
    }
    
    # Write rules file
    rules_file = meta_dir / "standard-selection-rules.json"
    rules_file.write_text(json.dumps(test_rules, indent=2))
    
    # Create sync config
    sync_config = {
        "repository": {
            "owner": "test",
            "name": "standards",
            "branch": "main"
        },
        "paths": {
            "standards": "standards/",
            "metadata": "metadata/"
        }
    }
    
    sync_file = standards_dir / "sync_config.yaml"
    sync_file.write_text(json.dumps(sync_config, indent=2))