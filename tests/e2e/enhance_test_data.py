#!/usr/bin/env python3
"""Enhance test data with more complete standards content."""

import json
from pathlib import Path


def enhance_standards():
    """Add more detailed content to test standards."""

    # Enhanced standards with full content
    enhanced_standards = {
        "react-18-patterns": {
            "id": "react-18-patterns",
            "name": "React 18 Best Practices",
            "version": "1.0.0",
            "category": "frontend",
            "tags": ["react", "javascript", "frontend", "components", "web"],
            "content": {
                "summary": "Modern React 18 patterns and best practices for building scalable applications.",
                "sections": [
                    {
                        "title": "Component Design",
                        "content": "Use functional components with hooks. Prefer composition over inheritance.",
                        "examples": [
                            {
                                "description": "Functional component with hooks",
                                "code": "const MyComponent = ({ data }) => {\n  const [state, setState] = useState(data);\n  return <div>{state}</div>;\n};",
                            }
                        ],
                    },
                    {
                        "title": "State Management",
                        "content": "Use useState for local state, useContext for cross-component state.",
                        "rules": [
                            "Avoid prop drilling by using Context API",
                            "Keep state as local as possible",
                            "Use useReducer for complex state logic",
                        ],
                    },
                    {
                        "title": "Performance Optimization",
                        "content": "Optimize rendering with React.memo, useMemo, and useCallback.",
                        "rules": [
                            "Memoize expensive computations",
                            "Use React.lazy for code splitting",
                            "Implement virtualization for long lists",
                        ],
                    },
                ],
            },
            "metadata": {
                "author": "React Team",
                "last_updated": "2024-01-01",
                "applies_to": ["react@>=18.0.0"],
                "nist_controls": ["SA-15", "SA-17"],
            },
            "validation_rules": [
                {
                    "id": "no-class-components",
                    "description": "Prefer functional components over class components",
                    "severity": "warning",
                }
            ],
        },
        "python-testing": {
            "id": "python-testing",
            "name": "Python Testing Standards",
            "version": "2.1.0",
            "category": "testing",
            "tags": ["python", "testing", "pytest", "quality", "backend"],
            "content": {
                "summary": "Comprehensive Python testing standards using pytest framework.",
                "sections": [
                    {
                        "title": "Test Structure",
                        "content": "Organize tests in a tests/ directory mirroring source structure.",
                        "rules": [
                            "Use descriptive test names starting with test_",
                            "Group related tests in classes",
                            "Keep tests independent and isolated",
                        ],
                    },
                    {
                        "title": "Pytest Best Practices",
                        "content": "Leverage pytest features for better tests.",
                        "examples": [
                            {
                                "description": "Parametrized testing",
                                "code": "@pytest.mark.parametrize('input,expected', [\n    (1, 2),\n    (2, 4),\n])\ndef test_double(input, expected):\n    assert double(input) == expected",
                            }
                        ],
                    },
                    {
                        "title": "Coverage Requirements",
                        "content": "Maintain minimum 80% code coverage.",
                        "rules": [
                            "Write tests for all public APIs",
                            "Test edge cases and error conditions",
                            "Use mocks for external dependencies",
                        ],
                    },
                ],
            },
            "metadata": {
                "author": "QA Team",
                "last_updated": "2024-01-15",
                "applies_to": ["python@>=3.8"],
                "tools": ["pytest", "coverage", "pytest-cov"],
            },
        },
        "javascript-es6-standards": {
            "id": "javascript-es6-standards",
            "name": "JavaScript ES6+ Standards",
            "version": "1.2.0",
            "category": "frontend",
            "tags": ["javascript", "es6", "frontend", "web", "modern"],
            "content": {
                "summary": "Modern JavaScript standards for ES6 and beyond.",
                "sections": [
                    {
                        "title": "ES6+ Features",
                        "content": "Use modern JavaScript features for cleaner code.",
                        "rules": [
                            "Use const/let instead of var",
                            "Prefer arrow functions for callbacks",
                            "Use template literals for string interpolation",
                            "Destructure objects and arrays",
                        ],
                    },
                    {
                        "title": "Async Programming",
                        "content": "Handle asynchronous operations properly.",
                        "examples": [
                            {
                                "description": "Async/await pattern",
                                "code": "async function fetchData() {\n  try {\n    const response = await fetch('/api/data');\n    return await response.json();\n  } catch (error) {\n    console.error('Failed to fetch:', error);\n  }\n}",
                            }
                        ],
                    },
                    {
                        "title": "Module System",
                        "content": "Use ES6 modules for code organization.",
                        "rules": [
                            "Use named exports for utilities",
                            "Use default exports for main components",
                            "Avoid circular dependencies",
                        ],
                    },
                ],
            },
            "metadata": {
                "author": "JavaScript Standards Committee",
                "last_updated": "2024-01-10",
                "applies_to": ["node@>=14.0.0", "es2020+"],
            },
        },
    }

    # Write enhanced standards to both locations - use cross-platform paths
    import tempfile

    locations = [
        Path(tempfile.gettempdir()) / "test_standards_data/standards/cache",
        Path.cwd()
        / "data/standards/cache",  # Use current working directory instead of hardcoded path
    ]

    for location in locations:
        if location.exists():
            for std_id, std_data in enhanced_standards.items():
                std_file = location / f"{std_id}.json"
                std_file.write_text(json.dumps(std_data, indent=2))
                print(f"Enhanced {std_id} in {location}")


if __name__ == "__main__":
    enhance_standards()
