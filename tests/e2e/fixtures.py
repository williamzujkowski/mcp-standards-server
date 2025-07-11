"""
Test fixtures and utilities for E2E testing.

Provides:
- Sample project contexts
- Mock standards data
- Test helpers for MCP client interaction
"""

import json
from pathlib import Path
from typing import Any

import pytest

# Sample project contexts for testing
SAMPLE_CONTEXTS = {
    "react_web_app": {
        "project_type": "web_application",
        "framework": "react",
        "language": "javascript",
        "requirements": ["accessibility", "performance", "seo"],
        "team_size": "medium",
        "deployment": "cloud",
    },
    "python_api": {
        "project_type": "api",
        "framework": "fastapi",
        "language": "python",
        "requirements": ["security", "scalability"],
        "database": "postgresql",
        "deployment": "kubernetes",
    },
    "mobile_app": {
        "project_type": "mobile_app",
        "framework": "react-native",
        "platform": ["ios", "android"],
        "requirements": ["offline-support", "push-notifications"],
        "backend": "firebase",
    },
    "microservice": {
        "project_type": "microservice",
        "language": "go",
        "requirements": ["high-performance", "observability"],
        "messaging": "kafka",
        "deployment": "kubernetes",
    },
    "data_pipeline": {
        "project_type": "data_pipeline",
        "framework": "apache-spark",
        "language": "python",
        "requirements": ["fault-tolerance", "scalability"],
        "storage": "s3",
        "orchestration": "airflow",
    },
    "ml_project": {
        "project_type": "machine_learning",
        "framework": "tensorflow",
        "language": "python",
        "requirements": ["reproducibility", "model-versioning"],
        "deployment": "mlflow",
    },
    "mcp_server": {
        "project_type": "mcp_server",
        "language": "typescript",
        "requirements": ["type-safety", "error-handling"],
        "tools": ["file-access", "web-search", "code-analysis"],
    },
}


# Mock standards data for testing
MOCK_STANDARDS = {
    "react-18-patterns": {
        "id": "react-18-patterns",
        "name": "React 18 Best Practices",
        "version": "1.0.0",
        "category": "frontend",
        "tags": ["react", "javascript", "frontend", "components"],
        "metadata": {
            "last_updated": "2025-01-01",
            "author": "Standards Team",
            "complexity": "intermediate",
        },
        "content": """
# React 18 Best Practices

## Component Patterns

### Functional Components
Always use functional components with hooks:
```javascript
const MyComponent = ({ title, children }) => {
    const [count, setCount] = useState(0);

    useEffect(() => {
        // Side effects
    }, []);

    return (
        <div>
            <h1>{title}</h1>
            {children}
        </div>
    );
};
```

### Performance Optimization
- Use React.memo for expensive components
- Implement useMemo and useCallback for costly computations
- Leverage React 18's automatic batching

## Accessibility
- Always include proper ARIA labels
- Ensure keyboard navigation works
- Test with screen readers
""",
    },
    "python-testing": {
        "id": "python-testing",
        "name": "Python Testing Standards",
        "version": "2.1.0",
        "category": "testing",
        "tags": ["python", "testing", "pytest", "quality"],
        "metadata": {
            "last_updated": "2025-01-15",
            "author": "QA Team",
            "complexity": "beginner",
        },
        "content": """
# Python Testing Standards

## Test Structure

### Directory Layout
```
tests/
├── unit/
│   ├── test_models.py
│   └── test_utils.py
├── integration/
│   └── test_api.py
└── conftest.py
```

### Using Pytest
```python
import pytest

def test_addition():
    assert 1 + 1 == 2

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6)
])
def test_double(input, expected):
    assert input * 2 == expected
```

## Best Practices
- Write descriptive test names
- Use fixtures for setup/teardown
- Aim for 80%+ code coverage
""",
    },
    "api-security": {
        "id": "api-security",
        "name": "API Security Standards",
        "version": "3.0.0",
        "category": "security",
        "tags": ["api", "security", "authentication", "authorization"],
        "metadata": {
            "last_updated": "2025-01-20",
            "author": "Security Team",
            "complexity": "advanced",
        },
        "content": """
# API Security Standards

## Authentication
- Implement OAuth 2.0 or JWT
- Use secure token storage
- Implement token refresh mechanisms

## Authorization
- Role-based access control (RBAC)
- Principle of least privilege
- Audit logging for all access

## Data Protection
- Always use HTTPS
- Encrypt sensitive data at rest
- Implement rate limiting
""",
    },
}


# Mock rules for testing
MOCK_RULES = [
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
        "standards": ["react-18-patterns", "frontend-accessibility"],
        "tags": ["frontend", "react"],
    },
    {
        "id": "python-api-rule",
        "name": "Python API Development",
        "priority": 8,
        "conditions": {
            "logic": "AND",
            "conditions": [
                {"field": "project_type", "operator": "equals", "value": "api"},
                {"field": "language", "operator": "equals", "value": "python"},
            ],
        },
        "standards": ["python-testing", "api-security"],
        "tags": ["backend", "python", "api"],
    },
]


class MockStandardsRepository:
    """Mock repository for standards data."""

    def __init__(self, standards: dict[str, Any] | None = None):
        self.standards = standards or MOCK_STANDARDS.copy()

    def get_standard(self, standard_id: str) -> dict[str, Any]:
        """Get a standard by ID."""
        if standard_id not in self.standards:
            raise ValueError(f"Standard '{standard_id}' not found")
        return self.standards[standard_id].copy()

    def list_standards(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all standards, optionally filtered by category."""
        results = []
        for std in self.standards.values():
            if category is None or std.get("category") == category:
                results.append(
                    {
                        "id": std["id"],
                        "name": std["name"],
                        "category": std["category"],
                        "tags": std["tags"],
                    }
                )
        return results

    def search_standards(self, query: str) -> list[dict[str, Any]]:
        """Simple search implementation."""
        query_lower = query.lower()
        results = []

        for std in self.standards.values():
            # Search in name, tags, and content
            name = std.get("name", "")
            content = std.get("content", "")
            tags = std.get("tags", [])

            if (
                (isinstance(name, str) and query_lower in name.lower())
                or any(query_lower in tag for tag in tags)
                or (isinstance(content, str) and query_lower in content.lower())
            ):
                results.append(std)

        return results


class TestDataGenerator:
    """Generate test data for various scenarios."""

    @staticmethod
    def generate_code_sample(language: str, pattern: str = "basic") -> str:
        """Generate code samples for testing."""
        samples = {
            "javascript": {
                "basic": """
                    function greet(name) {
                        console.log('Hello, ' + name);
                    }
                """,
                "react": """
                    import React, { useState } from 'react';

                    const Counter = () => {
                        const [count, setCount] = useState(0);
                        return (
                            <div>
                                <p>Count: {count}</p>
                                <button onClick={() => setCount(count + 1)}>
                                    Increment
                                </button>
                            </div>
                        );
                    };
                """,
                "async": """
                    async function fetchData(url) {
                        try {
                            const response = await fetch(url);
                            return await response.json();
                        } catch (error) {
                            console.error('Error:', error);
                        }
                    }
                """,
            },
            "python": {
                "basic": """
                    def greet(name):
                        print(f"Hello, {name}")
                """,
                "class": """
                    class User:
                        def __init__(self, name, email):
                            self.name = name
                            self.email = email

                        def __str__(self):
                            return f"User({self.name}, {self.email})"
                """,
                "async": """
                    import asyncio

                    async def fetch_data(url):
                        async with aiohttp.ClientSession() as session:
                            async with session.get(url) as response:
                                return await response.json()
                """,
            },
        }

        return samples.get(language, {}).get(pattern, "// No sample available")

    @staticmethod
    def generate_validation_cases() -> list[dict[str, Any]]:
        """Generate validation test cases."""
        return [
            {
                "code": "const x = 'test'",
                "standard": "javascript-es2025",
                "expected_violations": ["Use let or const appropriately"],
            },
            {
                "code": "def test():\n    pass",
                "standard": "python-testing",
                "expected_violations": ["Test function should start with test_"],
            },
            {
                "code": "<div onclick='alert(1)'>Click</div>",
                "standard": "frontend-security",
                "expected_violations": ["Avoid inline event handlers"],
            },
        ]


@pytest.fixture
def mock_standards_repo():
    """Provide mock standards repository."""
    return MockStandardsRepository()


@pytest.fixture
def test_data_generator():
    """Provide test data generator."""
    return TestDataGenerator()


@pytest.fixture
def sample_contexts():
    """Provide sample project contexts."""
    return SAMPLE_CONTEXTS.copy()


@pytest.fixture
def temp_standards_dir(tmp_path):
    """Create temporary directory with test standards."""
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()

    # Write mock standards to files
    for standard_id, content in MOCK_STANDARDS.items():
        file_path = standards_dir / f"{standard_id}.json"
        file_path.write_text(json.dumps(content, indent=2))

    # Write rules file
    rules_file = standards_dir / "rules.json"
    rules_file.write_text(json.dumps(MOCK_RULES, indent=2))

    return standards_dir


def create_test_mcp_config(data_dir: Path) -> dict[str, Any]:
    """Create test MCP server configuration."""
    return {
        "server": {
            "name": "mcp-standards-test",
            "version": "0.1.0",
            "description": "Test MCP Standards Server",
        },
        "data": {
            "standards_dir": str(data_dir / "standards"),
            "cache_dir": str(data_dir / "cache"),
            "rules_file": str(data_dir / "standards" / "rules.json"),
        },
        "sync": {"enabled": False, "interval": 3600},  # Disable sync for tests
        "search": {
            "enabled": True,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "cache_embeddings": True,
        },
    }
