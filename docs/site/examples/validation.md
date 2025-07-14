# Validation Examples

This guide provides practical examples of using the MCP Standards Server validation features.

## Basic Validation

### Command Line Validation

```bash
# Validate a single file
mcp-standards validate src/app.py --standard python-best-practices

# Validate entire directory
mcp-standards validate src/ --recursive

# Validate with specific standards
mcp-standards validate . --standards security,performance
```

### Python API Validation

```python
from src.core.standards import StandardsEngine

# Initialize engine
engine = StandardsEngine()

# Validate code
results = engine.validate_file(
    "src/app.py",
    standards=["python-async-patterns", "security-best-practices"]
)

# Process results
for violation in results.violations:
    print(f"{violation.file}:{violation.line} - {violation.message}")
```

## Custom Validation Rules

### Define Custom Rule

```python
# validators/custom_validator.py
from src.analyzers.base import BaseAnalyzer

class CustomValidator(BaseAnalyzer):
    def analyze(self, code: str) -> list:
        violations = []
        
        # Check for specific patterns
        if "TODO" in code:
            violations.append({
                "rule": "no-todos",
                "message": "Remove TODO comments before committing",
                "severity": "warning"
            })
        
        return violations
```

### Register Custom Validator

```yaml
# standards/custom-standard.yaml
standard:
  id: "custom-checks"
  name: "Custom Project Checks"
  
validators:
  - class: "validators.custom_validator.CustomValidator"
    rules:
      - id: "no-todos"
        enabled: true
```

## Validation Workflows

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Run validation before commit
mcp-standards validate --staged-only

if [ $? -ne 0 ]; then
    echo "Validation failed. Please fix issues before committing."
    exit 1
fi
```

### CI/CD Integration

```yaml
# .github/workflows/validate.yml
name: Standards Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run MCP Standards Validation
        run: |
          pip install mcp-standards-server
          mcp-standards validate . --format json > validation-results.json
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: validation-results
          path: validation-results.json
```

## Advanced Examples

### Context-Aware Validation

```python
# Validate with project context
context = {
    "project_type": "web",
    "framework": "django",
    "python_version": "3.11"
}

results = engine.validate_with_context("src/", context)
```

### Batch Validation

```python
# Validate multiple projects
projects = ["project1/", "project2/", "project3/"]

for project in projects:
    results = engine.validate_directory(project)
    report = engine.generate_report(results)
    
    with open(f"{project}/validation-report.html", "w") as f:
        f.write(report)
```

## Validation Output Formats

### JSON Format
```json
{
  "summary": {
    "total_files": 10,
    "files_with_violations": 3,
    "total_violations": 15
  },
  "violations": [...]
}
```

### SARIF Format
For GitHub Code Scanning integration.

### HTML Report
Interactive report with filtering and sorting.

## Related Documentation

- [Validation Rules](../api/validation-rules.md)
- [Custom Standards](./custom-standards.md)
- [CI/CD Integration](../guides/cicd-integration.md)