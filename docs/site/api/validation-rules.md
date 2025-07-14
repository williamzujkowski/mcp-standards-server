# Validation Rules API

This document describes the validation rules system and how to define custom validation logic.

## Overview

Validation rules define how code is checked against standards. Each rule can use pattern matching, AST analysis, or custom validators.

## Rule Types

### 1. Pattern-Based Rules
Simple regex or string pattern matching.

```yaml
rules:
  - id: "no-console-log"
    type: "pattern"
    pattern: "console\\.log\\("
    message: "Remove console.log statements"
    severity: "warning"
```

### 2. AST-Based Rules
Abstract Syntax Tree analysis for deeper code understanding.

```yaml
rules:
  - id: "no-unused-vars"
    type: "ast"
    ast_query: "//VariableDeclarator[not(referenced)]"
    message: "Variable '${name}' is declared but never used"
    severity: "error"
```

### 3. Custom Validators
Python-based custom validation logic.

```python
from src.core.validation import BaseValidator

class SecurityValidator(BaseValidator):
    def validate(self, code: str, context: dict) -> list:
        violations = []
        # Custom validation logic
        return violations
```

## Severity Levels

- **error**: Must fix - blocks deployment
- **warning**: Should fix - quality issue
- **info**: Consider fixing - best practice

## Rule Configuration

```yaml
rules:
  - id: "rule-id"
    enabled: true
    severity: "error"
    options:
      max_length: 80
      exceptions: ["test_*"]
```

## Built-in Validators

### Security Validators
- SQL injection detection
- XSS vulnerability scanning
- Hardcoded secrets detection

### Code Quality Validators
- Complexity analysis
- Naming conventions
- Documentation coverage

### Performance Validators
- N+1 query detection
- Memory leak patterns
- Inefficient algorithms

## Custom Validator API

```python
class CustomValidator:
    def __init__(self, options: dict):
        self.options = options
    
    def validate(self, file_path: str) -> ValidationResult:
        # Implementation
        pass
```

## Validation Result Format

```json
{
  "violations": [
    {
      "rule_id": "no-console-log",
      "file": "src/app.js",
      "line": 42,
      "column": 5,
      "severity": "warning",
      "message": "Remove console.log statements",
      "suggestion": "Use proper logging framework"
    }
  ],
  "summary": {
    "total": 1,
    "errors": 0,
    "warnings": 1,
    "info": 0
  }
}
```

## Related Documentation

- [Standards Format](./standards-format.md)
- [Writing Custom Validators](../contributing/validators.md)
- [Validation Examples](../examples/validation.md)