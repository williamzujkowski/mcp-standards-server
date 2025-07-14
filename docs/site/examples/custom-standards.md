# Custom Standards Examples

Learn how to create and use custom standards tailored to your project needs.

## Creating a Custom Standard

### Basic Standard Definition

```yaml
# standards/my-team-standard.yaml
standard:
  id: "my-team-python"
  name: "My Team Python Standards"
  version: "1.0.0"
  category: "coding"
  description: "Python coding standards for my team"

metadata:
  author: "Team Lead"
  team: "Backend Team"
  created: "2024-01-01"

applicability:
  languages: ["python"]
  project_types: ["api", "service"]

rules:
  - id: "function-naming"
    description: "Functions must use snake_case"
    pattern: "^[a-z_][a-z0-9_]*$"
    severity: "error"
    
  - id: "max-function-length"
    description: "Functions should not exceed 50 lines"
    type: "metric"
    max_value: 50
    severity: "warning"
```

### Advanced Standard with Custom Logic

```yaml
standard:
  id: "security-enhanced"
  name: "Enhanced Security Standards"
  
validators:
  - type: "custom"
    module: "validators.security"
    class: "EnhancedSecurityValidator"
    config:
      check_dependencies: true
      scan_depth: "deep"
```

## Implementing Custom Validators

### Simple Pattern Validator

```python
# validators/naming_validator.py
import re
from src.core.validation import BaseValidator

class NamingValidator(BaseValidator):
    def __init__(self, config):
        self.patterns = {
            'class': re.compile(r'^[A-Z][a-zA-Z0-9]*$'),
            'function': re.compile(r'^[a-z_][a-z0-9_]*$'),
            'constant': re.compile(r'^[A-Z_][A-Z0-9_]*$')
        }
    
    def validate(self, ast_node, context):
        violations = []
        
        if ast_node.type == 'class' and not self.patterns['class'].match(ast_node.name):
            violations.append({
                'rule': 'class-naming',
                'message': f"Class '{ast_node.name}' should use PascalCase"
            })
        
        return violations
```

### Complex Business Logic Validator

```python
# validators/business_logic.py
class BusinessLogicValidator(BaseValidator):
    def validate_api_endpoints(self, file_content):
        """Ensure all API endpoints follow REST conventions"""
        violations = []
        
        # Parse routes
        routes = self.extract_routes(file_content)
        
        for route in routes:
            # Check HTTP method matches operation
            if route.method == 'GET' and 'create' in route.path:
                violations.append({
                    'message': f"GET {route.path} should not contain 'create'"
                })
            
            # Check resource naming
            if not self.is_valid_resource_name(route.resource):
                violations.append({
                    'message': f"Resource '{route.resource}' should be plural"
                })
        
        return violations
```

## Standard Templates

### Microservice Standard Template

```yaml
standard:
  id: "microservice-template"
  name: "Microservice Standards Template"
  
includes:
  - "base-coding-standards"
  - "api-design-standards"
  - "security-standards"
  
overrides:
  - rule: "max-file-length"
    max_value: 300  # Smaller files for microservices
    
additions:
  - id: "health-check-required"
    description: "Service must implement /health endpoint"
    type: "structural"
    required_endpoints: ["/health", "/ready"]
```

### Frontend Component Standard

```yaml
standard:
  id: "react-component"
  name: "React Component Standards"
  
rules:
  - id: "component-structure"
    description: "Components must follow standard structure"
    required_sections:
      - "imports"
      - "types/interfaces"
      - "component-definition"
      - "exports"
      
  - id: "prop-types"
    description: "All props must be typed"
    severity: "error"
```

## Testing Custom Standards

```python
# tests/test_custom_standard.py
import pytest
from src.core.standards import StandardsEngine

def test_custom_standard_validation():
    engine = StandardsEngine()
    engine.load_standard('standards/my-team-standard.yaml')
    
    # Test compliant code
    good_code = """
    def calculate_total(items):
        return sum(item.price for item in items)
    """
    
    result = engine.validate_code(good_code, 'my-team-python')
    assert len(result.violations) == 0
    
    # Test non-compliant code
    bad_code = """
    def CalculateTotal(items):  # PascalCase function
        return sum(item.price for item in items)
    """
    
    result = engine.validate_code(bad_code, 'my-team-python')
    assert len(result.violations) == 1
    assert result.violations[0].rule_id == 'function-naming'
```

## Sharing Standards

### Publishing to Registry

```bash
# Package your standard
mcp-standards package my-team-standard.yaml

# Publish to registry
mcp-standards publish my-team-standard-1.0.0.tar.gz
```

### Using Shared Standards

```bash
# Install from registry
mcp-standards install team-standards/my-team-python

# Use in project
echo "extends: my-team-python" > .mcp-standards.yml
```

## Related Documentation

- [Standards Format](../api/standards-format.md)
- [Creating Standards Guide](../../CREATING_STANDARDS_GUIDE.md)
- [Validation Examples](./validation.md)