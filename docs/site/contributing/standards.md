# Contributing Standards Guide

Learn how to contribute new standards or improve existing ones in the MCP Standards Server.

## Understanding Standards

Standards are the core of the MCP Standards Server. They define best practices, coding conventions, and quality requirements for various aspects of software development.

## Standard Contribution Process

### 1. Identify the Need

Before creating a new standard:
- Check if a similar standard already exists
- Discuss in GitHub Issues or Discussions
- Gather community feedback

### 2. Choose Standard Type

- **Language Standards**: Python, JavaScript, Go, etc.
- **Framework Standards**: React, Django, Spring, etc.
- **Domain Standards**: Security, Performance, Testing
- **Process Standards**: CI/CD, Documentation, Code Review

### 3. Create Standard File

Use the appropriate template:

```yaml
# standards/YOUR_STANDARD_NAME.yaml
standard:
  id: "unique-standard-id"
  name: "Human Readable Name"
  version: "1.0.0"
  category: "coding|security|testing|architecture"
  description: |
    Clear description of what this standard covers and why it's important.

metadata:
  author: "Your Name"
  created: "2024-01-20"
  tags: ["relevant", "tags"]
  references:
    - "https://link-to-authoritative-source"

applicability:
  languages: ["python", "javascript"]
  project_types: ["web", "api"]
  frameworks: ["django", "fastapi"]

rules:
  - id: "rule-1"
    name: "Descriptive Rule Name"
    description: "What this rule checks"
    severity: "error"
    rationale: "Why this rule exists"
    
examples:
  good:
    - description: "Example following the standard"
      code: |
        def calculate_total(items: List[Item]) -> float:
            """Calculate total price of items."""
            return sum(item.price for item in items)
  
  bad:
    - description: "Example violating the standard"
      code: |
        def calc(i):
            t = 0
            for x in i: t += x.price
            return t
```

### 4. Add Validation Rules

Create validators for automated checking:

```python
# src/validators/your_standard_validator.py
from src.analyzers.base import BaseAnalyzer

class YourStandardValidator(BaseAnalyzer):
    """Validator for your standard."""
    
    def analyze(self, code: str, context: dict) -> List[Violation]:
        violations = []
        
        # Implement your validation logic
        if self.check_violation(code):
            violations.append(
                Violation(
                    rule_id="rule-1",
                    message="Descriptive error message",
                    line=line_number,
                    severity="error"
                )
            )
        
        return violations
```

### 5. Write Tests

```python
# tests/standards/test_your_standard.py
import pytest
from src.core.standards import StandardsEngine

class TestYourStandard:
    def test_validation_passes(self):
        """Test that compliant code passes validation."""
        engine = StandardsEngine()
        engine.load_standard("your-standard-id")
        
        good_code = """
        # Your compliant code example
        """
        
        result = engine.validate_code(good_code)
        assert result.passed
        assert len(result.violations) == 0
    
    def test_validation_fails(self):
        """Test that non-compliant code fails validation."""
        engine = StandardsEngine()
        engine.load_standard("your-standard-id")
        
        bad_code = """
        # Your non-compliant code example
        """
        
        result = engine.validate_code(bad_code)
        assert not result.passed
        assert len(result.violations) > 0
```

### 6. Document the Standard

Create comprehensive documentation:

```markdown
# standards/docs/YOUR_STANDARD_NAME.md

# Your Standard Name

## Overview
Explain what this standard covers and its importance.

## Rules

### Rule 1: Descriptive Name
- **What**: Explain what the rule checks
- **Why**: Explain the rationale
- **How**: Show how to comply

### Examples
[Include multiple examples]

## Migration Guide
If replacing an existing standard, provide migration steps.

## References
- Link to authoritative sources
- Related standards
- Further reading
```

## Quality Checklist

Before submitting your standard:

- [ ] **Unique ID**: Ensure the ID doesn't conflict
- [ ] **Clear Description**: Easy to understand purpose
- [ ] **Comprehensive Rules**: Cover all relevant cases
- [ ] **Good/Bad Examples**: Clear, realistic examples
- [ ] **Tests**: Unit tests for validators
- [ ] **Documentation**: Complete user guide
- [ ] **Performance**: Validators run efficiently
- [ ] **Compatibility**: Works with target languages/frameworks

## Submission Process

1. **Fork and Branch**
   ```bash
   git checkout -b standard/your-standard-name
   ```

2. **Add Your Standard**
   - Place YAML in `data/standards/`
   - Add validator in `src/validators/`
   - Add tests in `tests/standards/`
   - Add docs in `docs/standards/`

3. **Run Validation**
   ```bash
   # Validate your standard file
   mcp-standards validate-standard data/standards/YOUR_STANDARD.yaml
   
   # Run tests
   pytest tests/standards/test_your_standard.py
   ```

4. **Create Pull Request**
   - Use template: "New Standard: [Name]"
   - Reference related issues
   - Include motivation and examples

## Review Process

Your standard will be reviewed for:

1. **Relevance**: Addresses real needs
2. **Quality**: Well-defined and comprehensive
3. **Compatibility**: Works with existing standards
4. **Performance**: Efficient validation
5. **Documentation**: Clear and complete

## Maintenance

After your standard is merged:

- Monitor issues related to your standard
- Update based on community feedback
- Maintain backward compatibility
- Document breaking changes

## Examples of Good Standards

Study these well-crafted standards:

- `python-async-patterns`: Comprehensive async/await patterns
- `security-input-validation`: Security-focused with clear rationale
- `react-component-standards`: Framework-specific best practices

## Getting Help

- Ask questions in GitHub Discussions
- Review existing standards for examples
- Join our contributor community

Thank you for contributing to better software development practices! 🎉