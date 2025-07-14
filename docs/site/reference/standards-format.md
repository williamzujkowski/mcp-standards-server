# Standards Format Reference

This document provides a complete reference for the MCP Standards file format.

## Overview

Standards are defined in YAML or JSON format. YAML is preferred for readability.

## Complete Format Specification

```yaml
# Complete standard definition example
standard:
  id: "comprehensive-example"          # Required: Unique identifier
  name: "Comprehensive Example"        # Required: Human-readable name
  version: "2.1.0"                    # Required: Semantic version
  category: "coding"                  # Required: Category
  description: |                      # Required: Description
    This is a comprehensive example showing all possible fields
    and configurations for an MCP standard.
  enabled: true                       # Optional: Default true
  experimental: false                 # Optional: Mark as experimental
  deprecated: false                   # Optional: Mark as deprecated
  deprecation_message: ""             # Optional: Why deprecated

metadata:
  author: "Jane Doe"                  # Recommended: Author name
  email: "jane@example.com"          # Optional: Contact email
  organization: "ACME Corp"          # Optional: Organization
  created: "2024-01-01"              # Recommended: Creation date
  updated: "2024-01-20"              # Recommended: Last update
  license: "MIT"                     # Optional: License
  tags:                              # Recommended: Tags for search
    - "best-practices"
    - "security"
    - "performance"
  references:                        # Optional: External references
    - url: "https://example.com/guide"
      title: "Official Guide"
    - url: "https://standards.org/xyz"
      title: "Industry Standard XYZ"

dependencies:                        # Optional: Required standards
  - id: "base-standard"
    version: ">=1.0.0"
  - id: "security-baseline"
    version: "~2.0.0"

conflicts:                           # Optional: Conflicting standards
  - id: "legacy-standard"
    reason: "Incompatible rule sets"

extends:                             # Optional: Inherit from standard
  - "parent-standard"
  - "mixin-standard"

applicability:                       # Required: When to apply
  languages:                         # At least one required
    - "python"
    - "javascript"
  versions:                          # Optional: Language versions
    python: ["3.9", "3.10", "3.11", "3.12"]
    javascript: ["ES2020", "ES2021", "ES2022"]
  project_types:                     # Recommended
    - "web"
    - "api"
    - "cli"
    - "library"
  frameworks:                        # Optional
    - "django"
    - "fastapi"
    - "flask"
  environments:                      # Optional
    - "production"
    - "development"
  file_patterns:                     # Optional: Custom patterns
    - "src/**/*.py"
    - "!**/*_test.py"
  size_thresholds:                   # Optional
    min_lines: 10
    max_lines: 10000

compliance:                          # Optional: Compliance mapping
  frameworks:
    - framework: "NIST-800-53"
      controls:
        - "AC-2"
        - "AC-3"
        - "AU-2"
    - framework: "ISO-27001"
      controls:
        - "A.9.1.1"
        - "A.9.4.1"
  certifications:
    - "SOC2-Type2"
    - "PCI-DSS"

rules:                               # Required: Validation rules
  - id: "rule-001"                   # Required: Unique ID
    name: "Descriptive Rule Name"    # Required: Name
    description: |                   # Required: Description
      Detailed description of what this rule checks
      and why it's important.
    category: "style"                # Optional: Rule category
    severity: "error"                # Required: error/warning/info
    enabled: true                    # Optional: Default true
    tags: ["security", "owasp"]      # Optional: Rule tags
    
    # Pattern-based rule
    type: "pattern"                  # Rule type
    pattern: "TODO|FIXME|XXX"        # Regex pattern
    message: "Remove TODO comments"   # Error message
    
    # Advanced pattern with captures
    advanced_pattern:
      regex: "password\\s*=\\s*[\"'](.+?)[\"']"
      captures:
        - name: "password"
          index: 1
      message: "Hardcoded password: {password}"
    
    # AST-based rule
    ast_pattern:
      type: "function_def"
      conditions:
        - "len(body) > 50"
        - "complexity > 10"
    
    # Custom validator
    validator:
      module: "validators.custom"
      class: "CustomRuleValidator"
      method: "validate"
    
    # Configuration
    config:
      max_length: 100
      ignore_comments: true
      exceptions: ["test_*", "*_spec.py"]
    
    # Fix suggestion
    fix:
      type: "replace"                # replace/delete/insert
      suggestion: "Use logging instead"
      auto_fixable: true
      replacement_template: "logger.debug('{message}')"
    
    # Examples for this rule
    examples:
      good:
        - code: |
            logger.debug("Debug information")
          description: "Use proper logging"
      bad:
        - code: |
            # TODO: Fix this later
          description: "TODO comment"

  - id: "rule-002"
    name: "Complex Rule Example"
    description: "Shows advanced rule features"
    severity: "warning"
    
    # Conditional application
    conditions:
      - "file.is_test == false"
      - "file.size > 1000"
    
    # Multiple validators
    validators:
      - type: "pattern"
        pattern: "pattern1"
      - type: "ast"
        query: "//FunctionDef[...]"
    
    # Context-aware message
    message_template: |
      {rule_name} violation in {file}:{line}
      Found: {found}
      Expected: {expected}

# Global rule configuration
rule_config:
  default_severity: "warning"
  fail_fast: false
  parallel_execution: true
  ignore_generated_files: true

# Validation behavior
validation:
  pre_processors:                    # Optional: Before validation
    - "formatters.normalize_lineendings"
    - "formatters.expand_tabs"
  
  post_processors:                   # Optional: After validation
    - "reporters.summary"
    - "reporters.junit"
  
  options:
    stop_on_first_error: false
    report_ignored_files: true
    include_suggestions: true

# Code examples
examples:                            # Recommended: Usage examples
  description: "Examples showing standard compliance"
  
  good:                              # Compliant examples
    - name: "Well-structured function"
      description: "Shows proper function structure"
      language: "python"
      code: |
        def calculate_total(items: List[Item]) -> Decimal:
            """Calculate total price of items.
            
            Args:
                items: List of items to calculate
                
            Returns:
                Total price as Decimal
            """
            if not items:
                return Decimal('0.00')
            
            return sum(item.price for item in items)
      highlights:                    # Optional: Highlight lines
        - lines: [1]
          message: "Type hints"
        - lines: [2-8]
          message: "Docstring"
    
  bad:                               # Non-compliant examples
    - name: "Poor function structure"
      description: "Multiple violations"
      language: "python"
      code: |
        def calc(i):
            t = 0
            for x in i: t += x.price
            return t
      violations:
        - line: 1
          rule: "type-hints-required"
          message: "Missing type hints"
        - line: 1
          rule: "descriptive-names"
          message: "Non-descriptive function name"
        - line: 2-4
          rule: "no-single-letter-vars"
          message: "Single letter variables"

# Custom sections
custom_sections:                     # Optional: Extended metadata
  performance_impact: "low"
  review_required: true
  automation_level: "full"
  
  team_config:
    owner: "platform-team"
    reviewers: ["security-team", "qa-team"]
    sla_hours: 24

# Metrics and scoring
metrics:                             # Optional: Quality metrics
  thresholds:
    complexity: 10
    coverage: 80
    duplication: 5
  
  scoring:
    algorithm: "weighted"
    weights:
      security: 0.4
      maintainability: 0.3
      performance: 0.3

# Integration configuration
integrations:                        # Optional: Tool integrations
  github:
    status_checks: true
    pr_comments: true
    annotations: true
  
  slack:
    webhook: "${SLACK_WEBHOOK}"
    channels: ["#dev-standards"]
    notify_on: ["error", "warning"]
```

## Minimal Valid Standard

```yaml
standard:
  id: "minimal-example"
  name: "Minimal Example"
  version: "1.0.0"
  category: "coding"
  description: "Minimal valid standard"

applicability:
  languages: ["python"]

rules:
  - id: "rule-1"
    name: "Example Rule"
    description: "Example rule description"
    severity: "warning"
    pattern: "TODO"
    message: "Found TODO"
```

## Category Values

- `coding` - General coding standards
- `security` - Security-focused standards
- `performance` - Performance optimization
- `testing` - Testing standards
- `documentation` - Documentation standards
- `architecture` - Architecture patterns
- `process` - Development process standards

## Severity Levels

- `error` - Must fix, blocks deployment
- `warning` - Should fix, quality issue
- `info` - Consider fixing, suggestion

## Version Constraints

Uses semantic versioning with constraints:
- `1.0.0` - Exact version
- `>=1.0.0` - Minimum version
- `~1.2.0` - Compatible versions (1.2.x)
- `^1.2.3` - Compatible minor (1.x.x where x >= 2.3)
- `>1.0.0,<2.0.0` - Version range

## File Pattern Syntax

Uses gitignore-style patterns:
- `*.py` - All Python files
- `src/**/*.js` - All JS files under src/
- `!test_*.py` - Exclude test files
- `{src,lib}/**/*` - Multiple directories

## Related Documentation

- [Creating Standards Guide](../../CREATING_STANDARDS_GUIDE.md)
- [Validation Rules](../api/validation-rules.md)
- [Standards Examples](../examples/custom-standards.md)