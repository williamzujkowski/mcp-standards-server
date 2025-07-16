# Standards Format Specification

This document defines the JSON/YAML format for MCP standards files.

## Standard Structure

```yaml
standard:
  id: "unique-standard-id"
  name: "Human Readable Standard Name"
  version: "1.0.0"
  category: "coding|testing|security|architecture"
  description: "Brief description of the standard"
  
metadata:
  author: "Author Name"
  created: "2024-01-01"
  updated: "2024-01-15"
  tags: ["tag1", "tag2"]
  compliance:
    - "NIST-800-53-AC-2"
    - "ISO-27001-A.9"

applicability:
  project_types: ["web", "api", "mobile"]
  languages: ["python", "javascript", "go"]
  frameworks: ["react", "django", "fastapi"]
  
rules:
  - id: "rule-1"
    description: "Rule description"
    severity: "error|warning|info"
    pattern: "regex or code pattern"
    
examples:
  good:
    - description: "Good example description"
      code: |
        # Example code
  bad:
    - description: "Bad example description"
      code: |
        # Example code
```

## Field Definitions

### Required Fields

- **id**: Unique identifier for the standard
- **name**: Human-readable name
- **version**: Semantic version 1.0.0
- **category**: Primary category
- **description**: Clear description of the standard's purpose

### Optional Fields

- **metadata**: Additional information about the standard
- **applicability**: Conditions for when the standard applies
- **rules**: Specific validation rules
- **examples**: Code examples demonstrating compliance

## Validation Rules

1. All required fields must be present
2. Version must follow semantic versioning
3. Category must be from predefined list
4. Rules must have unique IDs within the standard

## File Naming Convention

Standards files should follow the pattern:
```
{CATEGORY}_{STANDARD_NAME}_STANDARDS.yaml
```

Example: `CODING_PYTHON_ASYNC_STANDARDS.yaml`

## Related Documentation

- [Validation Rules](./validation-rules.md)
- [Creating Standards Guide](../../CREATING_STANDARDS_GUIDE.md)
- [Standards Engine](../architecture/standards-engine.md)