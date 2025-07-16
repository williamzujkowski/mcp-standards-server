# Creating Standards Guide

This guide explains how to create new standards for the MCP Standards Server using the built-in template system and generation tools.

## Overview

The MCP Standards Server provides a comprehensive template-based system for creating new standards. This ensures consistency, quality, and completeness across all standards in the ecosystem.

## Quick Start

### 1. Identify the Standard Type

First, determine what kind of standard you're creating:

- **Technical Standards**: Programming languages, frameworks, tools
- **Process Standards**: Workflows, reviews, operations
- **Domain Standards**: Specialized areas like AI/ML, blockchain, gaming
- **Compliance Standards**: Security, privacy, regulatory requirements

### 2. Choose a Template

Select the appropriate template from `templates/`:

- **base.j2**: General-purpose standard template
- **technical.j2**: Technical implementation standards
- **process.j2**: Process and workflow standards
- **compliance.j2**: Compliance and regulatory standards
- **operational.j2**: Operations and monitoring standards
- **review_process.j2**: Review and approval workflows

See [TEMPLATE_INDEX.md](../templates/TEMPLATE_INDEX.md) for detailed template descriptions.

### 3. Create Your Standard

```bash
# Using the CLI (recommended)
mcp-standards generate --template technical --name "GraphQL API Standards"

# Or manually create a YAML configuration
cat > my_standard.yaml << EOF
title: "GraphQL API Standards"
version: "1.0.0"
category: "API Design"
domain: "Web Development"
description: "Best practices for GraphQL API design and implementation"

sections:
  overview:
    purpose: "Standardize GraphQL API development across teams"
    scope: "All GraphQL APIs in production systems"
  
  requirements:
    - "Use schema-first design approach"
    - "Implement proper error handling"
    - "Include comprehensive documentation"
  
  best_practices:
    - title: "Schema Design"
      content: "Design schemas with clear types and relationships"
    - title: "Query Optimization"
      content: "Implement DataLoader pattern for N+1 prevention"
EOF
```

## Standard Structure

### Required Sections

Every standard must include:

1. **Metadata Header**
   ```yaml
   title: "Standard Title"
   version: "1.0.0"
   category: "Category Name"
   domain: "Domain Name"
   description: "Brief description"
   last_updated: "2025-01-08"
   authors:
     - "Your Name"
   ```

2. **Overview Section**
   - Purpose and objectives
   - Scope and applicability
   - Key principles

3. **Requirements**
   - Mandatory requirements
   - Compliance criteria
   - Success metrics

4. **Best Practices**
   - Recommended approaches
   - Examples and patterns
   - Anti-patterns to avoid

### Optional Sections

Depending on the standard type, include:

- **Implementation Examples**: Code samples and configurations
- **Testing Guidelines**: How to verify compliance
- **Tool Recommendations**: Specific tools and configurations
- **Migration Guide**: For updating from previous versions
- **FAQ**: Common questions and answers
- **Glossary**: Domain-specific terminology

## Quality Assurance

### Validation Checklist

Before submitting your standard, ensure it:

- [ ] Has complete metadata with version 1.0.0
- [ ] Includes clear overview and scope
- [ ] Contains actionable requirements
- [ ] Provides concrete examples
- [ ] References related standards
- [ ] Maps to NIST controls (if applicable)
- [ ] Includes testing/validation methods
- [ ] Has proper formatting and structure

### Quality Metrics

The system automatically scores standards on:

1. **Completeness** (20%): All required sections present
2. **Clarity** (20%): Clear, unambiguous language
3. **Actionability** (20%): Specific, implementable guidelines
4. **Examples** (15%): Working code/configuration examples
5. **References** (15%): Links to resources and related standards
6. **Testability** (10%): Validation methods included

Aim for a quality score of 80% or higher.

## Adding to the System

### 1. Local Testing

```bash
# Validate your standard
mcp-standards validate my_standard.yaml

# Test with the rule engine
mcp-standards query --context "project_type=api,framework=graphql"
```

### 2. Integration

Place your standard in the appropriate directory:

```
data/standards/
├── GRAPHQL_API_STANDARDS.md     # Generated from template
└── GRAPHQL_API_STANDARDS.yaml   # Source configuration
```

### 3. Update Rule Engine

Add detection rules to `data/standards/meta/enhanced-selection-rules.json`:

```json
{
  "id": "graphql-api",
  "name": "GraphQL API Standards",
  "description": "Standards for GraphQL API development",
  "priority": 15,
  "conditions": {
    "logic": "OR",
    "conditions": [
      {
        "field": "framework",
        "operator": "contains",
        "value": "graphql"
      },
      {
        "field": "project_type",
        "operator": "equals",
        "value": "graphql_api"
      }
    ]
  },
  "standards": ["GRAPHQL_API_STANDARDS"],
  "tags": ["api", "graphql", "web"]
}
```

## Publishing Standards

### Community Review Process

1. **Draft Phase**
   - Create initial standard
   - Self-review against checklist
   - Run validation tools

2. **Review Phase**
   - Submit PR to repository
   - Address reviewer feedback
   - Update based on community input

3. **Publication**
   - Merge approved standard
   - Update catalog and index
   - Announce to community

### Version Management

Follow semantic versioning:

- **1.0.0**: Initial release
- **1.1.0**: New features or sections
- **1.0.1**: Fixes and clarifications
- **2.0.0**: Breaking changes

## Examples

### Technical Standard Example

See `data/standards/ADVANCED_API_DESIGN_STANDARDS.md` for a complete example of a technical standard covering REST, GraphQL, and gRPC.

### Process Standard Example

See `data/standards/CODE_REVIEW_STANDARDS.md` for a process-oriented standard with workflows and checklists.

### Compliance Standard Example

See `data/standards/DATA_PRIVACY_COMPLIANCE_STANDARDS.md` for a compliance standard with regulatory mappings.

## Best Practices

### Do's

- ✅ Be specific and actionable
- ✅ Include real-world examples
- ✅ Provide tool configurations
- ✅ Reference authoritative sources
- ✅ Consider different skill levels
- ✅ Include migration guidance
- ✅ Test your examples

### Don'ts

- ❌ Be vague or ambiguous
- ❌ Assume prior knowledge
- ❌ Ignore edge cases
- ❌ Skip validation
- ❌ Forget version 1.0.0
- ❌ Omit testing methods

## Getting Help

- Review existing standards for patterns
- Check the [TEMPLATE_INDEX.md](../templates/TEMPLATE_INDEX.md)
- Use the validation tools
- Ask in community forums
- Submit issues for clarification

## Contributing Templates

If you need a new template type:

1. Identify the gap in existing templates
2. Create the template following Jinja2 syntax
3. Add comprehensive documentation
4. Include at least two examples
5. Submit PR with rationale

## Resources

- [Template Index](../templates/TEMPLATE_INDEX.md)
- [Standards Catalog](../STANDARDS_COMPLETE_CATALOG.md)
- [Rule Engine Documentation](../src/core/standards/README_RULE_ENGINE.md)
- [Quality Assurance Framework](../src/generators/quality_assurance.py)