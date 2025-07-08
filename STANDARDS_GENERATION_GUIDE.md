# Standards Generation System Guide

The MCP Standards Server includes a comprehensive standards generation system that enables rapid creation of new specialty standards while maintaining consistency with existing standards and NIST compliance requirements.

## Overview

The standards generation system provides:

- **Template Engine**: Jinja2-based template system with inheritance and customization
- **Base Templates**: Common standard structures for technical, compliance, process, and architecture standards
- **Domain-Specific Templates**: Specialized templates for AI/ML, Blockchain, IoT, Gaming, and API design
- **CLI Integration**: Command-line tools for interactive standard creation
- **Quality Assurance**: Automated validation and quality scoring
- **NIST Integration**: Built-in NIST control mapping and compliance checking

## Architecture

```
src/generators/
├── __init__.py              # Main module exports
├── base.py                  # StandardsGenerator class
├── engine.py                # Jinja2 template engine
├── metadata.py              # Metadata schema and validation
├── validator.py             # Standards validation framework
└── quality_assurance.py     # Quality assessment system

templates/
├── standards/               # Base standard templates
│   ├── base.j2             # Base template with common structure
│   ├── technical.j2        # Technical implementation template
│   ├── compliance.j2       # Compliance-focused template
│   ├── process.j2          # Process methodology template
│   └── architecture.j2     # Architecture pattern template
└── domains/                # Domain-specific templates
    ├── ai_ml.j2            # AI/ML with model lifecycle
    ├── blockchain.j2       # Blockchain/Web3 with security
    ├── iot.j2              # IoT/Edge with connectivity
    ├── gaming.j2           # Gaming with performance optimization
    └── api.j2              # API design with REST/GraphQL
```

## Quick Start

### 1. Generate a Standard Interactively

```bash
# Interactive mode with guided wizard
mcp-standards generate --interactive

# Quick generation with minimal parameters
mcp-standards generate \
  --title "Secure API Development" \
  --domain api \
  --author "Security Team" \
  --description "Guidelines for secure API development"
```

### 2. List Available Templates

```bash
# List all templates with descriptions
mcp-standards generate list-templates

# Get detailed information about a specific template
mcp-standards generate template-info domains/ai_ml.j2
```

### 3. Generate from Configuration File

```bash
# Create a configuration file
cat > api_standard_config.yaml << EOF
title: "RESTful API Security Standard"
version: "1.0.0"
domain: "api"
type: "technical"
author: "API Security Team"
description: "Comprehensive security guidelines for RESTful API development"
tags:
  - "api"
  - "security"
  - "rest"
nist_controls:
  - "AC-3"
  - "SC-8"
  - "SI-10"
compliance_frameworks:
  - "NIST"
  - "OWASP"
api_types:
  - "REST"
  - "GraphQL"
authentication_methods:
  - "OAuth 2.0"
  - "JWT"
EOF

# Generate standard from configuration
mcp-standards generate --config api_standard_config.yaml
```

### 4. Validate Existing Standards

```bash
# Validate an existing standard document
mcp-standards generate validate my_standard.md --report validation_report.yaml
```

## Template System

### Template Inheritance

The template system uses Jinja2 inheritance to promote reuse and consistency:

```jinja2
{# Domain-specific template extending base technical template #}
{% extends "standards/technical.j2" %}

{% block purpose %}
This AI/ML standard defines requirements for {{ title.lower() }}.
It provides guidance for machine learning model lifecycle management.
{% endblock %}

{% block implementation %}
{{ super() }}  {# Include base implementation #}

### ML-Specific Requirements
- Model versioning and registry
- Training data governance
- Bias detection and mitigation
{% endblock %}
```

### Custom Filters

Built-in Jinja2 filters for standards formatting:

```jinja2
# NIST control formatting
{{ nist_controls | join(', ') | format_nist_control }}
# Output: NIST-AC-1, NIST-AU-2

# Risk level formatting  
{{ risk_level | format_compliance_level }}
# Output: High Impact

# Version formatting
{{ version | format_version }}
# Output: v1.0.0
```

### Metadata Schema

Each template includes a comprehensive metadata schema:

```yaml
# Base template metadata
name: "Base Standard Template"
description: "Foundation template for all standards"
version: "1.0.0"
author: "MCP Standards Team"
category: "base"
required_variables:
  - title
  - version
  - domain
  - type
optional_variables:
  - nist_controls
  - compliance_frameworks
  - risk_level
customization_points:
  - purpose
  - scope
  - implementation
```

## Domain-Specific Templates

### AI/ML Template

Specialized for machine learning and AI systems:

```bash
mcp-standards generate \
  --domain ai_ml \
  --title "ML Model Deployment Standard" \
  --config ml_config.yaml
```

Features:
- Model lifecycle management
- Data quality validation
- Bias detection frameworks
- MLOps pipeline integration
- Ethical AI considerations

### Blockchain/Web3 Template

Focused on blockchain and decentralized applications:

```bash
mcp-standards generate \
  --domain blockchain \
  --title "Smart Contract Security Standard"
```

Features:
- Smart contract security patterns
- DeFi protocol development
- Cross-chain infrastructure
- Web3 integration patterns
- Cryptographic security

### IoT/Edge Template

Designed for Internet of Things and edge computing:

```bash
mcp-standards generate \
  --domain iot \
  --title "IoT Device Security Standard"
```

Features:
- Device lifecycle management
- Connectivity protocols
- Edge computing architecture
- Security frameworks
- Performance optimization

### Gaming Template

Optimized for game development:

```bash
mcp-standards generate \
  --domain gaming \
  --title "Multiplayer Game Architecture Standard"
```

Features:
- Game engine architecture
- Performance optimization
- Multiplayer networking
- Player experience design
- Analytics integration

### API Design Template

Comprehensive API development guidance:

```bash
mcp-standards generate \
  --domain api \
  --title "GraphQL API Design Standard"
```

Features:
- REST and GraphQL patterns
- Authentication and authorization
- API versioning strategies
- Documentation standards
- Performance monitoring

## Quality Assurance System

The built-in QA system provides automated quality assessment:

### Quality Metrics

- **Completeness** (25%): Required sections, content depth, examples
- **Consistency** (15%): Terminology, formatting, structure
- **Clarity** (20%): Sentence length, active voice, readability
- **Compliance Coverage** (15%): NIST controls, framework alignment
- **Implementation Guidance** (15%): Code examples, step-by-step instructions
- **Maintainability** (10%): Version info, change history, review process

### Example Quality Report

```bash
mcp-standards generate validate my_standard.md

# Output:
Validation Results for: my_standard.md
==================================================
✓ Validation passed

Quality Score: 87/100

Score Breakdown:
  completeness: 92.0
  consistency: 85.0
  clarity: 89.0
  compliance_coverage: 78.0
  implementation_guidance: 95.0
  maintainability: 83.0

Recommendations:
  - Add risk assessment content
  - Include more NIST control references
  - Improve heading hierarchy consistency
```

## Advanced Usage

### Custom Template Creation

Create custom templates based on existing ones:

```bash
# Create custom template interactively
mcp-standards generate customize \
  --template standards/technical.j2 \
  --name my_custom_template \
  --interactive

# Create from configuration
mcp-standards generate customize \
  --template domains/api.j2 \
  --name microservices_api \
  --config customization.yaml
```

### Batch Generation

Generate multiple standards from a configuration directory:

```bash
# Generate standards for all configurations in a directory
for config in configs/*.yaml; do
  mcp-standards generate --config "$config" --output "standards/$(basename "$config" .yaml).md"
done
```

### Integration with CI/CD

Include standards generation in your CI/CD pipeline:

```yaml
# .github/workflows/standards.yml
name: Generate Standards
on:
  push:
    paths: ['standards/configs/**']

jobs:
  generate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Generate standards
        run: |
          for config in standards/configs/*.yaml; do
            mcp-standards generate --config "$config" --validate
          done
      - name: Commit generated standards
        run: |
          git add standards/generated/
          git commit -m "Auto-generate standards" || exit 0
          git push
```

## Best Practices

### Template Development

1. **Inherit from Base Templates**: Always extend existing templates when possible
2. **Use Semantic Blocks**: Define clear, reusable template blocks
3. **Include Comprehensive Metadata**: Provide detailed template descriptions and schemas
4. **Add Validation Rules**: Include custom validation for domain-specific requirements

### Standard Generation

1. **Start with Configuration Files**: Use YAML configs for reproducible generation
2. **Validate Early and Often**: Run validation during development
3. **Include Comprehensive Metadata**: Fill out all relevant metadata fields
4. **Map to NIST Controls**: Include appropriate NIST control mappings
5. **Add Implementation Guidance**: Include practical examples and code samples

### Quality Assurance

1. **Set Quality Thresholds**: Establish minimum quality scores for standards
2. **Review Generated Content**: Always review generated standards before approval
3. **Iterate Based on Feedback**: Use quality recommendations to improve standards
4. **Maintain Template Quality**: Regularly update and improve templates

## Troubleshooting

### Common Issues

**Template Not Found**
```bash
Error: Template 'domains/custom.j2' not found
```
Solution: Check template path and ensure file exists in templates directory.

**Validation Errors**
```bash
Error: Required field 'title' is missing
```
Solution: Ensure all required metadata fields are provided.

**Schema Validation Failed**
```bash
Error: Version must follow semantic versioning (e.g., 1.0.0)
```
Solution: Update version to use semantic versioning format.

### Debug Mode

Enable verbose output for troubleshooting:

```bash
mcp-standards generate --verbose --preview \
  --title "Debug Standard" \
  --domain api
```

### Template Validation

Validate templates before using them:

```bash
mcp-standards generate template-info domains/ai_ml.j2
```

## Contributing

### Adding New Templates

1. Create template file in appropriate directory
2. Add corresponding metadata YAML file
3. Include JSON schema for validation
4. Add comprehensive examples and documentation
5. Test template with various configurations

### Improving Quality Checks

1. Add new quality metrics to `quality_assurance.py`
2. Include domain-specific validation rules
3. Update test cases for new quality checks
4. Document new quality criteria

## Support

For issues, questions, or contributions:

1. Check existing documentation and examples
2. Run built-in validation and quality checks
3. Review template metadata and schemas
4. Submit issues with detailed reproduction steps

The standards generation system is designed to be extensible and maintainable, enabling teams to create high-quality, consistent standards efficiently while ensuring compliance with security and regulatory requirements.