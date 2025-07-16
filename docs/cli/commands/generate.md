# generate

Generate standards from templates.

## Synopsis

```bash
mcp-standards generate [options]
mcp-standards generate list-templates
mcp-standards generate template-info <template_name>
mcp-standards generate customize [options]
mcp-standards generate validate <standard_file>
```

## Description

The `generate` command provides a comprehensive system for creating new standards using pre-defined templates. It supports multiple workflows for standard creation, from simple generation to advanced customization.

## Options

### Main Generate Options

#### `--template, -t <name>`
Specify the template to use for generation.

```bash
mcp-standards generate --template technical --title "GraphQL API Standards"
```

#### `--domain, -d <name>`
Use a domain-specific template.

```bash
mcp-standards generate --domain ai_ml --title "Machine Learning Pipeline Standards"
```

#### `--output, -o <path>`
Specify the output file path.

```bash
mcp-standards generate --template process --output ./standards/code-review.md
```

#### `--title <title>`
Set the standard title.

```bash
mcp-standards generate --template technical --title "REST API Design Standards"
```

#### `--version 1.0.0
Set the standard version 1.0.0

```bash
mcp-standards generate --template technical --version 1.0.0
```

#### `--author <name>`
Specify the standard author.

```bash
mcp-standards generate --template technical --author "Jane Doe"
```

#### `--description <desc>`
Provide a brief description.

```bash
mcp-standards generate --template technical --description "Standards for GraphQL API development"
```

#### `--interactive, -i`
Run in interactive mode for guided generation.

```bash
mcp-standards generate --interactive
```

#### `--preview, -p`
Preview the generated standard without saving.

```bash
mcp-standards generate --template technical --preview
```

#### `--validate`
Validate the generated standard (default: true).

```bash
mcp-standards generate --template technical --no-validate
```

#### `--config-file <path>`
Use a configuration file for generation.

```bash
mcp-standards generate --config-file ./my-standard-config.yaml
```

## Subcommands

### list-templates

List all available templates.

```bash
mcp-standards generate list-templates
```

Output example:
```
Available Templates:

Base Templates:
  - base: General-purpose standard template
  - technical: Technical implementation standards
  - process: Process and workflow standards
  - compliance: Compliance and regulatory standards
  - operational: Operations and monitoring standards

Domain Templates:
  - ai_ml: AI/ML operations standards
  - blockchain: Blockchain/Web3 standards
  - gaming: Gaming development standards
  - iot: IoT/Edge computing standards
```

### template-info

Get detailed information about a specific template.

```bash
mcp-standards generate template-info technical
```

Output example:
```
Template: technical
Description: Template for technical implementation standards
Version: 1.2.0

Required Fields:
  - title: Standard title
  - version: Standard version
  - category: Technical category
  - domain: Technical domain

Optional Fields:
  - frameworks: Applicable frameworks
  - languages: Programming languages
  - tools: Required tools
```

### customize

Create a custom template based on an existing one.

```bash
mcp-standards generate customize --template base --name my-custom-template
```

Options:
- `--template, -t`: Base template to customize (required)
- `--name, -n`: Name for the custom template (required)
- `--config`: Configuration file for customization
- `--interactive, -i`: Interactive customization mode

### validate

Validate an existing standard file.

```bash
mcp-standards generate validate ./standards/my-standard.md
```

Options:
- `--report, -r`: Output validation report to file

## Examples

### Basic Generation

Generate a technical standard:
```bash
mcp-standards generate \
  --template technical \
  --title "Python Async Programming Standards" \
  --author "John Smith" \
  --output ./standards/python-async.md
```

### Interactive Mode

Use interactive mode for guided generation:
```bash
mcp-standards generate --interactive

? Select template type: technical
? Enter standard title: GraphQL API Standards
? Enter version 1.0.0
? Enter author name: Jane Doe
? Enter description: Best practices for GraphQL API design
? Select category: API Design
? Add frameworks? Yes
? Enter frameworks (comma-separated): Apollo, GraphQL-JS
? Preview before saving? Yes
```

### Using Configuration File

Create a configuration file (`graphql-standard.yaml`):
```yaml
template: technical
title: "GraphQL API Standards"
version: "1.0.0"
author: "API Team"
category: "API Design"
domain: "Web Development"
description: "Comprehensive GraphQL API design and implementation standards"
metadata:
  frameworks:
    - Apollo Server
    - GraphQL-JS
  languages:
    - JavaScript
    - TypeScript
  compliance:
    - "NIST-800-53:AC-2"
    - "NIST-800-53:AC-3"
```

Generate using the config:
```bash
mcp-standards generate --config-file graphql-standard.yaml
```

### Domain-Specific Generation

Generate an AI/ML standard:
```bash
mcp-standards generate \
  --domain ai_ml \
  --title "ML Model Deployment Standards" \
  --description "Standards for deploying ML models in production"
```

### Preview Mode

Preview without saving:
```bash
mcp-standards generate \
  --template operational \
  --title "Incident Response Standards" \
  --preview
```

### Batch Generation

Generate multiple standards using a script:
```bash
#!/bin/bash
templates=("technical" "process" "operational")
titles=("API Standards" "Code Review Standards" "Monitoring Standards")

for i in ${!templates[@]}; do
  mcp-standards generate \
    --template ${templates[$i]} \
    --title "${titles[$i]}" \
    --output "./standards/${templates[$i]}-standard.md"
done
```

## Template Selection Guide

Choose the appropriate template based on your needs:

- **base**: General standards, custom domains
- **technical**: Code, APIs, frameworks, languages
- **process**: Workflows, reviews, methodologies
- **compliance**: Security, privacy, regulations
- **operational**: Monitoring, incidents, SRE

Domain-specific templates:
- **ai_ml**: Machine learning pipelines, model management
- **blockchain**: Smart contracts, DeFi, Web3
- **gaming**: Game engines, multiplayer, performance
- **iot**: Edge computing, device management

## Configuration File Format

Configuration files support all generation options:

```yaml
# Template selection
template: technical  # or domain template
domain: web_development

# Basic information
title: "Standard Title"
version: "1.0.0"
author: "Author Name"
description: "Brief description"

# Categories and tags
category: "API Design"
tags:
  - rest
  - graphql
  - api

# Metadata
metadata:
  frameworks:
    - Express.js
    - FastAPI
  languages:
    - JavaScript
    - Python
  tools:
    - Postman
    - Swagger
  
# Compliance mapping
compliance:
  - "NIST-800-53:AC-2"
  - "ISO-27001:A.9"

# Custom sections
sections:
  requirements:
    - "Use OpenAPI 3.0 specification"
    - "Implement rate limiting"
  best_practices:
    - title: "API Versioning"
      content: "Use semantic versioning in URLs"
    - title: "Error Handling"
      content: "Return consistent error responses"
```

## Quality Validation

Generated standards are automatically validated for:

1. **Completeness**: All required sections present
2. **Formatting**: Proper markdown structure
3. **Metadata**: Valid frontmatter
4. **References**: Working links and citations
5. **Compliance**: Valid NIST control mappings

To skip validation:
```bash
mcp-standards generate --template technical --no-validate
```

To validate existing standards:
```bash
mcp-standards generate validate ./my-standard.md --report validation-report.json
```

## Tips

1. **Start with templates**: Use existing templates rather than starting from scratch
2. **Use interactive mode**: Helpful for first-time users or complex standards
3. **Preview first**: Always preview before saving, especially for complex standards
4. **Version control**: Commit generated standards to version 1.0.0
5. **Customize templates**: Create custom templates for repeated use

## See Also

- [Template Index](../../templates/TEMPLATE_INDEX.md)
- [Creating Standards Guide](../../CREATING_STANDARDS_GUIDE.md)
- [sync](./sync.md) - Sync standards from repository