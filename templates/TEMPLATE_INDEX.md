# Standards Templates Index

This document provides an overview of all available templates for creating standards in the MCP Standards Server.

## Template Categories

### 1. Process-Oriented Templates (`templates/standards/`)

These templates focus on operational processes, workflows, and methodologies.

#### operational.j2
- **Purpose**: For monitoring, incident response, and SRE standards
- **Use Cases**: 
  - Production monitoring standards
  - Alerting configurations
  - Incident response procedures
  - SRE practices
- **Key Sections**:
  - Service Level Objectives (SLOs)
  - Monitoring and Alerting configurations
  - Incident Response procedures
  - Runbooks
  - Automation tasks

#### review_process.j2
- **Purpose**: For code review, security review, and approval process standards
- **Use Cases**:
  - Code review processes
  - Security review procedures
  - Architecture review standards
  - Change approval workflows
- **Key Sections**:
  - Workflow diagrams
  - Review checklists
  - Roles and responsibilities
  - Automation integrations
  - Metrics and KPIs

#### content_creation.j2
- **Purpose**: For documentation, blog posts, and tutorial standards
- **Use Cases**:
  - Technical documentation standards
  - Blog writing guidelines
  - Tutorial creation processes
  - Marketing content standards
- **Key Sections**:
  - Writing style guides
  - SEO requirements
  - Content templates
  - Publishing workflows
  - Localization guidelines

#### planning.j2
- **Purpose**: For project planning, estimation, and roadmap standards
- **Use Cases**:
  - Project planning methodologies
  - Estimation guidelines
  - Roadmap creation standards
  - Resource planning
- **Key Sections**:
  - Planning frameworks
  - Estimation techniques
  - Risk management
  - Communication plans
  - Metrics and tracking

### 2. Domain-Specific Templates (`templates/domains/`)

These templates are tailored for specific technical domains and advanced practices.

#### testing_advanced.j2
- **Purpose**: For advanced testing methodologies and practices
- **Use Cases**:
  - Performance testing standards
  - Security testing procedures
  - Chaos engineering practices
  - Test automation frameworks
- **Key Sections**:
  - Testing strategies and pyramid
  - Advanced testing techniques
  - Performance baselines
  - Security testing categories
  - Chaos experiments

#### security_process.j2
- **Purpose**: For security reviews, audits, and compliance processes
- **Use Cases**:
  - Security review procedures
  - Vulnerability management
  - Compliance auditing
  - Incident response for security
- **Key Sections**:
  - Security controls
  - Risk assessment matrices
  - Vulnerability management
  - Incident response procedures
  - Compliance requirements

#### technical_writing.j2
- **Purpose**: For technical documentation and content creation
- **Use Cases**:
  - API documentation standards
  - Technical guide creation
  - Documentation maintenance
  - Developer documentation
- **Key Sections**:
  - Documentation standards
  - Content structure
  - Technical accuracy guidelines
  - Visual elements standards
  - Accessibility requirements

#### operations.j2
- **Purpose**: For production operations and infrastructure management
- **Use Cases**:
  - Production operations standards
  - Infrastructure management
  - Deployment procedures
  - Backup and recovery
- **Key Sections**:
  - Service definitions and SLAs
  - Infrastructure management
  - Deployment operations
  - Performance management
  - Cost optimization

### 3. Enhanced Base Template

#### base_enhanced.j2
- **Purpose**: Extended version of the base template with additional metadata support
- **New Features**:
  - Workflow diagram support
  - Comprehensive checklists
  - Tool recommendations sections
  - Metrics and KPIs tracking
  - Automation configuration
  - Training and certification requirements
  - Extended glossary support

## Template Selection Guide

### When to Use Each Template

1. **Use `operational.j2` when**:
   - Defining monitoring and alerting standards
   - Creating incident response procedures
   - Establishing SRE practices
   - Documenting runbooks

2. **Use `review_process.j2` when**:
   - Standardizing code review processes
   - Creating security review procedures
   - Defining approval workflows
   - Establishing quality gates

3. **Use `content_creation.j2` when**:
   - Setting documentation standards
   - Creating content guidelines
   - Defining publishing processes
   - Establishing SEO requirements

4. **Use `planning.j2` when**:
   - Defining project planning standards
   - Creating estimation guidelines
   - Establishing roadmap processes
   - Setting resource planning standards

5. **Use `testing_advanced.j2` when**:
   - Defining comprehensive testing strategies
   - Creating performance testing standards
   - Establishing security testing procedures
   - Implementing chaos engineering

6. **Use `security_process.j2` when**:
   - Creating security review processes
   - Defining vulnerability management
   - Establishing compliance procedures
   - Setting incident response standards

7. **Use `technical_writing.j2` when**:
   - Creating documentation standards
   - Defining API documentation requirements
   - Setting technical writing guidelines
   - Establishing content quality standards

8. **Use `operations.j2` when**:
   - Defining production operations standards
   - Creating infrastructure management procedures
   - Establishing deployment processes
   - Setting performance benchmarks

## Template Features

### Common Features Across All Templates

- **Metadata Support**: Version, author, dates, tags
- **Compliance Integration**: NIST controls, frameworks
- **Metrics and KPIs**: Measurement and tracking
- **Tool Recommendations**: Required and optional tools
- **Process Workflows**: Visual diagrams and steps
- **Checklists**: Actionable verification items
- **References**: External resources and dependencies

### Advanced Features

1. **Workflow Diagrams**: Mermaid diagram support for visual processes
2. **Automation Integration**: CI/CD configurations and automated checks
3. **Role-Based Sections**: Clear responsibilities and authorities
4. **Metric Dashboards**: Links to monitoring and reporting
5. **Compliance Mapping**: Direct mapping to standards and frameworks
6. **Multi-language Support**: Localization guidelines where applicable

## Using Templates

### Basic Usage

1. Choose the appropriate template based on your standard type
2. Copy the template to your working directory
3. Fill in the required fields in the YAML frontmatter
4. Customize the sections as needed
5. Remove any sections that don't apply

### Example Structure

```yaml
title: "Your Standard Title"
version: "1.0.0"
category: "Category"
domain: "Domain"
description: "Brief description"

# Template-specific fields...
```

### Best Practices

1. **Be Specific**: Provide concrete examples and clear guidelines
2. **Include Metrics**: Define measurable success criteria
3. **Add Automation**: Include CI/CD configurations where applicable
4. **Provide Tools**: Recommend specific tools and configurations
5. **Create Checklists**: Make standards actionable with checklists
6. **Include Examples**: Provide real-world examples and use cases

## Examples

See the `templates/examples/` directory for complete examples:

- `operational_standard.yaml`: Production monitoring standard example
- `code_review_standard.yaml`: Code review process example

## Contributing New Templates

When creating new templates:

1. Identify the gap in existing templates
2. Define the target use case clearly
3. Include all necessary sections
4. Provide comprehensive documentation
5. Create at least one example
6. Update this index

## Template Versioning

Templates follow semantic versioning:
- **Major**: Breaking changes to template structure
- **Minor**: New sections or features added
- **Patch**: Bug fixes or clarifications

Current template versions are tracked in each template's header comments.