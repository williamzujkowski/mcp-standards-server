# Common Workflows

This guide covers typical workflows and use cases for the MCP Standards Server.

## Table of Contents

1. [New Project Setup](#new-project-setup)
2. [Daily Development Workflow](#daily-development-workflow)
3. [Code Review Workflow](#code-review-workflow)
4. [CI/CD Integration](#cicd-integration)
5. [Team Collaboration](#team-collaboration)
6. [Migration Workflow](#migration-workflow)
7. [Compliance Checking](#compliance-checking)

## New Project Setup

### 1. Project Analysis and Standards Selection

When starting a new project, first analyze your requirements:

```bash
# Create project context file
cat > .mcp-context.json << 'EOF'
{
  "project_type": "web-application",
  "name": "E-commerce Platform",
  "languages": ["typescript", "python"],
  "frameworks": ["react", "nextjs", "fastapi"],
  "infrastructure": ["docker", "kubernetes", "aws"],
  "requirements": {
    "accessibility": "wcag-2.2",
    "security": "owasp-top-10",
    "performance": "core-web-vitals",
    "compliance": ["pci-dss", "gdpr"]
  },
  "team_size": "medium",
  "timeline": "6-months"
}
EOF

# Query applicable standards
mcp-standards query --context .mcp-context.json --detailed > project-standards.md
```

### 2. Generate Project Scaffold

Based on the standards, generate initial project structure:

```bash
# Get scaffold templates
mcp-standards generate --type scaffold --context .mcp-context.json

# This creates:
# - Project directory structure
# - Configuration files
# - CI/CD templates
# - Documentation templates
```

### 3. Configure Development Environment

Set up project-specific standards configuration:

```bash
# Generate project config
mcp-standards config --generate-project > .mcp-standards.yaml

# Edit to customize
vim .mcp-standards.yaml
```

Example `.mcp-standards.yaml`:
```yaml
# Project: E-commerce Platform
project:
  name: ecommerce-platform
  type: web-application
  
standards:
  enforce:
    - react-18-patterns
    - typescript-strict
    - fastapi-best-practices
    - k8s-security
    - wcag-2.2-accessibility
    
validation:
  pre_commit: true
  ci_pipeline: true
  severity: error
  
  rules:
    overrides:
      # Project-specific overrides
      max-line-length:
        options:
          limit: 120
      
      import-order:
        options:
          groups: ["builtin", "external", "internal", "parent", "sibling"]
```

### 4. Set Up Git Hooks

Install pre-commit hooks:

```bash
# Install pre-commit
pip install pre-commit

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: local
    hooks:
      - id: mcp-standards-validate
        name: MCP Standards Validation
        entry: mcp-standards validate
        language: system
        files: \.(js|jsx|ts|tsx|py)$
        pass_filenames: true
        
      - id: mcp-standards-sync-check
        name: Standards Sync Check
        entry: mcp-standards sync --check
        language: system
        pass_filenames: false
        always_run: true
EOF

# Install hooks
pre-commit install
```

## Daily Development Workflow

### Morning Routine

```bash
#!/bin/bash
# morning-setup.sh

echo "🌅 Good morning! Setting up development environment..."

# 1. Update standards
echo "📚 Checking for standards updates..."
if mcp-standards sync --check | grep -q "outdated"; then
    echo "📥 Updating standards..."
    mcp-standards sync
fi

# 2. Check project status
echo "📊 Project standards status:"
mcp-standards status --summary

# 3. Validate workspace
echo "✅ Validating workspace..."
mcp-standards validate . --severity error --quiet || echo "⚠️  Issues found!"

# 4. Start MCP server
echo "🚀 Starting MCP server..."
mcp-standards serve --daemon

echo "✨ Ready to code!"
```

### Before Committing

```bash
#!/bin/bash
# pre-commit-check.sh

# 1. Validate changed files
CHANGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -n "$CHANGED_FILES" ]; then
    echo "Validating changed files..."
    mcp-standards validate $CHANGED_FILES --fix
    
    # Re-add fixed files
    git add $CHANGED_FILES
fi

# 2. Run comprehensive check
mcp-standards validate . --fail-on warning

# 3. Generate compliance report
mcp-standards report --type compliance --output compliance-report.json
```

### Feature Development

When working on a new feature:

```bash
# 1. Query relevant standards for the feature
mcp-standards query --semantic "implementing user authentication with OAuth2" > auth-standards.md

# 2. Generate code templates
mcp-standards generate --template oauth2-provider --framework fastapi

# 3. Validate as you code
# In another terminal, watch for changes
watch -n 5 'mcp-standards validate src/auth/ --format short'
```

## Code Review Workflow

### For Reviewers

Create a code review checklist based on standards:

```bash
#!/bin/bash
# review-pr.sh

PR_NUMBER=$1
REPO="owner/repo"

# 1. Fetch PR changes
gh pr checkout $PR_NUMBER

# 2. Run comprehensive validation
echo "Running standards validation..."
mcp-standards validate . --format json > validation-report.json

# 3. Check for security issues
echo "Security check..."
mcp-standards validate . --standards security-* --severity error

# 4. Generate review comment
if [ -s validation-report.json ]; then
    echo "## MCP Standards Review" > review-comment.md
    echo "" >> review-comment.md
    mcp-standards report --input validation-report.json --format markdown >> review-comment.md
    
    # Post comment
    gh pr comment $PR_NUMBER --body-file review-comment.md
fi
```

### For PR Authors

Before requesting review:

```bash
# Pre-review checklist
mcp-standards checklist --type pr-ready

# Output:
# ✅ All files validated
# ✅ No security vulnerabilities
# ✅ Accessibility standards met
# ✅ Performance budgets satisfied
# ✅ Documentation updated
# ⚠️  Test coverage: 78% (target: 80%)
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/mcp-standards.yml
name: MCP Standards Validation

on:
  pull_request:
    types: [opened, synchronize, reopened]
  push:
    branches: [main, develop]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Cache MCP Standards
        uses: actions/cache@v3
        with:
          path: ~/.cache/mcp-standards
          key: ${{ runner.os }}-mcp-${{ hashFiles('.mcp-standards.yaml') }}
      
      - name: Install MCP Standards
        run: |
          pip install mcp-standards-server
          mcp-standards --version
      
      - name: Sync Standards
        env:
          MCP_STANDARDS_REPOSITORY_AUTH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          mcp-standards sync
      
      - name: Validate Code
        id: validate
        run: |
          mcp-standards validate . \
            --format sarif \
            --output results.sarif \
            --fail-on error
      
      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: results.sarif
      
      - name: Generate Report
        if: always()
        run: |
          mcp-standards report \
            --type summary \
            --format markdown > $GITHUB_STEP_SUMMARY
      
      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('validation-summary.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
```

### GitLab CI/CD

```yaml
# .gitlab-ci.yml
stages:
  - validate
  - report

variables:
  MCP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/mcp-standards"

before_script:
  - pip install mcp-standards-server
  - mcp-standards sync

standards-validation:
  stage: validate
  script:
    - mcp-standards validate . --format junit --output standards-report.xml
  artifacts:
    reports:
      junit: standards-report.xml
    paths:
      - validation-report.json
    expire_in: 1 week
  cache:
    key: "$CI_COMMIT_REF_SLUG-mcp"
    paths:
      - .cache/mcp-standards

compliance-report:
  stage: report
  dependencies:
    - standards-validation
  script:
    - mcp-standards report --type compliance --output compliance.html
  artifacts:
    paths:
      - compliance.html
    expire_in: 30 days
  only:
    - main
    - develop
```

## Team Collaboration

### Shared Standards Configuration

Set up team-wide standards:

```bash
# 1. Create shared configuration repository
git init team-standards
cd team-standards

# 2. Create team standards
cat > team-standards.yaml << 'EOF'
# Company-wide development standards
team:
  name: "ACME Corp Engineering"
  
standards:
  base:
    - coding-best-practices
    - security-baseline
    - accessibility-wcag-2.2
  
  by_language:
    javascript:
      - javascript-es2025
      - typescript-strict
    python:
      - python-3.11
      - type-hints-required
  
  by_framework:
    react:
      - react-18-patterns
      - react-performance
    django:
      - django-security
      - django-rest-framework

validation:
  rules:
    # Company-wide rules
    no-console-log:
      severity: error
    secure-headers:
      severity: error
    accessibility-alt-text:
      severity: error
EOF

# 3. Share with team
git add .
git commit -m "Initial team standards"
git remote add origin https://github.com/acme/team-standards
git push
```

### Team Member Setup

```bash
# Clone team standards
git clone https://github.com/acme/team-standards ~/.config/mcp-standards/team

# Link to team config
mcp-standards config --set team.config_path ~/.config/mcp-standards/team/team-standards.yaml

# Verify
mcp-standards config --show | grep team
```

### Standards Updates Workflow

```bash
#!/bin/bash
# update-team-standards.sh

# 1. Pull latest team standards
cd ~/.config/mcp-standards/team
git pull

# 2. Sync from upstream
mcp-standards sync

# 3. Notify team of changes
CHANGES=$(git log --oneline HEAD~1..HEAD)
if [ -n "$CHANGES" ]; then
    echo "Standards updated:"
    echo "$CHANGES"
    # Send notification (Slack, email, etc.)
fi
```

## Migration Workflow

### Migrating Existing Project

```bash
#!/bin/bash
# migrate-to-standards.sh

PROJECT_PATH=$1

echo "🔄 Migrating project to MCP Standards..."

# 1. Analyze existing project
cd $PROJECT_PATH
echo "📊 Analyzing project..."
mcp-standards analyze . > analysis-report.json

# 2. Generate migration plan
echo "📋 Creating migration plan..."
mcp-standards migrate --plan \
  --input analysis-report.json \
  --output migration-plan.md

# 3. Review plan
echo "👀 Review migration plan:"
cat migration-plan.md

read -p "Proceed with migration? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# 4. Apply automated fixes
echo "🔧 Applying automated fixes..."
mcp-standards migrate --apply \
  --plan migration-plan.md \
  --auto-fix

# 5. Generate report
echo "📄 Generating migration report..."
mcp-standards report --type migration \
  --before analysis-report.json \
  --output migration-report.html

echo "✅ Migration complete! Review migration-report.html"
```

### Gradual Migration Strategy

For large codebases:

```yaml
# .mcp-migration.yaml
migration:
  strategy: gradual
  phases:
    - name: "Phase 1: Critical Security"
      duration: "2 weeks"
      standards:
        - security-vulnerabilities
        - authentication-patterns
      paths:
        - src/auth/**
        - src/api/**
      
    - name: "Phase 2: Core Business Logic"
      duration: "1 month"
      standards:
        - coding-best-practices
        - error-handling
        - logging-standards
      paths:
        - src/core/**
        - src/services/**
    
    - name: "Phase 3: Frontend"
      duration: "1 month"
      standards:
        - react-18-patterns
        - accessibility-wcag-2.2
        - performance-optimization
      paths:
        - src/components/**
        - src/pages/**
```

## Compliance Checking

### Regular Compliance Audits

```bash
#!/bin/bash
# compliance-audit.sh

DATE=$(date +%Y-%m-%d)
REPORT_DIR="compliance-reports/$DATE"
mkdir -p $REPORT_DIR

echo "🔍 Running compliance audit..."

# 1. Security compliance
mcp-standards audit --type security \
  --standards "owasp-*,security-*" \
  --output "$REPORT_DIR/security-audit.json"

# 2. Accessibility compliance  
mcp-standards audit --type accessibility \
  --standards "wcag-2.2-*" \
  --output "$REPORT_DIR/accessibility-audit.json"

# 3. Performance compliance
mcp-standards audit --type performance \
  --standards "performance-*,optimization-*" \
  --output "$REPORT_DIR/performance-audit.json"

# 4. Generate executive summary
mcp-standards report --type compliance-summary \
  --input "$REPORT_DIR/*.json" \
  --format pdf \
  --output "$REPORT_DIR/executive-summary.pdf"

echo "✅ Audit complete. Reports in $REPORT_DIR/"
```

### Continuous Compliance Monitoring

```yaml
# docker-compose.yml for compliance dashboard
version: '3.8'

services:
  mcp-monitor:
    image: mcp-standards-monitor
    environment:
      - MCP_MODE=monitor
      - CHECK_INTERVAL=3600
      - ALERT_WEBHOOK=${SLACK_WEBHOOK}
    volumes:
      - ./src:/app/src:ro
      - ./compliance-data:/data
    
  compliance-dashboard:
    image: mcp-compliance-dashboard
    ports:
      - "8080:80"
    depends_on:
      - mcp-monitor
    volumes:
      - ./compliance-data:/data:ro
```

## Best Practices

1. **Automate Everything**: Use scripts and CI/CD for consistency
2. **Start Small**: Begin with critical standards, expand gradually
3. **Team Buy-in**: Involve team in standards selection
4. **Regular Updates**: Keep standards current with industry best practices
5. **Measure Impact**: Track metrics before/after standards adoption
6. **Customize Thoughtfully**: Override only when necessary
7. **Document Decisions**: Record why certain standards were chosen/modified