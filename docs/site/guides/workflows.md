# Common Workflows

Common workflows and best practices for using MCP Standards Server in various development scenarios.

## Daily Development Workflow

### Individual Developer Workflow

```
1. Start Development Session
   ↓
2. Sync Latest Standards
   ↓
3. Code Development
   ↓
4. Real-time Validation (IDE)
   ↓
5. Pre-commit Validation
   ↓
6. Commit & Push
   ↓
7. CI/CD Validation
```

**Step-by-step:**

```bash
# 1. Start development session
mcp-standards status

# 2. Sync latest standards (if needed)
mcp-standards sync

# 3. Code development with IDE integration
# (Real-time validation happens automatically)

# 4. Pre-commit validation
mcp-standards validate --fix .

# 5. Commit changes
git add .
git commit -m "Add new feature"
git push

# 6. Monitor CI/CD results
# Check GitHub Actions or CI system
```

### Team Lead Workflow

```bash
# Daily standards review
mcp-standards query list --recently-updated

# Team compliance report
mcp-standards validate ./team-projects --format json --output compliance-report.json

# Update team standards
mcp-standards config set validation.standards "team-baseline,security-strict"

# Distribute configuration
cp ~/.mcp-standards/config.yaml ./team-config.yaml
git add team-config.yaml
```

## Project Setup Workflows

### New Project Setup

```bash
# 1. Initialize project
mkdir my-new-project
cd my-new-project

# 2. Detect project type and get recommendations
mcp-standards query applicable --project-type web_application --framework react

# 3. Create project configuration
cat > .mcp-standards.json << EOF
{
  "projectType": "web_application",
  "framework": "react",
  "language": "typescript",
  "standards": {
    "required": [
      "react-patterns",
      "typescript-strict",
      "accessibility-wcag"
    ]
  }
}
EOF

# 4. Setup IDE integration
# (Install VS Code extension, configure settings)

# 5. Setup CI/CD
cp ~/.mcp-standards/templates/github-actions.yml .github/workflows/standards.yml

# 6. Initial validation
mcp-standards validate .
```

### Existing Project Integration

```bash
# 1. Analyze existing project
cd existing-project
mcp-standards query applicable .

# 2. Run baseline validation
mcp-standards validate --format sarif --output baseline.sarif .

# 3. Gradual adoption strategy
# Start with warnings only
mcp-standards config set validation.severity_level warning

# 4. Fix critical issues first
mcp-standards validate --severity error --fix .

# 5. Incremental improvement
# Weekly: mcp-standards validate --since "1 week ago" .
```

## Standards Management Workflows

### Custom Standards Development

```bash
# 1. Create custom standard
mkdir -p ./custom-standards/python-company-style

# 2. Use template
mcp-standards generate standard \
  --name "python-company-style" \
  --language python \
  --template company-style

# 3. Edit standard
vim ./custom-standards/python-company-style/standard.yaml

# 4. Test standard
mcp-standards validate \
  --standard ./custom-standards/python-company-style \
  ./test-project

# 5. Publish to team repository
git add custom-standards/
git commit -m "Add company Python style standard"
git push origin main

# 6. Team adoption
mcp-standards config set standards.repository_url https://github.com/company/standards
mcp-standards sync
```

### Standards Review Process

```bash
# 1. Review proposed standard changes
mcp-standards diff --standard react-patterns --version 1.0.0

# 2. Test impact on codebase
mcp-standards validate \
  --standard react-patterns:latest \
  --dry-run \
  ./production-code

# 3. Gradual rollout
# Phase 1: New projects only
mcp-standards config set standards.adoption_policy new_projects_only

# Phase 2: Existing projects (warnings)
mcp-standards config set validation.severity_level warning

# Phase 3: Full enforcement
mcp-standards config set validation.severity_level error
```

## Code Review Workflows

### Pre-Review Validation

```bash
# Author workflow before creating PR
# 1. Validate changed files only
git diff --name-only HEAD~1 | xargs mcp-standards validate

# 2. Auto-fix issues
mcp-standards validate --fix $(git diff --name-only HEAD~1)

# 3. Generate review comments
mcp-standards validate \
  --format review-comments \
  --output review.md \
  $(git diff --name-only HEAD~1)

# 4. Include in PR description
cat review.md >> pr-description.md
```

### Reviewer Workflow

```bash
# 1. Get PR validation results
gh pr view 123 --json checks

# 2. Review standards compliance
mcp-standards validate \
  --standard security-review \
  --format detailed \
  $(gh pr diff 123 --name-only)

# 3. Suggest improvements
mcp-standards suggest \
  --context "code review" \
  $(gh pr diff 123 --name-only)
```

## CI/CD Integration Workflows

### GitHub Actions Workflow

**Basic validation:**
```yaml
# .github/workflows/standards.yml
name: Standards Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install MCP Standards
      run: pip install mcp-standards-server
    - name: Validate Code
      run: mcp-standards validate --format sarif --output results.sarif
    - name: Upload Results
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: results.sarif
```

**Advanced workflow with caching:**
```bash
# Setup workflow with performance optimizations
mcp-standards generate workflow \
  --platform github-actions \
  --with-caching \
  --matrix-strategy \
  --security-scanning
```

### Multi-Environment Workflow

```bash
# Development environment
mcp-standards config set environment development
mcp-standards config set validation.auto_fix true
mcp-standards config set validation.severity_level info

# Staging environment
mcp-standards config set environment staging
mcp-standards config set validation.auto_fix false
mcp-standards config set validation.severity_level warning

# Production environment
mcp-standards config set environment production
mcp-standards config set validation.severity_level error
mcp-standards config set security.authentication.enabled true
```

## Compliance & Audit Workflows

### Security Audit Workflow

```bash
# 1. Comprehensive security scan
mcp-standards validate \
  --standard security-comprehensive \
  --format sarif \
  --output security-audit-$(date +%Y%m%d).sarif \
  .

# 2. Generate compliance report
mcp-standards report compliance \
  --framework NIST-800-53 \
  --output nist-compliance-report.pdf

# 3. Track remediation
mcp-standards track \
  --baseline security-audit-20240101.sarif \
  --current security-audit-$(date +%Y%m%d).sarif
```

### Accessibility Audit Workflow

```bash
# 1. WCAG compliance check
mcp-standards validate \
  --standard accessibility-wcag-2.1 \
  --format accessibility \
  .

# 2. Generate accessibility report
mcp-standards report accessibility \
  --level AA \
  --output accessibility-report.html

# 3. Integration with testing
npm run test:e2e:accessibility
mcp-standards validate --standard accessibility-testing .
```

## Performance Optimization Workflows

### Cache Management

```bash
# Daily cache maintenance
#!/bin/bash
# cache-maintenance.sh

# Clear old cache entries
mcp-standards cache clear --older-than "7 days"

# Warm cache with frequently used standards
mcp-standards cache warm --standards "$(mcp-standards analytics top-standards --limit 10)"

# Optimize cache performance
mcp-standards cache optimize

# Report cache statistics
mcp-standards cache stats --format json > cache-stats-$(date +%Y%m%d).json
```

### Performance Monitoring

```bash
# Monitor validation performance
mcp-standards benchmark \
  --duration 60s \
  --concurrent-users 10 \
  --output performance-$(date +%Y%m%d).json

# Profile memory usage
mcp-standards profile memory \
  --validate ./large-codebase \
  --output memory-profile.json

# Optimize for large codebases
mcp-standards config set performance.batch_size 50
mcp-standards config set performance.max_workers 8
```

## Team Collaboration Workflows

### Standards Committee Workflow

```bash
# Monthly standards review meeting preparation
# 1. Generate usage analytics
mcp-standards analytics standards-usage \
  --period "last 30 days" \
  --output monthly-usage.json

# 2. Identify improvement opportunities
mcp-standards analytics violations \
  --top 10 \
  --suggest-standards

# 3. Draft new standards proposals
mcp-standards generate proposal \
  --based-on violations \
  --template committee-review

# 4. Distribute for review
mcp-standards share proposal \
  --committee "standards-committee@company.com"
```

### Onboarding New Team Members

```bash
# Create onboarding package
mcp-standards package onboarding \
  --role developer \
  --team backend \
  --output onboarding-package.zip

# Generate learning path
mcp-standards generate learning-path \
  --standards "$(mcp-standards query applicable --project-type api --language python)" \
  --output learning-path.md

# Setup development environment
mcp-standards setup \
  --profile team-backend \
  --ide vscode \
  --auto-configure
```

## Troubleshooting Workflows

### Validation Issues

```bash
# Debug validation failures
mcp-standards validate --debug --verbose ./problematic-file.py

# Check standards conflicts
mcp-standards analyze conflicts \
  --standards "pep8,company-style,security-strict"

# Validate standards themselves
mcp-standards validate-standards ./custom-standards/

# Reset to known good state
mcp-standards reset --keep-config
mcp-standards sync --force
```

### Performance Issues

```bash
# Diagnose slow validation
mcp-standards profile validation \
  --target ./slow-project \
  --report performance-issues.json

# Check cache health
mcp-standards cache health-check

# Optimize for current workload
mcp-standards optimize \
  --based-on ./recent-validations.log
```

## Advanced Workflows

### Multi-Repository Management

```bash
# Setup workspace validation
for repo in $(cat repositories.txt); do
  cd $repo
  mcp-standards validate \
    --format json \
    --output ../results/${repo}-results.json
  cd ..
done

# Aggregate results
mcp-standards aggregate \
  --input "./results/*-results.json" \
  --output workspace-compliance.json

# Generate dashboard
mcp-standards dashboard \
  --data workspace-compliance.json \
  --output compliance-dashboard.html
```

### Standards Evolution Tracking

```bash
# Track standards evolution
mcp-standards track evolution \
  --standard react-patterns \
  --from "2024-01-01" \
  --to "2024-12-31" \
  --output react-patterns-evolution.json

# Impact analysis
mcp-standards analyze impact \
  --change "react-patterns:v2.0" \
  --codebase ./production-apps \
  --estimate-effort

# Migration planning
mcp-standards plan migration \
  --from "react-patterns:v1.0" \
  --to "react-patterns:v2.0" \
  --output migration-plan.md
```

### Custom Integration Development

```bash
# Create custom integration
mcp-standards create integration \
  --type webhook \
  --target slack \
  --template notification

# Test integration
mcp-standards test integration \
  --config ./slack-integration.yaml \
  --dry-run

# Deploy integration
mcp-standards deploy integration \
  --config ./slack-integration.yaml \
  --environment production
```

## Best Practices Summary

### Daily Practices
1. Use IDE integration for real-time feedback
2. Run validation before commits
3. Keep standards synchronized
4. Monitor CI/CD validation results

### Team Practices
1. Establish team-wide configuration
2. Regular standards review meetings
3. Gradual adoption of new standards
4. Document custom standards

### Organizational Practices
1. Centralized standards repository
2. Compliance monitoring and reporting
3. Regular security and accessibility audits
4. Performance optimization monitoring

### Troubleshooting Practices
1. Enable debug logging for issues
2. Regular cache maintenance
3. Monitor system health
4. Keep standards and tools updated

---

For more specific integration guides, see:
- [IDE Integration](./ide-integration.md)
- [CI/CD Integration](./cicd-integration.md)
- [Configuration Guide](./configuration.md)
