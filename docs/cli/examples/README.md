# CLI Examples

This directory contains practical examples of using the MCP Standards Server CLI in various scenarios.

## Quick Examples

### Basic Commands

```bash
# Initialize configuration
mcp-standards config --init

# Sync standards
mcp-standards sync

# Check status
mcp-standards status

# Query standards for a React project
mcp-standards query --project-type web-application --framework react

# Validate current directory
mcp-standards validate .

# Start MCP server
mcp-standards serve
```

### Common Use Cases

```bash
# Validate and auto-fix issues
mcp-standards validate src/ --fix

# Check for standards updates without downloading
mcp-standards sync --check

# Export standards for documentation
mcp-standards query --project-type api --format markdown > api-standards.md

# Clear outdated cache entries
mcp-standards cache --clear-outdated

# Validate with custom config
mcp-standards -c custom-config.yaml validate
```

## Example Scripts

### Daily Development Script

```bash
#!/bin/bash
# dev-start.sh - Start development with standards check

echo "🚀 Starting development environment..."

# Update standards if needed
if mcp-standards sync --check | grep -q "outdated"; then
    echo "📥 Updating standards..."
    mcp-standards sync
fi

# Validate workspace
echo "✅ Checking code standards..."
mcp-standards validate . --severity warning --quiet

# Start MCP server in background
echo "🖥️  Starting MCP server..."
mcp-standards serve --daemon

echo "✨ Ready to code! MCP server running on http://localhost:3000"
```

### Pre-commit Validation

```bash
#!/bin/bash
# pre-commit.sh - Validate before committing

# Get staged files
STAGED=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(js|jsx|ts|tsx|py)$')

if [ -z "$STAGED" ]; then
    echo "No files to validate"
    exit 0
fi

echo "Validating staged files..."
echo "$STAGED" | xargs mcp-standards validate --fix

# Re-stage fixed files
echo "$STAGED" | xargs git add

# Final validation
if ! echo "$STAGED" | xargs mcp-standards validate --fail-on error; then
    echo "❌ Validation failed. Please fix errors before committing."
    exit 1
fi

echo "✅ All checks passed!"
```

### Project Setup Script

```bash
#!/bin/bash
# setup-project.sh - Set up new project with standards

PROJECT_NAME=$1
PROJECT_TYPE=${2:-web-application}

if [ -z "$PROJECT_NAME" ]; then
    echo "Usage: $0 <project-name> [project-type]"
    exit 1
fi

echo "🏗️  Setting up project: $PROJECT_NAME"

# Create project directory
mkdir -p "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Initialize git
git init

# Create MCP context
cat > .mcp-context.json << EOF
{
  "project_type": "$PROJECT_TYPE",
  "name": "$PROJECT_NAME",
  "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
}
EOF

# Query and save applicable standards
echo "📋 Fetching applicable standards..."
mcp-standards query --context .mcp-context.json --format markdown > PROJECT_STANDARDS.md

# Create project config
cat > .mcp-standards.yaml << EOF
# MCP Standards Configuration for $PROJECT_NAME
project:
  name: $PROJECT_NAME
  type: $PROJECT_TYPE

validation:
  on_save: true
  severity: error
  
sync:
  auto_sync: true
EOF

# Set up pre-commit hooks
cat > .pre-commit-config.yaml << EOF
repos:
  - repo: local
    hooks:
      - id: mcp-standards
        name: MCP Standards Check
        entry: mcp-standards validate
        language: system
        files: '\.(js|jsx|ts|tsx|py)$'
        pass_filenames: true
EOF

echo "✅ Project setup complete!"
echo "📄 See PROJECT_STANDARDS.md for applicable standards"
```

### CI/CD Integration Script

```bash
#!/bin/bash
# ci-validate.sh - Validation script for CI/CD

set -e  # Exit on error

# Configuration
SEVERITY=${MCP_SEVERITY:-error}
FORMAT=${MCP_FORMAT:-json}
OUTPUT_FILE="validation-results.$FORMAT"

echo "🔍 MCP Standards Validation"
echo "=========================="

# Ensure standards are synced
echo "📚 Syncing standards..."
mcp-standards sync

# Run validation
echo "✅ Validating code..."
if mcp-standards validate . \
    --severity "$SEVERITY" \
    --format "$FORMAT" \
    --output "$OUTPUT_FILE"; then
    echo "✨ Validation passed!"
    EXIT_CODE=0
else
    echo "❌ Validation failed!"
    EXIT_CODE=1
fi

# Generate report
echo "📊 Generating report..."
mcp-standards report \
    --input "$OUTPUT_FILE" \
    --format html \
    --output standards-report.html

# Show summary
echo ""
echo "Summary:"
echo "--------"
mcp-standards report \
    --input "$OUTPUT_FILE" \
    --format summary

exit $EXIT_CODE
```

### Batch Processing Script

```bash
#!/bin/bash
# batch-validate.sh - Validate multiple projects

PROJECTS_DIR=${1:-./projects}
REPORT_DIR=${2:-./reports}

mkdir -p "$REPORT_DIR"

echo "🔄 Batch validation of projects in $PROJECTS_DIR"
echo "Reports will be saved to $REPORT_DIR"
echo ""

# Find all projects with .mcp-standards.yaml
find "$PROJECTS_DIR" -name ".mcp-standards.yaml" -type f | while read -r config_file; do
    PROJECT_DIR=$(dirname "$config_file")
    PROJECT_NAME=$(basename "$PROJECT_DIR")
    
    echo "📁 Processing: $PROJECT_NAME"
    
    # Run validation
    if mcp-standards -c "$config_file" validate "$PROJECT_DIR" \
        --format json \
        --output "$REPORT_DIR/$PROJECT_NAME.json" 2>/dev/null; then
        echo "  ✅ Passed"
    else
        echo "  ❌ Failed"
    fi
done

# Generate summary report
echo ""
echo "📊 Generating summary report..."
mcp-standards report \
    --merge "$REPORT_DIR"/*.json \
    --format html \
    --output "$REPORT_DIR/summary.html"

echo "✨ Batch validation complete!"
echo "📄 View summary: $REPORT_DIR/summary.html"
```

## Integration Examples

### Git Hooks

```bash
# .git/hooks/pre-commit
#!/bin/bash
# Validate before commit

# Only validate if MCP standards is installed
if ! command -v mcp-standards &> /dev/null; then
    echo "Warning: mcp-standards not installed, skipping validation"
    exit 0
fi

# Get changed files
CHANGED=$(git diff --cached --name-only --diff-filter=ACM)

if [ -n "$CHANGED" ]; then
    echo "$CHANGED" | xargs mcp-standards validate --fail-on error
fi
```

### VS Code Tasks

```json
// .vscode/tasks.json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Validate Current File",
      "type": "shell",
      "command": "mcp-standards validate ${file}",
      "problemMatcher": "$tsc",
      "presentation": {
        "reveal": "silent",
        "focus": false,
        "panel": "shared"
      }
    },
    {
      "label": "Fix Current File",
      "type": "shell",
      "command": "mcp-standards validate --fix ${file}",
      "presentation": {
        "reveal": "always",
        "focus": false
      }
    },
    {
      "label": "Query Standards",
      "type": "shell",
      "command": "mcp-standards query --context .mcp-context.json --format markdown",
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

### Shell Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc

# Quick validation
alias mcpv='mcp-standards validate'
alias mcpvf='mcp-standards validate --fix'

# Query shortcuts
alias mcpq='mcp-standards query'
alias mcpqa='mcp-standards query --project-type api'
alias mcpqw='mcp-standards query --project-type web-application'

# Status and sync
alias mcps='mcp-standards status'
alias mcpsync='mcp-standards sync'

# Function for validating specific file types
mcpvalidate() {
    local file_type="${1:-js}"
    find . -name "*.$file_type" -type f | xargs mcp-standards validate
}

# Function for quick project analysis
mcpanalyze() {
    echo "🔍 Analyzing project..."
    mcp-standards query --context . --format json | jq -r '.results[] | "- \(.title): \(.summary)"'
}
```

### Docker Integration

```dockerfile
# Dockerfile with MCP validation
FROM node:18 AS validator

# Install MCP Standards
RUN pip install mcp-standards-server

# Copy source code
COPY . /app
WORKDIR /app

# Validate during build
RUN mcp-standards validate . --fail-on error

# Continue with regular build...
FROM node:18-slim
COPY --from=validator /app /app
WORKDIR /app
RUN npm ci --only=production
CMD ["npm", "start"]
```

### Makefile Integration

```makefile
# Makefile with MCP standards

.PHONY: validate fix sync standards help

validate: ## Validate code against standards
	@echo "🔍 Validating code..."
	@mcp-standards validate src/ --fail-on error

fix: ## Auto-fix standards violations
	@echo "🔧 Fixing violations..."
	@mcp-standards validate src/ --fix

sync: ## Sync latest standards
	@echo "📥 Syncing standards..."
	@mcp-standards sync

standards: ## Show applicable standards
	@echo "📋 Applicable standards:"
	@mcp-standards query --context .mcp-context.json --format summary

check: validate ## Run all checks

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
```

## Advanced Examples

### Custom Validation Pipeline

```python
#!/usr/bin/env python3
# validate_pipeline.py - Custom validation pipeline

import subprocess
import json
import sys
from pathlib import Path

def run_validation(path, standards=None):
    """Run MCP validation with specific standards."""
    cmd = ["mcp-standards", "validate", str(path), "--format", "json"]
    
    if standards:
        cmd.extend(["--standards", ",".join(standards)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return json.loads(result.stdout) if result.stdout else {}

def main():
    # Define validation stages
    stages = [
        {
            "name": "Security",
            "standards": ["security-*", "auth-*"],
            "severity": "error"
        },
        {
            "name": "Performance",
            "standards": ["performance-*", "optimization-*"],
            "severity": "warning"
        },
        {
            "name": "Accessibility",
            "standards": ["wcag-*", "a11y-*"],
            "severity": "error"
        }
    ]
    
    all_passed = True
    
    for stage in stages:
        print(f"\n🔍 Running {stage['name']} validation...")
        result = run_validation(".", stage['standards'])
        
        issues = result.get('issues', [])
        errors = [i for i in issues if i['severity'] == 'error']
        
        if errors and stage['severity'] == 'error':
            print(f"❌ {stage['name']}: {len(errors)} errors found")
            all_passed = False
        else:
            print(f"✅ {stage['name']}: Passed")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()
```

### Monitoring Script

```bash
#!/bin/bash
# monitor-standards.sh - Monitor standards compliance over time

METRICS_FILE="standards-metrics.csv"
TIMESTAMP=$(date +%s)

# Run validation and extract metrics
RESULT=$(mcp-standards validate . --format json)
TOTAL_FILES=$(echo "$RESULT" | jq '.summary.files_scanned')
ERRORS=$(echo "$RESULT" | jq '.summary.errors')
WARNINGS=$(echo "$RESULT" | jq '.summary.warnings')

# Append to metrics file
echo "$TIMESTAMP,$TOTAL_FILES,$ERRORS,$WARNINGS" >> "$METRICS_FILE"

# Generate trend report if enough data
if [ $(wc -l < "$METRICS_FILE") -gt 10 ]; then
    echo "📊 Compliance Trend (last 10 checks):"
    tail -10 "$METRICS_FILE" | awk -F, '
        BEGIN { print "Errors | Warnings" }
        { printf "%6d | %8d\n", $3, $4 }
    '
fi

# Alert if errors increase
if [ -f ".last_error_count" ]; then
    LAST_ERRORS=$(cat .last_error_count)
    if [ "$ERRORS" -gt "$LAST_ERRORS" ]; then
        echo "⚠️  Error count increased from $LAST_ERRORS to $ERRORS"
    fi
fi
echo "$ERRORS" > .last_error_count
```

## Tips and Tricks

1. **Use JSON output** for scripting and automation
2. **Cache standards** to avoid repeated downloads
3. **Create project-specific configs** for consistent validation
4. **Integrate with existing tools** using format converters
5. **Monitor trends** to track improvement over time
6. **Automate fixes** where possible to save time
7. **Document exceptions** when overriding standards