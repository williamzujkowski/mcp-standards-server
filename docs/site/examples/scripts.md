# Integration Scripts Examples

Practical scripts for integrating MCP Standards Server into your development workflow.

## Setup Scripts

### Project Initialization Script

```bash
#!/bin/bash
# scripts/init-standards.sh

echo "Initializing MCP Standards for project..."

# Detect project type
if [ -f "package.json" ]; then
    PROJECT_TYPE="javascript"
elif [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
    PROJECT_TYPE="python"
elif [ -f "go.mod" ]; then
    PROJECT_TYPE="go"
else
    PROJECT_TYPE="generic"
fi

# Create configuration
cat > .mcp-standards.yml << EOF
version: 1.0
project:
  type: $PROJECT_TYPE
  name: $(basename $PWD)

standards:
  - ${PROJECT_TYPE}-best-practices
  - security-baseline
  - testing-standards

validation:
  exclude:
    - "node_modules/"
    - "venv/"
    - ".git/"
    - "dist/"
EOF

echo "Created .mcp-standards.yml with $PROJECT_TYPE defaults"
```

### Batch Validation Script

```python
#!/usr/bin/env python3
# scripts/validate-all.py

import os
import sys
import json
from pathlib import Path
from src.core.standards import StandardsEngine

def validate_projects(root_dir):
    """Validate all projects in a directory"""
    engine = StandardsEngine()
    results = {}
    
    for project_dir in Path(root_dir).iterdir():
        if project_dir.is_dir() and (project_dir / '.mcp-standards.yml').exists():
            print(f"Validating {project_dir.name}...")
            
            result = engine.validate_directory(str(project_dir))
            results[project_dir.name] = {
                'passed': result.passed,
                'violations': len(result.violations),
                'summary': result.summary
            }
    
    # Generate report
    with open('validation-report.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    total_projects = len(results)
    passed_projects = sum(1 for r in results.values() if r['passed'])
    
    print(f"\nValidation Summary:")
    print(f"Total Projects: {total_projects}")
    print(f"Passed: {passed_projects}")
    print(f"Failed: {total_projects - passed_projects}")

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    validate_projects(root)
```

## CI/CD Integration Scripts

### GitHub Actions Integration

```yaml
# .github/actions/mcp-validate/action.yml
name: 'MCP Standards Validation'
description: 'Validate code against MCP standards'

inputs:
  config-file:
    description: 'Path to MCP standards config'
    default: '.mcp-standards.yml'
  
  fail-on-violation:
    description: 'Fail build on violations'
    default: 'true'

runs:
  using: 'composite'
  steps:
    - name: Install MCP Standards Server
      shell: bash
      run: |
        pip install mcp-standards-server
    
    - name: Run Validation
      shell: bash
      run: |
        mcp-standards validate \
          --config ${{ inputs.config-file }} \
          --format github \
          $([ "${{ inputs.fail-on-violation }}" = "true" ] && echo "--fail-on-violation")
```

### GitLab CI Integration

```yaml
# .gitlab-ci.yml
standards-validation:
  stage: test
  image: python:3.11
  
  before_script:
    - pip install mcp-standards-server
  
  script:
    - mcp-standards validate . --format gitlab > validation-report.json
  
  artifacts:
    reports:
      codequality: validation-report.json
    paths:
      - validation-report.json
    expire_in: 1 week
```

## Migration Scripts

### Standards Migration Script

```python
#!/usr/bin/env python3
# scripts/migrate-standards.py

import yaml
import json
from pathlib import Path

def migrate_old_config(old_config_path, new_config_path):
    """Migrate from old standard format to new format"""
    
    # Read old config
    with open(old_config_path) as f:
        if old_config_path.suffix == '.json':
            old_config = json.load(f)
        else:
            old_config = yaml.safe_load(f)
    
    # Transform to new format
    new_config = {
        'version': '2.0',
        'standard': {
            'id': old_config.get('id', 'migrated-standard'),
            'name': old_config.get('name', 'Migrated Standard'),
            'version': '1.0.0',
            'category': old_config.get('type', 'general')
        },
        'rules': []
    }
    
    # Migrate rules
    for old_rule in old_config.get('rules', []):
        new_rule = {
            'id': old_rule['name'],
            'description': old_rule.get('description', ''),
            'severity': old_rule.get('level', 'warning'),
            'enabled': old_rule.get('active', True)
        }
        new_config['rules'].append(new_rule)
    
    # Save new config
    with open(new_config_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"Migrated {old_config_path} to {new_config_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: migrate-standards.py <old-config> <new-config>")
        sys.exit(1)
    
    migrate_old_config(Path(sys.argv[1]), Path(sys.argv[2]))
```

## Reporting Scripts

### HTML Report Generator

```python
#!/usr/bin/env python3
# scripts/generate-report.py

from jinja2 import Template
from datetime import datetime
import json

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MCP Standards Validation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .summary { background: #f0f0f0; padding: 10px; margin: 20px 0; }
        .passed { color: green; }
        .failed { color: red; }
        .violation { margin: 10px 0; padding: 10px; background: #ffe0e0; }
    </style>
</head>
<body>
    <h1>Validation Report</h1>
    <div class="summary">
        <h2>Summary</h2>
        <p>Generated: {{ timestamp }}</p>
        <p>Total Files: {{ summary.total_files }}</p>
        <p>Files with Violations: {{ summary.files_with_violations }}</p>
        <p>Total Violations: {{ summary.total_violations }}</p>
    </div>
    
    <h2>Violations</h2>
    {% for violation in violations %}
    <div class="violation">
        <strong>{{ violation.file }}:{{ violation.line }}</strong><br>
        Rule: {{ violation.rule_id }}<br>
        {{ violation.message }}
    </div>
    {% endfor %}
</body>
</html>
"""

def generate_html_report(validation_results, output_path):
    template = Template(HTML_TEMPLATE)
    
    html = template.render(
        timestamp=datetime.now().isoformat(),
        summary=validation_results['summary'],
        violations=validation_results['violations']
    )
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Report generated: {output_path}")

if __name__ == '__main__':
    with open('validation-results.json') as f:
        results = json.load(f)
    
    generate_html_report(results, 'validation-report.html')
```

## Automation Scripts

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Get list of staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM)

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Run validation only on staged files
echo "Running MCP Standards validation..."

for file in $STAGED_FILES; do
    if [[ "$file" =~ \.(py|js|go|java|rs|ts)$ ]]; then
        mcp-standards validate "$file" --quiet
        if [ $? -ne 0 ]; then
            echo "❌ Validation failed for $file"
            echo "Run 'mcp-standards validate $file' for details"
            exit 1
        fi
    fi
done

echo "✅ All files passed validation"
```

### Auto-fix Script

```bash
#!/bin/bash
# scripts/auto-fix.sh

echo "Running MCP Standards auto-fix..."

# Find all source files
find . -name "*.py" -o -name "*.js" -o -name "*.go" | while read file; do
    # Skip vendor/node_modules
    if [[ "$file" =~ (vendor|node_modules|\.git) ]]; then
        continue
    fi
    
    # Run auto-fix
    mcp-standards fix "$file" --backup
    
    if [ $? -eq 0 ]; then
        echo "✓ Fixed: $file"
    else
        echo "✗ Could not auto-fix: $file"
    fi
done

echo "Auto-fix complete. Review changes before committing."
```

## Related Documentation

- [CLI Commands](../cli/commands/README.md)
- [CI/CD Integration](../guides/cicd-integration.md)
- [Validation Examples](./validation.md)