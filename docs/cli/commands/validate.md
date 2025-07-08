# validate Command

Validate code against MCP standards.

## Synopsis

```bash
mcp-standards validate [options] [path...]
```

## Description

The `validate` command checks your code against applicable standards, providing detailed feedback on compliance issues, suggestions for improvement, and automated fixes where possible.

## Options

### `--standards <file>`
Use specific standards file or query result.

```bash
mcp-standards validate --standards standards.json
```

### `--auto-detect`
Automatically detect applicable standards (default).

```bash
mcp-standards validate --auto-detect
```

### `--fix`
Automatically fix issues where possible.

```bash
mcp-standards validate --fix
```

### `--dry-run`
Show what would be fixed without making changes.

```bash
mcp-standards validate --fix --dry-run
```

### `--format <format>`
Output format (text, json, junit, sarif).

```bash
mcp-standards validate --format junit
```

### `--severity <level>`
Minimum severity level to report (error, warning, info).

```bash
mcp-standards validate --severity warning
```

### `--ignore <pattern>`
Ignore files matching pattern.

```bash
mcp-standards validate --ignore "*.test.js" --ignore "dist/*"
```

### `--config <file>`
Use custom validation configuration.

```bash
mcp-standards validate --config .mcp-validate.yaml
```

### `--parallel <n>`
Number of parallel validation workers.

```bash
mcp-standards validate --parallel 4
```

### `--fail-on <level>`
Exit with error code if issues found at level.

```bash
mcp-standards validate --fail-on error
```

## Examples

### Basic Validation

```bash
mcp-standards validate src/
```

Output:
```
Detecting project context...
Project Type: web-application
Frameworks: react, tailwind
Languages: javascript, typescript

Loading applicable standards...
✓ React 18 Patterns
✓ TypeScript Best Practices
✓ Web Accessibility Standards
✓ JavaScript ES2025 Standards

Validating files...
[████████████████████] 100% | 45/45 files

Results:
========

src/components/Button.tsx
  Line 15: ERROR - Missing accessible label
    Standard: wcag-2.2-accessibility
    Rule: interactive-elements-labels
    
    <button onClick={handleClick}>
      {icon}
    </button>
    
    Fix: Add aria-label or visible text content
    
  Line 23: WARNING - Using deprecated pattern
    Standard: react-18-patterns
    Rule: no-default-props
    
    Button.defaultProps = { size: 'medium' }
    
    Fix: Use default parameters in function signature

src/api/client.js
  Line 8: WARNING - Missing error boundary
    Standard: javascript-error-handling
    Rule: async-error-handling
    
    async function fetchData(url) {
      const response = await fetch(url);
      return response.json();
    }
    
    Fix: Add try-catch block or .catch() handler

src/styles/global.css
  Line 145: INFO - Consider using CSS custom properties
    Standard: modern-css-architecture
    Rule: prefer-custom-properties
    
    .theme-dark { background: #000; color: #fff; }
    
    Suggestion: Use CSS variables for theme values

Summary:
  Files scanned: 45
  Issues found: 12
    Errors: 3
    Warnings: 7
    Info: 2
  
  Standards applied: 4
  Time: 2.34s

Exit code: 1 (errors found)
```

### Auto-Fix Issues

```bash
mcp-standards validate --fix src/
```

Output:
```
Validating and fixing issues...

Fixed: src/components/Button.tsx
  ✓ Added aria-label to button (line 15)
  ✓ Converted defaultProps to default parameters (line 23)

Fixed: src/api/client.js
  ✓ Added error handling to async function (line 8)

Could not auto-fix:
  src/styles/global.css - Manual review required for CSS architecture

Summary:
  Files fixed: 2
  Issues fixed: 3
  Issues remaining: 1

Please review the changes before committing.
```

### Dry Run Mode

```bash
mcp-standards validate --fix --dry-run
```

Output:
```
DRY RUN MODE - No files will be modified

Would fix: src/components/Button.tsx
  - Line 15: Add aria-label="Submit" to button
  - Line 23: Convert to: function Button({ size = 'medium' })

Would fix: src/api/client.js
  - Line 8-12: Wrap in try-catch block:
    
    async function fetchData(url) {
      try {
        const response = await fetch(url);
        return response.json();
      } catch (error) {
        console.error('Failed to fetch data:', error);
        throw error;
      }
    }

Total changes that would be made: 3 fixes in 2 files
```

### JSON Output for CI/CD

```bash
mcp-standards validate --format json src/ > validation-results.json
```

Output (validation-results.json):
```json
{
  "summary": {
    "files_scanned": 45,
    "total_issues": 12,
    "errors": 3,
    "warnings": 7,
    "info": 2,
    "standards_applied": 4,
    "duration_ms": 2340
  },
  "issues": [
    {
      "file": "src/components/Button.tsx",
      "line": 15,
      "column": 5,
      "severity": "error",
      "standard": "wcag-2.2-accessibility",
      "rule": "interactive-elements-labels",
      "message": "Missing accessible label",
      "code": "A11Y001",
      "snippet": "<button onClick={handleClick}>",
      "fix": {
        "available": true,
        "description": "Add aria-label attribute",
        "diff": "+ <button onClick={handleClick} aria-label=\"Submit\">"
      }
    }
  ],
  "standards": [
    {
      "id": "wcag-2.2-accessibility",
      "title": "Web Accessibility Standards",
      "rules_applied": 15,
      "issues_found": 3
    }
  ]
}
```

### JUnit Format for CI

```bash
mcp-standards validate --format junit > test-results.xml
```

### SARIF Format for GitHub

```bash
mcp-standards validate --format sarif > results.sarif
```

### Custom Validation Config

```yaml
# .mcp-validate.yaml
validation:
  # Override detected standards
  standards:
    - react-18-patterns
    - typescript-strict
    - security-best-practices
  
  # Ignore patterns
  ignore:
    - "**/*.test.*"
    - "**/*.spec.*"
    - "build/**"
    - "dist/**"
    - "node_modules/**"
  
  # Rule overrides
  rules:
    # Disable specific rules
    no-console: off
    no-default-props: warning  # Downgrade from error
    
    # Configure rule options
    max-line-length:
      severity: warning
      options:
        limit: 100
        ignore-comments: true
  
  # Auto-fix settings
  fix:
    enabled: true
    safe-only: true  # Only apply safe fixes
    
  # Reporting
  report:
    severity: warning  # Minimum level to report
    fail-on: error    # Exit code 1 if errors found
```

### Validate Specific Standards

```bash
# First, query standards
mcp-standards query --project-type api --format json > api-standards.json

# Then validate against them
mcp-standards validate --standards api-standards.json src/api/
```

### Parallel Validation

```bash
# Use multiple workers for large codebases
mcp-standards validate --parallel 8 .
```

Output:
```
Starting validation with 8 workers...

Worker 1: Scanning src/components/...
Worker 2: Scanning src/api/...
Worker 3: Scanning src/utils/...
Worker 4: Scanning src/styles/...
[... parallel progress ...]

Merged results from all workers.
Total issues: 45
```

## Integration Examples

### Pre-commit Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit

# Validate staged files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -E '\.(js|jsx|ts|tsx)$')

if [ -n "$STAGED_FILES" ]; then
    echo "Validating staged files..."
    mcp-standards validate $STAGED_FILES --fail-on error
    
    if [ $? -ne 0 ]; then
        echo "Validation failed. Fix errors before committing."
        exit 1
    fi
fi
```

### GitHub Actions

```yaml
name: Standards Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup MCP Standards
        run: pip install mcp-standards-server
      
      - name: Sync Standards
        run: mcp-standards sync
      
      - name: Validate Code
        run: |
          mcp-standards validate \
            --format sarif \
            --fail-on error \
            . > results.sarif
      
      - name: Upload SARIF
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: results.sarif
```

### VS Code Task

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Validate Current File",
      "type": "shell",
      "command": "mcp-standards",
      "args": [
        "validate",
        "${file}",
        "--format", "json"
      ],
      "problemMatcher": {
        "pattern": {
          "regexp": "^(.+):(\\d+):(\\d+):\\s+(error|warning)\\s+(.+)$",
          "file": 1,
          "line": 2,
          "column": 3,
          "severity": 4,
          "message": 5
        }
      }
    }
  ]
}
```

### Custom Validator Plugin

```python
# validators/custom_security.py
from mcp_standards.validators import BaseValidator

class CustomSecurityValidator(BaseValidator):
    """Custom security validation rules."""
    
    def validate_file(self, file_path, content):
        issues = []
        
        # Check for hardcoded secrets
        if 'api_key' in content.lower():
            issues.append({
                'severity': 'error',
                'message': 'Possible hardcoded API key',
                'line': self.find_line_number('api_key', content),
                'rule': 'no-hardcoded-secrets'
            })
        
        return issues
```

## Performance Tips

1. **Use Parallel Workers**: For large codebases, increase parallel workers
2. **Ignore Patterns**: Exclude generated files and dependencies
3. **Incremental Validation**: Only validate changed files in CI
4. **Cache Standards**: Ensure standards are synced before validation
5. **Severity Filtering**: Focus on errors first, then warnings

## Exit Codes

- `0`: Validation passed, no issues found
- `1`: Validation failed, issues found at fail-on level
- `2`: Command line error
- `3`: Configuration error
- `4`: Standards loading error
- `5`: File access error

## Related Commands

- [query](./query.md) - Find applicable standards
- [sync](./sync.md) - Update standards
- [serve](./serve.md) - Run as MCP server for IDE integration