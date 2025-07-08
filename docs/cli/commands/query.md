# query Command

Query standards based on project context and requirements.

## Synopsis

```bash
mcp-standards query [options]
```

## Description

The `query` command allows you to search and retrieve applicable standards based on your project's context, technology stack, and specific requirements. It uses intelligent matching and can provide results in various formats.

## Options

### `--project-type <type>`
Specify the type of project.

```bash
mcp-standards query --project-type web-application
```

### `--framework <name>`
Specify frameworks being used.

```bash
mcp-standards query --framework react --framework express
```

### `--language <name>`
Specify programming languages.

```bash
mcp-standards query --language javascript --language typescript
```

### `--requirements <req>`
Specify special requirements.

```bash
mcp-standards query --requirements accessibility --requirements security
```

### `--tags <tag>`
Filter by specific tags.

```bash
mcp-standards query --tags frontend --tags testing
```

### `--format <format>`
Output format (text, json, yaml, markdown).

```bash
mcp-standards query --format json
```

### `--detailed`
Include detailed standard content.

```bash
mcp-standards query --detailed
```

### `--token-budget <number>`
Limit response to token budget.

```bash
mcp-standards query --token-budget 4000
```

### `--semantic`
Use semantic search for natural language queries.

```bash
mcp-standards query --semantic "How do I implement authentication in React?"
```

## Examples

### Basic Query

```bash
mcp-standards query --project-type web-application --framework react
```

Output:
```
Applicable Standards Found: 7

1. React 18 Patterns (react-18-patterns.yaml)
   Tags: frontend, react, javascript, components
   Priority: HIGH
   Summary: Modern React patterns including hooks, Server Components, and performance optimization

2. JavaScript ES2025 Standards (javascript-es2025.yaml)
   Tags: javascript, ecmascript, language
   Priority: HIGH
   Summary: Modern JavaScript language features and best practices

3. Web Accessibility Standards (wcag-2.2-accessibility.yaml)
   Tags: accessibility, a11y, web
   Priority: MEDIUM
   Summary: WCAG 2.2 compliance guidelines and ARIA patterns

4. Frontend Testing Standards (frontend-testing.yaml)
   Tags: testing, frontend, jest, react-testing-library
   Priority: MEDIUM
   Summary: Testing patterns for React components and applications

[... more results ...]

Use --detailed to see full content or --format json for programmatic access
```

### Detailed Query with Content

```bash
mcp-standards query --project-type api --language python --detailed
```

Output:
```
Applicable Standards Found: 5

==============================
1. Python API Standards
==============================
File: python-api-standards.yaml
Tags: python, api, backend, rest
Priority: HIGH

## Overview
Standards for building RESTful APIs with Python frameworks like FastAPI and Flask.

## Key Standards

### API Structure
- Use consistent URL patterns: /api/v1/resources
- Implement proper HTTP methods (GET, POST, PUT, DELETE)
- Return appropriate status codes

### Error Handling
```python
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "type": "validation_error"}
    )
```

### Authentication
- Use JWT tokens for stateless authentication
- Implement OAuth2 for third-party integrations
- Always use HTTPS in production

[... full standard content ...]

==============================
2. Python Testing Standards
==============================
[... next standard ...]
```

### JSON Output for Integration

```bash
mcp-standards query --project-type web --framework vue --format json
```

Output:
```json
{
  "query": {
    "project_type": "web",
    "frameworks": ["vue"],
    "timestamp": "2025-07-08T10:30:00Z"
  },
  "results": [
    {
      "id": "vue-3-composition-api",
      "file": "vue-3-composition-api.yaml",
      "title": "Vue 3 Composition API Standards",
      "tags": ["frontend", "vue", "javascript", "composition-api"],
      "priority": "HIGH",
      "relevance_score": 0.95,
      "summary": "Best practices for Vue 3 including Composition API and performance patterns",
      "content_preview": "Standards for modern Vue 3 development...",
      "size_bytes": 18432,
      "last_updated": "2025-07-01T00:00:00Z"
    },
    {
      "id": "javascript-es2025",
      "file": "javascript-es2025.yaml",
      "title": "JavaScript ES2025 Standards",
      "tags": ["javascript", "ecmascript", "language"],
      "priority": "HIGH",
      "relevance_score": 0.88,
      "summary": "Modern JavaScript language features and best practices"
    }
  ],
  "metadata": {
    "total_results": 8,
    "query_time_ms": 45,
    "cache_hit": true,
    "semantic_search_used": false
  }
}
```

### Semantic Search Query

```bash
mcp-standards query --semantic "How do I implement secure authentication in a Node.js API?"
```

Output:
```
Semantic Search Results for: "How do I implement secure authentication in a Node.js API?"

1. Node.js Security Standards (relevance: 94%)
   - JWT implementation patterns
   - bcrypt for password hashing
   - Session management best practices
   - OAuth2 integration examples

2. API Authentication Standards (relevance: 89%)
   - Token-based authentication
   - API key management
   - Rate limiting and throttling
   - CORS configuration

3. Express.js Security Middleware (relevance: 85%)
   - Helmet.js configuration
   - Session security
   - CSRF protection
   - Input validation

Showing top 3 results. Use --detailed for full content.
```

### Token-Limited Query

```bash
mcp-standards query --project-type web --token-budget 2000
```

Output:
```
Applicable Standards (Token-Optimized Summary - 1,847 tokens):

1. **React 18 Patterns** (250 tokens)
   - Use functional components with hooks
   - Implement React.memo for performance
   - Use Suspense for data fetching
   - Server Components for SSR

2. **Web Performance Standards** (300 tokens)
   - Achieve Core Web Vitals targets
   - Implement lazy loading
   - Optimize bundle sizes
   - Use CDN for static assets

3. **Accessibility Checklist** (200 tokens)
   - ARIA labels for interactive elements
   - Keyboard navigation support
   - Color contrast ratios
   - Screen reader compatibility

[... more condensed standards ...]

Note: Content condensed to fit token budget. Use --detailed for full versions.
```

### Query with Multiple Filters

```bash
mcp-standards query \
  --project-type web-application \
  --framework react \
  --framework tailwind \
  --language typescript \
  --requirements accessibility \
  --requirements security \
  --tags testing
```

### Export Query Results

```bash
# Export to Markdown for documentation
mcp-standards query --project-type api --format markdown > api-standards.md

# Export to YAML for processing
mcp-standards query --framework django --format yaml > django-standards.yaml
```

## Advanced Queries

### Complex Context Object

```bash
# Using a context file
cat > context.json << EOF
{
  "project_type": "microservice",
  "languages": ["python", "go"],
  "frameworks": ["fastapi", "gin"],
  "infrastructure": ["kubernetes", "docker"],
  "requirements": {
    "compliance": ["pci-dss", "gdpr"],
    "performance": "high-throughput",
    "security": "critical"
  },
  "team_size": "medium",
  "timeline": "6-months"
}
EOF

mcp-standards query --context context.json
```

### Query with Rule Engine

```bash
# Show how standards were selected
mcp-standards query --project-type web --show-rules
```

Output:
```
Applied Selection Rules:

Rule: web-application-base
  Condition: project_type == "web-application"
  Applied Standards: [html5-standards, css3-standards, javascript-es2025]
  
Rule: react-ecosystem
  Condition: framework.includes("react")
  Applied Standards: [react-18-patterns, jsx-best-practices]
  
Rule: accessibility-required
  Condition: requirements.includes("accessibility")
  Applied Standards: [wcag-2.2-accessibility]

Final Standards: 6 (after de-duplication and priority sorting)
```

### Batch Queries

```bash
# Query multiple contexts at once
cat > queries.json << EOF
[
  {
    "name": "frontend",
    "project_type": "web",
    "framework": "react"
  },
  {
    "name": "backend", 
    "project_type": "api",
    "language": "python"
  }
]
EOF

mcp-standards query --batch queries.json
```

## Integration Examples

### IDE Integration

```bash
# VS Code task
{
  "label": "Get Project Standards",
  "type": "shell",
  "command": "mcp-standards query --context ${workspaceFolder}/.mcp-context.json --format json",
  "problemMatcher": []
}
```

### Git Hook

```bash
#!/bin/bash
# .git/hooks/pre-commit
# Ensure code follows applicable standards

STANDARDS=$(mcp-standards query --project-type web --format json)
# Process standards and run validators
```

### CI/CD Pipeline

```yaml
# GitHub Actions example
- name: Query Applicable Standards
  run: |
    mcp-standards query \
      --project-type ${{ matrix.project-type }} \
      --framework ${{ matrix.framework }} \
      --format json > standards.json
    
- name: Validate Against Standards
  run: |
    mcp-standards validate --standards standards.json
```

## Query Performance

- **Caching**: Query results are cached for repeated queries
- **Indexing**: Standards are indexed for fast retrieval
- **Semantic Search**: Optional embeddings for natural language queries
- **Token Optimization**: Results can be condensed to fit token budgets

## Related Commands

- [validate](./validate.md) - Validate code against queried standards
- [sync](./sync.md) - Ensure standards are up to date
- [serve](./serve.md) - Start MCP server for programmatic queries