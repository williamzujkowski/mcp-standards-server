# MCP Tools Reference

This document describes all MCP tools provided by the MCP Standards Server.

## Tool: load_standards

Load standards based on natural language or notation queries.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Natural language or notation query"
    },
    "context": {
      "type": "string",
      "description": "Optional context for better matching"
    },
    "version": {
      "type": "string",
      "default": "latest"
    },
    "token_limit": {
      "type": "integer",
      "description": "Maximum tokens to return"
    }
  },
  "required": ["query"]
}
```

### Examples
```json
// Natural language query
{
  "query": "secure api design"
}

// Direct notation
{
  "query": "CS:api + SEC:authentication"
}

// With context
{
  "query": "authentication",
  "context": "Building a REST API with OAuth2"
}
```

### Response
Returns loaded standards with metadata including token count and version.

## Tool: analyze_code

Analyze code for NIST control implementations.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "code": {
      "type": "string",
      "description": "Code to analyze"
    },
    "language": {
      "type": "string",
      "enum": ["python", "javascript", "typescript", "go", "java"],
      "description": "Programming language"
    },
    "filename": {
      "type": "string",
      "description": "Optional filename for context"
    }
  },
  "required": ["code", "language"]
}
```

### Examples
```json
{
  "code": "def authenticate_user(username, password):\n    # Implementation",
  "language": "python",
  "filename": "auth.py"
}
```

### Response
Returns detected NIST controls, evidence, suggestions, and compliance score.

## Tool: suggest_controls

Get NIST control recommendations based on requirements.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "description": {
      "type": "string",
      "description": "Description of what you're building"
    },
    "components": {
      "type": "array",
      "items": {"type": "string"},
      "description": "System components"
    },
    "security_level": {
      "type": "string",
      "enum": ["low", "moderate", "high"],
      "default": "moderate"
    }
  },
  "required": ["description"]
}
```

### Examples
```json
{
  "description": "Building user authentication system with MFA",
  "components": ["web-app", "api", "database"],
  "security_level": "high"
}
```

### Response
Returns recommended NIST controls with implementation guidance.

## Tool: generate_template

Generate NIST-compliant code templates.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "template_type": {
      "type": "string",
      "enum": ["api-endpoint", "auth-module", "logging-setup", "encryption-utils"],
      "description": "Type of template to generate"
    },
    "language": {
      "type": "string",
      "enum": ["python", "javascript", "typescript", "go", "java"]
    },
    "controls": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Specific NIST controls to implement"
    }
  },
  "required": ["template_type", "language"]
}
```

### Examples
```json
{
  "template_type": "api-endpoint",
  "language": "python",
  "controls": ["AC-3", "AU-2", "SI-10"]
}
```

### Response
Returns generated code template with NIST annotations.

## Tool: validate_compliance

Validate code/project against NIST compliance requirements.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "File or directory path to validate"
    },
    "profile": {
      "type": "string",
      "enum": ["low", "moderate", "high"],
      "default": "moderate"
    },
    "controls": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Specific controls to check"
    }
  },
  "required": ["path"]
}
```

### Examples
```json
{
  "path": "/project/src",
  "profile": "moderate",
  "controls": ["AC-3", "AU-2"]
}
```

### Response
Returns compliance report with gaps, recommendations, and coverage metrics.

## Tool: scan_with_llm

Enhanced scanning with LLM analysis for deeper insights.

### Input Schema
```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to scan"
    },
    "focus_areas": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Areas to focus on"
    },
    "output_format": {
      "type": "string",
      "enum": ["summary", "detailed", "oscal"],
      "default": "summary"
    }
  },
  "required": ["path"]
}
```

### Response
Returns enhanced analysis with LLM insights and recommendations.

## Error Handling

All tools follow consistent error handling:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid input: query is required",
    "details": {
      "field": "query",
      "reason": "missing"
    }
  }
}
```

Error codes:
- `VALIDATION_ERROR`: Invalid input parameters
- `NOT_FOUND`: Resource not found
- `ANALYSIS_ERROR`: Error during analysis
- `LIMIT_EXCEEDED`: Token or rate limit exceeded
- `INTERNAL_ERROR`: Server error

## Rate Limiting

Tools are subject to rate limiting:
- Default: 100 requests per minute
- Analysis tools: 20 requests per minute
- Generation tools: 10 requests per minute

## Best Practices

1. **Use specific queries**: More specific queries yield better results
2. **Provide context**: Context helps with natural language queries
3. **Check token limits**: Be aware of token limits for large responses
4. **Handle errors**: Always handle potential errors gracefully
5. **Cache results**: Cache standards loading for better performance