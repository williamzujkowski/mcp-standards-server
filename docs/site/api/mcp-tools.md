# MCP Tools Reference

The MCP Standards Server exposes several tools through the Model Context Protocol, enabling AI assistants and development tools to interact with standards programmatically.

> **Note**: This documentation covers the primary tools most commonly used. The actual MCP server implementation includes additional tools for advanced features like token optimization, cross-referencing, and analytics. See the source code at `src/mcp_server.py` for the complete list of 20+ available tools.

## Primary Tools

These are the most commonly used tools for interacting with standards:

### 1. get_applicable_standards

Retrieve standards applicable to a given project context.

#### Parameters

```typescript
{
  context: {
    project_type?: string;        // "web-application" | "api" | "cli" | "library" | "mobile" | "desktop"
    languages?: string[];         // ["javascript", "python", "go", ...]
    frameworks?: string[];        // ["react", "django", "express", ...]
    requirements?: string[];      // ["accessibility", "security", "performance", ...]
    infrastructure?: string[];    // ["docker", "kubernetes", "aws", ...]
    team_size?: string;          // "small" | "medium" | "large"
    compliance?: string[];       // ["pci-dss", "gdpr", "hipaa", ...]
  };
  include_resolution_details?: boolean;  // Include why standards were selected
  format?: "full" | "condensed" | "reference";
  token_budget?: number;        // Maximum tokens for response
}
```

#### Response

```typescript
{
  standards: [
    {
      id: string;
      file: string;
      title: string;
      tags: string[];
      priority: "HIGH" | "MEDIUM" | "LOW";
      relevance_score: number;
      summary: string;
      content?: string;  // If requested
      reason?: string;   // If include_resolution_details is true
    }
  ];
  resolution_details?: {
    rules_applied: string[];
    context_matches: object;
    priority_ordering: string[];
  };
  metadata: {
    total_standards: number;
    query_time_ms: number;
    cache_hit: boolean;
    token_count?: number;
  };
}
```

#### Example Usage

```json
{
  "tool": "get_applicable_standards",
  "arguments": {
    "context": {
      "project_type": "web-application",
      "frameworks": ["react", "nextjs"],
      "languages": ["typescript"],
      "requirements": ["accessibility", "performance"]
    },
    "include_resolution_details": true,
    "format": "condensed",
    "token_budget": 4000
  }
}
```

### 2. validate_code

Validate code snippet or file against standards.

#### Parameters

```typescript
{
  code?: string;              // Code snippet to validate
  file_path?: string;         // OR path to file to validate
  language: string;           // Programming language
  standards?: string[];       // Specific standards to validate against (optional)
  fix?: boolean;              // Return fixed code if possible
  severity?: "error" | "warning" | "info";  // Minimum severity
}
```

#### Response

```typescript
{
  valid: boolean;
  issues: [
    {
      line: number;
      column: number;
      severity: "error" | "warning" | "info";
      standard: string;
      rule: string;
      message: string;
      fix?: {
        description: string;
        old_code: string;
        new_code: string;
      };
    }
  ];
  fixed_code?: string;  // If fix was requested
  summary: {
    total_issues: number;
    errors: number;
    warnings: number;
    info: number;
    fixable: number;
  };
}
```

#### Example Usage

```json
{
  "tool": "validate_code",
  "arguments": {
    "code": "const Button = ({onClick}) => <button onClick={onClick}>Click</button>",
    "language": "javascript",
    "standards": ["react-18-patterns", "wcag-2.2-accessibility"],
    "fix": true
  }
}
```

### 3. search_standards

Search standards using natural language queries.

#### Parameters

```typescript
{
  query: string;              // Natural language search query
  limit?: number;             // Maximum results (default: 10)
  include_content?: boolean;  // Include full content in results
  filters?: {
    tags?: string[];          // Filter by tags
    categories?: string[];    // Filter by categories
    languages?: string[];     // Filter by applicable languages
  };
}
```

#### Response

```typescript
{
  results: [
    {
      id: string;
      file: string;
      title: string;
      relevance: number;        // 0-100 relevance score
      excerpt: string;          // Relevant excerpt
      highlights: string[];     // Highlighted matching sections
      content?: string;         // Full content if requested
      tags: string[];
    }
  ];
  query_interpretation: {
    identified_topics: string[];
    expanded_terms: string[];
    filters_applied: object;
  };
  metadata: {
    total_results: number;
    search_time_ms: number;
    model_used: string;
  };
}
```

#### Example Usage

```json
{
  "tool": "search_standards",
  "arguments": {
    "query": "How do I implement secure authentication in a Node.js API?",
    "limit": 5,
    "include_content": true,
    "filters": {
      "tags": ["security", "authentication"],
      "languages": ["javascript", "nodejs"]
    }
  }
}
```

### 4. get_standard_content

Retrieve specific standard content with formatting options.

#### Parameters

```typescript
{
  standard_id: string;        // Standard identifier
  format?: "full" | "condensed" | "outline" | "examples";
  token_budget?: number;      // Maximum tokens
  sections?: string[];        // Specific sections to include
  include_examples?: boolean; // Include code examples
}
```

#### Response

```typescript
{
  id: string;
  title: string;
  content: string;            // Formatted based on request
  metadata: {
    version: string;
    last_updated: string;
    tags: string[];
    languages: string[];
    frameworks: string[];
  };
  sections?: {
    [key: string]: string;
  };
  examples?: [
    {
      title: string;
      description: string;
      code: string;
      language: string;
    }
  ];
  token_count: number;
}
```

#### Example Usage

```json
{
  "tool": "get_standard_content",
  "arguments": {
    "standard_id": "react-18-patterns",
    "format": "condensed",
    "token_budget": 2000,
    "sections": ["hooks", "performance"],
    "include_examples": true
  }
}
```

### 5. check_compliance

Check project or code for compliance with specific requirements.

#### Parameters

```typescript
{
  target: string;             // File path or directory
  compliance_type: string;    // "security" | "accessibility" | "performance" | "all"
  standards?: string[];       // Specific standards to check against
  detailed?: boolean;         // Include detailed findings
  generate_report?: boolean;  // Generate compliance report
}
```

#### Response

```typescript
{
  compliant: boolean;
  score: number;              // 0-100 compliance score
  findings: {
    passed: [
      {
        standard: string;
        rule: string;
        description: string;
      }
    ];
    failed: [
      {
        standard: string;
        rule: string;
        description: string;
        severity: string;
        locations: string[];
        remediation: string;
      }
    ];
  };
  summary: {
    total_checks: number;
    passed: number;
    failed: number;
    warnings: number;
  };
  report_url?: string;        // If report was generated
}
```

#### Example Usage

```json
{
  "tool": "check_compliance",
  "arguments": {
    "target": "/src",
    "compliance_type": "accessibility",
    "standards": ["wcag-2.2-accessibility"],
    "detailed": true,
    "generate_report": true
  }
}
```

## Additional Tools

The MCP server includes many additional tools:

### Standards Management
- `list_available_standards` - List all available standards
- `sync_standards` - Synchronize standards from repository
- `get_sync_status` - Check synchronization status

### Token Optimization
- `get_optimized_standard` - Get token-optimized version 1.0.0
- `auto_optimize_standards` - Automatically optimize based on context
- `progressive_load_standard` - Load standard progressively
- `estimate_token_usage` - Estimate tokens for standards

### Standards Generation
- `generate_standard` - Generate new standard from template
- `validate_standard` - Validate standard structure
- `list_templates` - List available templates

### Cross-References & Analytics
- `get_cross_references` - Get related standards
- `generate_cross_references` - Generate cross-reference map
- `get_standards_analytics` - Get usage analytics
- `track_standards_usage` - Track standard usage
- `get_recommendations` - Get personalized recommendations

### Code Improvement
- `suggest_improvements` - Get improvement suggestions based on standards

## Error Handling

All tools follow a consistent error response format:

```typescript
{
  error: {
    code: string;             // Error code
    message: string;          // Human-readable message
    details?: object;         // Additional error details
    suggestions?: string[];   // Possible solutions
  };
}
```

Common error codes:
- `INVALID_PARAMETERS`: Missing or invalid parameters
- `STANDARD_NOT_FOUND`: Requested standard doesn't exist
- `VALIDATION_FAILED`: Code validation encountered an error
- `QUOTA_EXCEEDED`: Token budget exceeded
- `SERVER_ERROR`: Internal server error

## Rate Limiting

The MCP server implements rate limiting to ensure fair usage:

- **Default limit**: 60 requests per minute
- **Burst capacity**: 100 requests
- **Headers returned**:
  - `X-RateLimit-Limit`: Request limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Reset timestamp

## Authentication

Authentication is optional but recommended for production use:

### Token Authentication

```json
{
  "auth": {
    "type": "bearer",
    "token": "mcp_token_xxxxxxxxxxxx"
  }
}
```

### OAuth 2.0

```json
{
  "auth": {
    "type": "oauth2",
    "access_token": "xxxxxxxxxxxx"
  }
}
```

## WebSocket Support

For real-time validation and streaming responses:

```javascript
const ws = new WebSocket('ws://localhost:3000/mcp');

ws.onopen = () => {
  ws.send(JSON.stringify({
    id: '123',
    tool: 'validate_code',
    arguments: {
      code: 'const x = 1;',
      language: 'javascript',
      stream: true
    }
  }));
};

ws.onmessage = (event) => {
  const response = JSON.parse(event.data);
  // Handle streaming response
};
```

## Integration Examples

### Claude Desktop

```json
{
  "mcpServers": {
    "standards": {
      "command": "mcp-standards",
      "args": ["serve", "--stdio"],
      "env": {
        "MCP_STANDARDS_CONFIG": "/path/to/config.yaml"
      }
    }
  }
}
```

### Custom Client

```python
import json
import subprocess

class MCPStandardsClient:
    def __init__(self):
        self.process = subprocess.Popen(
            ['mcp-standards', 'serve', '--stdio'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    def call_tool(self, tool_name, arguments):
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": f"tools/{tool_name}",
            "params": arguments
        }
        
        self.process.stdin.write(json.dumps(request) + '\n')
        self.process.stdin.flush()
        
        response = self.process.stdout.readline()
        return json.loads(response)

# Usage
client = MCPStandardsClient()
result = client.call_tool('get_applicable_standards', {
    'context': {
        'project_type': 'api',
        'languages': ['python']
    }
})
```

## Best Practices

1. **Use token budgets** to control response size
2. **Cache responses** when possible to reduce API calls
3. **Batch related queries** for better performance
4. **Handle errors gracefully** with fallback behavior
5. **Specify context** for more accurate results
6. **Use streaming** for large responses
7. **Implement retries** for transient failures

## Performance Tips

- **Warm cache**: Pre-sync standards for better performance
- **Use condensed format**: When full content isn't needed
- **Filter results**: Use specific tags and filters
- **Parallel requests**: Tools support concurrent calls
- **Connection pooling**: Reuse connections for multiple requests

## Versioning

The MCP API follows semantic versioning:

- **v1**: Current stable version
- **Breaking changes**: Will increment major version
- **New tools**: Will increment minor version
- **Bug fixes**: Will increment patch version

Check version 1.0.0

```json
{
  "jsonrpc": "2.0",
  "method": "mcp/version",
  "id": 1
}
```

Response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "version": "1.0.0",
    "protocol_version": "2024-11-05",
    "capabilities": ["tools", "streaming", "websocket"]
  },
  "id": 1
}
```