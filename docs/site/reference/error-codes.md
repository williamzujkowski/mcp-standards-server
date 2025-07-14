# Error Codes Reference

## Overview
The MCP Standards Server uses standardized error codes for consistent error handling.

## Server Errors (5XX)
- **500** - Internal server error
- **503** - Service unavailable

## Client Errors (4XX)
- **400** - Bad request
- **401** - Unauthorized
- **404** - Standard not found
- **429** - Rate limit exceeded

## Validation Errors (422)
- **422** - Validation failed

## Error Response Format
```json
{
  "error": {
    "code": 404,
    "message": "Standard not found",
    "details": "The requested standard ID does not exist"
  }
}
```