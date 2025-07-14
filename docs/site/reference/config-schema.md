# Configuration Schema

## Overview
The MCP Standards Server uses YAML configuration files with the following schema.

## Main Configuration

```yaml
server:
  host: localhost
  port: 8000
  
cache:
  type: redis
  host: localhost
  port: 6379
  
standards:
  source: github
  repository: williamzujkowski/standards
  
logging:
  level: INFO
  format: json
```

## Schema Validation
All configuration files are validated against a JSON schema at startup.

## Environment Variables
Configuration can be overridden using environment variables with the prefix `MCP_`.