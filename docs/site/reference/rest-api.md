# REST API Reference

## Overview
The MCP Standards Server provides a REST API for integration with external systems.

## Endpoints

### Health Check
- **GET** `/health` - Returns server health status

### Standards
- **GET** `/standards` - List all available standards
- **GET** `/standards/{id}` - Get specific standard
- **POST** `/standards/search` - Search standards

### Validation
- **POST** `/validate` - Validate code against standards

## Authentication
Currently the API uses basic authentication. Future versions will support JWT tokens.

## Rate Limiting
Default rate limits apply to all endpoints.