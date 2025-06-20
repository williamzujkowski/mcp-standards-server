# MCP Standards Server Examples

This directory contains example implementations demonstrating NIST 800-53r5 compliance patterns across different technologies and use cases.

## Available Examples

### 🐍 Python API (`python-api/`)
A Flask-based REST API demonstrating:
- JWT authentication with role-based access control
- Comprehensive input validation
- Security event logging
- Error handling best practices
- HTTPS enforcement

**Key Controls**: AC-3, AU-2, IA-2, SC-8, SI-10, SI-11

### 🌐 JavaScript Frontend (`javascript-frontend/`)
A secure frontend application showing:
- Client-side authentication and session management
- Input validation and XSS prevention
- Content Security Policy implementation
- Audit logging and error handling
- Session timeout mechanisms

**Key Controls**: AC-3, AU-2, IA-2, SC-8, SI-10, SI-11

### 🗄️ Secure Database (`secure-database/`)
Database security patterns including:
- Parameterized queries to prevent SQL injection
- Database connection security
- Data encryption at rest
- Access logging and audit trails
- Backup and recovery procedures

**Key Controls**: AC-3, AU-2, SC-8, SC-28, SI-10

## Usage

Each example includes:
- **Complete source code** with NIST control annotations
- **README.md** with setup instructions and security notes
- **Compliance validation** commands
- **Best practices** documentation

## Getting Started

1. Choose an example that matches your technology stack
2. Review the README.md for setup instructions
3. Examine the NIST control annotations in the code
4. Run compliance validation:
   ```bash
   mcp-standards scan examples/<example-name>/
   ```

## Compliance Validation

All examples are designed to pass MCP Standards compliance scanning:

```bash
# Scan all examples
mcp-standards scan examples/

# Scan specific example
mcp-standards scan examples/python-api/

# Generate compliance report
mcp-standards scan examples/ --output-format oscal --output-file compliance-report.json
```

## Implementation Patterns

### Authentication & Authorization
- JWT-based authentication
- Role-based access control (RBAC)
- Session management and timeout
- Multi-factor authentication examples

### Input Validation
- Schema-based validation
- Sanitization and filtering
- SQL injection prevention
- XSS protection

### Audit & Logging
- Security event logging
- Structured log formats
- Log integrity protection
- Retention policies

### Encryption & Data Protection
- TLS/HTTPS enforcement
- Data encryption at rest
- Secure key management
- Certificate handling

### Error Handling
- Secure error responses
- Information disclosure prevention
- Graceful degradation
- User-friendly messages

## Contributing

When adding new examples:

1. Include comprehensive NIST control annotations
2. Add a detailed README.md
3. Ensure compliance validation passes
4. Follow the established directory structure
5. Include both positive and negative test cases

## Security Notes

These examples are for educational purposes and demonstrate security best practices. When implementing in production:

- Use proper secrets management
- Implement comprehensive logging
- Regular security updates
- Professional security review
- Proper certificate management
- Environment-specific configuration

## Standards Compliance

All examples implement relevant NIST 800-53r5 controls and are validated using the MCP Standards Server. The goal is to provide practical, real-world examples of secure coding practices that meet compliance requirements.