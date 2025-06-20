# NIST-Compliant Python API Example

This example demonstrates a Flask API with comprehensive NIST 800-53r5 compliance controls.

## Implemented Controls

- **AC-3**: Access Enforcement - Role-based authorization decorators
- **AU-2**: Audit Events - Comprehensive security event logging
- **IA-2**: Identification and Authentication - JWT-based authentication
- **SC-8**: Transmission Confidentiality - HTTPS enforcement
- **SI-10**: Information Input Validation - Request validation decorators
- **SI-11**: Error Handling - Secure error responses

## Features

- JWT-based authentication with role-based access control
- Input validation and sanitization
- Comprehensive audit logging
- Secure error handling
- HTTPS enforcement

## Setup

1. Install dependencies:
```bash
pip install flask pyjwt werkzeug
```

2. Run the application:
```bash
python app.py
```

3. Test authentication:
```bash
# Login
curl -X POST https://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "SecurePassword123!"}'

# Access protected endpoint
curl -H "Authorization: Bearer <token>" \
  https://localhost:5000/api/profile
```

## Security Notes

1. **Passwords**: Use strong, unique passwords in production
2. **Secret Key**: Generate a secure random secret key
3. **SSL/TLS**: Use proper SSL certificates, not self-signed
4. **Database**: Replace mock database with secure storage
5. **Rate Limiting**: Add rate limiting to prevent brute force attacks

## Compliance Validation

Run compliance validation:
```bash
mcp-standards scan examples/python-api/
```

This should detect all implemented NIST controls and provide a compliance score.