# Secure Flask API Example

This example demonstrates a Flask API with comprehensive NIST 800-53r5 controls.

## NIST Controls Implemented

- **AC-2**: Account Management - User registration and management
- **AC-3**: Access Enforcement - Role-based access control
- **AU-2**: Audit Events - Comprehensive logging
- **IA-2**: Identification and Authentication - JWT-based auth with MFA
- **SC-8**: Transmission Confidentiality - HTTPS enforcement
- **SC-13**: Cryptographic Protection - Password hashing with bcrypt
- **SI-10**: Information Input Validation - Request validation

## Project Structure

```
python-flask-api/
├── app.py              # Main application
├── auth.py             # Authentication module
├── models.py           # Data models
├── validators.py       # Input validation
├── security.py         # Security utilities
├── requirements.txt    # Dependencies
└── .mcp-standards/     # MCP configuration
```

## Running the Example

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Initialize MCP standards:
```bash
mcp-standards init
```

3. Run compliance scan:
```bash
mcp-standards scan
```

4. Start the server:
```bash
python app.py
```

## Security Features

### Authentication Flow
1. User registers with email/password
2. Password is hashed using bcrypt
3. MFA setup using TOTP
4. JWT tokens issued on successful auth
5. Token refresh mechanism

### Authorization
- Role-based access control (RBAC)
- Permission decorators
- Resource-level access control

### Audit Logging
- All authentication attempts
- Authorization decisions
- Data access events
- Security incidents

## API Endpoints

### Public Endpoints
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Token refresh

### Protected Endpoints
- `GET /api/profile` - User profile (requires auth)
- `GET /api/users` - List users (requires admin role)
- `POST /api/data` - Create data (requires specific permission)

## Compliance Testing

Generate OSCAL SSP:
```bash
mcp-standards ssp --output flask-api-ssp.json
```

This will analyze the codebase and generate a System Security Plan documenting all implemented controls.