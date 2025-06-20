# NIST-Compliant JavaScript Frontend Example

This example demonstrates a secure JavaScript frontend application with comprehensive NIST 800-53r5 compliance controls.

## Implemented Controls

- **AC-3**: Access Enforcement - Session management and role-based access
- **AU-2**: Audit Events - Client-side security event logging
- **IA-2**: Identification and Authentication - Secure login with JWT
- **SC-8**: Transmission Confidentiality - HTTPS enforcement and secure API calls
- **SI-10**: Information Input Validation - Comprehensive input validation
- **SI-11**: Error Handling - Secure error handling without information disclosure

## Features

- JWT-based authentication with automatic logout
- Input validation and sanitization
- Content Security Policy (CSP) implementation
- Session timeout handling
- Comprehensive audit logging
- Secure error handling
- HTTPS enforcement

## Security Features

### Authentication & Authorization
- JWT token management
- Automatic session timeout (30 minutes)
- Secure logout with data clearing

### Input Validation
- Email format validation
- Password strength requirements
- Username format validation
- XSS prevention

### Security Headers
- Content Security Policy
- Secure request headers

### Audit Logging
- All security events logged
- Local audit trail maintenance
- Comprehensive event details

## Setup

1. Create an HTML file to load the application:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Secure App</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .login-container, .dashboard { max-width: 400px; margin: 0 auto; }
        input { display: block; width: 100%; margin: 10px 0; padding: 8px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
    </style>
</head>
<body>
    <script src="app.js"></script>
</body>
</html>
```

2. Serve over HTTPS (required for security features)
3. Configure the API base URL in the constructor

## Configuration

Update the `apiBase` URL in the SecureApp constructor to point to your backend API.

## Password Requirements

- Minimum 8 characters
- At least one lowercase letter
- At least one uppercase letter
- At least one number
- At least one special character

## Security Notes

1. **HTTPS**: Application enforces HTTPS for all communication
2. **CSP**: Content Security Policy prevents XSS attacks
3. **Session Management**: Automatic logout on inactivity
4. **Input Validation**: All inputs are validated and sanitized
5. **Error Handling**: Generic error messages prevent information disclosure

## Compliance Validation

Run compliance validation:
```bash
mcp-standards scan examples/javascript-frontend/
```

This should detect all implemented NIST controls and provide a compliance score.