# NIST 800-53r5 Control Mappings

This document provides a comprehensive mapping of NIST 800-53r5 controls to code patterns and implementation strategies.

## Control Families

### AC - Access Control

#### AC-2: Account Management
**Implementation Patterns:**
- User account creation/deletion functions
- Account lifecycle management
- Automated account provisioning

**Code Indicators:**
```python
# @nist-controls: AC-2
# @evidence: User account management with approval workflow
def create_user_account(user_data, approver):
    # Implementation
```

#### AC-3: Access Enforcement
**Implementation Patterns:**
- Role-based access control (RBAC)
- Attribute-based access control (ABAC)
- Permission checking middleware

**Code Indicators:**
- `@authorize`, `@permission_required` decorators
- `check_permission()`, `has_role()` functions
- Access control lists (ACLs)

#### AC-4: Information Flow Enforcement
**Implementation Patterns:**
- Data classification checks
- Network segmentation
- API gateway filtering

### AU - Audit and Accountability

#### AU-2: Audit Events
**Implementation Patterns:**
- Security event logging
- Authentication/authorization logging
- Data access logging

**Code Indicators:**
```javascript
// @nist-controls: AU-2, AU-3
// @evidence: Comprehensive audit logging
logger.security({
  event: 'authentication',
  user: userId,
  timestamp: new Date(),
  outcome: 'success'
});
```

#### AU-3: Content of Audit Records
**Requirements:**
- Type of event
- When event occurred
- Where event occurred
- Source of event
- Outcome of event
- Identity of subjects/objects

### IA - Identification and Authentication

#### IA-2: Identification and Authentication
**Implementation Patterns:**
- Multi-factor authentication (MFA)
- Single Sign-On (SSO)
- Biometric authentication

**Code Indicators:**
```go
// @nist-controls: IA-2
// @evidence: MFA implementation with TOTP
func ValidateMFA(user User, token string) error {
    return totp.Validate(token, user.MFASecret)
}
```

#### IA-5: Authenticator Management
**Implementation Patterns:**
- Password complexity requirements
- Password history checking
- Secure credential storage

### SC - System and Communications Protection

#### SC-8: Transmission Confidentiality
**Implementation Patterns:**
- TLS/SSL implementation
- End-to-end encryption
- VPN connections

#### SC-13: Cryptographic Protection
**Implementation Patterns:**
- Encryption algorithm usage
- Key management
- Digital signatures

**Code Indicators:**
```java
// @nist-controls: SC-13, SC-28
// @evidence: AES-256 encryption for sensitive data
@Encrypt(algorithm = "AES-256-GCM")
private String sensitiveData;
```

### SI - System and Information Integrity

#### SI-10: Information Input Validation
**Implementation Patterns:**
- Input sanitization
- Schema validation
- SQL injection prevention

**Code Indicators:**
- Parameterized queries
- Input validation libraries
- Regular expression validation

## Control Implementation Matrix

| Control | Python | JavaScript | Go | Java |
|---------|---------|------------|-----|------|
| AC-2 | Django User model | Express middleware | Custom struct | Spring Security |
| AC-3 | @permission_required | RBAC middleware | Casbin | @PreAuthorize |
| AU-2 | logging module | Winston/Bunyan | logrus/zap | Log4j/SLF4J |
| IA-2 | django-allauth | Passport.js | JWT-go | Spring OAuth2 |
| SC-8 | ssl module | HTTPS module | crypto/tls | SSLContext |
| SI-10 | Django forms | Joi/Yup | validator | Bean Validation |

## Best Practices

### 1. Explicit Annotations
Always use explicit NIST control annotations when implementing security features:

```python
# @nist-controls: AC-3, AU-2
# @evidence: RBAC with audit logging
# @oscal-component: auth-service
```

### 2. Comprehensive Evidence
Provide clear evidence statements that explain how the control is implemented.

### 3. Control Inheritance
Document control inheritance in component hierarchies.

### 4. Continuous Monitoring
Implement automated scanning to ensure controls remain in place.

## Common Control Combinations

### Authentication System
- IA-2: Identification and Authentication
- IA-5: Authenticator Management
- AC-7: Unsuccessful Login Attempts
- AU-2: Audit Events

### API Security
- AC-3: Access Enforcement
- AC-4: Information Flow Enforcement
- SC-8: Transmission Confidentiality
- SI-10: Information Input Validation

### Data Protection
- SC-13: Cryptographic Protection
- SC-28: Protection of Information at Rest
- MP-5: Media Transport
- AU-9: Protection of Audit Information

## Compliance Validation

Use the CLI to validate control implementation:

```bash
# Check specific control
mcp-standards scan --control AC-3

# Generate control coverage report
mcp-standards scan --output-format oscal
```

## Resources

- [NIST SP 800-53r5](https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final)
- [NIST Control Catalog](https://csrc.nist.gov/projects/risk-management/sp800-53-controls/release-search)
- [OSCAL Documentation](https://pages.nist.gov/OSCAL/)