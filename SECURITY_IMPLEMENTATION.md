# Security Implementation Guide

## Overview

This document outlines the comprehensive security implementation for the MCP Standards Server. The security framework is designed with defense-in-depth principles, providing multiple layers of protection against common security threats.

## Security Architecture

### 1. Input Validation and Sanitization

#### Implementation
- **Location**: `src/core/validation.py`, `src/core/security.py`
- **Features**:
  - JSON Schema validation using Pydantic
  - Input sanitization with character filtering
  - SQL injection detection and prevention
  - Script injection detection and prevention
  - Request size limits (10MB default)
  - JSON depth limits (100 levels)
  - Array length limits (10,000 items)
  - String length limits (1MB)

#### Usage
```python
from src.core.validation import get_input_validator

validator = get_input_validator()
validated_data = validator.validate_tool_input("tool_name", raw_input)
```

### 2. Serialization Security

#### Pickle Deserialization Protection
- **Status**: ✅ DISABLED
- **Implementation**: `src/core/cache/redis_client.py`
- **Security Measures**:
  - Pickle deserialization completely disabled
  - Safe alternatives: msgpack (primary), JSON (fallback)
  - Detailed error logging for pickle attempts
  - Backward compatibility for reading old data

#### Safe Serialization Stack
1. **Primary**: msgpack (fast, secure, handles most Python types)
2. **Fallback**: JSON with custom encoder (safe, handles basic types)
3. **Prohibited**: pickle (removed due to RCE vulnerability)

### 3. Authentication and Authorization

#### JWT Token Security
- **Implementation**: `src/core/auth.py`
- **Features**:
  - HS256 algorithm (configurable)
  - Token expiration (24 hours default)
  - Token revocation support
  - Scope-based permissions
  - API key authentication alternative

#### Configuration
```python
from src.core.auth import AuthConfig, AuthManager

config = AuthConfig(
    enabled=True,
    secret_key="your-secret-key",
    token_expiry_hours=24
)
auth_manager = AuthManager(config)
```

### 4. Rate Limiting

#### Multi-Tier Rate Limiting
- **Implementation**: `src/core/rate_limiter.py`
- **Tiers**:
  - **Minute**: 100 requests/minute
  - **Hour**: 5,000 requests/hour
  - **Day**: 50,000 requests/day

#### Adaptive Rate Limiting
- **Features**:
  - Reputation-based limits
  - Automatic adjustment based on behavior
  - Higher limits for good users (1.5x)
  - Lower limits for problematic users (0.5x)

#### Usage
```python
from src.core.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
is_allowed, limit_info = limiter.check_all_limits(client_id)
```

### 5. Error Handling and Information Leakage Prevention

#### Secure Error Handler
- **Implementation**: `src/core/errors.py`
- **Features**:
  - Sensitive information sanitization
  - Stack trace removal
  - SQL error masking
  - File path redaction
  - IP address redaction
  - Production vs development modes

#### Sanitization Patterns
- File paths: `/home/user/file.txt` → `[REDACTED]`
- IP addresses: `192.168.1.1` → `[REDACTED]`
- Passwords/tokens: `password123` → `[REDACTED]`
- Stack traces: `Traceback...` → `[STACK_TRACE_REDACTED]`

### 6. Security Headers

#### HTTP Security Headers
- **Implementation**: `src/core/security.py`
- **Headers Applied**:
  - `X-Content-Type-Options: nosniff`
  - `X-Frame-Options: DENY`
  - `X-XSS-Protection: 1; mode=block`
  - `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  - `Content-Security-Policy: default-src 'self'...`
  - `Referrer-Policy: strict-origin-when-cross-origin`

### 7. Request Security

#### Content Security
- **Maximum request size**: 10MB
- **JSON depth limit**: 100 levels
- **Array length limit**: 10,000 items
- **String length limit**: 1MB
- **Allowed content types**: application/json, application/x-msgpack, text/plain

#### Injection Prevention
- **SQL Injection**: Pattern-based detection
- **Script Injection**: XSS and code injection prevention
- **Command Injection**: Dangerous function detection
- **Path Traversal**: Directory traversal prevention

## Security Testing

### Test Coverage

#### 1. Redis Security Tests
- **File**: `tests/unit/core/cache/test_redis_security.py`
- **Coverage**:
  - Pickle deserialization blocking
  - Safe serialization methods
  - Cache key security
  - Compression security
  - Circuit breaker security

#### 2. Security Middleware Tests
- **File**: `tests/unit/core/test_security.py`
- **Coverage**:
  - Input validation and sanitization
  - Injection detection
  - Request size limits
  - Security headers
  - Error handling

#### 3. Error Handling Tests
- **File**: `tests/unit/core/test_error_handling.py`
- **Coverage**:
  - Information leakage prevention
  - Error message sanitization
  - Stack trace removal
  - Sensitive data redaction

#### 4. Rate Limiting Tests
- **File**: `tests/unit/core/test_rate_limiting.py`
- **Coverage**:
  - Multi-tier rate limiting
  - Adaptive rate limiting
  - Reputation management
  - Security integration

### Running Security Tests

```bash
# Run all security tests
python -m pytest tests/unit/core/test_security.py -v
python -m pytest tests/unit/core/cache/test_redis_security.py -v
python -m pytest tests/unit/core/test_error_handling.py -v
python -m pytest tests/unit/core/test_rate_limiting.py -v

# Run with coverage
python -m pytest tests/unit/core/test_security.py --cov=src/core/security --cov-report=term-missing
```

## Security Configuration

### Environment Variables

```bash
# Authentication
export MCP_AUTH_ENABLED=true
export MCP_JWT_SECRET=your-secret-key-here

# Error handling
export MCP_MASK_ERRORS=true  # false for development

# Rate limiting
export MCP_RATE_LIMIT_ENABLED=true
```

### Configuration Classes

#### SecurityConfig
```python
from src.core.security import SecurityConfig

config = SecurityConfig(
    max_request_size=10 * 1024 * 1024,  # 10MB
    max_json_depth=100,
    max_array_length=10000,
    max_string_length=1000000,
    enable_security_headers=True,
    sanitize_inputs=True,
    mask_errors=True
)
```

#### CacheConfig
```python
from src.core.cache.redis_client import CacheConfig

config = CacheConfig(
    enable_compression=True,
    compression_threshold=1024,
    circuit_breaker_threshold=5,
    circuit_breaker_timeout=30
)
```

## Security Best Practices

### 1. Input Validation
- **Always validate inputs** at the entry point
- **Use type hints** and Pydantic models
- **Sanitize user data** before processing
- **Limit input sizes** to prevent DoS attacks

### 2. Error Handling
- **Never expose sensitive information** in error messages
- **Log security events** for monitoring
- **Use generic error messages** in production
- **Implement proper error boundaries**

### 3. Authentication
- **Use strong JWT secrets** (>32 characters)
- **Implement token expiration** (24 hours max)
- **Support token revocation** for compromised tokens
- **Use HTTPS only** in production

### 4. Rate Limiting
- **Implement multi-tier limits** (minute, hour, day)
- **Use adaptive limits** based on user behavior
- **Monitor rate limit violations** for security events
- **Implement circuit breakers** for resilience

### 5. Caching
- **Never use pickle** for serialization
- **Use safe serialization methods** (msgpack, JSON)
- **Implement cache key namespacing** to prevent collisions
- **Use compression** for large data

## Security Monitoring

### Logging

Security events are logged with appropriate levels:

```python
import logging

logger = logging.getLogger(__name__)

# Security violations
logger.warning("Security violation: injection detected")

# Authentication failures
logger.error("Authentication failed for user: %s", user_id)

# Rate limiting
logger.info("Rate limit exceeded for client: %s", client_id)
```

### Metrics

Monitor these security metrics:

- **Authentication failures per minute**
- **Rate limit violations per hour**
- **Security validation failures**
- **Malicious request attempts**
- **Cache security events**

## Incident Response

### Security Incident Procedure

1. **Immediate Response**
   - Block malicious IP addresses
   - Revoke compromised tokens
   - Increase rate limits temporarily

2. **Investigation**
   - Review security logs
   - Identify attack patterns
   - Assess impact and scope

3. **Recovery**
   - Apply security patches
   - Update security rules
   - Monitor for continued threats

4. **Prevention**
   - Update security policies
   - Improve detection rules
   - Conduct security training

### Contact Information

For security issues:
- **Security Team**: security@example.com
- **Emergency**: Use incident response process
- **Vulnerability Reports**: Follow responsible disclosure

## Security Audit Checklist

### Pre-Deployment Security Checklist

- [ ] All input validation is implemented
- [ ] Pickle deserialization is disabled
- [ ] Security headers are configured
- [ ] Rate limiting is enabled
- [ ] Error handling masks sensitive data
- [ ] Authentication is properly configured
- [ ] Security tests pass
- [ ] Logging is configured for security events
- [ ] HTTPS is enforced
- [ ] Secrets are properly managed

### Regular Security Reviews

- [ ] Review security logs weekly
- [ ] Update security dependencies monthly
- [ ] Conduct penetration testing quarterly
- [ ] Review authentication tokens annually
- [ ] Update security documentation continuously

## Security Dependencies

### Core Security Libraries

- **Pydantic**: Input validation and serialization
- **msgpack**: Safe binary serialization
- **Redis**: Secure caching with proper client
- **JWT**: Token-based authentication
- **bcrypt**: Password hashing (if needed)

### Security-Related Dependencies

```txt
pydantic>=2.0.0
msgpack>=1.0.0
redis>=4.0.0
PyJWT>=2.0.0
cryptography>=3.0.0
```

## Compliance

### Security Standards

This implementation follows:

- **OWASP Top 10** security practices
- **NIST Cybersecurity Framework**
- **CWE Common Weakness Enumeration**
- **SANS Top 25** software errors prevention

### Vulnerability Management

- **CVE monitoring**: Automated dependency scanning
- **Security updates**: Regular patch management
- **Penetration testing**: Quarterly assessments
- **Code review**: Security-focused reviews

## Future Enhancements

### Planned Security Improvements

1. **Web Application Firewall (WAF)** integration
2. **Advanced threat detection** with machine learning
3. **Zero-trust architecture** implementation
4. **Container security** scanning
5. **API security** gateway integration

### Security Roadmap

- **Q1**: Implement WAF integration
- **Q2**: Add advanced threat detection
- **Q3**: Implement zero-trust architecture
- **Q4**: Add container security scanning

---

**Note**: This security implementation is continuously evolving. Please refer to the latest version of this document and conduct regular security reviews to ensure ongoing protection against emerging threats.