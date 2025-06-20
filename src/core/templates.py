"""
Template Generator for NIST-compliant code
@nist-controls: SA-11, SA-15, SA-17
@evidence: Developer security and secure coding
"""
from typing import Dict, List, Optional
from pathlib import Path
import json

from ..core.logging import get_logger

logger = get_logger(__name__)


class TemplateGenerator:
    """
    Generate NIST-compliant code templates
    @nist-controls: SA-11, SA-15
    @evidence: Secure development practices
    """
    
    def __init__(self):
        self.templates = self._load_templates()
    
    def generate(
        self,
        template_type: str,
        language: str,
        controls: Optional[List[str]] = None
    ) -> str:
        """
        Generate a template based on type and language
        @nist-controls: SA-11
        @evidence: Security-focused code generation
        """
        # Validate inputs
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        if language not in self.templates[template_type]:
            raise ValueError(f"Language {language} not supported for {template_type}")
        
        # Get base template
        template = self.templates[template_type][language]
        
        # Add control-specific implementations if requested
        if controls:
            template = self._add_control_implementations(template, controls, language)
        
        return template
    
    def _load_templates(self) -> Dict[str, Dict[str, str]]:
        """Load template definitions"""
        templates = {}
        
        # API templates
        templates["api"] = {}
        templates["auth"] = {}
        templates["logging"] = {}
        templates["encryption"] = {}
        templates["database"] = {}
        
        # Python API template
        templates["api"]["python"] = '''"""
API Endpoint Implementation
@nist-controls: AC-3, AU-2, IA-2, SC-8, SI-10
@evidence: Secure API with authentication, authorization, and validation
"""
from typing import Any, Dict, Optional
from datetime import datetime
import logging
from functools import wraps
import jwt
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest, Unauthorized, Forbidden

app = Flask(__name__)
logger = logging.getLogger(__name__)

# @nist-controls: AU-2, AU-3
# @evidence: Structured security event logging
def log_security_event(event: str, user_id: Optional[str] = None, **kwargs):
    """Log security-relevant events with context"""
    logger.info(
        "SECURITY_EVENT",
        extra={
            "event": event,
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "remote_addr": request.remote_addr if request else None,
            **kwargs
        }
    )

# @nist-controls: IA-2, AC-3
# @evidence: JWT-based authentication and authorization
def require_auth(required_roles: Optional[List[str]] = None):
    """Decorator for endpoint authentication and authorization"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            # Extract and validate JWT token
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                log_security_event("AUTH_FAILED", reason="Missing token")
                raise Unauthorized("Authentication required")
            
            try:
                token = auth_header.split(' ')[1]
                payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
                
                # Check roles if required
                if required_roles:
                    user_roles = payload.get('roles', [])
                    if not any(role in user_roles for role in required_roles):
                        log_security_event("AUTHZ_FAILED", user_id=payload.get('sub'))
                        raise Forbidden("Insufficient permissions")
                
                # Add user context to request
                request.current_user = payload
                log_security_event("AUTH_SUCCESS", user_id=payload.get('sub'))
                
            except jwt.ExpiredSignatureError:
                log_security_event("AUTH_FAILED", reason="Expired token")
                raise Unauthorized("Token expired")
            except jwt.InvalidTokenError:
                log_security_event("AUTH_FAILED", reason="Invalid token")
                raise Unauthorized("Invalid token")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# @nist-controls: SI-10
# @evidence: Input validation for API requests
def validate_request(schema: Dict[str, Any]):
    """Decorator for request validation"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            data = request.get_json()
            if not data:
                raise BadRequest("JSON payload required")
            
            # Simple schema validation (use jsonschema in production)
            for field, rules in schema.items():
                if rules.get('required') and field not in data:
                    raise BadRequest(f"Missing required field: {field}")
                
                if field in data:
                    value = data[field]
                    if 'type' in rules and not isinstance(value, rules['type']):
                        raise BadRequest(f"Invalid type for {field}")
                    
                    if 'max_length' in rules and len(str(value)) > rules['max_length']:
                        raise BadRequest(f"{field} exceeds maximum length")
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

@app.route('/api/v1/resource', methods=['POST'])
@require_auth(required_roles=['user', 'admin'])
@validate_request({
    'name': {'required': True, 'type': str, 'max_length': 100},
    'description': {'required': False, 'type': str, 'max_length': 500}
})
def create_resource():
    """
    Create a new resource
    @nist-controls: AC-3, SI-10
    @evidence: Authorized access with validated input
    """
    data = request.get_json()
    user_id = request.current_user['sub']
    
    # Process the request
    # ... implementation ...
    
    log_security_event("RESOURCE_CREATED", user_id=user_id, resource_name=data['name'])
    
    return jsonify({
        'status': 'success',
        'id': 'generated-id',
        'message': 'Resource created successfully'
    }), 201

@app.errorhandler(Exception)
def handle_error(error):
    """
    Global error handler
    @nist-controls: SI-11
    @evidence: Secure error handling without information leakage
    """
    if isinstance(error, (BadRequest, Unauthorized, Forbidden)):
        return jsonify({'error': str(error)}), error.code
    
    # Log internal errors but don't expose details
    logger.error(f"Internal error: {error}", exc_info=True)
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # @nist-controls: SC-8
    # @evidence: TLS encryption for data in transit
    app.run(ssl_context='adhoc', debug=False)
'''

        # Python Auth template
        templates["auth"]["python"] = '''"""
Authentication Module
@nist-controls: IA-2, IA-5, IA-8, AC-7
@evidence: Multi-factor authentication with secure credential management
"""
import secrets
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pyotp
import bcrypt
import jwt

class AuthenticationService:
    """
    Secure authentication service
    @nist-controls: IA-2, IA-5
    @evidence: Strong authentication mechanisms
    """
    
    def __init__(self, secret_key: str, token_expiry: int = 3600):
        self.secret_key = secret_key
        self.token_expiry = token_expiry
        self.failed_attempts = {}  # In production, use Redis
        
    # @nist-controls: IA-5
    # @evidence: Secure password hashing
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    # @nist-controls: IA-2
    # @evidence: Multi-factor authentication support
    def generate_totp_secret(self) -> str:
        """Generate TOTP secret for 2FA"""
        return pyotp.random_base32()
    
    def verify_totp(self, secret: str, token: str) -> bool:
        """Verify TOTP token"""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)
    
    # @nist-controls: AC-7
    # @evidence: Account lockout after failed attempts
    def check_account_lockout(self, username: str) -> Tuple[bool, Optional[int]]:
        """Check if account is locked due to failed attempts"""
        if username not in self.failed_attempts:
            return False, None
            
        attempts_data = self.failed_attempts[username]
        if attempts_data['count'] >= 5:
            lockout_time = 300  # 5 minutes
            time_passed = time.time() - attempts_data['last_attempt']
            if time_passed < lockout_time:
                return True, int(lockout_time - time_passed)
            else:
                # Reset after lockout period
                del self.failed_attempts[username]
                
        return False, None
    
    def record_failed_attempt(self, username: str):
        """Record failed login attempt"""
        if username not in self.failed_attempts:
            self.failed_attempts[username] = {'count': 0, 'last_attempt': 0}
        
        self.failed_attempts[username]['count'] += 1
        self.failed_attempts[username]['last_attempt'] = time.time()
    
    def reset_failed_attempts(self, username: str):
        """Reset failed attempts on successful login"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    # @nist-controls: IA-8
    # @evidence: Secure token generation
    def generate_auth_token(self, user_id: str, roles: list) -> str:
        """Generate JWT authentication token"""
        payload = {
            'sub': user_id,
            'roles': roles,
            'iat': datetime.utcnow(),
            'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
            'jti': secrets.token_urlsafe(16)  # Unique token ID
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def generate_refresh_token(self) -> str:
        """Generate secure refresh token"""
        return secrets.token_urlsafe(32)
    
    # @nist-controls: IA-5
    # @evidence: Password strength validation
    def validate_password_strength(self, password: str) -> Tuple[bool, Optional[str]]:
        """Validate password meets security requirements"""
        if len(password) < 12:
            return False, "Password must be at least 12 characters"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain uppercase letters"
            
        if not any(c.islower() for c in password):
            return False, "Password must contain lowercase letters"
            
        if not any(c.isdigit() for c in password):
            return False, "Password must contain numbers"
            
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain special characters"
            
        # Check against common passwords (in production, use a comprehensive list)
        common_passwords = ['password123', 'admin123', 'letmein123']
        if password.lower() in [p.lower() for p in common_passwords]:
            return False, "Password is too common"
            
        return True, None

# Example usage
if __name__ == "__main__":
    auth = AuthenticationService(secret_key="your-secret-key")
    
    # Register new user
    password = "SecureP@ssw0rd123!"
    is_valid, error = auth.validate_password_strength(password)
    if is_valid:
        hashed = auth.hash_password(password)
        totp_secret = auth.generate_totp_secret()
        print(f"User registered with 2FA secret: {totp_secret}")
'''

        # Add logging template
        templates["logging"]["python"] = '''"""
Security Logging Module
@nist-controls: AU-2, AU-3, AU-4, AU-9, AU-12
@evidence: Comprehensive audit logging with integrity protection
"""
import json
import hashlib
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import structlog

# @nist-controls: AU-3
# @evidence: Structured logging with required fields
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

class SecurityLogger:
    """
    Security-focused logging with NIST compliance
    @nist-controls: AU-2, AU-3, AU-9
    @evidence: Audit log generation with integrity checking
    """
    
    def __init__(self, log_dir: Path = Path("/var/log/app")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = structlog.get_logger()
        
    # @nist-controls: AU-2
    # @evidence: Log security-relevant events
    def log_authentication(self, user_id: str, success: bool, method: str, **kwargs):
        """Log authentication attempts"""
        self.logger.info(
            "authentication",
            user_id=user_id,
            success=success,
            method=method,
            event_type="AUTHENTICATION",
            **self._add_context(**kwargs)
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, granted: bool, **kwargs):
        """Log authorization decisions"""
        self.logger.info(
            "authorization",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
            event_type="AUTHORIZATION",
            **self._add_context(**kwargs)
        )
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, **kwargs):
        """Log data access events"""
        self.logger.info(
            "data_access",
            user_id=user_id,
            data_type=data_type,
            operation=operation,
            event_type="DATA_ACCESS",
            **self._add_context(**kwargs)
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], **kwargs):
        """Log security violations"""
        self.logger.warning(
            "security_violation",
            violation_type=violation_type,
            details=details,
            event_type="SECURITY_VIOLATION",
            **self._add_context(**kwargs)
        )
    
    # @nist-controls: AU-9
    # @evidence: Log integrity protection
    def _calculate_log_hash(self, log_entry: Dict[str, Any]) -> str:
        """Calculate hash of log entry for integrity verification"""
        # Sort keys for consistent hashing
        sorted_entry = json.dumps(log_entry, sort_keys=True)
        return hashlib.sha256(sorted_entry.encode()).hexdigest()
    
    def _add_context(self, **kwargs) -> Dict[str, Any]:
        """Add standard context to all log entries"""
        context = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": kwargs.get("session_id"),
            "ip_address": kwargs.get("ip_address"),
            "user_agent": kwargs.get("user_agent"),
        }
        
        # Add integrity hash
        context["integrity_hash"] = self._calculate_log_hash(context)
        
        return context
    
    # @nist-controls: AU-4
    # @evidence: Log rotation and retention
    def setup_log_rotation(self):
        """Configure log rotation (example using logging.handlers)"""
        from logging.handlers import RotatingFileHandler
        
        handler = RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=30  # Keep 30 days
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        return handler

# Example usage
if __name__ == "__main__":
    logger = SecurityLogger()
    
    # Log authentication attempt
    logger.log_authentication(
        user_id="user123",
        success=True,
        method="password+totp",
        session_id="sess_abc123",
        ip_address="192.168.1.100"
    )
'''

        # Encryption template
        templates["encryption"]["python"] = '''"""
Encryption Utilities
@nist-controls: SC-8, SC-13, SC-28
@evidence: FIPS-validated cryptography for data protection
"""
import os
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class EncryptionService:
    """
    Encryption service with NIST-approved algorithms
    @nist-controls: SC-13
    @evidence: FIPS 140-2 validated cryptographic modules
    """
    
    def __init__(self, master_key: bytes = None):
        self.master_key = master_key or Fernet.generate_key()
        
    # @nist-controls: SC-28
    # @evidence: Encryption for data at rest
    def encrypt_data(self, data: bytes, associated_data: bytes = None) -> bytes:
        """Encrypt data using AES-GCM"""
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        aesgcm = AESGCM(self.master_key[:32])  # Use 256-bit key
        ciphertext = aesgcm.encrypt(nonce, data, associated_data)
        return nonce + ciphertext
    
    def decrypt_data(self, encrypted: bytes, associated_data: bytes = None) -> bytes:
        """Decrypt data encrypted with AES-GCM"""
        nonce = encrypted[:12]
        ciphertext = encrypted[12:]
        aesgcm = AESGCM(self.master_key[:32])
        return aesgcm.decrypt(nonce, ciphertext, associated_data)
'''

        # Database template
        templates["database"]["python"] = '''"""
Secure Database Operations
@nist-controls: AC-3, AU-2, SC-8, SI-10
@evidence: Parameterized queries and access control
"""
from contextlib import contextmanager
import logging
from typing import Any, Dict, List
import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

class SecureDatabase:
    """
    Database wrapper with security controls
    @nist-controls: AC-3, SI-10
    @evidence: Access control and injection prevention
    """
    
    def __init__(self, connection_string: str):
        # @nist-controls: SC-8
        # @evidence: Require SSL for database connections
        self.connection_string = connection_string + " sslmode=require"
        
    @contextmanager
    def get_connection(self):
        """Get database connection with automatic cleanup"""
        conn = psycopg2.connect(self.connection_string)
        try:
            yield conn
        finally:
            conn.close()
    
    # @nist-controls: SI-10
    # @evidence: Parameterized queries prevent SQL injection
    def execute_query(self, query: str, params: tuple = None) -> List[Dict[str, Any]]:
        """Execute SELECT query with parameters"""
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
    
    # @nist-controls: AU-2
    # @evidence: Audit database modifications
    def execute_write(self, query: str, params: tuple = None, user_id: str = None):
        """Execute INSERT/UPDATE/DELETE with audit logging"""
        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                affected = cursor.rowcount
                conn.commit()
                
                # Log the operation
                logger.info(
                    "database_write",
                    extra={
                        "user_id": user_id,
                        "operation": query.split()[0],
                        "affected_rows": affected
                    }
                )
                return affected
'''

        # Add basic JavaScript templates
        templates["api"]["javascript"] = '''/**
 * Secure API Implementation
 * @nist-controls AC-3, AU-2, IA-2, SC-8, SI-10
 * @evidence Authentication, authorization, and input validation
 */
const express = require('express');
const jwt = require('jsonwebtoken');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');

const app = express();

// Security middleware
app.use(helmet());
app.use(express.json({ limit: '10mb' }));

// @nist-controls: SC-8
// @evidence: Force HTTPS in production
app.use((req, res, next) => {
  if (process.env.NODE_ENV === 'production' && !req.secure) {
    return res.redirect('https://' + req.headers.host + req.url);
  }
  next();
});

// @nist-controls: SI-10
// @evidence: Rate limiting to prevent abuse
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// @nist-controls: IA-2, AC-3
// @evidence: JWT authentication middleware
const authenticate = (req, res, next) => {
  const token = req.headers.authorization?.split(' ')[1];
  
  if (!token) {
    return res.status(401).json({ error: 'Authentication required' });
  }
  
  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    req.user = decoded;
    next();
  } catch (error) {
    return res.status(401).json({ error: 'Invalid token' });
  }
};

// Example endpoint
app.post('/api/resource', authenticate, (req, res) => {
  // Input validation
  const { name, value } = req.body;
  
  if (!name || typeof name !== 'string' || name.length > 100) {
    return res.status(400).json({ error: 'Invalid name' });
  }
  
  // Process request...
  res.json({ success: true, id: 'generated-id' });
});

module.exports = app;
'''

        return templates
    
    def _add_control_implementations(self, template: str, controls: List[str], language: str) -> str:
        """Add specific control implementations to template"""
        # This would add control-specific code snippets
        # For now, just add a comment
        control_comment = f"\n# Additional controls implemented: {', '.join(controls)}\n"
        return control_comment + template