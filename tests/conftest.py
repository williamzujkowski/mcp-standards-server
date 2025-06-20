"""
Pytest Configuration and Shared Fixtures
@nist-controls: SA-11, CA-7
@evidence: Test infrastructure for compliance validation
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import jwt  # This is from the pyjwt package
import pytest
from cryptography.fernet import Fernet

from src.compliance.scanner import ComplianceScanner
from src.core.mcp.models import (
    AuthenticationLevel,
    ComplianceContext,
    MCPMessage,
    SessionInfo,
)
from src.core.standards.engine import StandardsEngine
from src.core.standards.models import StandardLoadResult


# Configure pytest-asyncio
pytest_plugins = ["pytest_asyncio"]


# Using pytest-asyncio default event loop implementation


@pytest.fixture
def test_config():
    """Standard test configuration"""
    return {
        "cors_origins": ["http://localhost:3000"],
        "jwt_secret": "test-secret-key-for-testing",
        "encryption_key": Fernet.generate_key(),
        "redis_url": None,  # No Redis in tests
        "log_level": "DEBUG"
    }


@pytest.fixture
def compliance_context():
    """Standard compliance context for testing"""
    return ComplianceContext(
        user_id="test-user",
        organization_id="test-org",
        session_id="test-session-123",
        request_id="test-request-456",
        timestamp=time.time(),
        ip_address="127.0.0.1",
        user_agent="test-client/1.0",
        auth_method="jwt",
        risk_score=0.1
    )


@pytest.fixture
def mcp_message():
    """Standard MCP message for testing"""
    return MCPMessage(
        id="test-msg-789",
        method="test.method",
        params={"test": "value"},
        timestamp=time.time()
    )


@pytest.fixture
def session_info():
    """Standard session info for testing"""
    return SessionInfo(
        session_id="test-session-123",
        user_id="test-user",
        created_at=datetime.now(),
        last_activity=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=1),
        auth_level=AuthenticationLevel.BASIC,
        permissions=["read", "write"],
        metadata={"client": "test"}
    )


@pytest.fixture
def valid_jwt_token(test_config):
    """Generate valid JWT token"""
    payload = {
        "sub": "test-user",
        "org": "test-org",
        "session_id": "test-session-123",
        "permissions": ["read", "write"],
        "exp": datetime.utcnow() + timedelta(hours=1)
    }
    return jwt.encode(
        payload,
        test_config["jwt_secret"],
        algorithm="HS256"
    )


@pytest.fixture
def expired_jwt_token(test_config):
    """Generate expired JWT token"""
    payload = {
        "sub": "test-user",
        "org": "test-org",
        "session_id": "test-session-123",
        "exp": datetime.utcnow() - timedelta(hours=1)
    }
    return jwt.encode(
        payload,
        test_config["jwt_secret"],
        algorithm="HS256"
    )


@pytest.fixture
def mock_standards_engine():
    """Mock standards engine for testing"""
    engine = MagicMock(spec=StandardsEngine)
    
    # Default response
    engine.load_standards = AsyncMock(
        return_value=StandardLoadResult(
            standards=[
                {"id": "CS.api", "content": "API design standards..."},
                {"id": "SEC.auth", "content": "Authentication standards..."},
                {"id": "TS.types", "content": "Type system standards..."}
            ],
            metadata={
                "query": "test query",
                "token_count": 1500,
                "version": "latest",
                "timestamp": time.time()
            }
        )
    )
    
    engine.get_catalog = MagicMock(
        return_value=["CS", "SEC", "TS", "FE", "DE", "CN", "OBS"]
    )
    
    return engine


@pytest.fixture
def mock_compliance_scanner():
    """Mock compliance scanner for testing"""
    scanner = MagicMock(spec=ComplianceScanner)
    
    # Default scan result
    scanner.scan = AsyncMock(
        return_value={
            "findings": [
                {
                    "control": "AC-3",
                    "status": "implemented",
                    "evidence": "Role-based access control found"
                },
                {
                    "control": "AU-2",
                    "status": "partial",
                    "evidence": "Some audit events missing"
                }
            ],
            "summary": {
                "total_controls": 10,
                "implemented": 5,
                "partial": 3,
                "missing": 2
            }
        }
    )
    
    return scanner


@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary data directory structure"""
    data_dir = tmp_path / "data"
    standards_dir = data_dir / "standards"
    standards_dir.mkdir(parents=True)
    
    # Create sample standards files
    cs_dir = standards_dir / "CS"
    cs_dir.mkdir()
    
    api_file = cs_dir / "api.yaml"
    api_file.write_text("""
id: CS.api
title: API Design Standards
content: |
  API design best practices for RESTful services.
tags:
  - api
  - rest
  - design
""")
    
    return data_dir


@pytest.fixture
def sample_code_snippets():
    """Sample code snippets for testing"""
    return {
        "python_auth": """
def authenticate_user(username: str, password: str) -> dict:
    \"\"\"
    Authenticate user and return JWT token
    @nist-controls: IA-2, IA-5
    @evidence: Multi-factor authentication with strong passwords
    \"\"\"
    user = db.get_user(username)
    if user and verify_password(password, user.password_hash):
        # Log successful authentication
        audit_log('auth.success', user_id=user.id)
        
        # Generate MFA challenge
        if user.mfa_enabled:
            return {"status": "mfa_required", "challenge": generate_mfa_challenge(user)}
        
        # Generate JWT token
        token = generate_jwt(user)
        return {"status": "success", "token": token}
    
    # Log failed authentication
    audit_log('auth.failed', username=username)
    return {"status": "failed"}
""",
        "javascript_api": """
// @nist-controls: AC-3, AC-4
// @evidence: Role-based access control for API endpoints
router.post('/api/users', authenticate, authorize(['admin']), async (req, res) => {
    try {
        // Validate input
        const { error } = userSchema.validate(req.body);
        if (error) {
            return res.status(400).json({ error: error.details[0].message });
        }
        
        // Create user
        const user = await createUser(req.body);
        
        // Audit log
        await auditLog({
            action: 'user.created',
            userId: req.user.id,
            targetUserId: user.id,
            ip: req.ip
        });
        
        res.status(201).json(user);
    } catch (err) {
        logger.error('Failed to create user', err);
        res.status(500).json({ error: 'Internal server error' });
    }
});
""",
        "go_encryption": """
// EncryptData encrypts sensitive data at rest
// @nist-controls: SC-8, SC-13, SC-28
// @evidence: AES-256-GCM encryption for data at rest
func EncryptData(plaintext []byte, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, fmt.Errorf("failed to create cipher: %w", err)
    }
    
    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, fmt.Errorf("failed to create GCM: %w", err)
    }
    
    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, fmt.Errorf("failed to generate nonce: %w", err)
    }
    
    ciphertext := gcm.Seal(nonce, nonce, plaintext, nil)
    
    // Log encryption operation
    log.Info("Data encrypted", "size", len(plaintext))
    
    return ciphertext, nil
}
"""
    }


@pytest.fixture
def mock_redis_client():
    """Mock Redis client for testing"""
    client = MagicMock()
    client.get = MagicMock(return_value=None)
    client.set = MagicMock(return_value=True)
    client.setex = MagicMock(return_value=True)
    client.delete = MagicMock(return_value=1)
    client.exists = MagicMock(return_value=0)
    client.expire = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_logger():
    """Mock logger for testing"""
    logger = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger


# Markers for test categorization
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take more than 1 second"
    )
    config.addinivalue_line(
        "markers", "security: Security-related tests"
    )