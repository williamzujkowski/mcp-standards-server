"""
Unit tests for authentication module.
"""

import time
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from src.core.auth import AuthConfig, AuthManager, TokenPayload


class TestAuthManager:
    """Test authentication manager functionality."""

    @pytest.fixture
    def auth_config(self):
        """Create test auth configuration."""
        return AuthConfig(
            enabled=True,
            secret_key="test_secret_key_for_testing_only",
            algorithm="HS256",
            token_expiry_hours=1,
            api_keys={"test_api_key": "test_user"},
        )

    @pytest.fixture
    def auth_manager(self, auth_config):
        """Create auth manager with test config."""
        return AuthManager(auth_config)

    def test_auth_disabled_by_default(self):
        """Test that auth is disabled by default."""
        manager = AuthManager()
        assert not manager.is_enabled()

    def test_auth_enabled_via_env(self, monkeypatch):
        """Test enabling auth via environment variable."""
        monkeypatch.setenv("MCP_AUTH_ENABLED", "true")
        monkeypatch.setenv("MCP_JWT_SECRET", "env_secret")

        manager = AuthManager()
        assert manager.is_enabled()
        assert manager.config.secret_key == "env_secret"

    def test_generate_token(self, auth_manager):
        """Test JWT token generation."""
        user_id = "test_user"
        token = auth_manager.generate_token(user_id)

        assert isinstance(token, str)
        assert len(token) > 0

        # Decode to verify structure
        payload = jwt.decode(
            token,
            auth_manager.config.secret_key,
            algorithms=[auth_manager.config.algorithm],
        )

        assert payload["sub"] == user_id
        assert "exp" in payload
        assert "iat" in payload
        assert "jti" in payload
        assert payload["scope"] == "mcp:tools"

    def test_verify_valid_token(self, auth_manager):
        """Test verification of valid token."""
        user_id = "test_user"
        token = auth_manager.generate_token(user_id)

        is_valid, payload, error = auth_manager.verify_token(token)

        assert is_valid
        assert payload is not None
        assert error is None
        assert payload.sub == user_id

    def test_verify_expired_token(self, auth_manager):
        """Test verification of expired token."""
        # Create token with past expiration
        now = datetime.now(timezone.utc)
        exp = now - timedelta(hours=1)

        payload = {
            "sub": "test_user",
            "exp": int(exp.timestamp()),
            "iat": int(now.timestamp()),
            "jti": "test_jti",
            "scope": "mcp:tools",
        }

        token = jwt.encode(
            payload,
            auth_manager.config.secret_key,
            algorithm=auth_manager.config.algorithm,
        )

        is_valid, payload, error = auth_manager.verify_token(token)

        assert not is_valid
        assert payload is None
        assert "expired" in error.lower()

    def test_verify_invalid_token(self, auth_manager):
        """Test verification of invalid token."""
        is_valid, payload, error = auth_manager.verify_token("invalid.token.here")

        assert not is_valid
        assert payload is None
        assert "invalid token" in error.lower()

    def test_verify_token_wrong_secret(self, auth_manager):
        """Test verification with wrong secret."""
        token = jwt.encode(
            {"sub": "test_user", "exp": int(time.time()) + 3600},
            "wrong_secret",
            algorithm="HS256",
        )

        is_valid, payload, error = auth_manager.verify_token(token)

        assert not is_valid
        assert payload is None
        assert error is not None

    def test_revoke_token(self, auth_manager):
        """Test token revocation."""
        user_id = "test_user"
        token = auth_manager.generate_token(user_id)

        # First verification should succeed
        is_valid, payload, _ = auth_manager.verify_token(token)
        assert is_valid

        # Revoke the token
        auth_manager.revoke_token(payload.jti)

        # Second verification should fail
        is_valid, _, error = auth_manager.verify_token(token)
        assert not is_valid
        assert "revoked" in error.lower()

    def test_verify_api_key_valid(self, auth_manager):
        """Test valid API key verification."""
        is_valid, user_id, error = auth_manager.verify_api_key("test_api_key")

        assert is_valid
        assert user_id == "test_user"
        assert error is None

    def test_verify_api_key_invalid(self, auth_manager):
        """Test invalid API key verification."""
        is_valid, user_id, error = auth_manager.verify_api_key("invalid_key")

        assert not is_valid
        assert user_id is None
        assert "invalid api key" in error.lower()

    def test_extract_bearer_token(self, auth_manager):
        """Test extracting Bearer token from headers."""
        headers = {"Authorization": "Bearer test_token_123"}

        auth_type, credential = auth_manager.extract_auth_from_headers(headers)

        assert auth_type == "bearer"
        assert credential == "test_token_123"

    def test_extract_api_key_header(self, auth_manager):
        """Test extracting API key from Authorization header."""
        headers = {"Authorization": "ApiKey test_api_key"}

        auth_type, credential = auth_manager.extract_auth_from_headers(headers)

        assert auth_type == "api_key"
        assert credential == "test_api_key"

    def test_extract_api_key_x_header(self, auth_manager):
        """Test extracting API key from X-API-Key header."""
        headers = {"X-API-Key": "test_api_key"}

        auth_type, credential = auth_manager.extract_auth_from_headers(headers)

        assert auth_type == "api_key"
        assert credential == "test_api_key"

    def test_extract_no_auth(self, auth_manager):
        """Test extracting auth from headers with no auth."""
        headers = {"Content-Type": "application/json"}

        auth_type, credential = auth_manager.extract_auth_from_headers(headers)

        assert auth_type is None
        assert credential is None

    def test_check_permission_valid(self, auth_manager):
        """Test permission checking with valid scope."""
        payload = TokenPayload(
            sub="test_user",
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            jti="test_jti",
            scope="mcp:tools mcp:admin",
        )

        assert auth_manager.check_permission(payload, "mcp:tools")
        assert auth_manager.check_permission(payload, "mcp:admin")

    def test_check_permission_invalid(self, auth_manager):
        """Test permission checking with invalid scope."""
        payload = TokenPayload(
            sub="test_user",
            exp=int(time.time()) + 3600,
            iat=int(time.time()),
            jti="test_jti",
            scope="mcp:tools",
        )

        assert not auth_manager.check_permission(payload, "mcp:admin")

    def test_auth_disabled_allows_all(self):
        """Test that disabled auth allows all operations."""
        manager = AuthManager(AuthConfig(enabled=False))

        # All operations should succeed
        is_valid, _, _ = manager.verify_token("any_token")
        assert is_valid

        is_valid, _, _ = manager.verify_api_key("any_key")
        assert is_valid

        assert manager.check_permission(None, "any_scope")
