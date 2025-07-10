"""
Authentication and authorization module for MCP server.

Implements JWT-based authentication with support for API keys.
"""

import os
import time
from datetime import datetime, timedelta, timezone

import jwt
from pydantic import BaseModel, Field


class AuthConfig(BaseModel):
    """Authentication configuration."""

    enabled: bool = Field(default=False, description="Enable authentication")
    secret_key: str = Field(default="", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    token_expiry_hours: int = Field(default=24, description="Token expiry in hours")
    api_keys: dict[str, str] = Field(
        default_factory=dict, description="API key to user mapping"
    )


class TokenPayload(BaseModel):
    """JWT token payload structure."""

    sub: str  # Subject (user ID)
    exp: int  # Expiration timestamp
    iat: int  # Issued at timestamp
    jti: str  # JWT ID for revocation
    scope: str = "mcp:tools"  # Token scope


class AuthManager:
    """Manages authentication and authorization for MCP server."""

    def __init__(self, config: AuthConfig | None = None) -> None:
        """Initialize auth manager with configuration."""
        self.config = config or AuthConfig()

        # Use environment variables if available
        if os.getenv("MCP_AUTH_ENABLED", "false").lower() == "true":
            self.config.enabled = True

        jwt_secret = os.getenv("MCP_JWT_SECRET")
        if jwt_secret:
            self.config.secret_key = jwt_secret
        elif self.config.enabled and not self.config.secret_key:
            # Generate a random secret key if auth is enabled but no key provided
            import secrets

            self.config.secret_key = secrets.token_urlsafe(32)

        # Track revoked tokens (in production, use Redis)
        self._revoked_tokens: set = set()

    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self.config.enabled

    def generate_token(self, user_id: str, scope: str = "mcp:tools") -> str:
        """Generate a JWT token for a user."""
        if not self.config.enabled:
            raise RuntimeError("Authentication is not enabled")

        now = datetime.now(timezone.utc)
        exp = now + timedelta(hours=self.config.token_expiry_hours)

        payload = TokenPayload(
            sub=user_id,
            exp=int(exp.timestamp()),
            iat=int(now.timestamp()),
            jti=f"{user_id}:{int(now.timestamp())}",
            scope=scope,
        )

        token = jwt.encode(
            payload.model_dump(),
            self.config.secret_key,
            algorithm=self.config.algorithm,
        )

        return str(token)

    def verify_token(self, token: str) -> tuple[bool, TokenPayload | None, str | None]:
        """
        Verify a JWT token.

        Returns:
            Tuple of (is_valid, payload, error_message)
        """
        if not self.config.enabled:
            return True, None, None  # Auth disabled, allow all

        try:
            # Decode and verify token
            payload_dict = jwt.decode(
                token, self.config.secret_key, algorithms=[self.config.algorithm]
            )

            payload = TokenPayload(**payload_dict)

            # Check if token is revoked
            if payload.jti in self._revoked_tokens:
                return False, None, "Token has been revoked"

            # Check expiration (jwt.decode handles this, but being explicit)
            if payload.exp < time.time():
                return False, None, "Token has expired"

            return True, payload, None

        except jwt.ExpiredSignatureError:
            return False, None, "Token has expired"
        except jwt.InvalidTokenError as e:
            return False, None, f"Invalid token: {str(e)}"
        except Exception as e:
            return False, None, f"Token verification failed: {str(e)}"

    def verify_api_key(self, api_key: str) -> tuple[bool, str | None, str | None]:
        """
        Verify an API key.

        Returns:
            Tuple of (is_valid, user_id, error_message)
        """
        if not self.config.enabled:
            return True, "anonymous", None

        if api_key in self.config.api_keys:
            return True, self.config.api_keys[api_key], None

        return False, None, "Invalid API key"

    def revoke_token(self, jti: str) -> None:
        """Revoke a token by its JTI."""
        self._revoked_tokens.add(jti)

    def extract_auth_from_headers(
        self, headers: dict[str, str]
    ) -> tuple[str | None, str | None]:
        """
        Extract authentication credentials from headers.

        Returns:
            Tuple of (auth_type, credential)
        """
        auth_header = headers.get("Authorization", "")

        if auth_header.startswith("Bearer "):
            return "bearer", auth_header[7:]
        elif auth_header.startswith("ApiKey "):
            return "api_key", auth_header[7:]
        elif "X-API-Key" in headers:
            return "api_key", headers["X-API-Key"]

        return None, None

    def check_permission(
        self, payload: TokenPayload | None, required_scope: str
    ) -> bool:
        """Check if a token payload has the required scope."""
        if not self.config.enabled:
            return True

        if not payload:
            return False

        # Simple scope checking - in production, use more sophisticated logic
        return required_scope in payload.scope


# Singleton instance
_auth_manager: AuthManager | None = None


def get_auth_manager() -> AuthManager:
    """Get the singleton auth manager instance."""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
