"""
MCP Protocol Data Models
@nist-controls: AC-4, AU-2, AU-3
@evidence: Data validation and audit trail models
"""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class MCPMessage(BaseModel):
    """
    Base MCP message structure
    @nist-controls: SI-10
    @evidence: Input validation for all MCP messages
    """
    id: str = Field(..., description="Unique message ID", min_length=1, max_length=128)
    method: str = Field(..., description="RPC method name", min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    timestamp: float = Field(..., description="Unix timestamp")

    @field_validator('method')
    @classmethod
    def validate_method(cls, v: str) -> str:
        """Validate method name format"""
        if not v.replace('_', '').replace('.', '').isalnum():
            raise ValueError('Method name must be alphanumeric with underscores or dots')
        return v

    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: float) -> float:
        """Ensure timestamp is reasonable"""
        now = datetime.now().timestamp()
        if v > now + 300:  # 5 minutes in future
            raise ValueError('Timestamp cannot be more than 5 minutes in the future')
        if v < now - 86400:  # 24 hours in past
            raise ValueError('Timestamp cannot be more than 24 hours in the past')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "msg_123",
                "method": "load_standards",
                "params": {"query": "CS:api + SEC:*"},
                "timestamp": 1234567890.123
            }
        }
    }


class MCPResponse(BaseModel):
    """
    MCP response structure
    @nist-controls: AU-10
    @evidence: Non-repudiation through response tracking
    """
    id: str = Field(..., description="Message ID this responds to")
    result: dict[str, Any] | None = Field(None, description="Success result")
    error: dict[str, Any] | None = Field(None, description="Error details")
    timestamp: float = Field(..., description="Response timestamp")

    @field_validator('error')
    @classmethod
    def validate_error_or_result(cls, v: dict[str, Any] | None, info: Any) -> dict[str, Any] | None:
        """Ensure either result or error is present, not both"""
        if v is not None and info.data.get('result') is not None:
            raise ValueError('Cannot have both result and error')
        if v is None and info.data.get('result') is None:
            raise ValueError('Must have either result or error')
        return v


@dataclass
class ComplianceContext:
    """
    Tracks compliance context for requests
    @nist-controls: AU-2, AU-3, AU-12
    @evidence: Comprehensive audit logging context
    """
    user_id: str
    organization_id: str
    session_id: str
    request_id: str
    timestamp: float
    ip_address: str
    user_agent: str
    auth_method: str = "jwt"
    risk_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "user_id": self.user_id,
            "organization_id": self.organization_id,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "auth_method": self.auth_method,
            "risk_score": self.risk_score
        }


class AuthenticationLevel(Enum):
    """
    Authentication strength levels
    @nist-controls: IA-2, IA-8
    @evidence: Multi-factor authentication levels
    """
    NONE = "none"
    BASIC = "basic"
    MFA = "mfa"
    CERTIFICATE = "certificate"
    FEDERATED = "federated"


@dataclass
class SessionInfo:
    """
    Session management information
    @nist-controls: AC-12, SC-10
    @evidence: Session timeout and management
    """
    session_id: str
    user_id: str
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    auth_level: AuthenticationLevel
    permissions: list[str]
    metadata: dict[str, Any]

    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.now() > self.expires_at

    def is_idle_timeout(self, idle_minutes: int = 30) -> bool:
        """Check if session has been idle too long"""
        idle_time = datetime.now() - self.last_activity
        return idle_time.total_seconds() > (idle_minutes * 60)


class MCPError(BaseModel):
    """
    Standardized error response
    @nist-controls: SI-11
    @evidence: Secure error handling without information disclosure
    """
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: dict[str, Any] | None = Field(None, description="Additional error context")

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        """Remove sensitive information from error messages"""
        # Remove file paths
        import re
        v = re.sub(r'(/[a-zA-Z0-9_\-./]+)+', '[PATH]', v)
        # Remove potential secrets
        v = re.sub(r'(password|token|key|secret)=\S+', '[REDACTED]', v, flags=re.IGNORECASE)
        return v
