"""
Test MCP Models
@nist-controls: SA-11, CA-7
@evidence: Unit tests for data models
"""
import pytest
from datetime import datetime
from pydantic import ValidationError

from src.core.mcp.models import (
    MCPMessage, MCPResponse, ComplianceContext,
    AuthenticationLevel, SessionInfo, MCPError
)


class TestMCPMessage:
    """Test MCPMessage model"""
    
    def test_valid_message(self):
        """Test creating valid MCP message"""
        msg = MCPMessage(
            id="test_123",
            method="load_standards",
            params={"query": "CS:api"},
            timestamp=datetime.now().timestamp()
        )
        assert msg.id == "test_123"
        assert msg.method == "load_standards"
        assert msg.params["query"] == "CS:api"
    
    def test_invalid_method_name(self):
        """Test validation of method name"""
        with pytest.raises(ValidationError):
            MCPMessage(
                id="test",
                method="invalid-method!",  # Invalid characters
                params={},
                timestamp=datetime.now().timestamp()
            )
    
    def test_future_timestamp_rejected(self):
        """Test that future timestamps are rejected"""
        future_time = datetime.now().timestamp() + 3600  # 1 hour future
        with pytest.raises(ValidationError):
            MCPMessage(
                id="test",
                method="test_method",
                params={},
                timestamp=future_time
            )


class TestMCPResponse:
    """Test MCPResponse model"""
    
    def test_success_response(self):
        """Test creating success response"""
        resp = MCPResponse(
            id="msg_123",
            result={"status": "success"},
            error=None,
            timestamp=datetime.now().timestamp()
        )
        assert resp.result["status"] == "success"
        assert resp.error is None
    
    def test_error_response(self):
        """Test creating error response"""
        resp = MCPResponse(
            id="msg_123",
            result=None,
            error={"code": "ERROR", "message": "Test error"},
            timestamp=datetime.now().timestamp()
        )
        assert resp.result is None
        assert resp.error["code"] == "ERROR"
    
    def test_cannot_have_both_result_and_error(self):
        """Test that response cannot have both result and error"""
        with pytest.raises(ValidationError):
            MCPResponse(
                id="msg_123",
                result={"data": "test"},
                error={"code": "ERROR"},
                timestamp=datetime.now().timestamp()
            )


class TestComplianceContext:
    """Test ComplianceContext"""
    
    def test_context_creation(self):
        """Test creating compliance context"""
        ctx = ComplianceContext(
            user_id="user123",
            organization_id="org456",
            session_id="sess789",
            request_id="req000",
            timestamp=datetime.now().timestamp(),
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        assert ctx.user_id == "user123"
        assert ctx.auth_method == "jwt"  # Default
        assert ctx.risk_score == 0.0  # Default
    
    def test_context_to_dict(self):
        """Test converting context to dictionary"""
        ctx = ComplianceContext(
            user_id="user123",
            organization_id="org456",
            session_id="sess789",
            request_id="req000",
            timestamp=1234567890.0,
            ip_address="192.168.1.1",
            user_agent="TestAgent/1.0"
        )
        d = ctx.to_dict()
        assert d["user_id"] == "user123"
        assert d["timestamp"] == 1234567890.0
        assert "risk_score" in d


class TestSessionInfo:
    """Test SessionInfo"""
    
    def test_session_expiry(self):
        """Test session expiry check"""
        now = datetime.now()
        session = SessionInfo(
            session_id="test",
            user_id="user123",
            created_at=now,
            last_activity=now,
            expires_at=now,  # Already expired
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )
        assert session.is_expired()
    
    def test_idle_timeout(self):
        """Test idle timeout check"""
        now = datetime.now()
        old_activity = datetime(2020, 1, 1)  # Very old
        session = SessionInfo(
            session_id="test",
            user_id="user123",
            created_at=old_activity,
            last_activity=old_activity,
            expires_at=now,  # Not expired yet
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )
        assert session.is_idle_timeout(idle_minutes=30)


class TestMCPError:
    """Test MCPError model"""
    
    def test_error_message_sanitization(self):
        """Test that sensitive info is removed from errors"""
        error = MCPError(
            code="FILE_ERROR",
            message="Error reading /home/user/secret/file.txt with password=secret123",
            details=None
        )
        assert "[PATH]" in error.message
        assert "[REDACTED]" in error.message
        assert "secret123" not in error.message
        assert "/home/user" not in error.message