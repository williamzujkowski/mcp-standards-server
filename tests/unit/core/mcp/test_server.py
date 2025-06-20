"""
Test Core MCP Server Implementation
@nist-controls: SA-11, CA-7
@evidence: Unit tests for MCP protocol server
"""

import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import jwt
import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

from src.core.mcp.handlers import MCPHandler
from src.core.mcp.models import (
    AuthenticationLevel,
    ComplianceContext,
    MCPMessage,
    SessionInfo,
)
from src.core.mcp.server import MCPServer, create_app


class MockHandler(MCPHandler):
    """Mock handler for testing"""

    required_permissions = ["test.permission"]

    async def handle(self, message: MCPMessage, context: ComplianceContext) -> dict[str, Any]:
        return {"result": "success", "method": message.method}


class TestMCPServer:
    """Test MCP Server core functionality"""

    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {
            "cors_origins": ["http://localhost:3000"],
            "jwt_secret": "test-secret-key",
            "encryption_key": Fernet.generate_key()
        }

    @pytest.fixture
    def server(self, config):
        """Create test server instance"""
        return MCPServer(config)

    @pytest.fixture
    def client(self, server):
        """Create test client"""
        return TestClient(server.app)

    @pytest.fixture
    def valid_token(self, config):
        """Generate valid JWT token"""
        payload = {
            "sub": "test-user",
            "org": "test-org",
            "session_id": "test-session",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, config["jwt_secret"], algorithm="HS256")

    @pytest.fixture
    def expired_token(self, config):
        """Generate expired JWT token"""
        payload = {
            "sub": "test-user",
            "org": "test-org",
            "session_id": "test-session",
            "exp": datetime.utcnow() - timedelta(hours=1)
        }
        return jwt.encode(payload, config["jwt_secret"], algorithm="HS256")

    def test_server_initialization(self, server):
        """Test server initializes correctly"""
        assert server.handler_registry is not None
        assert server.sessions == {}
        assert server.cipher is not None
        assert server.jwt_secret == "test-secret-key"

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == "0.1.0"

    @pytest.mark.asyncio
    async def test_authenticate_valid_token(self, server, valid_token):
        """Test authentication with valid token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=valid_token
        )

        with patch('src.core.mcp.server.log_security_event') as mock_log:
            context = await server.authenticate(credentials)

            assert context.user_id == "test-user"
            assert context.organization_id == "test-org"
            assert context.session_id == "test-session"
            assert context.auth_method == "jwt"

            # Verify security event was logged
            mock_log.assert_called_once()
            call_args = mock_log.call_args
            assert call_args[0][1] == "authentication.success"

    @pytest.mark.asyncio
    async def test_authenticate_expired_token(self, server, expired_token):
        """Test authentication with expired token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=expired_token
        )

        with patch('src.core.mcp.server.log_security_event'):
            with pytest.raises(HTTPException) as exc_info:
                await server.authenticate(credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Token expired"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_token(self, server):
        """Test authentication with invalid token"""
        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid-token"
        )

        with patch('src.core.mcp.server.log_security_event'):
            with pytest.raises(HTTPException) as exc_info:
                await server.authenticate(credentials)

            assert exc_info.value.status_code == 401
            assert exc_info.value.detail == "Invalid token"

    def test_register_handler(self, server):
        """Test handler registration"""
        handler = MockHandler()
        server.register_handler("test.method", handler)

        assert server.handler_registry.get_handler("test.method") == handler

    @pytest.mark.asyncio
    async def test_handle_message_success(self, server):
        """Test successful message handling"""
        # Register handler
        handler = MockHandler()
        handler.check_permissions = MagicMock(return_value=True)
        server.register_handler("test.method", handler)

        # Create message and context
        message = MCPMessage(
            id="test-123",
            method="test.method",
            params={},
            timestamp=time.time()
        )

        context = ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

        with patch('src.core.mcp.server.log_security_event'):
            result = await server.handle_message(message, context)

            assert result["result"] == "success"
            assert result["method"] == "test.method"

    @pytest.mark.asyncio
    async def test_handle_message_unknown_method(self, server):
        """Test handling unknown method"""
        message = MCPMessage(
            id="test-123",
            method="unknown.method",
            params={},
            timestamp=time.time()
        )

        context = ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

        with patch('src.core.mcp.server.log_security_event'):
            with pytest.raises(ValueError) as exc_info:
                await server.handle_message(message, context)

            assert "Unknown method: unknown.method" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_handle_message_permission_denied(self, server):
        """Test handling with insufficient permissions"""
        handler = MockHandler()
        handler.check_permissions = MagicMock(return_value=False)
        server.register_handler("test.method", handler)

        message = MCPMessage(
            id="test-123",
            method="test.method",
            params={},
            timestamp=time.time()
        )

        context = ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

        with patch('src.core.mcp.server.log_security_event'):
            with pytest.raises(PermissionError) as exc_info:
                await server.handle_message(message, context)

            assert "Insufficient permissions" in str(exc_info.value)

    def test_list_methods_requires_auth(self, client):
        """Test list methods endpoint requires authentication"""
        response = client.get("/api/methods")
        assert response.status_code == 403  # No auth header

    def test_list_methods_with_auth(self, client, valid_token):
        """Test list methods endpoint with authentication"""
        headers = {"Authorization": f"Bearer {valid_token}"}
        response = client.get("/api/methods", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_session_management(self, server):
        """Test session creation and management"""
        session = SessionInfo(
            session_id="test-session",
            user_id="test-user",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )

        server.sessions[session.session_id] = session
        assert len(server.sessions) == 1
        assert server.sessions["test-session"] == session

    @pytest.mark.asyncio
    async def test_session_cleanup(self, server):
        """Test expired session cleanup"""
        # Create expired session
        expired_session = SessionInfo(
            session_id="expired-session",
            user_id="test-user",
            created_at=datetime.now() - timedelta(hours=2),
            last_activity=datetime.now() - timedelta(hours=2),
            expires_at=datetime.now() - timedelta(hours=1),
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )

        # Create valid session
        valid_session = SessionInfo(
            session_id="valid-session",
            user_id="test-user",
            created_at=datetime.now(),
            last_activity=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1),
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )

        server.sessions["expired-session"] = expired_session
        server.sessions["valid-session"] = valid_session

        # Manually trigger cleanup
        expired_sessions = []
        for session_id, session in server.sessions.items():
            if session.is_expired():
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del server.sessions[session_id]

        assert len(server.sessions) == 1
        assert "valid-session" in server.sessions
        assert "expired-session" not in server.sessions

    def test_security_headers(self, client):
        """Test security headers are set correctly"""
        response = client.get("/health")

        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["X-XSS-Protection"] == "1; mode=block"
        assert response.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"

    def test_create_app_with_default_config(self):
        """Test app creation with default config"""
        app = create_app()
        assert app is not None
        assert app.title == "MCP Standards Server"

    def test_create_app_with_custom_config(self):
        """Test app creation with custom config"""
        custom_config = {
            "cors_origins": ["http://example.com"],
            "jwt_secret": "custom-secret",
            "encryption_key": Fernet.generate_key()
        }

        app = create_app(custom_config)
        assert app is not None
        assert app.title == "MCP Standards Server"


class TestWebSocketEndpoint:
    """Test WebSocket MCP endpoint"""

    @pytest.fixture
    def server(self):
        """Create test server"""
        config = {
            "cors_origins": ["http://localhost:3000"],
            "jwt_secret": "test-secret-key",
            "encryption_key": Fernet.generate_key()
        }
        return MCPServer(config)

    @pytest.fixture
    def valid_token(self):
        """Generate valid JWT token"""
        payload = {
            "sub": "test-user",
            "org": "test-org",
            "session_id": "test-session",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        return jwt.encode(payload, "test-secret-key", algorithm="HS256")

    @pytest.mark.asyncio
    async def test_websocket_missing_auth(self, server):
        """Test WebSocket connection without authentication"""
        client = TestClient(server.app)

        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect("/mcp") as websocket:
                # Should disconnect immediately
                websocket.receive_text()

    @pytest.mark.asyncio
    async def test_websocket_with_auth(self, server, valid_token):
        """Test WebSocket connection with authentication"""
        # Register a test handler
        handler = MockHandler()
        handler.check_permissions = MagicMock(return_value=True)
        server.register_handler("test.method", handler)

        client = TestClient(server.app)

        with patch('src.core.mcp.server.log_security_event'), \
             client.websocket_connect(f"/mcp?token={valid_token}") as websocket:
                # Send test message
                message = MCPMessage(
                    id="test-123",
                    method="test.method",
                    params={},
                    timestamp=time.time()
                )

                websocket.send_text(message.json())

                # Receive response
                response_data = websocket.receive_json()

                assert response_data["id"] == "test-123"
                assert response_data["result"]["result"] == "success"
                assert response_data["error"] is None

    @pytest.mark.asyncio
    async def test_websocket_error_handling(self, server, valid_token):
        """Test WebSocket error handling"""
        client = TestClient(server.app)

        with patch('src.core.mcp.server.log_security_event'), \
             client.websocket_connect(f"/mcp?token={valid_token}") as websocket:
                # Send invalid message
                websocket.send_text("invalid json")

                # Receive error response
                response_data = websocket.receive_json()

                assert response_data["result"] is None
                assert response_data["error"] is not None
                assert "code" in response_data["error"]
                assert "message" in response_data["error"]
