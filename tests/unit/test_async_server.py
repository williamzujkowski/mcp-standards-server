"""
Unit tests for the Async MCP Server implementation.

Tests the async server functionality including WebSocket handling,
message routing, and connection management.
"""

import json
import sys
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

# Mock aioredis to avoid import errors in Python 3.12
sys.modules["aioredis"] = MagicMock()

# Import after mock setup
from src.core.mcp.async_server import (  # noqa: E402
    AsyncMCPServer,
    MCPSession,
    ServerConfig,
)


class TestAsyncMCPServer:
    """Test cases for Async MCP Server."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ServerConfig(
            host="localhost",
            port=3000,
            max_connections=100,
            enable_authentication=False,
        )

    @pytest.fixture
    def server(self, config):
        """Create async server instance."""
        server = AsyncMCPServer(config)
        return server

    def test_server_initialization(self, config):
        """Test server initialization."""
        server = AsyncMCPServer(config)

        assert server.config == config
        assert hasattr(server, "connection_manager")
        assert hasattr(server, "request_batcher")
        assert hasattr(server, "metrics")

    async def test_start_server(self, server):
        """Test server startup."""
        with patch("aiohttp.web.AppRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner_class.return_value = mock_runner

            with patch("aiohttp.web.TCPSite") as mock_site_class:
                mock_site = AsyncMock()
                mock_site_class.return_value = mock_site

                await server.start()

                assert server.running is True
                mock_runner.setup.assert_called_once()
                mock_site.start.assert_called_once()

    async def test_stop_server(self, server):
        """Test server shutdown."""
        # Mock server state
        server.running = True
        server.server = AsyncMock()

        # Add mock sessions
        mock_session = Mock(spec=MCPSession)
        mock_session.close = AsyncMock()
        server.sessions = {"session1": mock_session}

        await server.stop()

        assert server.running is False
        mock_session.close.assert_called_once()
        assert len(server.sessions) == 0

    async def test_handle_client_connection(self, server):
        """Test handling new client connections."""
        # Mock reader and writer
        mock_reader = AsyncMock()
        mock_writer = AsyncMock()
        mock_writer.get_extra_info = Mock(return_value=("127.0.0.1", 12345))

        # Mock session creation
        with patch.object(server, "_create_session") as mock_create_session:
            mock_session = Mock(spec=MCPSession)
            mock_session.handle = AsyncMock()
            mock_create_session.return_value = mock_session

            await server._handle_client(mock_reader, mock_writer)

            mock_create_session.assert_called_once()
            mock_session.handle.assert_called_once()

    def test_create_session(self, server):
        """Test session creation."""
        mock_reader = Mock()
        mock_writer = Mock()
        mock_writer.get_extra_info = Mock(return_value=("127.0.0.1", 12345))

        session = server._create_session(mock_reader, mock_writer)

        assert isinstance(session, MCPSession)
        assert session.id in server.sessions
        assert server.sessions[session.id] == session

    async def test_remove_session(self, server):
        """Test session removal."""
        # Create mock session
        session_id = "test-session"
        mock_session = Mock(spec=MCPSession)
        mock_session.id = session_id
        server.sessions[session_id] = mock_session

        await server._remove_session(session_id)

        assert session_id not in server.sessions

    async def test_broadcast_message(self, server):
        """Test broadcasting message to all sessions."""
        # Create mock sessions
        mock_session1 = Mock(spec=MCPSession)
        mock_session1.send_message = AsyncMock()
        mock_session2 = Mock(spec=MCPSession)
        mock_session2.send_message = AsyncMock()

        server.sessions = {"session1": mock_session1, "session2": mock_session2}

        message = {"type": "broadcast", "data": "test"}
        await server.broadcast_message(message)

        mock_session1.send_message.assert_called_once_with(message)
        mock_session2.send_message.assert_called_once_with(message)

    async def test_get_server_stats(self, server):
        """Test getting server statistics."""
        # Mock sessions and connection manager
        server.sessions = {"s1": Mock(), "s2": Mock()}
        server.running = True

        # Mock the connection manager to return expected values
        server.connection_manager.connections = {"c1": Mock(), "c2": Mock()}

        stats = server.get_stats()

        assert stats["running"] is True
        assert stats["sessions"] == 2
        assert stats["active_connections"] == 2


class TestMCPSession:
    """Test cases for MCP Session."""

    @pytest.fixture
    def mock_server(self):
        """Create mock server."""
        server = Mock(spec=AsyncMCPServer)
        server.mcp_server = Mock()
        server.mcp_server._execute_tool = AsyncMock(return_value={"standards": []})
        server.config = {"message_timeout": 60}
        server._remove_session = AsyncMock()
        return server

    @pytest.fixture
    def mock_reader(self):
        """Create mock reader."""
        reader = AsyncMock()
        reader.read = AsyncMock()
        return reader

    @pytest.fixture
    def mock_writer(self):
        """Create mock writer."""
        writer = AsyncMock()
        writer.write = Mock()
        writer.drain = AsyncMock()
        writer.close = Mock()
        writer.wait_closed = AsyncMock()
        writer.is_closing = Mock(return_value=False)  # Return boolean, not AsyncMock
        writer.get_extra_info = Mock(return_value=("127.0.0.1", 12345))
        return writer

    @pytest.fixture
    def session(self, mock_server, mock_reader, mock_writer):
        """Create session instance."""
        return MCPSession("test-session", mock_server, mock_reader, mock_writer)

    def test_session_initialization(self, mock_server, mock_reader, mock_writer):
        """Test session initialization."""
        session = MCPSession("test-id", mock_server, mock_reader, mock_writer)

        assert session.id == "test-id"
        assert session.server == mock_server
        assert session.reader == mock_reader
        assert session.writer == mock_writer
        assert session.authenticated is False
        assert session.client_info["address"] == ("127.0.0.1", 12345)

    async def test_handle_session(self, session):
        """Test session message handling loop."""
        # Mock message reading
        messages = [
            json.dumps({"type": "hello", "version": "1.0"}).encode() + b"\n",
            json.dumps({"type": "request", "method": "test"}).encode() + b"\n",
            b"",  # EOF
        ]
        session.reader.readline = AsyncMock(side_effect=messages)

        with patch.object(session, "_process_message") as mock_process:
            mock_process.return_value = AsyncMock()

            await session.handle()

            assert mock_process.call_count == 2

    async def test_send_message(self, session):
        """Test sending message to client."""
        message = {"type": "response", "data": "test"}

        await session.send_message(message)

        expected_data = json.dumps(message).encode() + b"\n"
        session.writer.write.assert_called_once_with(expected_data)
        session.writer.drain.assert_called_once()

    async def test_process_hello_message(self, session):
        """Test processing hello message."""
        message = {
            "type": "hello",
            "version": "1.0",
            "capabilities": ["tools", "resources"],
        }

        with patch.object(session, "send_message") as mock_send:
            await session._process_message(message)

            # Should send hello response
            mock_send.assert_called_once()
            response = mock_send.call_args[0][0]
            assert response["type"] == "hello"
            assert "version" in response

    async def test_process_request_message(self, session):
        """Test processing request message."""
        message = {
            "type": "request",
            "id": "req-123",
            "method": "get_applicable_standards",
            "params": {
                "project_context": {"languages": ["python"]},
                "requirements": ["testing"],
            },
        }

        # Mock tool execution
        session.server.mcp_server._get_applicable_standards = AsyncMock(
            return_value={"standards": []}
        )

        with patch.object(session, "send_message") as mock_send:
            await session._process_message(message)

            # Should send response
            mock_send.assert_called_once()
            response = mock_send.call_args[0][0]
            assert response["type"] == "response"
            assert response["id"] == "req-123"

    async def test_process_invalid_message(self, session):
        """Test processing invalid message."""
        message = {"invalid": "message"}

        with patch.object(session, "send_message") as mock_send:
            await session._process_message(message)

            # Should send error response
            mock_send.assert_called_once()
            response = mock_send.call_args[0][0]
            assert response["type"] == "error"

    async def test_close_session(self, session):
        """Test closing session."""
        session.server._remove_session = AsyncMock()

        await session.close()

        session.writer.close.assert_called_once()
        session.server._remove_session.assert_called_once_with("test-session")

    async def test_authentication(self, session):
        """Test session authentication."""
        # Enable auth in server config
        session.server.config["auth"] = {"enabled": True}

        # Test unauthenticated request
        message = {
            "type": "request",
            "id": "req-123",
            "method": "get_applicable_standards",
        }

        with patch.object(session, "send_message") as mock_send:
            await session._process_message(message)

            # Should send error for unauthenticated request
            response = mock_send.call_args[0][0]
            assert response["type"] == "error"
            assert "authentication" in response["error"].lower()

    async def test_heartbeat(self, session):
        """Test heartbeat handling."""
        message = {"type": "ping"}

        with patch.object(session, "send_message") as mock_send:
            await session._process_message(message)

            # Should send pong response
            mock_send.assert_called_once()
            response = mock_send.call_args[0][0]
            assert response["type"] == "pong"

    def test_get_session_info(self, session):
        """Test getting session information."""
        info = session.get_info()

        assert info["id"] == "test-session"
        assert info["authenticated"] is False
        assert info["client_info"]["address"] == ("127.0.0.1", 12345)
        assert "connected_at" in info


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
