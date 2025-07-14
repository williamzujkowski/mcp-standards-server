"""
Unit tests for the main entry point and CombinedServer.

Tests the application startup, configuration loading,
and server orchestration.
"""

import os
import signal
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.main import CombinedServer


class TestCombinedServer:
    """Test cases for CombinedServer."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "host": "localhost",
            "port": 3000,
            "auth": {"enabled": False},
        }

    @pytest.fixture
    def server(self, config):
        """Create combined server instance."""
        with (
            patch("src.main.MCPStandardsServer"),
            patch("src.main.start_http_server"),
            patch("src.main.init_logging"),
        ):
            return CombinedServer(config)

    def test_server_initialization(self, config):
        """Test server initialization."""
        with (
            patch("src.main.MCPStandardsServer"),
            patch("src.main.start_http_server"),
            patch("src.main.init_logging"),
        ):

            server = CombinedServer(config)

            assert server.mcp_config == config
            assert server.running is False
            assert server.mcp_server is None
            assert server.http_runner is None

    async def test_start_http_server(self, server):
        """Test starting HTTP server."""
        mock_runner = AsyncMock()

        with patch(
            "src.main.start_http_server", return_value=mock_runner
        ) as mock_start:
            await server.start_http_server()

            assert server.http_runner == mock_runner
            mock_start.assert_called_once_with("127.0.0.1", 8080)

    async def test_start_mcp_server(self, server):
        """Test starting MCP server."""
        mock_mcp_server = Mock()
        mock_mcp_server.run = AsyncMock()

        with patch("src.main.MCPStandardsServer", return_value=mock_mcp_server):
            await server.start_mcp_server()

            assert server.mcp_server == mock_mcp_server
            mock_mcp_server.run.assert_called_once()

    async def test_run_both_servers(self, server):
        """Test running both servers."""
        with (
            patch.object(server, "start_http_server") as mock_start_http,
            patch.object(server, "start_mcp_server") as mock_start_mcp,
            patch.object(server, "shutdown") as mock_shutdown,
        ):

            mock_start_http.return_value = AsyncMock()
            mock_start_mcp.return_value = AsyncMock()
            mock_shutdown.return_value = AsyncMock()

            await server.run()

            mock_start_http.assert_called_once()
            mock_start_mcp.assert_called_once()
            mock_shutdown.assert_called_once()

    async def test_run_http_only(self, server):
        """Test running only HTTP server."""
        with patch.dict(os.environ, {"HTTP_ONLY": "true"}):
            with (
                patch.object(server, "start_http_server") as mock_start_http,
                patch.object(server, "start_mcp_server") as mock_start_mcp,
                patch.object(server, "shutdown") as mock_shutdown,
                patch("asyncio.sleep", side_effect=KeyboardInterrupt),
            ):

                mock_start_http.return_value = AsyncMock()
                mock_shutdown.return_value = AsyncMock()

                await server.run()

                mock_start_http.assert_called_once()
                mock_start_mcp.assert_not_called()
                mock_shutdown.assert_called_once()

    async def test_shutdown(self, server):
        """Test server shutdown."""
        # Mock server state
        server.running = True
        server.http_runner = AsyncMock()
        server.http_runner.cleanup = AsyncMock()
        server.mcp_server = AsyncMock()

        # Store reference to runner before shutdown sets it to None
        http_runner_mock = server.http_runner

        await server.shutdown()

        # Verify cleanup was called and runner was set to None
        http_runner_mock.cleanup.assert_called_once()
        assert server.http_runner is None

    def test_signal_handler_functionality(self, server):
        """Test signal handler functionality."""
        server.running = True

        # Get the signal handler function
        with patch("signal.signal") as mock_signal:
            server.setup_signal_handlers()
            handler = mock_signal.call_args_list[0][0][1]

        # Execute the handler
        handler(signal.SIGINT, None)

        assert server.running is False

    def test_server_attributes(self, server):
        """Test server attributes after initialization."""
        assert hasattr(server, "mcp_config")
        assert hasattr(server, "mcp_server")
        assert hasattr(server, "http_runner")
        assert hasattr(server, "running")
        assert hasattr(server, "start_time")


class TestSignalHandling:
    """Test signal handling."""

    def test_signal_handler_setup(self):
        """Test signal handler setup."""
        with (
            patch("src.main.MCPStandardsServer"),
            patch("src.main.start_http_server"),
            patch("src.main.init_logging"),
            patch("signal.signal") as mock_signal,
        ):
            CombinedServer({})

            # Check SIGINT and SIGTERM are handled
            calls = mock_signal.call_args_list
            assert len(calls) >= 2
            assert any(call[0][0] == signal.SIGINT for call in calls)
            assert any(call[0][0] == signal.SIGTERM for call in calls)

    def test_signal_handler_execution(self):
        """Test signal handler execution."""
        with (
            patch("src.main.MCPStandardsServer"),
            patch("src.main.start_http_server"),
            patch("src.main.init_logging"),
        ):
            server = CombinedServer({})
            server.running = True

            # Get the signal handler
            with patch("signal.signal") as mock_signal:
                server.setup_signal_handlers()
                handler = mock_signal.call_args_list[0][0][1]

            # Execute the handler
            handler(signal.SIGINT, None)

            assert server.running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
