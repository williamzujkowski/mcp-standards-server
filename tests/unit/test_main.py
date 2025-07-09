"""
Unit tests for the main entry point and CombinedServer.

Tests the application startup, configuration loading,
and server orchestration.
"""

import pytest
import asyncio
import os
from unittest.mock import Mock, AsyncMock, patch, call, MagicMock
from typing import Dict, Any

from src.main import CombinedServer


class TestCombinedServer:
    """Test cases for CombinedServer."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "mcp": {
                "host": "localhost",
                "port": 3000,
                "auth": {"enabled": False}
            },
            "http": {
                "host": "localhost",
                "port": 8080
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            }
        }
    
    @pytest.fixture
    def server(self, config):
        """Create combined server instance."""
        with patch('src.main.AsyncMCPServer'), \
             patch('src.main.HTTPServer'), \
             patch('src.main.initialize_performance_monitor'):
            return CombinedServer(config)
    
    def test_server_initialization(self, config):
        """Test server initialization."""
        with patch('src.main.AsyncMCPServer') as mock_mcp, \
             patch('src.main.HTTPServer') as mock_http, \
             patch('src.main.initialize_performance_monitor'):
            
            server = CombinedServer(config)
            
            assert server.config == config
            assert server.running is False
            assert server.mcp_server is None
            assert server.http_server is None
            assert server.http_runner is None
    
    async def test_start_http_server(self, server):
        """Test starting HTTP server."""
        mock_http_server = Mock()
        mock_runner = AsyncMock()
        mock_http_server.start = AsyncMock(return_value=mock_runner)
        
        with patch('src.main.HTTPServer', return_value=mock_http_server):
            await server.start_http_server()
            
            assert server.http_server == mock_http_server
            assert server.http_runner == mock_runner
            mock_http_server.start.assert_called_once()
    
    async def test_start_mcp_server(self, server):
        """Test starting MCP server."""
        mock_mcp_server = Mock()
        mock_mcp_server.start = AsyncMock()
        
        with patch('src.main.AsyncMCPServer', return_value=mock_mcp_server):
            await server.start_mcp_server()
            
            assert server.mcp_server == mock_mcp_server
            mock_mcp_server.start.assert_called_once()
    
    async def test_start_both_servers(self, server):
        """Test starting both servers."""
        with patch.object(server, 'start_http_server') as mock_start_http, \
             patch.object(server, 'start_mcp_server') as mock_start_mcp:
            
            mock_start_http.return_value = AsyncMock()
            mock_start_mcp.return_value = AsyncMock()
            
            await server.start()
            
            mock_start_http.assert_called_once()
            mock_start_mcp.assert_called_once()
            assert server.running is True
    
    async def test_start_http_only(self, server):
        """Test starting only HTTP server."""
        with patch.dict(os.environ, {"HTTP_ONLY": "true"}):
            with patch.object(server, 'start_http_server') as mock_start_http, \
                 patch.object(server, 'start_mcp_server') as mock_start_mcp:
                
                mock_start_http.return_value = AsyncMock()
                
                await server.start()
                
                mock_start_http.assert_called_once()
                mock_start_mcp.assert_not_called()
                assert server.running is True
    
    async def test_shutdown(self, server):
        """Test server shutdown."""
        # Mock server state
        server.running = True
        server.http_runner = AsyncMock()
        server.http_runner.cleanup = AsyncMock()
        server.mcp_server = AsyncMock()
        server.mcp_server.stop = AsyncMock()
        
        with patch('src.main.shutdown_performance_monitor') as mock_shutdown_perf:
            mock_shutdown_perf.return_value = AsyncMock()
            
            await server.shutdown()
            
            assert server.running is False
            server.http_runner.cleanup.assert_called_once()
            server.mcp_server.stop.assert_called_once()
            mock_shutdown_perf.assert_called_once()
    
    async def test_signal_handler(self, server):
        """Test signal handler."""
        server.running = True
        
        with patch.object(server, 'shutdown') as mock_shutdown:
            mock_shutdown.return_value = AsyncMock()
            
            await server.signal_handler()
            
            mock_shutdown.assert_called_once()
    
    def test_get_status(self, server):
        """Test getting server status."""
        server.running = True
        server.http_server = Mock()
        server.mcp_server = Mock()
        
        status = server.get_status()
        
        assert status["running"] is True
        assert status["servers"]["http"] is True
        assert status["servers"]["mcp"] is True
        assert "uptime" in status


class TestSignalHandling:
    """Test signal handling."""
    
    def test_signal_handler_setup(self, server):
        """Test signal handler setup."""
        with patch('signal.signal') as mock_signal:
            server.setup_signal_handlers()
            
            # Check SIGINT and SIGTERM are handled
            calls = mock_signal.call_args_list
            assert len(calls) >= 2
            assert any(call[0][0] == signal.SIGINT for call in calls)
            assert any(call[0][0] == signal.SIGTERM for call in calls)
    
    def test_signal_handler_execution(self, server):
        """Test signal handler execution."""
        server.running = True
        
        # Get the signal handler
        with patch('signal.signal') as mock_signal:
            server.setup_signal_handlers()
            handler = mock_signal.call_args_list[0][0][1]
        
        # Execute the handler
        handler(signal.SIGINT, None)
        
        assert server.running is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])