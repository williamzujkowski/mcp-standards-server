"""
Unit tests for the HTTP Server implementation.

Tests the HTTP server endpoints, health checks, metrics,
and API functionality.
"""

import json
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from src.http_server import HTTPServer


class TestHTTPServer(AioHTTPTestCase):
    """Test cases for HTTP Server using aiohttp test utilities."""

    async def get_application(self):
        """Create application for testing."""
        server = HTTPServer(host="127.0.0.1", port=8080)
        return server.app

    async def test_root_endpoint(self):
        """Test root endpoint returns service info."""
        resp = await self.client.request("GET", "/")
        assert resp.status == 200

        data = await resp.json()
        assert data["service"] == "MCP Standards Server"
        assert data["status"] == "running"
        assert "endpoints" in data

    async def test_status_endpoint(self):
        """Test status endpoint."""
        resp = await self.client.request("GET", "/status")
        assert resp.status == 200

        data = await resp.json()
        assert data["service"] == "mcp-standards-server"
        assert data["status"] == "running"
        assert "version" in data
        assert "environment" in data

    async def test_info_endpoint(self):
        """Test info endpoint."""
        resp = await self.client.request("GET", "/info")
        assert resp.status == 200

        data = await resp.json()
        assert data["name"] == "MCP Standards Server"
        assert "version" in data
        assert "endpoints" in data
        assert isinstance(data["endpoints"], dict)

    async def test_health_endpoint(self):
        """Test health endpoint."""
        with patch("src.core.health.get_health_checker") as mock_checker:
            mock_health = Mock()
            mock_health.check_health = AsyncMock(
                return_value={
                    "status": "healthy",
                    "checks": {},
                    "timestamp": "2024-01-01T00:00:00",
                }
            )
            mock_checker.return_value = mock_health

            resp = await self.client.request("GET", "/health")
            assert resp.status == 200

            data = await resp.json()
            assert data["status"] == "healthy"

    async def test_liveness_endpoint(self):
        """Test liveness endpoint."""
        with patch("src.core.health.liveness_check") as mock_liveness:
            mock_liveness.return_value = {"alive": True, "status": "healthy"}

            resp = await self.client.request("GET", "/health/live")
            assert resp.status == 200

            data = await resp.json()
            assert data["alive"] is True

    async def test_readiness_endpoint(self):
        """Test readiness endpoint."""
        # The endpoint might fail due to Redis not being available in test env
        resp = await self.client.request("GET", "/health/ready")

        # In test environment, readiness might return 503 if Redis is not available
        # but the response should still have the expected structure
        data = await resp.json()
        assert "ready" in data
        assert "timestamp" in data

        # If Redis is available, status should be 200
        # If not, it might be 503 but that's acceptable in test env
        assert resp.status in [200, 503]

    async def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        with patch("src.http_server.get_performance_monitor") as mock_monitor:
            mock_perf = Mock()
            mock_perf.get_prometheus_metrics = Mock(
                return_value="# HELP test\ntest_metric 1.0\n"
            )
            mock_monitor.return_value = mock_perf

            resp = await self.client.request("GET", "/metrics")
            assert resp.status == 200
            assert resp.content_type == "text/plain"

            text = await resp.text()
            assert "test_metric" in text

    async def test_list_standards_endpoint(self):
        """Test list standards API endpoint."""
        # Create a module if it doesn't exist
        if "src.core.standards" not in sys.modules:
            sys.modules["src.core.standards"] = Mock()
        if "src.core.standards.engine" not in sys.modules:
            sys.modules["src.core.standards.engine"] = Mock()

        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            # Create mock standard objects
            mock_standard = Mock()
            mock_standard.id = "test-standard"
            mock_standard.title = "Test Standard"
            mock_standard.category = "testing"
            mock_standard.description = "A test standard for testing purposes"

            mock_instance.list_standards = AsyncMock(return_value=[mock_standard])
            mock_engine.return_value = mock_instance

            resp = await self.client.request("GET", "/api/standards")
            assert resp.status == 200

            data = await resp.json()
            assert "standards" in data
            assert data["total"] == 1
            assert data["standards"][0]["id"] == "test-standard"

    async def test_get_standard_endpoint(self):
        """Test get specific standard endpoint."""
        # Create a module if it doesn't exist
        if "src.core.standards" not in sys.modules:
            sys.modules["src.core.standards"] = Mock()
        if "src.core.standards.engine" not in sys.modules:
            sys.modules["src.core.standards.engine"] = Mock()

        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            mock_instance.get_standard = AsyncMock(
                return_value={
                    "id": "test-standard",
                    "title": "Test Standard",
                    "content": "# Test Standard",
                }
            )
            mock_engine.return_value = mock_instance

            resp = await self.client.request("GET", "/api/standards/test-standard")
            assert resp.status == 200

            data = await resp.json()
            assert "standard" in data
            assert data["standard"]["id"] == "test-standard"

    async def test_get_standard_not_found(self):
        """Test get standard returns 404 for non-existent standard."""
        # Create a module if it doesn't exist
        if "src.core.standards" not in sys.modules:
            sys.modules["src.core.standards"] = Mock()
        if "src.core.standards.engine" not in sys.modules:
            sys.modules["src.core.standards.engine"] = Mock()

        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            mock_instance.get_standard = AsyncMock(return_value=None)
            mock_engine.return_value = mock_instance

            resp = await self.client.request("GET", "/api/standards/nonexistent")
            assert resp.status == 404

            data = await resp.json()
            assert "error" in data

    async def test_cors_headers(self):
        """Test CORS headers are present."""
        resp = await self.client.request("GET", "/")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "Access-Control-Allow-Methods" in resp.headers
        assert "Access-Control-Allow-Headers" in resp.headers

    async def test_options_request(self):
        """Test OPTIONS request for CORS preflight."""
        resp = await self.client.request("OPTIONS", "/health")
        assert resp.status == 200

    async def test_invalid_endpoint(self):
        """Test invalid endpoint returns 404."""
        resp = await self.client.request("GET", "/invalid/endpoint")
        # Server returns 405 for unmatched routes due to catch-all OPTIONS handler
        assert resp.status in [404, 405]


class TestHTTPServerUnit:
    """Unit tests for HTTP Server without running server."""

    @pytest.fixture
    def server(self):
        """Create HTTP server instance."""
        return HTTPServer(host="127.0.0.1", port=8080)

    def test_server_initialization(self):
        """Test server initialization."""
        server = HTTPServer(host="127.0.0.1", port=8000)

        assert server.host == "127.0.0.1"
        assert server.port == 8000
        assert server.app is not None
        assert isinstance(server.app, web.Application)

    def test_route_registration(self, server):
        """Test all routes are registered."""
        routes = [str(route.resource.canonical) for route in server.app.router.routes()]

        # Check main routes
        assert "/" in routes
        assert "/status" in routes
        assert "/info" in routes
        assert "/health" in routes
        assert "/health/live" in routes
        assert "/health/ready" in routes
        assert "/metrics" in routes
        assert "/api/standards" in routes
        assert "/api/standards/{standard_id}" in routes

    async def test_start_server(self, server):
        """Test server startup."""
        with patch("aiohttp.web.AppRunner") as mock_runner_class:
            mock_runner = AsyncMock()
            mock_runner.setup = AsyncMock()
            mock_runner_class.return_value = mock_runner

            with patch("aiohttp.web.TCPSite") as mock_site_class:
                mock_site = AsyncMock()
                mock_site.start = AsyncMock()
                mock_site_class.return_value = mock_site

                runner = await server.start()

                assert runner == mock_runner
                mock_runner.setup.assert_called_once()
                mock_site.start.assert_called_once()

    def test_add_cors_headers(self, server):
        """Test CORS header addition."""
        mock_response = Mock()
        mock_response.headers = {}

        server._add_cors_headers(mock_response)

        assert mock_response.headers["Access-Control-Allow-Origin"] == "*"
        assert (
            mock_response.headers["Access-Control-Allow-Methods"]
            == "GET, POST, OPTIONS"
        )
        assert (
            mock_response.headers["Access-Control-Allow-Headers"]
            == "Content-Type, Authorization"
        )

    async def test_error_handling_in_health(self, server):
        """Test error handling in health endpoint."""
        with patch("src.core.health.get_health_checker") as mock_checker:
            mock_checker.side_effect = Exception("Health check failed")

            request = Mock()
            response = await server.health(request)

            assert response.status == 503
            data = json.loads(response.text)
            assert data["status"] == "unhealthy"

    async def test_error_handling_in_metrics(self, server):
        """Test error handling in metrics endpoint."""
        with patch("src.http_server.get_performance_monitor") as mock_monitor:
            mock_monitor.side_effect = Exception("Metrics failed")

            request = Mock()
            response = await server.metrics(request)

            assert response.status == 500
            assert "Error exporting metrics" in response.text

    async def test_standards_api_error_handling(self, server):
        """Test error handling in standards API."""
        # Create a module if it doesn't exist
        if "src.core.standards" not in sys.modules:
            sys.modules["src.core.standards"] = Mock()
        if "src.core.standards.engine" not in sys.modules:
            sys.modules["src.core.standards.engine"] = Mock()

        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_engine.side_effect = Exception("Engine failed")

            request = Mock()
            response = await server.list_standards(request)

            assert response.status == 500
            data = json.loads(response.text)
            assert "error" in data

    def test_endpoint_documentation(self, server):
        """Test endpoint documentation in info endpoint."""
        endpoints = server._get_endpoint_info()

        # Check that all main endpoints are documented
        endpoint_paths = [ep["path"] for ep in endpoints]
        assert "/" in endpoint_paths
        assert "/health" in endpoint_paths
        assert "/metrics" in endpoint_paths
        assert "/api/standards" in endpoint_paths

        # Check endpoint details
        health_endpoint = next(ep for ep in endpoints if ep["path"] == "/health")
        assert health_endpoint["method"] == "GET"
        assert "description" in health_endpoint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
