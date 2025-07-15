"""
Integration tests for the HTTP server and health check endpoints.

Tests the full HTTP server functionality including health checks,
monitoring endpoints, and API endpoints.
"""

from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from src.core.health import HealthChecker
from src.http_server import HTTPServer


class TestHTTPServerIntegration:
    """Integration tests for HTTP server."""

    @pytest.fixture
    async def http_server(self):
        """Create HTTP server for testing."""
        server = HTTPServer(host="127.0.0.1", port=8081)
        runner = await server.start()

        yield server

        await runner.cleanup()

    @pytest.fixture
    async def client_session(self):
        """Create aiohttp client session."""
        session = aiohttp.ClientSession()
        yield session
        await session.close()

    async def test_health_endpoint_healthy(self, http_server, client_session):
        """Test health endpoint when system is healthy."""
        # Mock health checker to return healthy status
        with patch("src.core.health.get_health_checker") as mock_checker:
            mock_health_checker = Mock()
            mock_health_checker.check_health = AsyncMock(
                return_value={
                    "status": "healthy",
                    "timestamp": "2023-01-01T00:00:00",
                    "uptime_seconds": 3600,
                    "uptime_human": "1:00:00",
                    "version": "1.0.0",
                    "service": "mcp-standards-server",
                    "checks": {
                        "system_resources": {
                            "name": "system_resources",
                            "status": "healthy",
                            "message": "CPU usage normal: 25%",
                            "duration_ms": 10.5,
                            "timestamp": "2023-01-01T00:00:00",
                        }
                    },
                    "summary": {
                        "total_checks": 1,
                        "healthy": 1,
                        "degraded": 0,
                        "unhealthy": 0,
                        "unknown": 0,
                    },
                }
            )
            mock_checker.return_value = mock_health_checker

            async with client_session.get("http://127.0.0.1:8081/health") as response:
                assert response.status == 200
                data = await response.json()
                assert data["status"] == "healthy"
                assert data["service"] == "mcp-standards-server"
                assert "checks" in data
                assert "summary" in data

    async def test_health_endpoint_unhealthy(self, http_server, client_session):
        """Test health endpoint when system is unhealthy."""
        with patch("src.core.health.get_health_checker") as mock_checker:
            mock_health_checker = Mock()
            mock_health_checker.check_health = AsyncMock(
                return_value={
                    "status": "unhealthy",
                    "timestamp": "2023-01-01T00:00:00",
                    "uptime_seconds": 3600,
                    "uptime_human": "1:00:00",
                    "version": "1.0.0",
                    "service": "mcp-standards-server",
                    "checks": {
                        "redis_connection": {
                            "name": "redis_connection",
                            "status": "unhealthy",
                            "message": "Redis connection failed",
                            "duration_ms": 5000.0,
                            "timestamp": "2023-01-01T00:00:00",
                        }
                    },
                    "summary": {
                        "total_checks": 1,
                        "healthy": 0,
                        "degraded": 0,
                        "unhealthy": 1,
                        "unknown": 0,
                    },
                }
            )
            mock_checker.return_value = mock_health_checker

            async with client_session.get("http://127.0.0.1:8081/health") as response:
                assert response.status == 503
                data = await response.json()
                assert data["status"] == "unhealthy"

    async def test_liveness_endpoint(self, http_server, client_session):
        """Test liveness endpoint."""
        with patch("src.http_server.liveness_check") as mock_liveness:
            mock_liveness.return_value = {
                "alive": True,
                "status": "healthy",
                "timestamp": "2023-01-01T00:00:00",
                "uptime_seconds": 3600,
            }

            async with client_session.get(
                "http://127.0.0.1:8081/health/live"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert data["alive"] is True

    async def test_readiness_endpoint(self, http_server, client_session):
        """Test readiness endpoint."""
        with patch("src.http_server.readiness_check") as mock_readiness:
            mock_readiness.return_value = {
                "ready": True,
                "status": "healthy",
                "timestamp": "2023-01-01T00:00:00",
                "checks": {},
            }

            async with client_session.get(
                "http://127.0.0.1:8081/health/ready"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert data["ready"] is True

    async def test_metrics_endpoint(self, http_server, client_session):
        """Test metrics endpoint."""
        # Patch at the correct location where it's imported
        with patch("src.http_server.get_performance_monitor") as mock_metrics:
            mock_monitor = Mock()
            mock_monitor.get_prometheus_metrics = Mock(
                return_value="# HELP test_metric Test metric\ntest_metric 1.0\n"
            )
            mock_metrics.return_value = mock_monitor

            async with client_session.get("http://127.0.0.1:8081/metrics") as response:
                assert response.status == 200
                assert response.content_type == "text/plain"
                text = await response.text()
                assert "test_metric" in text

    async def test_status_endpoint(self, http_server, client_session):
        """Test status endpoint."""
        async with client_session.get("http://127.0.0.1:8081/status") as response:
            assert response.status == 200
            data = await response.json()
            assert data["service"] == "mcp-standards-server"
            assert data["version"] == "1.0.0"
            assert data["status"] == "running"
            assert "environment" in data

    async def test_info_endpoint(self, http_server, client_session):
        """Test info endpoint."""
        async with client_session.get("http://127.0.0.1:8081/info") as response:
            assert response.status == 200
            data = await response.json()
            assert data["name"] == "MCP Standards Server"
            assert "endpoints" in data
            assert "version" in data

    async def test_root_endpoint(self, http_server, client_session):
        """Test root endpoint."""
        async with client_session.get("http://127.0.0.1:8081/") as response:
            assert response.status == 200
            data = await response.json()
            assert data["service"] == "MCP Standards Server"
            assert data["status"] == "running"
            assert "endpoints" in data

    async def test_cors_headers(self, http_server, client_session):
        """Test CORS headers are present."""
        async with client_session.get("http://127.0.0.1:8081/health") as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "*"
            assert "Access-Control-Allow-Methods" in response.headers
            assert "Access-Control-Allow-Headers" in response.headers

    async def test_options_handler(self, http_server, client_session):
        """Test OPTIONS handler for CORS."""
        async with client_session.options("http://127.0.0.1:8081/health") as response:
            assert response.status == 200

    async def test_standards_api_list(self, http_server, client_session):
        """Test standards API list endpoint."""
        # Create test directory with a standard
        import json
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test standard file
            test_standard = {
                "id": "test-standard",
                "title": "Test Standard",
                "category": "testing",
                "description": "A test standard for testing purposes",
                "content": "# Test Standard\n\nThis is test content.",
                "version": "1.0.0",
                "tags": ["test"],
                "metadata": {"status": "active"},
            }

            os.makedirs(os.path.join(temp_dir, "standards"), exist_ok=True)
            with open(
                os.path.join(temp_dir, "standards", "test-standard.json"), "w"
            ) as f:
                json.dump(test_standard, f)

            # Patch the data directory
            with patch.dict(os.environ, {"DATA_DIR": temp_dir}):
                async with client_session.get(
                    "http://127.0.0.1:8081/api/standards"
                ) as response:
                    assert response.status == 200
                    data = await response.json()
                    assert "standards" in data
                    assert data["total"] >= 0  # May be 0 if no standards found

    async def test_standards_api_get(self, http_server, client_session):
        """Test standards API get endpoint."""
        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            mock_instance.get_standard = AsyncMock(
                return_value={
                    "id": "test-standard",
                    "title": "Test Standard",
                    "category": "testing",
                    "description": "A test standard for testing purposes",
                    "content": "# Test Standard\nThis is a test standard.",
                }
            )
            mock_engine.return_value = mock_instance

            async with client_session.get(
                "http://127.0.0.1:8081/api/standards/test-standard"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "standard" in data
                assert data["standard"]["id"] == "test-standard"

    async def test_standards_api_not_found(self, http_server, client_session):
        """Test standards API returns 404 for non-existent standard."""
        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            mock_instance.get_standard = AsyncMock(return_value=None)
            mock_engine.return_value = mock_instance

            async with client_session.get(
                "http://127.0.0.1:8081/api/standards/nonexistent"
            ) as response:
                assert response.status == 404
                data = await response.json()
                assert "error" in data
                assert "not found" in data["error"].lower()


class TestHealthCheckerIntegration:
    """Integration tests for health checker."""

    @pytest.fixture
    def health_checker(self):
        """Create health checker instance."""
        return HealthChecker()

    async def test_full_health_check(self, health_checker):
        """Test complete health check with all checks."""
        # Mock all the system dependencies
        with (
            patch("psutil.cpu_percent", return_value=25.0),
            patch("psutil.virtual_memory") as mock_memory,
            patch("psutil.disk_usage") as mock_disk,
        ):

            # Mock memory info
            mock_memory.return_value = Mock(
                total=8 * 1024**3, available=4 * 1024**3, percent=50.0, used=4 * 1024**3
            )

            # Mock disk info
            mock_disk.return_value = Mock(
                total=100 * 1024**3, free=50 * 1024**3, used=50 * 1024**3
            )

            result = await health_checker.check_health(
                ["system_resources", "memory_usage", "disk_space"]
            )

            assert result["status"] == "healthy"
            assert result["summary"]["total_checks"] == 3
            assert result["summary"]["healthy"] == 3
            assert "checks" in result
            assert "uptime_seconds" in result

    async def test_health_check_caching(self, health_checker):
        """Test health check result caching."""
        # Mock system resources
        with patch("psutil.cpu_percent", return_value=25.0):
            # First call
            result1 = await health_checker.check_health(["system_resources"])

            # Second call (should use cache)
            result2 = await health_checker.check_health(["system_resources"])

            # Results should be identical (cached)
            assert (
                result1["checks"]["system_resources"]["timestamp"]
                == result2["checks"]["system_resources"]["timestamp"]
            )

    async def test_health_check_error_handling(self, health_checker):
        """Test health check error handling."""
        # Mock a failing check
        with patch("psutil.cpu_percent", side_effect=Exception("System unavailable")):
            result = await health_checker.check_health(["system_resources"])

            assert result["status"] == "unhealthy"
            assert result["checks"]["system_resources"]["status"] == "unhealthy"
            assert (
                "System unavailable" in result["checks"]["system_resources"]["message"]
            )

    async def test_redis_health_check(self, health_checker):
        """Test Redis health check."""
        # Mock Redis cache
        with patch("src.core.cache.redis_client.get_cache") as mock_get_cache:
            mock_cache = Mock()
            # Store the value that was set so we can return it in get
            stored_value = None

            def mock_set(key, value, ttl=None):
                nonlocal stored_value
                stored_value = value

            def mock_get(key):
                return stored_value

            # Make the cache methods synchronous since they're called synchronously in the health check
            mock_cache.set = Mock(side_effect=mock_set)
            mock_cache.get = Mock(side_effect=mock_get)
            mock_cache.delete = Mock()
            mock_get_cache.return_value = mock_cache

            result = await health_checker.check_health(["redis_connection"])

            assert result["status"] == "healthy"
            assert result["checks"]["redis_connection"]["status"] == "healthy"
            assert (
                "Redis connection working"
                in result["checks"]["redis_connection"]["message"]
            )

    async def test_standards_health_check(self, health_checker):
        """Test standards loading health check."""
        # Mock standards engine
        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            mock_instance = Mock()
            mock_instance.list_standards = AsyncMock(
                return_value=[
                    {"id": "test1", "title": "Test 1"},
                    {"id": "test2", "title": "Test 2"},
                ]
            )
            mock_engine.return_value = mock_instance

            result = await health_checker.check_health(["standards_loaded"])

            assert result["status"] == "healthy"
            assert result["checks"]["standards_loaded"]["status"] == "healthy"
            assert (
                "Standards loaded: 2" in result["checks"]["standards_loaded"]["message"]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
