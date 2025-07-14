"""
Integration tests for the combined server (MCP + HTTP).

Tests the full system integration including both MCP and HTTP servers
running together, health checks, and end-to-end functionality.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from src.main import CombinedServer


class TestCombinedServerIntegration:
    """Integration tests for combined server."""

    @pytest.fixture
    def mcp_config(self):
        """Create MCP configuration."""
        return {
            "auth": {"enabled": False},  # Disable for testing
            "privacy": {
                "detect_pii": False,
                "redact_pii": False,
            },  # Disable for testing
            "rate_limit_window": 60,
            "rate_limit_max_requests": 1000,  # High limit for testing
        }

    @pytest.fixture
    async def combined_server(self, mcp_config):
        """Create combined server instance."""
        with patch.dict(
            os.environ,
            {"HTTP_HOST": "127.0.0.1", "HTTP_PORT": "8082", "HTTP_ONLY": "true"},
        ):
            server = CombinedServer(mcp_config)

            # Start the server in HTTP-only mode for testing
            await server.start_http_server()

            yield server

            # Cleanup
            await server.shutdown()

    @pytest.fixture
    async def client_session(self):
        """Create aiohttp client session."""
        session = aiohttp.ClientSession()
        yield session
        await session.close()

    async def test_combined_server_startup(self, combined_server):
        """Test that combined server starts up correctly."""
        # Check that HTTP server is running
        assert combined_server.http_runner is not None
        # Note: combined_server.running is only set to True in run() method,
        # not when starting just the HTTP server

    async def test_health_endpoints_work(self, combined_server, client_session):
        """Test that health endpoints work after startup."""
        # Test basic health endpoint
        async with client_session.get("http://127.0.0.1:8082/health") as response:
            assert response.status in [200, 503]  # May be unhealthy but should respond
            data = await response.json()
            assert "status" in data
            assert "timestamp" in data

        # Test liveness endpoint
        async with client_session.get("http://127.0.0.1:8082/health/live") as response:
            assert response.status in [200, 503]
            data = await response.json()
            assert "alive" in data

        # Test readiness endpoint
        async with client_session.get("http://127.0.0.1:8082/health/ready") as response:
            assert response.status in [200, 503]
            data = await response.json()
            assert "ready" in data

    async def test_service_info_endpoints(self, combined_server, client_session):
        """Test service information endpoints."""
        # Test status endpoint
        async with client_session.get("http://127.0.0.1:8082/status") as response:
            assert response.status == 200
            data = await response.json()
            assert data["service"] == "mcp-standards-server"
            assert data["status"] == "running"

        # Test info endpoint
        async with client_session.get("http://127.0.0.1:8082/info") as response:
            assert response.status == 200
            data = await response.json()
            assert data["name"] == "MCP Standards Server"
            assert "endpoints" in data

        # Test root endpoint
        async with client_session.get("http://127.0.0.1:8082/") as response:
            assert response.status == 200
            data = await response.json()
            assert data["service"] == "MCP Standards Server"
            assert data["status"] == "running"

    async def test_standards_api_endpoints(self, combined_server, client_session):
        """Test standards API endpoints."""
        # Mock standards engine
        with patch("src.core.standards.engine.StandardsEngine") as mock_engine:
            from src.core.standards.models import Standard

            mock_instance = Mock()
            mock_instance.list_standards = AsyncMock(
                return_value=[
                    Standard(
                        id="test-standard",
                        title="Test Standard",
                        category="testing",
                        description="A test standard for integration testing",
                        content="# Test Standard\n\nThis is a test standard.",
                    )
                ]
            )
            mock_instance.get_standard = AsyncMock(
                return_value={
                    "id": "test-standard",
                    "title": "Test Standard",
                    "category": "testing",
                    "description": "A test standard for integration testing",
                    "content": "# Test Standard\n\nThis is a test standard.",
                }
            )
            mock_engine.return_value = mock_instance

            # Test list standards
            async with client_session.get(
                "http://127.0.0.1:8082/api/standards"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "standards" in data
                assert data["total"] == 1
                assert data["standards"][0]["id"] == "test-standard"

            # Test get specific standard
            async with client_session.get(
                "http://127.0.0.1:8082/api/standards/test-standard"
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "standard" in data
                assert data["standard"]["id"] == "test-standard"

    async def test_metrics_endpoint(self, combined_server, client_session):
        """Test metrics endpoint."""
        with patch(
            "src.core.performance.metrics.get_performance_monitor"
        ) as mock_metrics:
            mock_monitor = Mock()
            mock_monitor.get_prometheus_metrics = Mock(
                return_value="""
# HELP mcp_requests_total Total number of MCP requests
# TYPE mcp_requests_total counter
mcp_requests_total{tool="get_applicable_standards"} 42
"""
            )
            mock_metrics.return_value = mock_monitor

            async with client_session.get("http://127.0.0.1:8082/metrics") as response:
                assert response.status == 200
                assert response.content_type == "text/plain"
                text = await response.text()
                assert "mcp_tool_calls_total" in text

    async def test_error_handling(self, combined_server, client_session):
        """Test error handling in API endpoints."""
        # Test non-existent standard
        async with client_session.get(
            "http://127.0.0.1:8082/api/standards/nonexistent"
        ) as response:
            assert response.status == 404
            data = await response.json()
            assert "error" in data

        # Test invalid endpoint - returns 405 because OPTIONS is available for all paths
        async with client_session.get("http://127.0.0.1:8082/api/invalid") as response:
            assert response.status == 405  # Method not allowed (OPTIONS is available)

    async def test_concurrent_requests(self, combined_server, client_session):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        tasks = []
        for _i in range(10):
            task = asyncio.create_task(
                client_session.get("http://127.0.0.1:8082/status")
            )
            tasks.append(task)

        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)

        # Verify all requests were handled
        assert len(responses) == 10
        for response in responses:
            assert response.status == 200
            response.close()

    async def test_graceful_shutdown(self, mcp_config):
        """Test graceful shutdown of combined server."""
        with patch.dict(
            os.environ,
            {"HTTP_HOST": "127.0.0.1", "HTTP_PORT": "8083", "HTTP_ONLY": "true"},
        ):
            server = CombinedServer(mcp_config)

            # Start server
            await server.start_http_server()
            # Note: running is only set to True in run() method, not start_http_server()
            assert server.http_runner is not None

            # Shutdown server
            await server.shutdown()

            # Verify cleanup
            assert server.http_runner is None  # Should be cleaned up


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create standards directory structure
            standards_dir = os.path.join(temp_dir, "standards")
            os.makedirs(standards_dir)

            # Create a test standard file
            test_standard = {
                "id": "test-workflow-standard",
                "title": "Test Workflow Standard",
                "category": "testing",
                "description": "Standard for testing end-to-end workflow",
                "content": "# Test Workflow Standard\n\nThis is a test standard for workflow testing.",
                "tags": ["testing", "workflow"],
                "applicability": {
                    "languages": ["python"],
                    "frameworks": ["pytest"],
                    "project_types": ["testing"],
                },
                "rules": {
                    "test_naming": {
                        "title": "Test Naming Convention",
                        "description": "Tests should follow naming conventions",
                        "severity": "warning",
                        "category": "naming",
                    }
                },
            }

            with open(
                os.path.join(standards_dir, "test-workflow-standard.json"), "w"
            ) as f:
                json.dump(test_standard, f)

            yield temp_dir

    async def test_full_workflow(self, temp_data_dir):
        """Test complete workflow from startup to API usage."""
        # Mock environment
        with patch.dict(
            os.environ,
            {
                "HTTP_HOST": "127.0.0.1",
                "HTTP_PORT": "8084",
                "HTTP_ONLY": "true",
                "DATA_DIR": temp_data_dir,
            },
        ):
            # Create and start server
            server = CombinedServer(
                {"auth": {"enabled": False}, "privacy": {"enabled": False}}
            )

            await server.start_http_server()

            try:
                # Create client session
                async with aiohttp.ClientSession() as session:
                    # Test 1: Check server is running
                    async with session.get("http://127.0.0.1:8084/status") as response:
                        assert response.status == 200
                        data = await response.json()
                        assert data["status"] == "running"

                    # Test 2: Check health
                    async with session.get("http://127.0.0.1:8084/health") as response:
                        # Health may be degraded but should respond
                        assert response.status in [200, 503]
                        data = await response.json()
                        assert "status" in data

                    # Test 3: List standards (with mocked engine)
                    with patch(
                        "src.core.standards.engine.StandardsEngine"
                    ) as mock_engine:
                        from src.core.standards.models import Standard

                        mock_instance = Mock()
                        mock_instance.list_standards = AsyncMock(
                            return_value=[
                                Standard(
                                    id="test-workflow-standard",
                                    title="Test Workflow Standard",
                                    category="testing",
                                    description="Standard for testing end-to-end workflow",
                                    content="# Test Workflow Standard\n\nThis is a test standard.",
                                )
                            ]
                        )
                        mock_engine.return_value = mock_instance

                        async with session.get(
                            "http://127.0.0.1:8084/api/standards"
                        ) as response:
                            assert response.status == 200
                            data = await response.json()
                            assert data["total"] == 1
                            assert (
                                data["standards"][0]["id"] == "test-workflow-standard"
                            )

                    # Test 4: Get specific standard
                    with patch(
                        "src.core.standards.engine.StandardsEngine"
                    ) as mock_engine:
                        mock_instance = Mock()
                        mock_instance.get_standard = AsyncMock(
                            return_value={
                                "id": "test-workflow-standard",
                                "title": "Test Workflow Standard",
                                "category": "testing",
                                "description": "Standard for testing end-to-end workflow",
                                "content": "# Test Workflow Standard\n\nThis is a test standard.",
                            }
                        )
                        mock_engine.return_value = mock_instance

                        async with session.get(
                            "http://127.0.0.1:8084/api/standards/test-workflow-standard"
                        ) as response:
                            assert response.status == 200
                            data = await response.json()
                            assert data["standard"]["id"] == "test-workflow-standard"

                    # Test 5: Check metrics
                    with patch(
                        "src.http_server.get_performance_monitor"
                    ) as mock_metrics:
                        mock_monitor = Mock()
                        mock_monitor.get_prometheus_metrics = Mock(
                            return_value="# Test metrics\nmcp_tool_calls_total 1\ntest_metric 1\n"
                        )
                        mock_metrics.return_value = mock_monitor

                        async with session.get(
                            "http://127.0.0.1:8084/metrics"
                        ) as response:
                            assert response.status == 200
                            assert response.content_type == "text/plain"
                            text = await response.text()
                            assert "mcp_tool_calls_total" in text

                    # Test 6: Check service info
                    async with session.get("http://127.0.0.1:8084/info") as response:
                        assert response.status == 200
                        data = await response.json()
                        assert data["name"] == "MCP Standards Server"
                        assert "endpoints" in data

            finally:
                # Cleanup
                await server.shutdown()

    async def test_performance_under_load(self, temp_data_dir):
        """Test server performance under load."""
        with patch.dict(
            os.environ,
            {
                "HTTP_HOST": "127.0.0.1",
                "HTTP_PORT": "8085",
                "HTTP_ONLY": "true",
                "DATA_DIR": temp_data_dir,
            },
        ):
            server = CombinedServer(
                {"auth": {"enabled": False}, "privacy": {"enabled": False}}
            )

            await server.start_http_server()

            try:
                # Create multiple concurrent sessions
                sessions = []
                for _i in range(5):
                    session = aiohttp.ClientSession()
                    sessions.append(session)

                # Create load test tasks
                tasks = []
                for session in sessions:
                    for _i in range(20):  # 20 requests per session
                        task = asyncio.create_task(
                            session.get("http://127.0.0.1:8085/status")
                        )
                        tasks.append(task)

                # Execute all tasks
                responses = await asyncio.gather(*tasks)

                # Verify all requests succeeded
                assert len(responses) == 100
                for response in responses:
                    assert response.status == 200
                    response.close()

                # Cleanup sessions
                for session in sessions:
                    await session.close()

            finally:
                await server.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
