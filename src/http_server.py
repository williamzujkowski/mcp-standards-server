"""
HTTP server for health checks and monitoring endpoints.

Provides REST API endpoints for health monitoring, metrics, and status checks
that can be used by load balancers, monitoring systems, and container orchestrators.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

from aiohttp import web
from aiohttp.web import Request, Response

from .core.health import health_check_endpoint, liveness_check, readiness_check
from .core.logging_config import get_logger
from .core.middleware.error_middleware import setup_error_handling
from .core.performance.metrics import get_performance_monitor

logger = get_logger(__name__)


class HTTPServer:
    """HTTP server for health checks and monitoring."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        self.host = host
        self.port = port
        self.app = web.Application()
        self.setup_routes()
        self.setup_middleware()

    def setup_middleware(self) -> None:
        """Setup middleware for error handling, logging, and CORS."""

        @web.middleware
        async def cors_middleware(request: Request, handler: Any) -> Any:
            """Add CORS headers."""
            response = await handler(request)
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )
            return response

        # Setup comprehensive error handling
        setup_error_handling(self.app)

        # Add CORS middleware
        self.app.middlewares.append(cors_middleware)

    def setup_routes(self) -> None:
        """Setup HTTP routes."""
        # Health check endpoints
        self.app.router.add_get("/health", self.health_check)
        self.app.router.add_get("/health/live", self.liveness_check)
        self.app.router.add_get("/health/ready", self.readiness_check)

        # Metrics endpoint
        self.app.router.add_get("/metrics", self.metrics)

        # Status endpoints
        self.app.router.add_get("/status", self.status)
        self.app.router.add_get("/info", self.info)

        # Standards endpoints
        self.app.router.add_get("/api/standards", self.list_standards)
        self.app.router.add_get("/api/standards/{standard_id}", self.get_standard)

        # Root endpoint
        self.app.router.add_get("/", self.root)

        # Handle OPTIONS for CORS
        self.app.router.add_options("/{path:.*}", self.options_handler)

    async def health_check(self, request: Request) -> Response:
        """Comprehensive health check endpoint."""
        try:
            # Get optional check names from query params
            check_names = (
                request.query.get("checks", "").split(",")
                if request.query.get("checks")
                else None
            )

            result = await health_check_endpoint(check_names)

            # Determine HTTP status code based on health status
            if result["status"] == "healthy":
                status_code = 200
            elif result["status"] == "degraded":
                status_code = 200  # Still serving traffic
            else:
                status_code = 503  # Service unavailable

            return web.json_response(result, status=status_code)

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return web.json_response(
                {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
                status=503,
            )

    async def liveness_check(self, request: Request) -> Response:
        """Kubernetes liveness probe endpoint."""
        try:
            result = await liveness_check()
            status_code = 200 if result["alive"] else 503
            return web.json_response(result, status=status_code)
        except Exception as e:
            logger.error(f"Liveness check failed: {e}")
            return web.json_response(
                {
                    "alive": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
                status=503,
            )

    async def readiness_check(self, request: Request) -> Response:
        """Kubernetes readiness probe endpoint."""
        try:
            result = await readiness_check()
            status_code = 200 if result["ready"] else 503
            return web.json_response(result, status=status_code)
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")
            return web.json_response(
                {
                    "ready": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                },
                status=503,
            )

    async def metrics(self, request: Request) -> Response:
        """Prometheus metrics endpoint."""
        try:
            metrics_collector = get_performance_monitor()

            # Generate Prometheus format metrics
            metrics_data = metrics_collector.get_prometheus_metrics()

            return web.Response(
                text=metrics_data,
                content_type="text/plain; version=0.0.4",
                charset="utf-8",
            )
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
            return web.Response(
                text="# Error exporting metrics\n",
                content_type="text/plain",
                status=500,
            )

    async def status(self, request: Request) -> Response:
        """Service status endpoint."""
        try:
            # Get basic service information
            status_info = {
                "service": "mcp-standards-server",
                "version": "1.0.0",
                "status": "running",
                "timestamp": datetime.now().isoformat(),
                "environment": {
                    "host": self.host,
                    "port": self.port,
                    "data_dir": os.environ.get("DATA_DIR", "data"),
                    "log_level": os.environ.get("LOG_LEVEL", "INFO"),
                },
            }

            return web.json_response(status_info)
        except Exception as e:
            logger.error(f"Status check failed: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
            )

    async def info(self, request: Request) -> Response:
        """Service information endpoint."""
        try:
            info = {
                "name": "MCP Standards Server",
                "description": "Model Context Protocol server for software development standards",
                "version": "1.0.0",
                "author": "MCP Standards Team",
                "endpoints": {
                    "health": "/health",
                    "liveness": "/health/live",
                    "readiness": "/health/ready",
                    "metrics": "/metrics",
                    "status": "/status",
                    "standards": "/api/standards",
                },
                "documentation": "https://github.com/williamzujkowski/mcp-standards-server",
            }

            return web.json_response(info)
        except Exception as e:
            logger.error(f"Info endpoint failed: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
            )

    async def list_standards(self, request: Request) -> Response:
        """List available standards."""
        try:
            from .core.standards.engine import StandardsEngine

            engine = StandardsEngine(data_dir="data/standards")
            standards = await engine.list_standards()

            # Convert to simple list format
            standards_list = [
                {
                    "id": std.get("id"),
                    "title": std.get("title"),
                    "category": std.get("category"),
                    "description": (
                        std.get("description", "")[:200] + "..."
                        if len(std.get("description", "")) > 200
                        else std.get("description", "")
                    ),
                }
                for std in standards
            ]

            return web.json_response(
                {
                    "standards": standards_list,
                    "total": len(standards_list),
                    "timestamp": datetime.now().isoformat(),
                }
            )
        except Exception as e:
            logger.error(f"List standards failed: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
            )

    async def get_standard(self, request: Request) -> Response:
        """Get specific standard details."""
        try:
            standard_id = request.match_info["standard_id"]

            from .core.standards.engine import StandardsEngine

            engine = StandardsEngine(data_dir="data/standards")
            standard = await engine.get_standard(standard_id)

            if not standard:
                return web.json_response(
                    {
                        "error": f"Standard not found: {standard_id}",
                        "timestamp": datetime.now().isoformat(),
                    },
                    status=404,
                )

            return web.json_response(
                {"standard": standard, "timestamp": datetime.now().isoformat()}
            )
        except Exception as e:
            logger.error(f"Get standard failed: {e}")
            return web.json_response(
                {"error": str(e), "timestamp": datetime.now().isoformat()}, status=500
            )

    async def root(self, request: Request) -> Response:
        """Root endpoint with service information."""
        return web.json_response(
            {
                "service": "MCP Standards Server",
                "status": "running",
                "version": "1.0.0",
                "endpoints": {
                    "health": "/health",
                    "info": "/info",
                    "standards": "/api/standards",
                    "metrics": "/metrics",
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

    async def options_handler(self, request: Request) -> Response:
        """Handle OPTIONS requests for CORS."""
        return web.Response(status=200)

    async def start(self) -> web.AppRunner:
        """Start the HTTP server."""
        runner = web.AppRunner(self.app)
        await runner.setup()

        site = web.TCPSite(runner, self.host, self.port)
        await site.start()

        logger.info(f"HTTP server started on http://{self.host}:{self.port}")
        return runner

    async def stop(self, runner: web.AppRunner) -> None:
        """Stop the HTTP server."""
        await runner.cleanup()
        logger.info("HTTP server stopped")


async def start_http_server(
    host: str | None = None, port: int | None = None
) -> web.AppRunner:
    """Start the HTTP server with environment variable support."""
    host = host or os.environ.get("HTTP_HOST", "127.0.0.1")
    port = port or int(os.environ.get("HTTP_PORT", "8080"))

    server = HTTPServer(host, port)
    return await server.start()


if __name__ == "__main__":

    async def main() -> None:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Start HTTP server
        runner = await start_http_server()

        try:
            # Keep the server running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down HTTP server...")
        finally:
            await runner.cleanup()

    asyncio.run(main())
