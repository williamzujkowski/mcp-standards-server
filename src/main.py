"""
Main entry point for the MCP Standards Server.

Starts both the MCP server (stdio) and HTTP server (for health checks and monitoring)
in a coordinated manner with proper shutdown handling.
"""

import asyncio
import logging
import os
import signal
import sys

from .core.decorators import with_error_handling, with_logging
from .core.errors import ErrorCode
from .core.logging_config import get_logger, init_logging
from .http_server import start_http_server
from .mcp_server import MCPStandardsServer

logger = get_logger(__name__)


class CombinedServer:
    """Combined server that runs both MCP and HTTP servers."""

    def __init__(self, mcp_config: dict | None = None):
        self.mcp_config = mcp_config or {}
        self.mcp_server = None
        self.http_runner = None
        self.running = False
        self.start_time = None

        # Initialize logging
        init_logging()

        # Setup signal handlers
        self.setup_signal_handlers()

    def setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, starting graceful shutdown...")
            self.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    @with_error_handling(error_code=ErrorCode.SYSTEM_INTERNAL_ERROR)
    @with_logging(level=logging.INFO)
    async def start_http_server(self):
        """Start the HTTP server for health checks and monitoring."""
        http_host = os.environ.get("HTTP_HOST", "127.0.0.1")
        http_port = int(os.environ.get("HTTP_PORT", "8080"))

        self.http_runner = await start_http_server(http_host, http_port)
        logger.info(f"HTTP server started on http://{http_host}:{http_port}")

    @with_error_handling(error_code=ErrorCode.SYSTEM_INTERNAL_ERROR)
    @with_logging(level=logging.INFO)
    async def start_mcp_server(self):
        """Start the MCP server."""
        try:
            self.mcp_server = MCPStandardsServer(self.mcp_config)
            logger.info("MCP server initialized")

            # Run MCP server in background
            await self.mcp_server.run()

        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            raise

    async def run(self):
        """Run both servers with proper coordination."""
        self.running = True

        try:
            # Start HTTP server first
            await self.start_http_server()

            # Check if we should run in HTTP-only mode
            if os.environ.get("HTTP_ONLY", "false").lower() == "true":
                logger.info("Running in HTTP-only mode")

                # Keep HTTP server running
                while self.running:
                    await asyncio.sleep(1)
            else:
                # Run MCP server (this will block until completion)
                await self.start_mcp_server()

        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self):
        """Gracefully shutdown both servers."""
        logger.info("Starting graceful shutdown...")

        # Stop HTTP server
        if self.http_runner:
            try:
                await self.http_runner.cleanup()
                logger.info("HTTP server stopped")
            except Exception as e:
                logger.error(f"Error stopping HTTP server: {e}")

        # MCP server shutdown is handled by its own cleanup
        logger.info("Shutdown complete")


async def main():
    """Main entry point."""
    # Setup logging
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/mcp-server.log') if os.path.exists('logs') else logging.StreamHandler()
        ]
    )

    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)

    logger.info("Starting MCP Standards Server")

    # Load MCP configuration
    mcp_config = {}
    config_path = os.environ.get("MCP_CONFIG_PATH", "config.json")
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            mcp_config = json.load(f)
        logger.info(f"Loaded configuration from {config_path}")

    # Create and run combined server
    combined_server = CombinedServer(mcp_config)

    try:
        await combined_server.run()
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)

    logger.info("MCP Standards Server stopped")


if __name__ == "__main__":
    asyncio.run(main())
