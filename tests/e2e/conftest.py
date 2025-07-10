"""E2E test configuration and shared fixtures - FIXED VERSION."""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest_asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tests.e2e.test_data_setup import setup_test_data

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MCPTestClient:
    """Test client for interacting with MCP server."""

    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session: ClientSession | None = None
        self._read = None
        self._write = None

    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server as an async context manager."""
        logger.debug("Connecting to MCP server...")
        try:
            async with stdio_client(self.server_params) as (read, write):
                self._read = read
                self._write = write
                try:
                    async with ClientSession(read, write) as session:
                        self.session = session
                        try:
                            # Initialize the session
                            logger.debug("Initializing MCP session...")
                            await session.initialize()
                            logger.debug("MCP session initialized successfully")
                            yield self
                        finally:
                            logger.debug("Closing MCP session...")
                            self.session = None
                except Exception as e:
                    # Log but don't re-raise asyncio cancellation errors
                    if "cancel scope" not in str(e):
                        logger.error(f"Error in ClientSession: {e}")
                        raise
                    else:
                        logger.warning(f"Ignoring asyncio cancellation error: {e}")
                finally:
                    self._read = None
                    self._write = None
        except Exception as e:
            # Log but don't re-raise asyncio cancellation errors during cleanup
            if "cancel scope" not in str(e):
                logger.error(f"Error in stdio_client: {e}")
                raise
            else:
                logger.warning(
                    f"Ignoring asyncio cancellation error during cleanup: {e}"
                )

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to server")

        logger.debug(f"Calling tool: {tool_name} with arguments: {arguments}")

        try:
            result = await self.session.call_tool(tool_name, arguments)
            logger.debug(f"Raw result from tool {tool_name}: {result}")

            # Extract the content from the result
            if hasattr(result, "content") and result.content:
                # Parse the JSON content from the first text content
                if result.content[0].text:
                    text_content = result.content[0].text
                    logger.debug(f"Text content from tool {tool_name}: {text_content}")
                    try:
                        parsed_result = json.loads(text_content)
                        logger.debug(
                            f"Parsed result from tool {tool_name}: {parsed_result}"
                        )
                        return parsed_result
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse JSON response from {tool_name}: {e}"
                        )
                        logger.error(f"Raw text was: {repr(text_content)}")
                        raise
                else:
                    logger.warning(f"Tool {tool_name} returned empty text content")
                    return {}
            else:
                logger.warning(f"Tool {tool_name} returned no content")
                return result

        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}", exc_info=True)
            raise


# Change from session scope to function scope to match event loop
@pytest_asyncio.fixture
async def mcp_server(tmp_path):
    """Fixture to start and stop MCP server for tests - function scoped."""
    logger.info(f"Starting MCP server with tmp_path: {tmp_path}")

    # Set up test data
    setup_test_data(tmp_path)

    # Set up test environment
    env = os.environ.copy()
    env["MCP_STANDARDS_DATA_DIR"] = str(tmp_path)
    env["MCP_CONFIG_PATH"] = str(Path(__file__).parent / "test_config.json")
    env["MCP_DISABLE_SEARCH"] = "true"  # Disable search to avoid heavy deps
    env["PYTHONPATH"] = str(Path(__file__).parent.parent.parent)  # Project root

    # Enable coverage in subprocess
    env["COVERAGE_PROCESS_START"] = str(
        Path(__file__).parent.parent.parent / ".coveragerc"
    )

    # Ensure sitecustomize.py is in PYTHONPATH for coverage subprocess
    project_root = Path(__file__).parent.parent.parent
    env["PYTHONPATH"] = f"{project_root}:{env.get('PYTHONPATH', '')}"

    # Enable coverage for subprocess
    env["COVERAGE_RUN"] = "true"

    # Log environment for debugging
    logger.debug(f"MCP_STANDARDS_DATA_DIR: {env['MCP_STANDARDS_DATA_DIR']}")
    logger.debug(f"MCP_CONFIG_PATH: {env['MCP_CONFIG_PATH']}")
    logger.debug(f"PYTHONPATH: {env['PYTHONPATH']}")

    # Server parameters - run directly without coverage subprocess
    # Coverage will be handled by the parent process
    server_params = StdioServerParameters(
        command=sys.executable,  # Use the same Python interpreter
        args=["-m", "src"],
        env=env,
    )

    # Start server process
    logger.info("Starting server process...")
    logger.debug(f"Command: {server_params.command} {' '.join(server_params.args)}")
    process = await asyncio.create_subprocess_exec(
        server_params.command,
        *server_params.args,
        env=server_params.env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    # Monitor server startup with reduced timeout
    startup_timeout = 3.0
    start_time = time.time()
    server_ready = False

    # Try to read initial output to ensure server started
    try:
        while time.time() - start_time < startup_timeout:
            # Check if process is still running
            if process.returncode is not None:
                # Process exited, read stderr for error info
                stderr_data = await process.stderr.read()
                stdout_data = await process.stdout.read()
                logger.error(f"Server process exited with code {process.returncode}")
                logger.error(f"Server stdout: {stdout_data.decode()}")
                logger.error(f"Server stderr: {stderr_data.decode()}")
                raise RuntimeError(f"Server failed to start: {stderr_data.decode()}")

            # Give server time to initialize
            await asyncio.sleep(0.5)

            # After initial delay, assume server is ready
            if time.time() - start_time > 1.0:
                server_ready = True
                break

    except asyncio.TimeoutError:
        logger.warning("Timeout waiting for server startup confirmation")
        server_ready = True  # Proceed anyway

    if not server_ready:
        logger.error("Server failed to start within timeout")
        if process.returncode is None:
            process.terminate()
            await process.wait()
        raise RuntimeError("MCP server failed to start")

    logger.info("MCP server started successfully")

    try:
        yield server_params
    finally:
        # Stop server
        logger.info("Stopping MCP server...")
        try:
            if process.returncode is None:
                process.terminate()
                await asyncio.wait_for(process.wait(), timeout=5.0)
                logger.info("MCP server stopped")
            else:
                logger.warning(f"Server already exited with code {process.returncode}")
        except (ProcessLookupError, asyncio.TimeoutError) as e:
            logger.warning(f"Error stopping server: {e}")
            # Process may have already exited


@pytest_asyncio.fixture
async def mcp_client(mcp_server):
    """Fixture to provide connected MCP client for tests."""
    logger.info("Creating MCP client...")
    client = MCPTestClient(mcp_server)
    async with client.connect() as connected_client:
        logger.info("MCP client connected")
        yield connected_client
        logger.info("MCP client disconnecting...")


def pytest_sessionfinish(session, exitstatus):
    """Combine coverage data after all tests."""
    # Always try to combine coverage data
    try:
        subprocess.run(["coverage", "combine"], check=False, capture_output=True)
    except FileNotFoundError:
        pass
