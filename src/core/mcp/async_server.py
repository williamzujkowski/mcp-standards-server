"""
Async MCP server implementation with enhanced performance and scalability.

This module provides:
- Fully async request handling with connection pooling
- Request batching and pipelining
- Connection lifecycle management
- Performance monitoring and metrics
- Enhanced error handling and recovery
- Request throttling and rate limiting
- WebSocket and HTTP support
- Graceful shutdown handling
"""

import asyncio
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, cast

from aiohttp import WSMsgType, web
from aiohttp.web_ws import WebSocketResponse

# Handle aioredis Python 3.12 compatibility issue
try:
    import aioredis
except (TypeError, ImportError) as e:
    # Python 3.12 compatibility issue with aioredis TimeoutError
    if "duplicate base class TimeoutError" in str(e):
        # Mock aioredis for compatibility
        class MockRedis:
            @staticmethod
            async def from_url(*args: Any, **kwargs: Any) -> None:
                return None

        aioredis = type(
            "aioredis", (), {"Redis": MockRedis, "from_url": MockRedis.from_url}
        )()
    else:
        raise

from ..errors import get_secure_error_handler
from ..performance.memory_manager import get_memory_manager
from ..performance.metrics import record_metric
from ..rate_limiter import get_rate_limiter
from ..security import get_security_middleware
from .handlers import StandardsHandler

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Connection states."""

    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ConnectionInfo:
    """Information about a client connection."""

    id: str
    remote_addr: str
    user_agent: str | None
    connection_time: float
    last_activity: float
    state: ConnectionState = ConnectionState.CONNECTING
    request_count: int = 0
    error_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0

    def update_activity(self) -> None:
        """Update last activity timestamp."""
        self.last_activity = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "remote_addr": self.remote_addr,
            "user_agent": self.user_agent,
            "connection_time": self.connection_time,
            "last_activity": self.last_activity,
            "state": self.state.value,
            "request_count": self.request_count,
            "error_count": self.error_count,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "duration": time.time() - self.connection_time,
        }


@dataclass
class ServerConfig:
    """Configuration for the async MCP server."""

    # Server settings
    host: str = "127.0.0.1"
    port: int = 8080
    max_connections: int = 1000
    connection_timeout: float = 300.0  # 5 minutes

    # WebSocket settings
    enable_websocket: bool = True
    websocket_path: str = "/ws"
    websocket_ping_interval: float = 30.0
    websocket_ping_timeout: float = 10.0

    # HTTP settings
    enable_http: bool = True
    http_path: str = "/mcp"
    max_request_size: int = 1024 * 1024  # 1MB

    # Request processing
    enable_request_batching: bool = True
    batch_size: int = 100
    batch_timeout: float = 0.1  # seconds
    max_concurrent_requests: int = 500

    # Performance settings
    enable_request_compression: bool = True
    compression_threshold: int = 1024
    enable_connection_pooling: bool = True
    pool_size: int = 20

    # Security settings
    enable_authentication: bool = False
    enable_rate_limiting: bool = True
    enable_cors: bool = True
    cors_origins: list[str] = field(default_factory=lambda: ["*"])

    # Monitoring settings
    enable_metrics: bool = True
    metrics_interval: float = 60.0
    enable_health_check: bool = True
    health_check_path: str = "/health"

    # Cleanup settings
    cleanup_interval: float = 300.0  # 5 minutes
    inactive_connection_timeout: float = 1800.0  # 30 minutes


@dataclass
class RequestMetrics:
    """Metrics for request processing."""

    total_requests: int = 0
    active_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0

    # Timing metrics
    request_times: deque = field(default_factory=lambda: deque(maxlen=1000))
    batch_times: deque = field(default_factory=lambda: deque(maxlen=100))

    # Connection metrics
    active_connections: int = 0
    total_connections: int = 0
    connection_errors: int = 0

    # Data metrics
    bytes_sent: int = 0
    bytes_received: int = 0

    # Method metrics
    method_counts: dict[str, int] = field(default_factory=dict)

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary."""
        avg_request_time = (
            sum(self.request_times) / len(self.request_times)
            if self.request_times
            else 0
        )
        avg_batch_time = (
            sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
        )

        error_rate = (
            self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        )

        return {
            "total_requests": self.total_requests,
            "active_requests": self.active_requests,
            "completed_requests": self.completed_requests,
            "failed_requests": self.failed_requests,
            "error_rate": error_rate,
            "average_request_time": avg_request_time,
            "average_batch_time": avg_batch_time,
            "active_connections": self.active_connections,
            "total_connections": self.total_connections,
            "connection_errors": self.connection_errors,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "method_counts": dict(self.method_counts),
        }


class RequestBatcher:
    """Batches requests for more efficient processing."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.batch_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.batch_worker_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()
        self.pending_requests: dict[str, asyncio.Future[Any]] = {}
        self.batch_metrics = RequestMetrics()

    async def start(self) -> None:
        """Start request batching."""
        if self.config.enable_request_batching:
            self.batch_worker_task = asyncio.create_task(self._batch_worker())

    async def stop(self) -> None:
        """Stop request batching."""
        self.shutdown_event.set()
        if self.batch_worker_task:
            self.batch_worker_task.cancel()
            try:
                await self.batch_worker_task
            except asyncio.CancelledError:
                pass

    async def queue_request(self, request: dict[str, Any], connection_id: str) -> Any:
        """Queue a request for batch processing."""
        if not self.config.enable_request_batching:
            return None

        # Create future for result
        result_future: asyncio.Future[Any] = asyncio.Future()
        request_id = str(uuid.uuid4())

        # Add to batch queue
        await self.batch_queue.put(
            {
                "id": request_id,
                "request": request,
                "connection_id": connection_id,
                "future": result_future,
                "timestamp": time.time(),
            }
        )

        # Store in pending requests
        self.pending_requests[request_id] = result_future

        try:
            # Wait for result
            return await result_future
        finally:
            # Clean up
            self.pending_requests.pop(request_id, None)

    async def _batch_worker(self) -> None:
        """Worker task for processing request batches."""
        while not self.shutdown_event.is_set():
            try:
                batch = []

                # Collect batch items
                try:
                    # Wait for first item
                    item = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=self.config.batch_timeout
                    )
                    batch.append(item)

                    # Collect additional items
                    while len(batch) < self.config.batch_size:
                        try:
                            item = await asyncio.wait_for(
                                self.batch_queue.get(),
                                timeout=0.001,  # Very short timeout
                            )
                            batch.append(item)
                        except asyncio.TimeoutError:
                            break

                except asyncio.TimeoutError:
                    continue

                # Process batch
                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch worker: {e}")
                await asyncio.sleep(0.1)

    async def _process_batch(self, batch: list[dict[str, Any]]) -> None:
        """Process a batch of requests."""
        start_time = time.time()

        try:
            # Group requests by type for optimization
            request_groups = defaultdict(list)
            for item in batch:
                method = item["request"].get("method", "unknown")
                request_groups[method].append(item)

            # Process each group
            for _method, items in request_groups.items():
                try:
                    # For now, process individually
                    # In a real implementation, you'd optimize for bulk operations
                    for item in items:
                        try:
                            # Mock processing
                            result = await self._process_single_request(item["request"])
                            item["future"].set_result(result)
                        except Exception as e:
                            item["future"].set_exception(e)

                except Exception as e:
                    # Set exception for all items in group
                    for item in items:
                        if not item["future"].done():
                            item["future"].set_exception(e)

            # Update metrics
            batch_time = time.time() - start_time
            self.batch_metrics.batch_times.append(batch_time)

        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Set exception for all items
            for item in batch:
                if not item["future"].done():
                    item["future"].set_exception(e)

    async def _process_single_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a single request."""
        # This would be replaced with actual request processing logic
        method = request.get("method", "unknown")

        # Simulate processing time
        await asyncio.sleep(0.001)

        return {
            "id": str(uuid.uuid4()),
            "result": f"Processed {method}",
            "timestamp": time.time(),
        }


class ConnectionManager:
    """Manages client connections and their lifecycle."""

    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.connections: dict[str, ConnectionInfo] = {}
        self.websockets: dict[str, WebSocketResponse] = {}
        self.connection_lock = threading.Lock()

        # Cleanup task
        self.cleanup_task: asyncio.Task[None] | None = None
        self.shutdown_event = asyncio.Event()

        # Connection limits
        self.max_connections = config.max_connections
        self.connection_semaphore = asyncio.Semaphore(self.max_connections)

    async def start(self) -> None:
        """Start connection management."""
        self.cleanup_task = asyncio.create_task(self._cleanup_worker())

    async def stop(self) -> None:
        """Stop connection management."""
        self.shutdown_event.set()

        # Close all connections
        await self._close_all_connections()

        # Stop cleanup task
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

    async def add_connection(
        self,
        connection_id: str,
        remote_addr: str,
        user_agent: str | None = None,
        websocket: WebSocketResponse | None = None,
    ) -> bool:
        """Add a new connection."""
        # Check connection limit
        if len(self.connections) >= self.max_connections:
            logger.warning(f"Connection limit reached: {len(self.connections)}")
            return False

        # Acquire semaphore
        await self.connection_semaphore.acquire()

        try:
            with self.connection_lock:
                if connection_id in self.connections:
                    return False

                # Create connection info
                connection_info = ConnectionInfo(
                    id=connection_id,
                    remote_addr=remote_addr,
                    user_agent=user_agent,
                    connection_time=time.time(),
                    last_activity=time.time(),
                    state=ConnectionState.CONNECTED,
                )

                self.connections[connection_id] = connection_info

                if websocket:
                    self.websockets[connection_id] = websocket

                logger.info(f"Connection added: {connection_id} from {remote_addr}")
                return True

        except Exception as e:
            logger.error(f"Error adding connection: {e}")
            self.connection_semaphore.release()
            return False

    async def remove_connection(self, connection_id: str) -> None:
        """Remove a connection."""
        with self.connection_lock:
            connection_info = self.connections.pop(connection_id, None)
            websocket = self.websockets.pop(connection_id, None)

        if connection_info:
            connection_info.state = ConnectionState.DISCONNECTED
            logger.info(f"Connection removed: {connection_id}")

        if websocket and not websocket.closed:
            await websocket.close()

        # Release semaphore
        self.connection_semaphore.release()

    def update_connection_activity(self, connection_id: str) -> None:
        """Update connection activity timestamp."""
        with self.connection_lock:
            if connection_id in self.connections:
                self.connections[connection_id].update_activity()

    def get_connection_info(self, connection_id: str) -> ConnectionInfo | None:
        """Get connection information."""
        with self.connection_lock:
            return self.connections.get(connection_id)

    def get_all_connections(self) -> list[ConnectionInfo]:
        """Get all connection information."""
        with self.connection_lock:
            return list(self.connections.values())

    async def _cleanup_worker(self) -> None:
        """Worker task for cleaning up inactive connections."""
        while not self.shutdown_event.is_set():
            try:
                await self._cleanup_inactive_connections()
                await asyncio.sleep(self.config.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(self.config.cleanup_interval)

    async def _cleanup_inactive_connections(self) -> None:
        """Clean up inactive connections."""
        current_time = time.time()
        inactive_connections = []

        with self.connection_lock:
            for connection_id, connection_info in self.connections.items():
                if (
                    current_time - connection_info.last_activity
                ) > self.config.inactive_connection_timeout:
                    inactive_connections.append(connection_id)

        # Remove inactive connections
        for connection_id in inactive_connections:
            logger.info(f"Removing inactive connection: {connection_id}")
            await self.remove_connection(connection_id)

    async def _close_all_connections(self) -> None:
        """Close all connections."""
        connection_ids = list(self.connections.keys())

        for connection_id in connection_ids:
            await self.remove_connection(connection_id)

    def get_connection_stats(self) -> dict[str, Any]:
        """Get connection statistics."""
        with self.connection_lock:
            total_connections = len(self.connections)
            active_connections = sum(
                1
                for conn in self.connections.values()
                if conn.state == ConnectionState.CONNECTED
            )

            # Calculate average connection duration
            current_time = time.time()
            durations = [
                current_time - conn.connection_time
                for conn in self.connections.values()
            ]
            avg_duration = sum(durations) / len(durations) if durations else 0

            return {
                "total_connections": total_connections,
                "active_connections": active_connections,
                "max_connections": self.max_connections,
                "average_connection_duration": avg_duration,
                "websocket_connections": len(self.websockets),
            }


class MCPSession:
    """MCP session management for individual client connections."""

    def __init__(self, session_id: str, server: Any, reader: Any, writer: Any) -> None:
        self.id = session_id
        self.server = server
        self.reader = reader
        self.writer = writer
        self.authenticated = False
        self.client_info = {
            "address": writer.get_extra_info("peername") if writer else None,
            "connected_at": time.time(),
        }
        self.last_activity = time.time()

    async def handle(self) -> None:
        """Handle session messages."""
        try:
            while True:
                try:
                    line = await self.reader.readline()
                    if not line:
                        break

                    # Parse message
                    message = json.loads(line.decode().strip())

                    # Process message
                    await self._process_message(message)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error handling session message: {e}")
                    await self.send_error_message(str(e))
        finally:
            await self.close()

    async def _process_message(self, message: dict[str, Any]) -> None:
        """Process a single message."""
        self.last_activity = time.time()

        msg_type = message.get("type", "unknown")

        if msg_type == "hello":
            await self.send_message(
                {
                    "type": "hello",
                    "version": "1.0.0",
                    "capabilities": ["tools", "resources"],
                }
            )
        elif msg_type == "ping":
            await self.send_message({"type": "pong"})
        elif msg_type == "request":
            await self._handle_request(message)
        else:
            await self.send_error_message(f"Unknown message type: {msg_type}")

    async def _handle_request(self, message: dict[str, Any]) -> None:
        """Handle a request message."""
        request_id = message.get("id")
        method = message.get("method")
        params = message.get("params", {})

        try:
            # Check authentication if required
            if (
                self.server.config.get("auth", {}).get("enabled", False)
                and not self.authenticated
            ):
                raise Exception("Authentication required")

            # Call tool on server
            result = await self.server.mcp_server._execute_tool(method, params)

            await self.send_message(
                {"type": "response", "id": request_id, "result": result}
            )
        except Exception as e:
            await self.send_message(
                {"type": "error", "id": request_id, "error": str(e)}
            )

    async def send_message(self, message: dict[str, Any]) -> None:
        """Send a message to the client."""
        if self.writer and not self.writer.is_closing():
            data = json.dumps(message).encode() + b"\n"
            self.writer.write(data)
            await self.writer.drain()

    async def send_error_message(self, error: str) -> None:
        """Send an error message to the client."""
        await self.send_message({"type": "error", "error": error})

    async def close(self) -> None:
        """Close the session."""
        if self.writer and not self.writer.is_closing():
            self.writer.close()
            await self.writer.wait_closed()

        # Remove from server
        await self.server._remove_session(self.id)

    def get_info(self) -> dict[str, Any]:
        """Get session information."""
        return {
            "id": self.id,
            "authenticated": self.authenticated,
            "client_info": self.client_info,
            "last_activity": self.last_activity,
            "connected_at": self.client_info.get("connected_at"),
        }


class AsyncMCPServer:
    """Async MCP server with enhanced performance."""

    def __init__(
        self, config: ServerConfig | None = None, standards_engine: Any = None
    ) -> None:
        self.config = config or ServerConfig()
        self.standards_engine = standards_engine

        # Core components
        self.connection_manager = ConnectionManager(self.config)
        self.request_batcher = RequestBatcher(self.config)
        self.metrics = RequestMetrics()

        # Session management
        self.sessions: dict[str, MCPSession] = {}

        # Handlers
        self.handlers = {}
        if standards_engine:
            self.handlers["standards"] = StandardsHandler(standards_engine)

        # Middleware
        self.security_middleware = get_security_middleware()
        self.rate_limiter = get_rate_limiter()
        self.error_handler = get_secure_error_handler()

        # HTTP app
        self.app: web.Application | None = None
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        # Tasks
        self.metrics_task: asyncio.Task[None] | None = None
        self.running = False
        self.shutdown_event = asyncio.Event()

        # Request semaphore
        self.request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Memory manager
        self.memory_manager = get_memory_manager()

    async def start(self) -> None:
        """Start the async MCP server."""
        if self.running:
            return

        logger.info("Starting async MCP server...")

        # Start components
        await self.connection_manager.start()
        await self.request_batcher.start()

        # Create HTTP app
        self.app = web.Application()

        # Add routes
        if self.config.enable_http:
            self.app.router.add_post(self.config.http_path, self._handle_http_request)

        if self.config.enable_websocket:
            self.app.router.add_get(self.config.websocket_path, self._handle_websocket)

        if self.config.enable_health_check:
            self.app.router.add_get(
                self.config.health_check_path, self._handle_health_check
            )

        # Add middleware
        if self.config.enable_cors:
            self.app.middlewares.append(self._cors_middleware)

        # Start HTTP server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(
            self.runner, host=self.config.host, port=self.config.port
        )
        await self.site.start()

        # Start metrics collection
        if self.config.enable_metrics:
            self.metrics_task = asyncio.create_task(self._metrics_worker())

        self.running = True
        logger.info(
            f"Async MCP server started on {self.config.host}:{self.config.port}"
        )

    async def stop(self) -> None:
        """Stop the async MCP server."""
        if not self.running:
            return

        logger.info("Stopping async MCP server...")

        # Signal shutdown
        self.shutdown_event.set()

        # Stop metrics collection
        if self.metrics_task:
            self.metrics_task.cancel()
            try:
                await self.metrics_task
            except asyncio.CancelledError:
                pass

        # Close all sessions
        session_close_tasks = []
        for session in list(self.sessions.values()):
            if hasattr(session, "close"):
                session_close_tasks.append(session.close())

        if session_close_tasks:
            await asyncio.gather(*session_close_tasks, return_exceptions=True)

        # Clear sessions
        self.sessions.clear()

        # Stop components
        await self.connection_manager.stop()
        await self.request_batcher.stop()

        # Stop HTTP server
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        self.running = False
        logger.info("Async MCP server stopped")

    async def _handle_http_request(self, request: web.Request) -> web.Response:
        """Handle HTTP MCP request."""
        connection_id = str(uuid.uuid4())

        try:
            # Add connection
            remote_addr = request.remote or "unknown"
            await self.connection_manager.add_connection(
                connection_id, remote_addr, request.headers.get("User-Agent")
            )

            # Read request data
            try:
                data = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

            # Process request
            result = await self._process_request(data, connection_id)

            # Create response
            response = web.json_response(result)

            # Add compression if enabled
            if self.config.enable_request_compression:
                response.enable_compression()

            return response

        except Exception as e:
            logger.error(f"Error handling HTTP request: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)
        finally:
            # Remove connection
            await self.connection_manager.remove_connection(connection_id)

    async def _handle_websocket(self, request: web.Request) -> WebSocketResponse:
        """Handle WebSocket MCP connection."""
        ws = WebSocketResponse()
        await ws.prepare(request)

        connection_id = str(uuid.uuid4())

        try:
            # Add connection
            remote_addr = request.remote or "unknown"
            success = await self.connection_manager.add_connection(
                connection_id, remote_addr, request.headers.get("User-Agent"), ws
            )

            if not success:
                await ws.close(code=1013, message=b"Server overloaded")
                return ws

            # Handle messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        result = await self._process_request(data, connection_id)
                        await ws.send_str(json.dumps(result))
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {e}")
                        await ws.send_str(
                            json.dumps({"error": "Processing error", "message": str(e)})
                        )
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {ws.exception()}")
                    break
                elif msg.type == WSMsgType.CLOSE:
                    break

        except Exception as e:
            logger.error(f"Error handling WebSocket: {e}")
        finally:
            # Remove connection
            await self.connection_manager.remove_connection(connection_id)

        return ws

    async def _handle_health_check(self, request: web.Request) -> web.Response:
        """Handle health check request."""
        try:
            health_data = await self.get_health_status()
            return web.json_response(health_data)
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)

    async def _process_request(
        self, request: dict[str, Any], connection_id: str
    ) -> dict[str, Any]:
        """Process an MCP request."""
        # Acquire request semaphore
        async with self.request_semaphore:
            start_time = time.time()

            try:
                # Update connection activity
                self.connection_manager.update_connection_activity(connection_id)

                # Update metrics
                self.metrics.total_requests += 1
                self.metrics.active_requests += 1

                # Apply security middleware
                try:
                    request = self.security_middleware.validate_and_sanitize_request(
                        request
                    )
                except Exception as e:
                    return {"error": "Security validation failed", "message": str(e)}

                # Apply rate limiting
                is_allowed, limit_info = self.rate_limiter.check_all_limits(
                    connection_id
                )
                if not is_allowed:
                    return {"error": "Rate limit exceeded", "rate_limit": limit_info}

                # Get method
                method = request.get("method")
                if not method:
                    return {"error": "Missing method"}

                # Update method metrics
                self.metrics.method_counts[method] = (
                    self.metrics.method_counts.get(method, 0) + 1
                )

                # Process request
                if self.config.enable_request_batching:
                    result = await self.request_batcher.queue_request(
                        request, connection_id
                    )
                else:
                    result = await self._handle_request_direct(request, connection_id)

                # Update metrics
                self.metrics.completed_requests += 1

                # Record request time
                request_time = time.time() - start_time
                self.metrics.request_times.append(request_time)

                # Record performance metrics
                record_metric(
                    "mcp_request_duration_seconds", request_time, {"method": method}
                )
                record_metric("mcp_request_count", 1, {"method": method})

                return cast(dict[str, Any], result)

            except Exception as e:
                # Update error metrics
                self.metrics.failed_requests += 1

                # Record error metrics
                record_metric(
                    "mcp_request_errors",
                    1,
                    {"method": method or "unknown", "error_type": type(e).__name__},
                )

                # Handle error
                return self.error_handler.handle_exception(
                    e, context={"method": method, "connection_id": connection_id}
                )
            finally:
                self.metrics.active_requests -= 1

    async def _handle_request_direct(
        self, request: dict[str, Any], connection_id: str
    ) -> dict[str, Any]:
        """Handle request directly without batching."""
        method = request.get("method")
        params = request.get("params", {})

        if method == "list_tools":
            return await self._list_tools()
        elif method == "call_tool":
            return await self._call_tool(params)
        else:
            return {"error": f"Unknown method: {method}"}

    async def _list_tools(self) -> dict[str, Any]:
        """List available tools."""
        tools = []

        for handler in self.handlers.values():
            if hasattr(handler, "get_tools"):
                handler_tools = await handler.get_tools()
                tools.extend(handler_tools)

        return {"tools": tools}

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        """Call a specific tool."""
        tool_name = params.get("name")
        tool_args = params.get("arguments", {})

        if not tool_name:
            return {"error": "Missing tool name"}

        # Find appropriate handler
        for handler in self.handlers.values():
            if hasattr(handler, "handle_tool"):
                result = await handler.handle_tool(tool_name, tool_args)
                if result is not None:
                    return result

        return {"error": f"Tool not found: {tool_name}"}

    @web.middleware
    async def _cors_middleware(
        self, request: web.Request, handler: Any
    ) -> web.Response:
        """CORS middleware."""
        response = await handler(request)

        if self.config.enable_cors:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "Content-Type, Authorization"
            )

        return cast(web.Response, response)

    async def _metrics_worker(self) -> None:
        """Worker task for metrics collection."""
        while not self.shutdown_event.is_set():
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.metrics_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.config.metrics_interval)

    async def _collect_metrics(self) -> None:
        """Collect server metrics."""
        # Connection metrics
        connection_stats = self.connection_manager.get_connection_stats()
        record_metric("mcp_active_connections", connection_stats["active_connections"])
        record_metric("mcp_total_connections", connection_stats["total_connections"])

        # Request metrics
        metrics_summary = self.metrics.get_summary()
        record_metric("mcp_total_requests", metrics_summary["total_requests"])
        record_metric("mcp_active_requests", metrics_summary["active_requests"])
        record_metric("mcp_request_error_rate", metrics_summary["error_rate"])

        # Memory metrics
        memory_stats = self.memory_manager.get_memory_stats()
        record_metric(
            "mcp_memory_usage_mb", memory_stats["memory_stats"]["current_usage_mb"]
        )

    async def get_health_status(self) -> dict[str, Any]:
        """Get server health status."""
        connection_stats = self.connection_manager.get_connection_stats()
        metrics_summary = self.metrics.get_summary()

        # Determine overall health
        health_status = "healthy"
        if metrics_summary["error_rate"] > 0.1:  # 10% error rate
            health_status = "degraded"
        if connection_stats["active_connections"] >= self.config.max_connections:
            health_status = "overloaded"

        return {
            "status": health_status,
            "timestamp": time.time(),
            "version": "1.0.0",
            "uptime": time.time()
            - (
                self.metrics.request_times[0]
                if self.metrics.request_times
                else time.time()
            ),
            "connections": connection_stats,
            "requests": metrics_summary,
            "memory": self.memory_manager.get_memory_stats()["memory_stats"],
            "config": {
                "max_connections": self.config.max_connections,
                "max_concurrent_requests": self.config.max_concurrent_requests,
                "enable_batching": self.config.enable_request_batching,
            },
        }

    def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "metrics": self.metrics.get_summary(),
            "connections": self.connection_manager.get_connection_stats(),
            "batch_metrics": self.request_batcher.batch_metrics.get_summary(),
            "memory": self.memory_manager.get_memory_stats(),
        }

    def get_active_connections(self) -> list[dict[str, Any]]:
        """Get information about active connections."""
        connections = self.connection_manager.get_all_connections()
        return [conn.to_dict() for conn in connections]

    def _create_session(self, reader: Any, writer: Any) -> MCPSession:
        """Create a new MCP session."""
        session_id = str(uuid.uuid4())
        session = MCPSession(session_id, self, reader, writer)
        self.sessions[session_id] = session
        return session

    async def _remove_session(self, session_id: str) -> None:
        """Remove a session from the server."""
        if session_id in self.sessions:
            session = self.sessions.pop(session_id)
            try:
                if hasattr(session, "close"):
                    await session.close()
            except Exception:
                pass  # Session already closed or error  # nosec B110

    async def _handle_client(self, reader: Any, writer: Any) -> None:
        """Handle a new client connection."""
        try:
            # Create and handle session
            session = self._create_session(reader, writer)
            await session.handle()
        except Exception as e:
            logger.error(f"Error handling client connection: {e}")
            if writer and not writer.is_closing():
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass  # Ignore cleanup errors  # nosec B110

    async def broadcast_message(self, message: dict[str, Any]) -> None:
        """Broadcast a message to all active sessions."""
        tasks = []
        for session in self.sessions.values():
            if hasattr(session, "send_message"):
                tasks.append(session.send_message(message))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def get_stats(self) -> dict[str, Any]:
        """Get server statistics."""
        return {
            "running": self.running,
            "sessions": len(self.sessions),
            "active_connections": len(self.connection_manager.connections),
            "total_requests": self.metrics.completed_requests
            + self.metrics.failed_requests,
            "completed_requests": self.metrics.completed_requests,
            "failed_requests": self.metrics.failed_requests,
            "active_requests": self.metrics.active_requests,
        }


# Factory function
async def create_async_mcp_server(
    config: ServerConfig | None = None,
    standards_engine: Any = None,
    auto_start: bool = True,
) -> AsyncMCPServer:
    """Create and optionally start an async MCP server."""
    server = AsyncMCPServer(config, standards_engine)

    if auto_start:
        await server.start()

    return server
