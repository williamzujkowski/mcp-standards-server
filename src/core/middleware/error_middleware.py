"""
Error handling middleware for HTTP and WebSocket servers.

Provides consistent error handling, logging, and response formatting
across all server endpoints.
"""

import logging
import time
import traceback
from collections.abc import Awaitable, Callable
from typing import Any, cast

from aiohttp import web

from ..errors import ErrorCode, MCPError, get_secure_error_handler
from ..logging_config import ContextFilter
from ..metrics import get_mcp_metrics

logger = logging.getLogger(__name__)


@web.middleware
async def error_handling_middleware(
    request: web.Request,
    handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
) -> web.StreamResponse:
    """
    Middleware for handling errors in HTTP requests.

    Features:
    - Catches and formats all exceptions
    - Logs errors with context
    - Records metrics
    - Provides secure error responses
    """
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", f"{time.time()}")

    # Set logging context
    ContextFilter.set_context(
        request_id=request_id,
        method=request.method,
        path=request.path,
        remote=request.remote,
    )

    metrics = get_mcp_metrics()

    try:
        # Process request
        response = await handler(request)

        # Log successful request
        duration = time.time() - start_time
        logger.info(
            f"Request completed: {request.method} {request.path}",
            extra={
                "status": response.status,
                "duration": duration,
                "size": response.content_length,
            },
        )

        # Record metrics
        metrics.record_http_request(
            method=request.method,
            path=request.path,
            status=response.status,
            duration=duration,
        )

        return response

    except web.HTTPException as e:
        # Handle aiohttp HTTP exceptions
        duration = time.time() - start_time

        logger.warning(
            f"HTTP exception: {request.method} {request.path}",
            extra={"status": e.status, "reason": e.reason, "duration": duration},
        )

        metrics.record_http_request(
            method=request.method, path=request.path, status=e.status, duration=duration
        )

        # Re-raise to let aiohttp handle it
        raise

    except MCPError as e:
        # Handle our custom errors
        duration = time.time() - start_time

        logger.error(
            f"MCP error: {request.method} {request.path}",
            extra={
                "error_code": e.code.value,
                "error_message": str(e),
                "duration": duration,
            },
        )

        metrics.record_http_request(
            method=request.method,
            path=request.path,
            status=_get_http_status_for_error(e),
            duration=duration,
            error=True,
        )

        metrics.record_error(error_type=type(e).__name__, error_code=e.code.value)

        # Return formatted error response
        return web.json_response(
            e.to_dict(),
            status=_get_http_status_for_error(e),
            headers={"X-Request-ID": request_id},
        )

    except Exception as e:
        # Handle unexpected errors
        duration = time.time() - start_time

        logger.exception(
            f"Unhandled error: {request.method} {request.path}",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "duration": duration,
                "traceback": traceback.format_exc(),
            },
        )

        metrics.record_http_request(
            method=request.method,
            path=request.path,
            status=500,
            duration=duration,
            error=True,
        )

        metrics.record_error(
            error_type=type(e).__name__,
            error_code=ErrorCode.SYSTEM_INTERNAL_ERROR.value,
        )

        # Use secure error handler for response
        error_handler = get_secure_error_handler()
        error_response = error_handler.handle_exception(
            e,
            context={
                "request": f"{request.method} {request.path}",
                "request_id": request_id,
            },
        )

        return web.json_response(
            error_response, status=500, headers={"X-Request-ID": request_id}
        )

    finally:
        # Clear logging context
        ContextFilter.clear_context()


@web.middleware
async def request_logging_middleware(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.Response]]
) -> web.Response:
    """
    Middleware for logging all requests and responses.

    Logs request details at the start and response details at the end.
    """
    # Log request
    logger.info(
        f"Request started: {request.method} {request.path}",
        extra={
            "headers": dict(request.headers),
            "query": dict(request.query),
            "content_type": request.content_type,
            "content_length": request.content_length,
        },
    )

    # Process request
    response = await handler(request)

    # Log response
    logger.info(
        f"Response sent: {request.method} {request.path}",
        extra={
            "status": response.status,
            "content_type": response.content_type,
            "content_length": response.content_length,
            "headers": dict(response.headers),
        },
    )

    return response


class WebSocketErrorHandler:
    """Error handler for WebSocket connections."""

    def __init__(self, ws: web.WebSocketResponse, request_id: str) -> None:
        self.ws = ws
        self.request_id = request_id
        self.logger = logging.getLogger(f"{__name__}.websocket")
        self.metrics = get_mcp_metrics()

    async def handle_error(
        self, error: Exception, context: dict[str, Any] | None = None
    ) -> None:
        """Handle error in WebSocket communication."""
        try:
            # Log error
            self.logger.error(
                f"WebSocket error: {type(error).__name__}",
                extra={
                    "request_id": self.request_id,
                    "error_message": str(error),
                    "context": context,
                    "traceback": (
                        traceback.format_exc()
                        if not isinstance(error, MCPError)
                        else None
                    ),
                },
            )

            # Record metrics
            self.metrics.record_error(
                error_type=type(error).__name__,
                error_code=getattr(
                    error, "code", ErrorCode.SYSTEM_INTERNAL_ERROR
                ).value,
            )

            # Format error response
            if isinstance(error, MCPError):
                error_data = error.to_dict()
            else:
                error_handler = get_secure_error_handler()
                error_data = error_handler.handle_exception(error, context)

            # Send error to client
            error_data["request_id"] = self.request_id
            await self.ws.send_json(error_data)

        except Exception as e:
            # Last resort - log the error handling failure
            self.logger.exception(
                "Failed to handle WebSocket error",
                extra={"original_error": str(error), "handler_error": str(e)},
            )


def _get_http_status_for_error(error: MCPError) -> int:
    """Map MCP error codes to HTTP status codes."""
    status_map = {
        # Authentication errors -> 401
        ErrorCode.AUTH_REQUIRED: 401,
        ErrorCode.AUTH_INVALID_TOKEN: 401,
        ErrorCode.AUTH_EXPIRED_TOKEN: 401,
        ErrorCode.AUTH_INVALID_API_KEY: 401,
        # Authorization errors -> 403
        ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: 403,
        ErrorCode.RESOURCE_ACCESS_DENIED: 403,
        ErrorCode.SECURITY_BLOCKED_OPERATION: 403,
        # Validation errors -> 400
        ErrorCode.VALIDATION_INVALID_PARAMETERS: 400,
        ErrorCode.VALIDATION_MISSING_REQUIRED: 400,
        ErrorCode.VALIDATION_TYPE_MISMATCH: 400,
        ErrorCode.VALIDATION_OUT_OF_RANGE: 400,
        ErrorCode.VALIDATION_PATTERN_MISMATCH: 400,
        # Not found errors -> 404
        ErrorCode.TOOL_NOT_FOUND: 404,
        ErrorCode.RESOURCE_NOT_FOUND: 404,
        ErrorCode.STANDARDS_NOT_FOUND: 404,
        # Rate limit errors -> 429
        ErrorCode.SYSTEM_RATE_LIMIT_EXCEEDED: 429,
        ErrorCode.SECURITY_RATE_LIMIT_EXCEEDED: 429,
        # Service unavailable -> 503
        ErrorCode.SYSTEM_UNAVAILABLE: 503,
        ErrorCode.SYSTEM_MAINTENANCE: 503,
        # Request too large -> 413
        ErrorCode.SECURITY_REQUEST_TOO_LARGE: 413,
        # Default to 500 for system errors
        ErrorCode.SYSTEM_INTERNAL_ERROR: 500,
    }

    return status_map.get(error.code, 500)


def setup_error_handling(app: web.Application) -> None:
    """
    Setup error handling for the application.

    Args:
        app: aiohttp application instance
    """
    # Add middleware
    app.middlewares.append(cast(Any, error_handling_middleware))

    # Add request logging in development
    if logger.isEnabledFor(logging.DEBUG):
        app.middlewares.append(cast(Any, request_logging_middleware))

    # Note: Error handling is managed by the middleware above
    # No need to add catch-all routes which would conflict

    logger.info("Error handling middleware configured")
