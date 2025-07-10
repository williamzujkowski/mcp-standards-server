"""
Middleware components for MCP Standards Server.

Provides error handling, logging, authentication, and other
cross-cutting concerns.
"""

from .error_middleware import (
    WebSocketErrorHandler,
    error_handling_middleware,
    request_logging_middleware,
    setup_error_handling,
)

__all__ = [
    "error_handling_middleware",
    "request_logging_middleware",
    "WebSocketErrorHandler",
    "setup_error_handling",
]
