"""
Unit tests for error handling and logging system.

Tests the comprehensive error handling, logging configuration,
decorators, and middleware components.
"""

import asyncio
import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase

from src.core.decorators import (
    deprecated,
    with_context,
    with_error_handling,
    with_logging,
)
from src.core.errors import (
    AuthenticationError,
    ErrorCode,
    MCPError,
    SecureErrorHandler,
    ValidationError,
)
from src.core.logging_config import (
    ContextFilter,
    ErrorTrackingHandler,
    LoggingConfig,
    setup_logging,
)
from src.core.middleware.error_middleware import setup_error_handling


class TestMCPErrors:
    """Test MCP error classes."""

    def test_mcp_error_creation(self):
        """Test MCPError creation and attributes."""
        error = MCPError(
            code=ErrorCode.VALIDATION_INVALID_PARAMETERS,
            message="Invalid parameter",
            details={"field": "test"},
            field="test_field",
            suggestion="Fix the parameter",
        )

        assert error.code == ErrorCode.VALIDATION_INVALID_PARAMETERS
        assert str(error) == "Invalid parameter"
        assert error.error_detail.field == "test_field"
        assert error.error_detail.suggestion == "Fix the parameter"

    def test_mcp_error_to_dict(self):
        """Test MCPError to_dict conversion."""
        error = MCPError(
            code=ErrorCode.TOOL_NOT_FOUND,
            message="Tool not found",
            details={"tool_name": "test_tool"},
        )

        error_dict = error.to_dict()

        assert error_dict["error"]["code"] == ErrorCode.TOOL_NOT_FOUND.value
        assert error_dict["error"]["message"] == "Tool not found"
        assert error_dict["error"]["details"]["tool_name"] == "test_tool"

    def test_validation_error(self):
        """Test ValidationError specific functionality."""
        error = ValidationError(message="Invalid value", field="test_field")

        assert error.code == ErrorCode.VALIDATION_INVALID_PARAMETERS
        assert error.error_detail.field == "test_field"
        assert error.error_detail.suggestion is not None
        assert "Check the parameter requirements" in error.error_detail.suggestion

    def test_authentication_error(self):
        """Test AuthenticationError specific functionality."""
        error = AuthenticationError(
            message="Invalid token", code=ErrorCode.AUTH_INVALID_TOKEN
        )

        assert error.code == ErrorCode.AUTH_INVALID_TOKEN
        assert str(error) == "Invalid token"
        assert error.error_detail.suggestion is not None
        assert "authentication token" in error.error_detail.suggestion


class TestSecureErrorHandler:
    """Test secure error handler."""

    def test_sanitize_error_message(self):
        """Test error message sanitization."""
        handler = SecureErrorHandler(mask_errors=True)

        # Test file path sanitization - use cross-platform path
        import tempfile

        temp_dir = tempfile.gettempdir()
        secret_path = Path(temp_dir) / "secret.txt"
        message = f"File not found: {secret_path}"
        sanitized = handler.sanitize_error_message(message)
        assert str(secret_path) not in sanitized
        assert "[REDACTED]" in sanitized

        # Test email sanitization
        message = "Error for user test@example.com"
        sanitized = handler.sanitize_error_message(message)
        assert "test@example.com" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_handle_mcp_error(self):
        """Test handling of MCPError."""
        handler = SecureErrorHandler(mask_errors=False)
        error = ValidationError("Test error", "test_field")

        result = handler.handle_exception(error)

        assert result["error"]["code"] == ErrorCode.VALIDATION_INVALID_PARAMETERS.value
        assert result["error"]["message"] == "Test error"
        assert result["error"]["field"] == "test_field"

    def test_handle_generic_error_masked(self):
        """Test handling of generic error with masking."""
        handler = SecureErrorHandler(mask_errors=True)
        error = ValueError("Sensitive information")

        result = handler.handle_exception(error)

        assert result["error"]["code"] == ErrorCode.SYSTEM_INTERNAL_ERROR.value
        assert result["error"]["message"] == "An internal error occurred"
        assert "Sensitive information" not in result["error"]["message"]

    def test_handle_generic_error_unmasked(self):
        """Test handling of generic error without masking."""
        handler = SecureErrorHandler(mask_errors=False)
        error = ValueError("Debug information")

        result = handler.handle_exception(error)

        assert result["error"]["code"] == ErrorCode.SYSTEM_INTERNAL_ERROR.value
        assert "Debug information" in result["error"]["message"]


class TestLoggingConfig:
    """Test logging configuration."""

    def test_logging_config_creation(self):
        """Test LoggingConfig creation."""
        import tempfile
        from pathlib import Path

        # Use cross-platform temp directory
        temp_dir = tempfile.gettempdir()
        log_dir = Path(temp_dir) / "logs"

        config = LoggingConfig(level="DEBUG", format="json", log_dir=str(log_dir))

        assert config.level == logging.DEBUG
        assert config.format == "json"
        assert str(config.log_dir) == str(log_dir)

    def test_setup_logging(self):
        """Test logging setup."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(
                level="INFO",
                format="text",
                log_dir=temp_dir,
                enable_error_tracking=True,
            )

            # Store initial handler count to clean up properly
            initial_handlers = list(logging.getLogger().handlers)

            try:
                error_handler = setup_logging(config)

                assert isinstance(error_handler, ErrorTrackingHandler)
                assert logging.getLogger().level == logging.INFO
                assert len(logging.getLogger().handlers) > 0
            finally:
                # Clean up handlers to prevent Windows file locking issues
                root_logger = logging.getLogger()
                for handler in list(root_logger.handlers):
                    if handler not in initial_handlers:
                        handler.close()
                        root_logger.removeHandler(handler)

    def test_context_filter(self):
        """Test context filter functionality."""
        # Set context
        ContextFilter.set_context(request_id="test-123", user_id="user-456")

        # Create a test record directly
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Create and test the context filter
        context_filter = ContextFilter()
        context_filter.filter(record)

        # Check that context was added to record
        assert hasattr(record, "request_id")
        assert record.request_id == "test-123"
        assert hasattr(record, "user_id")
        assert record.user_id == "user-456"

        # Clear context
        ContextFilter.clear_context()

    def test_error_tracking_handler(self):
        """Test error tracking handler."""
        handler = ErrorTrackingHandler()

        # Create error record
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Test error",
            args=(),
            exc_info=None,
        )

        handler.emit(record)

        stats = handler.get_error_stats()
        assert stats["total_errors"] == 1
        assert stats["error_counts"]["test"] == 1
        assert len(stats["last_errors"]) == 1


class TestDecorators:
    """Test error handling decorators."""

    def test_with_error_handling_decorator(self):
        """Test with_error_handling decorator."""

        @with_error_handling(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED,
            raise_errors=False,
            default_return="default",
        )
        def failing_function():
            raise ValueError("Test error")

        result = failing_function()
        assert result == "default"

    def test_with_error_handling_decorator_async(self):
        """Test with_error_handling decorator with async function."""

        @with_error_handling(
            error_code=ErrorCode.TOOL_EXECUTION_FAILED, raise_errors=True
        )
        async def failing_async_function():
            raise ValueError("Test error")

        with pytest.raises(MCPError) as exc_info:
            asyncio.run(failing_async_function())

        assert exc_info.value.code == ErrorCode.TOOL_EXECUTION_FAILED
        assert "Test error" in str(exc_info.value)

    def test_with_logging_decorator(self):
        """Test with_logging decorator."""
        with patch("src.core.decorators.logger") as mock_logger:

            @with_logging(level=logging.DEBUG, log_args=True)
            def test_function(arg1, arg2="default"):
                return "result"

            result = test_function("test", arg2="value")

            assert result == "result"
            assert mock_logger.log.call_count >= 2  # Start and end

    def test_with_context_decorator(self):
        """Test with_context decorator."""

        @with_context(operation="test_op", module="test_module")
        def test_function():
            # Context should be set during execution
            context = ContextFilter._context.data
            assert context.get("operation") == "test_op"
            assert context.get("module") == "test_module"
            return "result"

        result = test_function()
        assert result == "result"

    def test_deprecated_decorator(self):
        """Test deprecated decorator."""
        with patch("src.core.decorators.logger") as mock_logger:

            @deprecated(
                reason="Use new_function instead",
                version="1.0.0",
                alternative="new_function",
            )
            def old_function():
                return "old"

            result = old_function()

            assert result == "old"
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "deprecated" in warning_msg
            assert "new_function" in warning_msg


class TestErrorMiddleware(AioHTTPTestCase):
    """Test error handling middleware."""

    async def get_application(self):
        """Create test application."""
        app = web.Application()
        setup_error_handling(app)

        # Add test routes
        app.router.add_get("/success", self.success_handler)
        app.router.add_get("/error", self.error_handler)
        app.router.add_get("/mcp-error", self.mcp_error_handler)

        return app

    async def success_handler(self, request):
        """Handler that returns success."""
        return web.json_response({"status": "success"})

    async def error_handler(self, request):
        """Handler that raises generic error."""
        raise ValueError("Test error")

    async def mcp_error_handler(self, request):
        """Handler that raises MCP error."""
        raise ValidationError("Invalid input", "test_field")

    async def test_success_response(self):
        """Test successful request handling."""
        resp = await self.client.request("GET", "/success")
        assert resp.status == 200

        data = await resp.json()
        assert data["status"] == "success"

    async def test_generic_error_handling(self):
        """Test generic error handling."""
        resp = await self.client.request("GET", "/error")
        assert resp.status == 500

        data = await resp.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.SYSTEM_INTERNAL_ERROR.value

    async def test_mcp_error_handling(self):
        """Test MCP error handling."""
        resp = await self.client.request("GET", "/mcp-error")
        assert resp.status == 400

        data = await resp.json()
        assert "error" in data
        assert data["error"]["code"] == ErrorCode.VALIDATION_INVALID_PARAMETERS.value
        assert data["error"]["field"] == "test_field"


class TestIntegration:
    """Integration tests for error handling system."""

    def test_end_to_end_error_flow(self):
        """Test complete error handling flow."""
        # Setup logging
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(
                level="INFO",
                format="json",
                log_dir=temp_dir,
                enable_error_tracking=True,
            )

            # Store initial handler count to clean up properly
            initial_handlers = list(logging.getLogger().handlers)

            try:
                error_handler = setup_logging(config)

                # Create function with error handling
                @with_error_handling(
                    error_code=ErrorCode.TOOL_EXECUTION_FAILED, log_errors=True
                )
                def test_function():
                    raise ValueError("Test error")

                # Execute and verify error is handled
                with pytest.raises(MCPError) as exc_info:
                    test_function()

                assert exc_info.value.code == ErrorCode.TOOL_EXECUTION_FAILED

                # Check error was tracked
                stats = error_handler.get_error_stats()
                assert stats["total_errors"] > 0
            finally:
                # Clean up handlers to prevent Windows file locking issues
                root_logger = logging.getLogger()
                for handler in list(root_logger.handlers):
                    if handler not in initial_handlers:
                        handler.close()
                        root_logger.removeHandler(handler)

    def test_logging_context_propagation(self):
        """Test that logging context is properly propagated."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = LoggingConfig(level="DEBUG", format="json", log_dir=temp_dir)

            # Store initial handler count to clean up properly
            initial_handlers = list(logging.getLogger().handlers)

            try:
                setup_logging(config)

                # Set context and log
                ContextFilter.set_context(request_id="test-123")

                # Create a test record directly
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="test.py",
                    lineno=1,
                    msg="Test message",
                    args=(),
                    exc_info=None,
                )

                # Create and test the context filter
                context_filter = ContextFilter()
                context_filter.filter(record)

                assert hasattr(record, "request_id")
                assert record.request_id == "test-123"
            finally:
                # Clean up handlers to prevent Windows file locking issues
                root_logger = logging.getLogger()
                for handler in list(root_logger.handlers):
                    if handler not in initial_handlers:
                        handler.close()
                        root_logger.removeHandler(handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
