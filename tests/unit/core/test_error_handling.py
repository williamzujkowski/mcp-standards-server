"""Tests for enhanced error handling."""

import os
from unittest.mock import patch

import pytest

from src.core.errors import (
    ErrorCode,
    SecureErrorHandler,
    SecurityError,
    ValidationError,
    get_secure_error_handler,
    init_secure_error_handler,
)


class TestSecureErrorHandler:
    """Test secure error handler functionality."""

    @pytest.fixture
    def dev_handler(self):
        """Create error handler for development mode."""
        return SecureErrorHandler(mask_errors=False, log_errors=False)

    @pytest.fixture
    def prod_handler(self):
        """Create error handler for production mode."""
        return SecureErrorHandler(mask_errors=True, log_errors=False)

    def test_sensitive_pattern_sanitization(self, prod_handler):
        """Test that sensitive patterns are sanitized."""
        test_cases = [
            ("Error in /home/user/app/config.py", "Error in [REDACTED]"),
            ("Connection to localhost:5432 failed", "Connection to [REDACTED] failed"),
            ("Invalid password for user", "Invalid [REDACTED] for user"),
            ("Token abc123 expired", "[REDACTED] abc123 expired"),
            ("Secret key not found", "[REDACTED] [REDACTED] not found"),
            ("User email: user@example.com", "User email: [REDACTED]"),
            ("Server IP: 192.168.1.100", "Server IP: [REDACTED]"),
            ("File C:\\Users\\admin\\config.ini", "File [REDACTED]"),
        ]

        for original, _expected in test_cases:
            result = prod_handler.sanitize_error_message(original)
            assert "[REDACTED]" in result, f"Failed to sanitize: {original}"

    def test_stack_trace_removal(self, prod_handler):
        """Test that stack traces are removed."""
        error_with_traceback = """
        Exception occurred:
        Traceback (most recent call last):
          File "/app/main.py", line 10, in <module>
            raise ValueError("Test error")
        ValueError: Test error
        """

        result = prod_handler.sanitize_error_message(error_with_traceback)
        assert "Traceback" not in result
        assert "[STACK_TRACE_REDACTED]" in result

    def test_sql_error_sanitization(self, prod_handler):
        """Test that SQL errors are sanitized."""
        sql_error = "SQL error: SELECT * FROM users WHERE id = 1"
        result = prod_handler.sanitize_error_message(sql_error)
        assert "SELECT * FROM users" not in result
        assert "[SQL_REDACTED]" in result

    def test_file_system_error_sanitization(self, prod_handler):
        """Test that file system errors are sanitized."""
        fs_error = "No such file or directory: /etc/passwd"
        result = prod_handler.sanitize_error_message(fs_error)
        assert result == "File not found"

    def test_development_mode_no_sanitization(self, dev_handler):
        """Test that development mode doesn't sanitize."""
        sensitive_error = "Error in /home/user/secret.key"
        result = dev_handler.sanitize_error_message(sensitive_error)
        assert result == sensitive_error

    def test_mcp_error_handling(self, prod_handler):
        """Test handling of MCP errors."""
        error = ValidationError("Invalid input", "field_name")
        result = prod_handler.handle_exception(error)

        assert "error" in result
        assert result["error"]["code"] == ErrorCode.VALIDATION_INVALID_PARAMETERS.value
        assert "field_name" in result["error"]["field"]

    def test_security_error_detection(self, prod_handler):
        """Test detection of security-related errors."""
        security_errors = [
            ValueError("Potential injection detected"),
            RuntimeError("Malicious content blocked"),
            Exception("Dangerous operation prevented"),
        ]

        for error in security_errors:
            result = prod_handler.handle_exception(error)
            assert result["error"]["code"] == ErrorCode.SECURITY_INVALID_INPUT.value
            assert "Security validation failed" in result["error"]["message"]

    def test_generic_error_masking(self, prod_handler):
        """Test masking of generic errors in production."""
        generic_error = ValueError("Detailed internal error message")
        result = prod_handler.handle_exception(generic_error)

        assert result["error"]["code"] == ErrorCode.SYSTEM_INTERNAL_ERROR.value
        assert result["error"]["message"] == "An internal error occurred"
        assert "suggestion" in result["error"]

    def test_generic_error_development_mode(self, dev_handler):
        """Test that development mode shows actual errors."""
        generic_error = ValueError("Detailed internal error message")
        result = dev_handler.handle_exception(generic_error)

        assert result["error"]["code"] == ErrorCode.SYSTEM_INTERNAL_ERROR.value
        assert "Detailed internal error message" in result["error"]["message"]
        assert result["error"]["type"] == "ValueError"

    def test_security_violation_handling(self, prod_handler):
        """Test handling of security violations."""
        violation_types = [
            "injection",
            "size_limit",
            "malicious_content",
            "rate_limit",
            "blocked_operation",
        ]

        for violation_type in violation_types:
            result = prod_handler.handle_security_violation(violation_type)
            assert "error" in result
            assert result["error"]["code"].startswith("SEC_")
            assert violation_type in result["error"]["message"]

    def test_pydantic_validation_error_handling(self, prod_handler):
        """Test handling of Pydantic validation errors."""

        # Mock Pydantic validation error
        class MockValidationError(Exception):
            def errors(self):
                return [
                    {
                        "loc": ["field1"],
                        "msg": "Field required",
                        "type": "value_error.missing",
                    },
                    {
                        "loc": ["field2", "nested"],
                        "msg": "Invalid type",
                        "type": "type_error.int",
                    },
                ]

        error = MockValidationError("Validation failed")
        result = prod_handler.handle_exception(error)

        assert result["error"]["code"] == ErrorCode.VALIDATION_INVALID_PARAMETERS.value
        assert "validation_errors" in result["error"]["details"]
        assert len(result["error"]["details"]["validation_errors"]) == 2

    def test_logging_behavior(self):
        """Test that errors are logged when enabled."""
        with patch("src.core.errors.logger") as mock_logger:
            handler = SecureErrorHandler(mask_errors=True, log_errors=True)
            error = ValueError("Test error")

            handler.handle_exception(error)

            # Should log the error
            mock_logger.error.assert_called_once()

            # Test security violation logging
            handler.handle_security_violation("injection")
            mock_logger.warning.assert_called_once()


class TestSecurityError:
    """Test security error class."""

    def test_security_error_creation(self):
        """Test security error creation."""
        error = SecurityError("Input validation failed")

        assert error.code == ErrorCode.SECURITY_INVALID_INPUT
        assert error.error_detail.message == "Input validation failed"
        assert (
            error.error_detail.suggestion
            == "Ensure your input follows security guidelines"
        )

    def test_security_error_with_custom_code(self):
        """Test security error with custom code."""
        error = SecurityError(
            "Injection detected", ErrorCode.SECURITY_INJECTION_DETECTED
        )

        assert error.code == ErrorCode.SECURITY_INJECTION_DETECTED
        assert error.error_detail.message == "Injection detected"


class TestGlobalErrorHandler:
    """Test global error handler management."""

    def test_get_secure_error_handler(self):
        """Test getting global error handler."""
        handler1 = get_secure_error_handler()
        handler2 = get_secure_error_handler()

        # Should return same instance
        assert handler1 is handler2

    def test_init_secure_error_handler(self):
        """Test initializing global error handler."""
        handler = init_secure_error_handler(mask_errors=False, log_errors=False)

        assert handler.mask_errors is False
        assert handler.log_errors is False
        assert get_secure_error_handler() is handler

    def test_environment_variable_configuration(self):
        """Test configuration from environment variables."""
        # Test with MCP_MASK_ERRORS=false
        with patch.dict(os.environ, {"MCP_MASK_ERRORS": "false"}):
            # Reset global handler
            import src.core.errors

            src.core.errors._secure_error_handler = None

            handler = get_secure_error_handler()
            assert handler.mask_errors is False

        # Test with MCP_MASK_ERRORS=true (default)
        with patch.dict(os.environ, {"MCP_MASK_ERRORS": "true"}):
            # Reset global handler
            src.core.errors._secure_error_handler = None

            handler = get_secure_error_handler()
            assert handler.mask_errors is True


class TestErrorCodeExtensions:
    """Test new security error codes."""

    def test_security_error_codes_exist(self):
        """Test that security error codes are defined."""
        security_codes = [
            ErrorCode.SECURITY_INVALID_INPUT,
            ErrorCode.SECURITY_INJECTION_DETECTED,
            ErrorCode.SECURITY_REQUEST_TOO_LARGE,
            ErrorCode.SECURITY_MALICIOUS_CONTENT,
            ErrorCode.SECURITY_RATE_LIMIT_EXCEEDED,
            ErrorCode.SECURITY_BLOCKED_OPERATION,
        ]

        for code in security_codes:
            assert code.value.startswith("SEC_")
            assert isinstance(code, ErrorCode)


class TestIntegrationWithExistingErrors:
    """Test integration with existing error handling."""

    def test_validation_error_sanitization(self):
        """Test that validation errors are sanitized."""
        handler = SecureErrorHandler(mask_errors=True, log_errors=False)

        # Validation error with sensitive information
        error = ValidationError("Invalid value in /home/user/config.py", "config_file")
        result = handler.handle_exception(error)

        # Should sanitize the message
        assert "[REDACTED]" in result["error"]["message"]
        assert result["error"]["field"] == "config_file"

    def test_mcp_error_preservation(self):
        """Test that MCP error structure is preserved."""
        handler = SecureErrorHandler(mask_errors=False, log_errors=False)

        error = ValidationError("Test validation error", "test_field")
        result = handler.handle_exception(error)

        expected_structure = {
            "error": {
                "code": ErrorCode.VALIDATION_INVALID_PARAMETERS.value,
                "message": "Test validation error",
                "details": None,
                "field": "test_field",
                "suggestion": "Check the parameter requirements in the tool schema",
            }
        }

        assert result == expected_structure
