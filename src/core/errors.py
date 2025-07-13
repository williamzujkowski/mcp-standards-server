"""
Structured error handling for MCP server.

Provides standardized error codes and error response formatting with security controls.
"""

import logging
import os
import re
from enum import Enum
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Standardized error codes for MCP operations."""

    # Authentication errors (1000-1099)
    AUTH_REQUIRED = "AUTH_001"
    AUTH_INVALID_TOKEN = "AUTH_002"  # nosec B105
    AUTH_EXPIRED_TOKEN = "AUTH_003"  # nosec B105
    AUTH_INVALID_API_KEY = "AUTH_004"
    AUTH_INSUFFICIENT_PERMISSIONS = "AUTH_005"

    # Validation errors (2000-2099)
    VALIDATION_INVALID_PARAMETERS = "VAL_001"
    VALIDATION_MISSING_REQUIRED = "VAL_002"
    VALIDATION_TYPE_MISMATCH = "VAL_003"
    VALIDATION_OUT_OF_RANGE = "VAL_004"
    VALIDATION_PATTERN_MISMATCH = "VAL_005"

    # Tool errors (3000-3099)
    TOOL_NOT_FOUND = "TOOL_001"
    TOOL_EXECUTION_FAILED = "TOOL_002"
    TOOL_TIMEOUT = "TOOL_003"
    TOOL_INVALID_RESPONSE = "TOOL_004"

    # Resource errors (4000-4099)
    RESOURCE_NOT_FOUND = "RES_001"
    RESOURCE_ACCESS_DENIED = "RES_002"
    RESOURCE_OPERATION_NOT_SUPPORTED = "RES_003"

    # System errors (5000-5099)
    SYSTEM_INTERNAL_ERROR = "SYS_001"
    SYSTEM_UNAVAILABLE = "SYS_002"
    SYSTEM_RATE_LIMIT_EXCEEDED = "SYS_003"
    SYSTEM_MAINTENANCE = "SYS_004"

    # Standards specific errors (6000-6099)
    STANDARDS_NOT_FOUND = "STD_001"
    STANDARDS_SYNC_FAILED = "STD_002"
    STANDARDS_INVALID_FORMAT = "STD_003"
    STANDARDS_RULE_EVALUATION_FAILED = "STD_004"

    # Security errors (7000-7099)
    SECURITY_INVALID_INPUT = "SEC_001"
    SECURITY_INJECTION_DETECTED = "SEC_002"
    SECURITY_REQUEST_TOO_LARGE = "SEC_003"
    SECURITY_MALICIOUS_CONTENT = "SEC_004"
    SECURITY_RATE_LIMIT_EXCEEDED = "SEC_005"
    SECURITY_BLOCKED_OPERATION = "SEC_006"


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: ErrorCode
    message: str
    details: dict[str, Any] | None = None
    field: str | None = None
    suggestion: str | None = None


class MCPError(Exception):
    """Base exception for MCP errors with structured information."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
        field: str | None = None,
        suggestion: str | None = None,
    ):
        super().__init__(message)
        self.error_detail = ErrorDetail(
            code=code,
            message=message,
            details=details,
            field=field,
            suggestion=suggestion,
        )

    @property
    def code(self) -> ErrorCode:
        """Access the error code for backward compatibility."""
        return self.error_detail.code

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        return {
            "error": {
                "code": self.error_detail.code.value,
                "message": self.error_detail.message,
                "details": self.error_detail.details,
                "field": self.error_detail.field,
                "suggestion": self.error_detail.suggestion,
            }
        }


class ValidationError(MCPError):
    """Validation error with field information."""

    def __init__(
        self,
        message: str,
        field: str,
        code: ErrorCode = ErrorCode.VALIDATION_INVALID_PARAMETERS,
    ):
        super().__init__(
            code=code,
            message=message,
            field=field,
            suggestion="Check the parameter requirements in the tool schema",
        )


class AuthenticationError(MCPError):
    """Authentication error."""

    def __init__(
        self,
        message: str = "Authentication required",
        code: ErrorCode = ErrorCode.AUTH_REQUIRED,
    ):
        super().__init__(
            code=code,
            message=message,
            suggestion="Provide a valid authentication token or API key",
        )


class AuthorizationError(MCPError):
    """Authorization error."""

    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_scope: str | None = None,
    ):
        super().__init__(
            code=ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            message=message,
            details={"required_scope": required_scope} if required_scope else None,
            suggestion="Request a token with the required permissions",
        )


class ToolNotFoundError(MCPError):
    """Tool not found error."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            code=ErrorCode.TOOL_NOT_FOUND,
            message=f"Tool '{tool_name}' not found",
            details={"tool_name": tool_name},
            suggestion="Use list_tools to see available tools",
        )


class ResourceNotFoundError(MCPError):
    """Resource not found error."""

    def __init__(self, resource_type: str, resource_id: str) -> None:
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} '{resource_id}' not found",
            details={"resource_type": resource_type, "resource_id": resource_id},
        )


class RateLimitError(MCPError):
    """Rate limit exceeded error."""

    def __init__(self, limit: int, window: str, retry_after: int | None = None) -> None:
        super().__init__(
            code=ErrorCode.SYSTEM_RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded: {limit} requests per {window}",
            details={"limit": limit, "window": window, "retry_after": retry_after},
            suggestion=(
                f"Retry after {retry_after} seconds"
                if retry_after
                else "Reduce request frequency"
            ),
        )


class SecurityError(MCPError):
    """Security-related error."""

    def __init__(
        self,
        message: str = "Security validation failed",
        code: ErrorCode = ErrorCode.SECURITY_INVALID_INPUT,
    ):
        super().__init__(
            code=code,
            message=message,
            suggestion="Ensure your input follows security guidelines",
        )


class SecureErrorHandler:
    """Handles errors with security-first approach to prevent information leakage."""

    def __init__(self, mask_errors: bool = True, log_errors: bool = True) -> None:
        self.mask_errors = mask_errors
        self.log_errors = log_errors
        self.sensitive_patterns = [
            r"/[A-Za-z]:/.*",  # Windows file paths
            r"/(?:home|root|etc|var|usr)/.*",  # Unix file paths
            r"[A-Za-z]:\\.*",  # Windows-style paths
            r"password",  # Password references
            r"token",  # Token references
            r"secret",  # Secret references
            r"key",  # Key references
            r"localhost",  # localhost references
            r"\d+\.\d+\.\d+\.\d+",  # IP addresses
            r":[0-9]+",  # Port numbers
            r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",  # Email addresses
        ]

    def sanitize_error_message(self, message: str) -> str:
        """Sanitize error message to remove sensitive information."""
        if not self.mask_errors:
            return message

        # Replace sensitive patterns with generic placeholders
        sanitized = message
        for pattern in self.sensitive_patterns:
            sanitized = re.sub(pattern, "[REDACTED]", sanitized, flags=re.IGNORECASE)

        # Remove stack traces
        sanitized = re.sub(
            r"Traceback.*?(?=\n\S|\Z)",
            "[STACK_TRACE_REDACTED]",
            sanitized,
            flags=re.DOTALL,
        )

        # Remove SQL error details
        sanitized = re.sub(
            r"SQL.*?(?=\n|\Z)", "[SQL_REDACTED]", sanitized, flags=re.IGNORECASE
        )

        # Remove file system references
        sanitized = re.sub(r"No such file or directory.*", "File not found", sanitized)

        return sanitized

    def handle_exception(
        self, exc: Exception, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle an exception and return a safe error response."""
        if self.log_errors:
            logger.error(
                f"Exception in {context or 'unknown'}: {str(exc)}", exc_info=True
            )

        # Handle known MCP errors
        if isinstance(exc, MCPError):
            error_dict = exc.to_dict()
            if self.mask_errors:
                error_dict["error"]["message"] = self.sanitize_error_message(
                    error_dict["error"]["message"]
                )
            return error_dict

        # Handle validation errors
        if hasattr(exc, "errors") and callable(exc.errors):
            # Pydantic validation error
            try:
                return format_validation_errors(exc.errors())
            except Exception as format_err:
                logger.warning(f"Failed to format validation errors: {format_err}")
                # Fallback to basic error message
                return {"error": str(exc), "type": "validation_error"}

        # Handle security-related errors
        if any(
            keyword in str(exc).lower()
            for keyword in ["injection", "malicious", "dangerous", "blocked"]
        ):
            return SecurityError(
                message="Security validation failed",
                code=ErrorCode.SECURITY_INVALID_INPUT,
            ).to_dict()

        # Handle generic exceptions
        if self.mask_errors:
            # Generic error response
            return {
                "error": {
                    "code": ErrorCode.SYSTEM_INTERNAL_ERROR.value,
                    "message": "An internal error occurred",
                    "suggestion": "Please try again later or contact support",
                }
            }
        else:
            # Development mode - show actual error
            return {
                "error": {
                    "code": ErrorCode.SYSTEM_INTERNAL_ERROR.value,
                    "message": self.sanitize_error_message(str(exc)),
                    "type": type(exc).__name__,
                }
            }

    def handle_security_violation(
        self, violation_type: str, details: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Handle a security violation."""
        if self.log_errors:
            logger.warning(f"Security violation: {violation_type}, details: {details}")

        # Map violation types to error codes
        error_code_map = {
            "injection": ErrorCode.SECURITY_INJECTION_DETECTED,
            "size_limit": ErrorCode.SECURITY_REQUEST_TOO_LARGE,
            "malicious_content": ErrorCode.SECURITY_MALICIOUS_CONTENT,
            "rate_limit": ErrorCode.SECURITY_RATE_LIMIT_EXCEEDED,
            "blocked_operation": ErrorCode.SECURITY_BLOCKED_OPERATION,
        }

        code = error_code_map.get(violation_type, ErrorCode.SECURITY_INVALID_INPUT)

        return SecurityError(
            message=f"Security violation: {violation_type}", code=code
        ).to_dict()


def format_validation_errors(errors: list) -> dict[str, Any]:
    """Format pydantic validation errors to MCP error format."""
    formatted_errors = []

    for error in errors:
        field = ".".join(str(loc) for loc in error.get("loc", []))
        formatted_errors.append(
            {
                "field": field,
                "message": error.get("msg", "Validation error"),
                "type": error.get("type", "unknown"),
            }
        )

    return {
        "error": {
            "code": ErrorCode.VALIDATION_INVALID_PARAMETERS.value,
            "message": "Validation failed for one or more parameters",
            "details": {"validation_errors": formatted_errors},
        }
    }


# Global secure error handler
_secure_error_handler: SecureErrorHandler | None = None


def get_secure_error_handler() -> SecureErrorHandler:
    """Get the global secure error handler."""
    global _secure_error_handler
    if _secure_error_handler is None:
        # Default to production mode (mask errors) unless explicitly disabled
        mask_errors = os.getenv("MCP_MASK_ERRORS", "true").lower() != "false"
        _secure_error_handler = SecureErrorHandler(mask_errors=mask_errors)
    return _secure_error_handler


def init_secure_error_handler(
    mask_errors: bool = True, log_errors: bool = True
) -> SecureErrorHandler:
    """Initialize the global secure error handler."""
    global _secure_error_handler
    _secure_error_handler = SecureErrorHandler(
        mask_errors=mask_errors, log_errors=log_errors
    )
    return _secure_error_handler
