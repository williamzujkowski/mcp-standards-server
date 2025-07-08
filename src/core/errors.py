"""
Structured error handling for MCP server.

Provides standardized error codes and error response formatting.
"""

from enum import Enum
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """Standardized error codes for MCP operations."""
    
    # Authentication errors (1000-1099)
    AUTH_REQUIRED = "AUTH_001"
    AUTH_INVALID_TOKEN = "AUTH_002"
    AUTH_EXPIRED_TOKEN = "AUTH_003"
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


class ErrorDetail(BaseModel):
    """Detailed error information."""
    code: ErrorCode
    message: str
    details: Optional[Dict[str, Any]] = None
    field: Optional[str] = None
    suggestion: Optional[str] = None


class MCPError(Exception):
    """Base exception for MCP errors with structured information."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ):
        super().__init__(message)
        self.error_detail = ErrorDetail(
            code=code,
            message=message,
            details=details,
            field=field,
            suggestion=suggestion
        )
    
    @property
    def code(self) -> ErrorCode:
        """Access the error code for backward compatibility."""
        return self.error_detail.code
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for JSON response."""
        return {
            "error": {
                "code": self.error_detail.code.value,
                "message": self.error_detail.message,
                "details": self.error_detail.details,
                "field": self.error_detail.field,
                "suggestion": self.error_detail.suggestion
            }
        }


class ValidationError(MCPError):
    """Validation error with field information."""
    
    def __init__(self, message: str, field: str, code: ErrorCode = ErrorCode.VALIDATION_INVALID_PARAMETERS):
        super().__init__(
            code=code,
            message=message,
            field=field,
            suggestion="Check the parameter requirements in the tool schema"
        )


class AuthenticationError(MCPError):
    """Authentication error."""
    
    def __init__(self, message: str = "Authentication required", code: ErrorCode = ErrorCode.AUTH_REQUIRED):
        super().__init__(
            code=code,
            message=message,
            suggestion="Provide a valid authentication token or API key"
        )


class AuthorizationError(MCPError):
    """Authorization error."""
    
    def __init__(self, message: str = "Insufficient permissions", required_scope: Optional[str] = None):
        super().__init__(
            code=ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS,
            message=message,
            details={"required_scope": required_scope} if required_scope else None,
            suggestion="Request a token with the required permissions"
        )


class ToolNotFoundError(MCPError):
    """Tool not found error."""
    
    def __init__(self, tool_name: str):
        super().__init__(
            code=ErrorCode.TOOL_NOT_FOUND,
            message=f"Tool '{tool_name}' not found",
            details={"tool_name": tool_name},
            suggestion="Use list_tools to see available tools"
        )


class ResourceNotFoundError(MCPError):
    """Resource not found error."""
    
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND,
            message=f"{resource_type} '{resource_id}' not found",
            details={"resource_type": resource_type, "resource_id": resource_id}
        )


class RateLimitError(MCPError):
    """Rate limit exceeded error."""
    
    def __init__(self, limit: int, window: str, retry_after: Optional[int] = None):
        super().__init__(
            code=ErrorCode.SYSTEM_RATE_LIMIT_EXCEEDED,
            message=f"Rate limit exceeded: {limit} requests per {window}",
            details={
                "limit": limit,
                "window": window,
                "retry_after": retry_after
            },
            suggestion=f"Retry after {retry_after} seconds" if retry_after else "Reduce request frequency"
        )


def format_validation_errors(errors: list) -> Dict[str, Any]:
    """Format pydantic validation errors to MCP error format."""
    formatted_errors = []
    
    for error in errors:
        field = ".".join(str(loc) for loc in error.get("loc", []))
        formatted_errors.append({
            "field": field,
            "message": error.get("msg", "Validation error"),
            "type": error.get("type", "unknown")
        })
        
    return {
        "error": {
            "code": ErrorCode.VALIDATION_INVALID_PARAMETERS.value,
            "message": "Validation failed for one or more parameters",
            "details": {"validation_errors": formatted_errors}
        }
    }