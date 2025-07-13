"""
Security middleware and utilities for MCP server.

Provides request size limits, security headers, and enhanced security features.
"""

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any

from .errors import ErrorCode, SecurityError, get_secure_error_handler

logger = logging.getLogger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration settings."""

    # Request size limits
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    max_json_depth: int = 100
    max_array_length: int = 10000
    max_string_length: int = 1000000  # 1MB
    max_object_keys: int = 10000

    # Rate limiting
    enable_rate_limiting: bool = True
    default_rate_limit: int = 100  # requests per minute

    # Security headers
    enable_security_headers: bool = True

    # Input validation
    enable_strict_validation: bool = True
    sanitize_inputs: bool = True

    # Error handling
    mask_errors: bool = True
    log_security_events: bool = True

    # Content security
    allowed_content_types: list[str] | None = None

    def __post_init__(self) -> None:
        if self.allowed_content_types is None:
            self.allowed_content_types = [
                "application/json",
                "application/x-msgpack",
                "text/plain",
            ]


class SecurityHeaders:
    """Security headers for HTTP responses."""

    @staticmethod
    def get_security_headers() -> dict[str, str]:
        """Get standard security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; font-src 'self'; object-src 'none'; media-src 'self'; form-action 'self'; frame-ancestors 'none';",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=(), usb=(), accelerometer=(), gyroscope=(), magnetometer=()",
        }


class InputSanitizer:
    """Sanitizes and validates input data."""

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config

    def sanitize_string(self, value: str, max_length: int | None = None) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            raise ValueError("Input must be a string")

        max_len = max_length or self.config.max_string_length

        # Remove null bytes and control characters
        value = value.replace("\x00", "")
        value = re.sub(r"[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", value)

        # Limit length
        if len(value) > max_len:
            logger.warning(
                f"String truncated from {len(value)} to {max_len} characters"
            )
            value = value[:max_len]

        return value

    def sanitize_dict(self, data: dict[str, Any], max_depth: int = 0) -> dict[str, Any]:
        """Sanitize dictionary data recursively."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")

        if max_depth > self.config.max_json_depth:
            raise ValueError(
                f"JSON depth exceeds maximum of {self.config.max_json_depth}"
            )

        if len(data) > self.config.max_object_keys:
            raise ValueError(
                f"Object has too many keys: {len(data)} > {self.config.max_object_keys}"
            )

        sanitized = {}
        for key, value in data.items():
            # Sanitize key
            if isinstance(key, str):
                key = self.sanitize_string(key, max_length=200)

            # Sanitize value based on type
            if isinstance(value, str):
                value = self.sanitize_string(value)
            elif isinstance(value, dict):
                value = self.sanitize_dict(value, max_depth + 1)
            elif isinstance(value, list):
                value = self.sanitize_list(value, max_depth + 1)
            elif isinstance(value, int | float | bool | type(None)):
                # These types are safe as-is
                pass
            else:
                # Convert unknown types to string
                value = str(value)
                value = self.sanitize_string(value)

            sanitized[key] = value

        return sanitized

    def sanitize_list(self, data: list[Any], max_depth: int = 0) -> list[Any]:
        """Sanitize list data recursively."""
        if not isinstance(data, list):
            raise ValueError("Input must be a list")

        if len(data) > self.config.max_array_length:
            raise ValueError(
                f"Array length exceeds maximum of {self.config.max_array_length}"
            )

        sanitized = []
        for item in data:
            if isinstance(item, str):
                item = self.sanitize_string(item)
            elif isinstance(item, dict):
                item = self.sanitize_dict(item, max_depth + 1)
            elif isinstance(item, list):
                item = self.sanitize_list(item, max_depth + 1)
            elif isinstance(item, int | float | bool | type(None)):
                # These types are safe as-is
                pass
            else:
                # Convert unknown types to string
                item = str(item)
                item = self.sanitize_string(item)

            sanitized.append(item)

        return sanitized


class SecurityValidator:
    """Validates requests for security issues."""

    def __init__(self, config: SecurityConfig) -> None:
        self.config = config
        self.sanitizer = InputSanitizer(config)

    def validate_request_size(self, data: Any) -> None:
        """Validate request size limits."""
        if isinstance(data, str):
            size = len(data.encode("utf-8"))
        elif isinstance(data, bytes):
            size = len(data)
        else:
            # Estimate size for complex objects
            import sys

            size = sys.getsizeof(data)

        if size > self.config.max_request_size:
            raise ValueError(
                f"Request size {size} exceeds maximum of {self.config.max_request_size}"
            )

    def validate_content_type(self, content_type: str) -> None:
        """Validate content type is allowed."""
        if (
            self.config.allowed_content_types
            and content_type not in self.config.allowed_content_types
        ):
            raise ValueError(f"Content type '{content_type}' not allowed")

    def validate_json_structure(self, data: Any, max_depth: int = 0) -> None:
        """Validate JSON structure for security issues."""
        if max_depth > self.config.max_json_depth:
            raise ValueError(
                f"JSON depth exceeds maximum of {self.config.max_json_depth}"
            )

        if isinstance(data, dict):
            if len(data) > self.config.max_object_keys:
                raise ValueError(f"Object has too many keys: {len(data)}")

            for key, value in data.items():
                if isinstance(key, str) and len(key) > 1000:
                    raise ValueError("Object key too long")
                self.validate_json_structure(value, max_depth + 1)

        elif isinstance(data, list):
            if len(data) > self.config.max_array_length:
                raise ValueError(f"Array length exceeds maximum: {len(data)}")

            for item in data:
                self.validate_json_structure(item, max_depth + 1)

        elif isinstance(data, str):
            if len(data) > self.config.max_string_length:
                raise ValueError(f"String length exceeds maximum: {len(data)}")

    def check_for_injection_patterns(self, data: Any) -> None:
        """Check for common injection patterns."""
        if isinstance(data, str):
            # SQL injection patterns
            sql_patterns = [
                r"(?i)\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b",
                r"(?i)\b(OR|AND)\s+\d+\s*=\s*\d+",
                r"(?i)\'\s*OR\s*\'\d+\'\s*=\s*\'\d+",
                r"(?i)\"\s*OR\s*\"\d+\"\s*=\s*\"\d+",
                r"(?i);\s*(DROP|DELETE|UPDATE|INSERT)",
            ]

            for pattern in sql_patterns:
                if re.search(pattern, data):
                    logger.warning(f"Potential SQL injection detected: {pattern}")
                    raise ValueError(
                        "Input contains potentially dangerous SQL patterns"
                    )

            # Script injection patterns
            script_patterns = [
                r"(?i)<script[^>]*>",
                r"(?i)javascript:",
                r"(?i)vbscript:",
                r"(?i)\bon[a-z]+\s*=",  # Match event handlers like onclick= but not const
                r"(?i)eval\s*\(",
                r"(?i)exec\s*\(",
                r"(?i)system\s*\(",
                r"(?i)__import__",
            ]

            for pattern in script_patterns:
                if re.search(pattern, data):
                    logger.warning(f"Potential script injection detected: {pattern}")
                    raise ValueError(
                        "Input contains potentially dangerous script patterns"
                    )

        elif isinstance(data, dict):
            for key, value in data.items():
                self.check_for_injection_patterns(key)
                self.check_for_injection_patterns(value)

        elif isinstance(data, list):
            for item in data:
                self.check_for_injection_patterns(item)


class SecurityMiddleware:
    """Main security middleware class."""

    def __init__(self, config: SecurityConfig | None = None) -> None:
        self.config = config or SecurityConfig()
        self.validator = SecurityValidator(self.config)
        self.sanitizer = InputSanitizer(self.config)
        self.error_handler = get_secure_error_handler()

    def validate_and_sanitize_request(self, data: Any) -> Any:
        """Validate and sanitize request data."""
        try:
            # Validate request size
            self.validator.validate_request_size(data)

            # Validate JSON structure
            self.validator.validate_json_structure(data)

            # Check for injection patterns
            if self.config.enable_strict_validation:
                self.validator.check_for_injection_patterns(data)

            # Sanitize input if enabled
            if self.config.sanitize_inputs:
                if isinstance(data, dict):
                    data = self.sanitizer.sanitize_dict(data)
                elif isinstance(data, list):
                    data = self.sanitizer.sanitize_list(data)
                elif isinstance(data, str):
                    data = self.sanitizer.sanitize_string(data)

            return data

        except Exception as e:
            if self.config.log_security_events:
                logger.error(f"Security validation failed: {str(e)}")
            # Convert to security error
            raise SecurityError(
                message=self.error_handler.sanitize_error_message(str(e)),
                code=ErrorCode.SECURITY_INVALID_INPUT,
            )

    def add_security_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Add security headers to response."""
        if self.config.enable_security_headers:
            headers.update(SecurityHeaders.get_security_headers())
        return headers

    def sanitize_error_response(self, error: Exception) -> dict[str, Any]:
        """Sanitize error response to prevent information leakage."""
        if self.config.mask_errors:
            # Generic error message
            return {
                "error": "An error occurred while processing your request",
                "code": "INTERNAL_ERROR",
                "timestamp": time.time(),
            }
        else:
            # Return actual error (for development)
            return {
                "error": str(error),
                "type": type(error).__name__,
                "timestamp": time.time(),
            }


def security_middleware(
    config: SecurityConfig | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for applying security middleware to functions."""
    middleware = SecurityMiddleware(config)

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Validate and sanitize inputs
            try:
                # Find dictionary arguments that might be request data
                args_list = list(args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, dict | list | str):
                        args_list[i] = middleware.validate_and_sanitize_request(arg)
                args = tuple(args_list)

                for key, value in kwargs.items():
                    if isinstance(value, dict | list | str):
                        kwargs[key] = middleware.validate_and_sanitize_request(value)

                # Call the original function
                result = func(*args, **kwargs)

                return result

            except Exception as e:
                # Log security event
                if middleware.config.log_security_events:
                    logger.error(f"Security middleware blocked request: {str(e)}")

                # Return sanitized error
                raise ValueError(middleware.sanitize_error_response(e))

        return wrapper

    return decorator


# Global security middleware instance
_security_middleware: SecurityMiddleware | None = None


def get_security_middleware() -> SecurityMiddleware:
    """Get the global security middleware instance."""
    global _security_middleware
    if _security_middleware is None:
        _security_middleware = SecurityMiddleware()
    return _security_middleware


def init_security_middleware(config: SecurityConfig) -> SecurityMiddleware:
    """Initialize security middleware with configuration."""
    global _security_middleware
    _security_middleware = SecurityMiddleware(config)
    return _security_middleware
