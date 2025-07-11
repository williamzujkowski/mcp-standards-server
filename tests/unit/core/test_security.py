"""Tests for security middleware."""

from typing import Any

import pytest

from src.core.security import (
    InputSanitizer,
    SecurityConfig,
    SecurityHeaders,
    SecurityMiddleware,
    SecurityValidator,
    get_security_middleware,
    security_middleware,
)


class TestSecurityConfig:
    """Test security configuration."""

    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()

        assert config.max_request_size == 10 * 1024 * 1024  # 10MB
        assert config.max_json_depth == 100
        assert config.max_array_length == 10000
        assert config.max_string_length == 1000000
        assert config.enable_security_headers is True
        assert config.sanitize_inputs is True
        assert config.mask_errors is True

    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            max_request_size=1024,
            max_json_depth=10,
            enable_security_headers=False,
            mask_errors=False,
        )

        assert config.max_request_size == 1024
        assert config.max_json_depth == 10
        assert config.enable_security_headers is False
        assert config.mask_errors is False


class TestSecurityHeaders:
    """Test security headers functionality."""

    def test_security_headers_present(self):
        """Test that all expected security headers are present."""
        headers = SecurityHeaders.get_security_headers()

        expected_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Referrer-Policy",
            "Content-Security-Policy",
            "Permissions-Policy",
        ]

        for header in expected_headers:
            assert header in headers

    def test_security_header_values(self):
        """Test security header values are correct."""
        headers = SecurityHeaders.get_security_headers()

        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"
        assert headers["X-XSS-Protection"] == "1; mode=block"
        assert "max-age=31536000" in headers["Strict-Transport-Security"]
        assert "includeSubDomains" in headers["Strict-Transport-Security"]


class TestInputSanitizer:
    """Test input sanitization."""

    @pytest.fixture
    def sanitizer(self):
        """Create sanitizer instance."""
        return InputSanitizer(SecurityConfig())

    def test_string_sanitization(self, sanitizer):
        """Test string sanitization."""
        # Test null byte removal
        result = sanitizer.sanitize_string("hello\x00world")
        assert result == "helloworld"

        # Test control character removal
        result = sanitizer.sanitize_string("hello\x07world\x1f")
        assert result == "helloworld"

        # Test length limiting
        long_string = "x" * 2000000
        result = sanitizer.sanitize_string(long_string, max_length=1000)
        assert len(result) == 1000

    def test_dict_sanitization(self, sanitizer):
        """Test dictionary sanitization."""
        data = {
            "clean_key": "clean_value",
            "dirty_key\x00": "dirty_value\x07",
            "nested": {"key": "value\x1f"},
        }

        result = sanitizer.sanitize_dict(data)

        assert "clean_key" in result
        assert "dirty_key" in result
        assert result["clean_key"] == "clean_value"
        assert result["dirty_key"] == "dirty_value"
        assert result["nested"]["key"] == "value"

    def test_list_sanitization(self, sanitizer):
        """Test list sanitization."""
        data = [
            "clean_item",
            "dirty_item\x00",
            {"nested": "value\x07"},
            ["nested", "list\x1f"],
        ]

        result = sanitizer.sanitize_list(data)

        assert result[0] == "clean_item"
        assert result[1] == "dirty_item"
        assert result[2]["nested"] == "value"
        assert result[3][1] == "list"

    def test_max_depth_validation(self, sanitizer):
        """Test maximum depth validation."""
        # Create deeply nested structure
        deep_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            "level1": {"level2": {"level3": {}}}
        }

        # Should work with default depth
        result = sanitizer.sanitize_dict(deep_data)
        assert "level1" in result

        # Should fail with low max depth
        config = SecurityConfig(max_json_depth=2)
        sanitizer = InputSanitizer(config)

        with pytest.raises(ValueError, match="JSON depth exceeds maximum"):
            sanitizer.sanitize_dict(deep_data)

    def test_max_array_length_validation(self, sanitizer):
        """Test maximum array length validation."""
        # Create long array
        long_array = list(range(20000))

        # Should fail with default config
        with pytest.raises(ValueError, match="Array length exceeds maximum"):
            sanitizer.sanitize_list(long_array)

    def test_max_object_keys_validation(self, sanitizer):
        """Test maximum object keys validation."""
        # Create object with many keys
        large_obj = {f"key_{i}": f"value_{i}" for i in range(15000)}

        # Should fail with default config
        with pytest.raises(ValueError, match="Object has too many keys"):
            sanitizer.sanitize_dict(large_obj)


class TestSecurityValidator:
    """Test security validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return SecurityValidator(SecurityConfig())

    def test_request_size_validation(self, validator):
        """Test request size validation."""
        # Small request should pass
        small_data = "small data"
        validator.validate_request_size(small_data)  # Should not raise

        # Large request should fail
        large_data = "x" * (20 * 1024 * 1024)  # 20MB
        with pytest.raises(ValueError, match="Request size.*exceeds maximum"):
            validator.validate_request_size(large_data)

    def test_content_type_validation(self, validator):
        """Test content type validation."""
        # Allowed content types should pass
        validator.validate_content_type("application/json")
        validator.validate_content_type("application/x-msgpack")

        # Disallowed content type should fail
        with pytest.raises(ValueError, match="Content type.*not allowed"):
            validator.validate_content_type("text/html")

    def test_json_structure_validation(self, validator):
        """Test JSON structure validation."""
        # Valid structure should pass
        valid_data = {"key": "value", "nested": {"array": [1, 2, 3]}}
        validator.validate_json_structure(valid_data)

        # Too deep structure should fail
        deep_data: dict[str, dict[str, dict[str, dict[str, Any]]]] = {
            "level1": {"level2": {"level3": {}}}
        }
        config = SecurityConfig(max_json_depth=2)
        validator = SecurityValidator(config)

        with pytest.raises(ValueError, match="JSON depth exceeds maximum"):
            validator.validate_json_structure(deep_data)

    def test_sql_injection_detection(self, validator):
        """Test SQL injection pattern detection."""
        # Safe strings should pass
        safe_data = "This is a normal text string"
        validator.check_for_injection_patterns(safe_data)

        # SQL injection patterns should fail
        malicious_patterns = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "UNION SELECT * FROM passwords",
            "1' OR 1=1 --",
        ]

        for pattern in malicious_patterns:
            with pytest.raises(ValueError, match="dangerous SQL patterns"):
                validator.check_for_injection_patterns(pattern)

    def test_script_injection_detection(self, validator):
        """Test script injection pattern detection."""
        # Safe strings should pass
        safe_data = "normal text content"
        validator.check_for_injection_patterns(safe_data)

        # Script injection patterns should fail
        malicious_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onload=alert('xss')",
            "eval(malicious_code)",
            "__import__('os').system('rm -rf /')",
        ]

        for pattern in malicious_patterns:
            with pytest.raises(ValueError, match="dangerous script patterns"):
                validator.check_for_injection_patterns(pattern)

    def test_nested_injection_detection(self, validator):
        """Test injection detection in nested structures."""
        # Malicious data in nested structure
        malicious_data = {
            "user": "normal_user",
            "query": "'; DROP TABLE users; --",
            "nested": {"script": "<script>alert('xss')</script>"},
        }

        with pytest.raises(ValueError, match="dangerous"):
            validator.check_for_injection_patterns(malicious_data)


class TestSecurityMiddleware:
    """Test security middleware."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        return SecurityMiddleware(SecurityConfig(mask_errors=False))

    def test_request_validation_and_sanitization(self, middleware):
        """Test request validation and sanitization."""
        # Test with valid data
        clean_data = {"key": "value", "number": 42}
        result = middleware.validate_and_sanitize_request(clean_data)
        assert result == clean_data

        # Test with data that needs sanitization
        dirty_data = {"key\x00": "value\x07", "array": ["item\x1f"]}
        result = middleware.validate_and_sanitize_request(dirty_data)
        assert result == {"key": "value", "array": ["item"]}

    def test_security_headers_addition(self, middleware):
        """Test security headers addition."""
        original_headers = {"Content-Type": "application/json"}
        result = middleware.add_security_headers(original_headers)

        assert "Content-Type" in result
        assert "X-Content-Type-Options" in result
        assert "X-Frame-Options" in result
        assert result["X-Content-Type-Options"] == "nosniff"

    def test_error_response_sanitization(self, middleware):
        """Test error response sanitization."""
        # Test with error masking disabled
        error = ValueError("Detailed error message")
        result = middleware.sanitize_error_response(error)

        assert result["error"] == "Detailed error message"
        assert result["type"] == "ValueError"
        assert "timestamp" in result

        # Test with error masking enabled
        masked_middleware = SecurityMiddleware(SecurityConfig(mask_errors=True))
        result = masked_middleware.sanitize_error_response(error)

        assert result["error"] == "An error occurred while processing your request"
        assert result["code"] == "INTERNAL_ERROR"
        assert "timestamp" in result

    def test_malicious_request_blocking(self, middleware):
        """Test that malicious requests are blocked."""
        from src.core.errors import SecurityError

        malicious_data = {
            "query": "'; DROP TABLE users; --",
            "script": "<script>alert('xss')</script>",
        }

        with pytest.raises(SecurityError):
            middleware.validate_and_sanitize_request(malicious_data)

    def test_oversized_request_blocking(self, middleware):
        """Test that oversized requests are blocked."""
        from src.core.errors import SecurityError

        # Create oversized request
        oversized_data = "x" * (20 * 1024 * 1024)  # 20MB

        with pytest.raises(SecurityError):
            middleware.validate_and_sanitize_request(oversized_data)


class TestSecurityDecorator:
    """Test security decorator."""

    def test_decorator_application(self):
        """Test that security decorator is applied correctly."""

        @security_middleware()
        def test_function(data):
            return data

        # Safe data should pass through
        safe_data = {"key": "value"}
        result = test_function(safe_data)
        assert result == safe_data

        # Unsafe data should be blocked
        unsafe_data = {"query": "'; DROP TABLE users; --"}

        with pytest.raises(ValueError):
            test_function(unsafe_data)

    def test_decorator_with_kwargs(self):
        """Test decorator with keyword arguments."""

        @security_middleware()
        def test_function(name, data=None):
            return {"name": name, "data": data}

        # Safe call should work
        result = test_function("test", data={"key": "value"})
        assert result["name"] == "test"
        assert result["data"] == {"key": "value"}

        # Unsafe kwarg should be blocked
        with pytest.raises(ValueError):
            test_function("test", data={"query": "'; DROP TABLE users; --"})


class TestGlobalSecurityMiddleware:
    """Test global security middleware management."""

    def test_get_security_middleware(self):
        """Test getting global security middleware."""
        middleware1 = get_security_middleware()
        middleware2 = get_security_middleware()

        # Should return same instance
        assert middleware1 is middleware2

    def test_security_middleware_configuration(self):
        """Test security middleware configuration."""
        middleware = get_security_middleware()

        # Should have default configuration
        assert middleware.config.enable_security_headers is True
        assert middleware.config.sanitize_inputs is True
        assert middleware.config.mask_errors is True
