"""
Unit tests for input validation module.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from src.core.errors import ErrorCode, ValidationError
from src.core.validation import (
    ContextModel,
    InputValidator,
    SearchStandardsInput,
    ValidateAgainstStandardInput,
    get_input_validator,
)


class TestContextModel:
    """Test context model validation."""

    def test_valid_context(self):
        """Test valid context creation."""
        context = ContextModel(
            project_type="web_application",
            language="python",
            framework="django",
            requirements=["security", "performance"],
            team_size="medium",
        )

        assert context.project_type == "web_application"
        assert context.language == "python"
        assert len(context.requirements) == 2

    def test_invalid_project_type(self):
        """Test invalid project type format."""
        with pytest.raises(PydanticValidationError):
            ContextModel(project_type="Web Application")  # Should be lowercase

    def test_invalid_team_size(self):
        """Test invalid team size value."""
        with pytest.raises(PydanticValidationError):
            ContextModel(team_size="huge")  # Should be small/medium/large

    def test_extra_fields_allowed(self):
        """Test that extra fields are allowed."""
        context = ContextModel(project_type="api", custom_field="custom_value")

        assert context.project_type == "api"
        assert hasattr(context, "custom_field")


class TestValidateAgainstStandardInput:
    """Test code validation input."""

    def test_valid_input(self):
        """Test valid validation input."""
        input_data = ValidateAgainstStandardInput(
            code="function test() { return true; }",
            standard="javascript-es6",
            language="javascript",
        )

        assert input_data.code == "function test() { return true; }"
        assert input_data.standard == "javascript-es6"

    def test_dangerous_code_patterns(self):
        """Test rejection of dangerous code patterns."""
        dangerous_codes = [
            "__import__('os').system('rm -rf /')",
            "eval('malicious code')",
            "exec(compile('bad', 'string', 'exec'))",
            "globals()['__builtins__']['eval']('bad')",
        ]

        for code in dangerous_codes:
            with pytest.raises(PydanticValidationError):
                ValidateAgainstStandardInput(
                    code=code, standard="python-security", language="python"
                )

    def test_code_size_limit(self):
        """Test code size validation."""
        huge_code = "x" * (1024 * 1024 + 1)  # Over 1MB

        with pytest.raises(PydanticValidationError):
            ValidateAgainstStandardInput(
                code=huge_code, standard="test", language="python"
            )


class TestSearchStandardsInput:
    """Test search input validation."""

    def test_valid_search(self):
        """Test valid search input."""
        input_data = SearchStandardsInput(
            query="React performance optimization", limit=20, min_relevance=0.7
        )

        assert input_data.query == "React performance optimization"
        assert input_data.limit == 20

    def test_query_sanitization(self):
        """Test query sanitization."""
        input_data = SearchStandardsInput(
            query="React <script>alert('xss')</script> optimization"
        )

        # Dangerous characters should be removed
        assert "<" not in input_data.query
        assert ">" not in input_data.query

    def test_limit_bounds(self):
        """Test limit validation."""
        # Too high
        with pytest.raises(PydanticValidationError):
            SearchStandardsInput(query="test", limit=1000)

        # Too low
        with pytest.raises(PydanticValidationError):
            SearchStandardsInput(query="test", limit=0)

    def test_relevance_bounds(self):
        """Test relevance score bounds."""
        # Valid
        input_data = SearchStandardsInput(query="test", min_relevance=0.5)
        assert input_data.min_relevance == 0.5

        # Too high
        with pytest.raises(PydanticValidationError):
            SearchStandardsInput(query="test", min_relevance=1.5)


class TestInputValidator:
    """Test the input validator class."""

    @pytest.fixture
    def validator(self):
        """Create input validator instance."""
        return InputValidator()

    def test_validate_get_applicable_standards(self, validator):
        """Test validation of get_applicable_standards input."""
        arguments = {
            "context": {"project_type": "web_application", "framework": "react"}
        }

        validated = validator.validate_tool_input("get_applicable_standards", arguments)

        assert validated["context"]["project_type"] == "web_application"
        assert validated["include_resolution_details"] is False  # Default

    def test_validate_missing_required_field(self, validator):
        """Test validation with missing required field."""
        arguments = {
            # Missing 'context' field
            "include_resolution_details": True
        }

        with pytest.raises(ValidationError) as exc_info:
            validator.validate_tool_input("get_applicable_standards", arguments)

        assert (
            exc_info.value.error_detail.code == ErrorCode.VALIDATION_INVALID_PARAMETERS
        )

    def test_validate_search_standards(self, validator):
        """Test validation of search_standards input."""
        arguments = {
            "query": "React hooks",
            "limit": 5,
            "filters": {
                "categories": ["frontend", "react"],
                "languages": ["javascript"],
            },
        }

        validated = validator.validate_tool_input("search_standards", arguments)

        assert validated["query"] == "React hooks"
        assert validated["limit"] == 5
        assert len(validated["filters"]["categories"]) == 2

    def test_validate_no_input_required(self, validator):
        """Test validation of tools with no input."""
        # get_sync_status has no input requirements
        arguments = {"any": "field"}

        validated = validator.validate_tool_input("get_sync_status", arguments)

        # Should return arguments as-is
        assert validated == arguments

    def test_sanitize_string(self, validator):
        """Test string sanitization."""
        # Test null byte removal
        sanitized = validator.sanitize_string("test\x00string")
        assert "\x00" not in sanitized

        # Test control character removal
        sanitized = validator.sanitize_string("test\x01\x02string")
        assert "\x01" not in sanitized
        assert "\x02" not in sanitized

        # Test length limiting
        long_string = "x" * 2000
        sanitized = validator.sanitize_string(long_string, max_length=100)
        assert len(sanitized) == 100

    def test_sanitize_identifier(self, validator):
        """Test identifier sanitization."""
        # Test special character removal
        sanitized = validator.sanitize_identifier("test@#$identifier")
        assert sanitized == "testidentifier"

        # Test starting with number
        sanitized = validator.sanitize_identifier("123test")
        assert sanitized == "id_123test"

        # Test length limiting
        long_id = "a" * 200
        sanitized = validator.sanitize_identifier(long_id)
        assert len(sanitized) == 100

    def test_singleton_instance(self):
        """Test that get_input_validator returns singleton."""
        validator1 = get_input_validator()
        validator2 = get_input_validator()

        assert validator1 is validator2
