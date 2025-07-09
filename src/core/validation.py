"""
Input validation module for MCP server.

Provides comprehensive validation for tool inputs using JSON Schema and Pydantic.
"""

import re
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict
import jsonschema
from jsonschema import Draft7Validator

from src.core.errors import ValidationError, ErrorCode, get_secure_error_handler
from src.core.security import get_security_middleware


# Common validation patterns
PATTERNS = {
    "identifier": r"^[a-zA-Z][a-zA-Z0-9_-]*$",
    "version": r"^\d+\.\d+\.\d+$",
    "language": r"^[a-z]+$",
    "category": r"^[a-z_]+$"
}

# Maximum sizes for various inputs
MAX_CODE_SIZE = 1024 * 1024  # 1MB
MAX_QUERY_LENGTH = 1000
MAX_ARRAY_SIZE = 1000


class ContextModel(BaseModel):
    """Validation model for project context."""
    model_config = ConfigDict(extra="allow")
    
    project_type: Optional[str] = Field(None, pattern=r"^[a-z_]+$")
    language: Optional[str] = Field(None, pattern=r"^[a-z]+$")
    framework: Optional[str] = Field(None, pattern=r"^[a-z0-9_-]+$")
    requirements: Optional[List[str]] = Field(None, max_length=100)
    languages: Optional[List[str]] = Field(None, max_length=50)
    team_size: Optional[str] = Field(None, pattern=r"^(small|medium|large)$")
    platform: Optional[List[str]] = Field(None, max_length=10)
    
    @field_validator("requirements", "languages", "platform")
    @classmethod
    def validate_string_list(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for item in v:
                if not isinstance(item, str) or len(item) > 100:
                    raise ValueError("List items must be strings with max length 100")
        return v


class GetApplicableStandardsInput(BaseModel):
    """Input validation for get_applicable_standards."""
    context: ContextModel
    include_resolution_details: bool = False


class ValidateAgainstStandardInput(BaseModel):
    """Input validation for validate_against_standard."""
    code: str = Field(..., min_length=1, max_length=MAX_CODE_SIZE)
    standard: str = Field(..., pattern=PATTERNS["identifier"])
    language: str = Field(..., pattern=PATTERNS["language"])
    
    @field_validator("code")
    @classmethod
    def validate_code_safety(cls, v: str) -> str:
        # Basic safety checks
        dangerous_patterns = [
            r"__import__",
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"globals\s*\(",
            r"locals\s*\("
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE):
                raise ValueError(f"Code contains potentially dangerous pattern: {pattern}")
                
        return v


class SearchStandardsInput(BaseModel):
    """Input validation for search_standards."""
    query: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    limit: int = Field(default=10, ge=1, le=100)
    min_relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: Optional[Dict[str, List[str]]] = None
    
    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        # Remove potential injection patterns
        v = re.sub(r'[<>\"\'`;]', '', v)
        return v.strip()


class GetStandardDetailsInput(BaseModel):
    """Input validation for get_standard_details."""
    standard_id: str = Field(..., pattern=PATTERNS["identifier"])


class ListAvailableStandardsInput(BaseModel):
    """Input validation for list_available_standards."""
    category: Optional[str] = Field(None, pattern=PATTERNS["category"])
    limit: int = Field(default=100, ge=1, le=1000)


class SuggestImprovementsInput(BaseModel):
    """Input validation for suggest_improvements."""
    code: str = Field(..., min_length=1, max_length=MAX_CODE_SIZE)
    context: ContextModel


class SyncStandardsInput(BaseModel):
    """Input validation for sync_standards."""
    force: bool = False


class GenerateCrossReferencesInput(BaseModel):
    """Input validation for generate_cross_references."""
    force_refresh: bool = False


class GetOptimizedStandardInput(BaseModel):
    """Input validation for get_optimized_standard."""
    standard_id: str = Field(..., pattern=PATTERNS["identifier"])
    format_type: str = Field(default="condensed", pattern=r"^(full|condensed|reference|summary)$")
    token_budget: Optional[int] = Field(None, ge=100, le=100000)
    required_sections: Optional[List[str]] = Field(None, max_length=50)
    context: Optional[Dict[str, Any]] = None
    

class AutoOptimizeStandardsInput(BaseModel):
    """Input validation for auto_optimize_standards."""
    standard_ids: List[str] = Field(..., min_length=1, max_length=MAX_ARRAY_SIZE)
    total_token_budget: int = Field(..., ge=1000, le=1000000)
    context: Optional[Dict[str, Any]] = None
    
    @field_validator("standard_ids")
    @classmethod
    def validate_standard_ids(cls, v: List[str]) -> List[str]:
        for std_id in v:
            if not re.match(PATTERNS["identifier"], std_id):
                raise ValueError(f"Invalid standard ID: {std_id}")
        return v


class ProgressiveLoadStandardInput(BaseModel):
    """Input validation for progressive_load_standard."""
    standard_id: str = Field(..., pattern=PATTERNS["identifier"])
    initial_sections: List[str] = Field(..., min_length=1, max_length=20)
    max_depth: int = Field(default=3, ge=1, le=10)


class EstimateTokenUsageInput(BaseModel):
    """Input validation for estimate_token_usage."""
    standard_ids: List[str] = Field(..., min_length=1, max_length=MAX_ARRAY_SIZE)
    format_types: Optional[List[str]] = None
    
    @field_validator("standard_ids")
    @classmethod
    def validate_standard_ids(cls, v: List[str]) -> List[str]:
        for std_id in v:
            if not re.match(PATTERNS["identifier"], std_id):
                raise ValueError(f"Invalid standard ID: {std_id}")
        return v
    
    @field_validator("format_types")
    @classmethod
    def validate_format_types(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            valid_formats = {"full", "condensed", "reference", "summary"}
            for fmt in v:
                if fmt not in valid_formats:
                    raise ValueError(f"Invalid format type: {fmt}")
        return v


class GenerateStandardInput(BaseModel):
    """Input validation for generate_standard."""
    template_name: str = Field(..., pattern=PATTERNS["identifier"])
    context: Dict[str, Any]
    title: str = Field(..., min_length=3, max_length=200)
    domain: Optional[str] = Field(None, pattern=PATTERNS["identifier"])


class ValidateStandardInput(BaseModel):
    """Input validation for validate_standard."""
    standard_content: str = Field(..., min_length=10, max_length=MAX_CODE_SIZE)
    format: str = Field(default="yaml", pattern=r"^(yaml|json)$")


class ListTemplatesInput(BaseModel):
    """Input validation for list_templates."""
    domain: Optional[str] = Field(None, pattern=PATTERNS["identifier"])


class GetCrossReferencesInput(BaseModel):
    """Input validation for get_cross_references."""
    standard_id: Optional[str] = Field(None, pattern=PATTERNS["identifier"])
    concept: Optional[str] = Field(None, min_length=1, max_length=200)
    max_depth: int = Field(default=2, ge=1, le=5)
    
    @field_validator("concept")
    @classmethod
    def sanitize_concept(cls, v: Optional[str]) -> Optional[str]:
        if v:
            # Remove potential injection patterns
            v = re.sub(r'[<>"\';]', '', v)
            return v.strip()
        return v


class GetStandardsAnalyticsInput(BaseModel):
    """Input validation for get_standards_analytics."""
    metric_type: str = Field(default="usage", pattern=r"^(usage|popularity|gaps)$")
    time_range: str = Field(default="30d", pattern=r"^\d+[dwmyh]$")
    standard_ids: Optional[List[str]] = Field(None, max_length=100)
    
    @field_validator("standard_ids")
    @classmethod
    def validate_standard_ids(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            for std_id in v:
                if not re.match(PATTERNS["identifier"], std_id):
                    raise ValueError(f"Invalid standard ID: {std_id}")
        return v


class TrackStandardsUsageInput(BaseModel):
    """Input validation for track_standards_usage."""
    standard_id: str = Field(..., pattern=PATTERNS["identifier"])
    usage_type: str = Field(..., pattern=r"^(view|apply|reference)$")
    section_id: Optional[str] = Field(None, pattern=PATTERNS["identifier"])
    context: Optional[Dict[str, Any]] = None


class GetRecommendationsInput(BaseModel):
    """Input validation for get_recommendations."""
    analysis_type: str = Field(default="gaps", pattern=r"^(gaps|quality|usage)$")
    context: Optional[Dict[str, Any]] = None


# Tool input validators mapping
TOOL_VALIDATORS = {
    "get_applicable_standards": GetApplicableStandardsInput,
    "validate_against_standard": ValidateAgainstStandardInput,
    "search_standards": SearchStandardsInput,
    "get_standard_details": GetStandardDetailsInput,
    "list_available_standards": ListAvailableStandardsInput,
    "suggest_improvements": SuggestImprovementsInput,
    "sync_standards": SyncStandardsInput,
    "get_sync_status": None,  # No input required
    "get_optimized_standard": GetOptimizedStandardInput,
    "auto_optimize_standards": AutoOptimizeStandardsInput,
    "progressive_load_standard": ProgressiveLoadStandardInput,
    "estimate_token_usage": EstimateTokenUsageInput,
    "generate_standard": GenerateStandardInput,
    "validate_standard": ValidateStandardInput,
    "list_templates": ListTemplatesInput,
    "get_cross_references": GetCrossReferencesInput,
    "generate_cross_references": GenerateCrossReferencesInput,
    "get_standards_analytics": GetStandardsAnalyticsInput,
    "track_standards_usage": TrackStandardsUsageInput,
    "get_recommendations": GetRecommendationsInput
}


class InputValidator:
    """Validates and sanitizes inputs for MCP tools."""
    
    def __init__(self):
        self.validators = TOOL_VALIDATORS
        self.error_handler = get_secure_error_handler()
        
    def validate_tool_input(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate tool input arguments.
        
        Returns:
            Validated and sanitized arguments
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            # Apply security middleware validation first
            security_middleware = get_security_middleware()
            arguments = security_middleware.validate_and_sanitize_request(arguments)
            
            validator_class = self.validators.get(tool_name)
            
            if validator_class is None:
                # Tool has no input requirements
                return arguments
                
            # Validate using Pydantic model
            validated = validator_class(**arguments)
            return validated.model_dump()
            
        except ValueError as e:
            # Extract field information from Pydantic error
            error_msg = str(e)
            field_match = re.search(r"validation error for (\w+)", error_msg)
            field = field_match.group(1) if field_match else "unknown"
            
            raise ValidationError(
                message=error_msg,
                field=field,
                code=ErrorCode.VALIDATION_INVALID_PARAMETERS
            )
        except Exception as e:
            # Handle security validation errors
            sanitized_message = self.error_handler.sanitize_error_message(str(e))
            raise ValidationError(
                message=sanitized_message,
                field="security",
                code=ErrorCode.VALIDATION_INVALID_PARAMETERS
            )
            
    def validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against a JSON schema.
        
        Returns:
            The validated data
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(data))
            
            if errors:
                # Format the first error
                error = errors[0]
                field = ".".join(str(p) for p in error.path)
                
                raise ValidationError(
                    message=error.message,
                    field=field or "root",
                    code=ErrorCode.VALIDATION_TYPE_MISMATCH
                )
                
            return data
            
        except jsonschema.exceptions.SchemaError as e:
            raise ValidationError(
                message=f"Invalid schema: {str(e)}",
                field="schema",
                code=ErrorCode.VALIDATION_INVALID_PARAMETERS
            )
            
    def sanitize_string(self, value: str, max_length: int = 1000) -> str:
        """Sanitize a string input."""
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Limit length
        if len(value) > max_length:
            value = value[:max_length]
            
        # Remove control characters except newlines and tabs
        value = re.sub(r'[\x01-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', value)
        
        return value
        
    def sanitize_identifier(self, value: str) -> str:
        """Sanitize an identifier (alphanumeric + dash/underscore)."""
        # Keep only allowed characters
        value = re.sub(r'[^a-zA-Z0-9_-]', '', value)
        
        # Ensure it starts with a letter
        if value and not value[0].isalpha():
            value = 'id_' + value
            
        return value[:100]  # Limit length


# Singleton instance
_input_validator: Optional[InputValidator] = None


def get_input_validator() -> InputValidator:
    """Get the singleton input validator instance."""
    global _input_validator
    if _input_validator is None:
        _input_validator = InputValidator()
    return _input_validator