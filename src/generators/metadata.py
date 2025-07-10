"""
Metadata Schema and Validation

Schema definitions and validation for standards metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StandardMetadata:
    """Standard metadata model with validation."""

    # Required fields
    title: str
    version: str
    domain: str
    type: str

    # Optional fields
    description: str = ""
    author: str = ""
    created_date: datetime | None = None
    updated_date: datetime | None = None
    tags: list[str] = field(default_factory=list)
    nist_controls: list[str] = field(default_factory=list)
    compliance_frameworks: list[str] = field(default_factory=list)
    risk_level: str = "moderate"
    maturity_level: str = "developing"

    # Technical fields
    implementation_guides: list[str] = field(default_factory=list)
    code_examples: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Validation fields
    review_status: str = "draft"
    reviewers: list[str] = field(default_factory=list)
    approval_date: datetime | None = None

    # Custom fields
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.updated_date is None:
            self.updated_date = datetime.now()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardMetadata":
        """Create StandardMetadata from dictionary."""
        # Handle datetime fields
        if "created_date" in data and isinstance(data["created_date"], str):
            data["created_date"] = datetime.fromisoformat(data["created_date"])
        if "updated_date" in data and isinstance(data["updated_date"], str):
            data["updated_date"] = datetime.fromisoformat(data["updated_date"])
        if "approval_date" in data and isinstance(data["approval_date"], str):
            data["approval_date"] = datetime.fromisoformat(data["approval_date"])

        # Extract known fields
        known_fields = {
            "title",
            "version",
            "domain",
            "type",
            "description",
            "author",
            "created_date",
            "updated_date",
            "tags",
            "nist_controls",
            "compliance_frameworks",
            "risk_level",
            "maturity_level",
            "implementation_guides",
            "code_examples",
            "dependencies",
            "review_status",
            "reviewers",
            "approval_date",
        }

        standard_data = {k: v for k, v in data.items() if k in known_fields}
        custom_fields = {k: v for k, v in data.items() if k not in known_fields}

        return cls(custom_fields=custom_fields, **standard_data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "title": self.title,
            "version": self.version,
            "domain": self.domain,
            "type": self.type,
            "description": self.description,
            "author": self.author,
            "created_date": (
                self.created_date.isoformat() if self.created_date else None
            ),
            "updated_date": (
                self.updated_date.isoformat() if self.updated_date else None
            ),
            "tags": self.tags,
            "nist_controls": self.nist_controls,
            "compliance_frameworks": self.compliance_frameworks,
            "risk_level": self.risk_level,
            "maturity_level": self.maturity_level,
            "implementation_guides": self.implementation_guides,
            "code_examples": self.code_examples,
            "dependencies": self.dependencies,
            "review_status": self.review_status,
            "reviewers": self.reviewers,
            "approval_date": (
                self.approval_date.isoformat() if self.approval_date else None
            ),
        }

        # Add custom fields
        result.update(self.custom_fields)

        return result

    def validate(self) -> dict[str, Any]:
        """Validate metadata fields."""
        errors = []
        warnings = []

        # Required field validation
        if not self.title:
            errors.append("Title is required")
        if not self.version:
            errors.append("Version is required")
        if not self.domain:
            errors.append("Domain is required")
        if not self.type:
            errors.append("Type is required")

        # Format validation
        if self.version and not self._is_valid_version(self.version):
            errors.append("Version must follow semantic versioning (e.g., 1.0.0)")

        if self.risk_level not in ["low", "moderate", "high"]:
            errors.append("Risk level must be 'low', 'moderate', or 'high'")

        if self.maturity_level not in [
            "planning",
            "developing",
            "testing",
            "production",
            "deprecated",
        ]:
            errors.append(
                "Maturity level must be one of: planning, developing, testing, production, deprecated"
            )

        if self.review_status not in ["draft", "review", "approved", "rejected"]:
            errors.append(
                "Review status must be one of: draft, review, approved, rejected"
            )

        # Warning validation
        if not self.description:
            warnings.append("Description is recommended")
        if not self.author:
            warnings.append("Author is recommended")
        if not self.nist_controls:
            warnings.append("NIST controls mapping is recommended")

        return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}

    def _is_valid_version(self, version: str) -> bool:
        """Check if version follows semantic versioning."""
        parts = version.split(".")
        if len(parts) != 3:
            return False

        try:
            [int(part) for part in parts]
            return True
        except ValueError:
            return False


class MetadataSchema:
    """Schema definitions for different types of standards."""

    BASE_SCHEMA: dict[str, Any] = {
        "type": "object",
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "version": {"type": "string", "pattern": r"^\d+\.\d+\.\d+$"},
            "domain": {"type": "string", "minLength": 1},
            "type": {
                "type": "string",
                "enum": ["technical", "compliance", "process", "architecture"],
            },
            "description": {"type": "string"},
            "author": {"type": "string"},
            "created_date": {"type": "string", "format": "date-time"},
            "updated_date": {"type": "string", "format": "date-time"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "nist_controls": {"type": "array", "items": {"type": "string"}},
            "compliance_frameworks": {"type": "array", "items": {"type": "string"}},
            "risk_level": {"type": "string", "enum": ["low", "moderate", "high"]},
            "maturity_level": {
                "type": "string",
                "enum": [
                    "planning",
                    "developing",
                    "testing",
                    "production",
                    "deprecated",
                ],
            },
            "implementation_guides": {"type": "array", "items": {"type": "string"}},
            "code_examples": {"type": "array", "items": {"type": "string"}},
            "dependencies": {"type": "array", "items": {"type": "string"}},
            "review_status": {
                "type": "string",
                "enum": ["draft", "review", "approved", "rejected"],
            },
            "reviewers": {"type": "array", "items": {"type": "string"}},
            "approval_date": {"type": "string", "format": "date-time"},
        },
        "required": ["title", "version", "domain", "type"],
        "additionalProperties": True,
    }

    TECHNICAL_SCHEMA: dict[str, Any] = {
        **BASE_SCHEMA,
        "properties": {
            **BASE_SCHEMA["properties"],
            "programming_languages": {"type": "array", "items": {"type": "string"}},
            "frameworks": {"type": "array", "items": {"type": "string"}},
            "tools": {"type": "array", "items": {"type": "string"}},
            "platforms": {"type": "array", "items": {"type": "string"}},
            "performance_requirements": {"type": "object"},
            "security_requirements": {"type": "object"},
        },
    }

    COMPLIANCE_SCHEMA: dict[str, Any] = {
        **BASE_SCHEMA,
        "properties": {
            **BASE_SCHEMA["properties"],
            "regulatory_requirements": {"type": "array", "items": {"type": "string"}},
            "audit_requirements": {"type": "object"},
            "evidence_requirements": {"type": "array", "items": {"type": "string"}},
            "compliance_metrics": {"type": "object"},
        },
    }

    PROCESS_SCHEMA: dict[str, Any] = {
        **BASE_SCHEMA,
        "properties": {
            **BASE_SCHEMA["properties"],
            "process_steps": {"type": "array", "items": {"type": "object"}},
            "roles_responsibilities": {"type": "object"},
            "decision_points": {"type": "array", "items": {"type": "object"}},
            "success_criteria": {"type": "array", "items": {"type": "string"}},
        },
    }

    ARCHITECTURE_SCHEMA: dict[str, Any] = {
        **BASE_SCHEMA,
        "properties": {
            **BASE_SCHEMA["properties"],
            "architecture_patterns": {"type": "array", "items": {"type": "string"}},
            "design_principles": {"type": "array", "items": {"type": "string"}},
            "technology_stack": {"type": "object"},
            "scalability_requirements": {"type": "object"},
            "availability_requirements": {"type": "object"},
        },
    }

    @classmethod
    def get_schema(cls, standard_type: str) -> dict[str, Any]:
        """Get schema for a specific standard type."""
        schema_map = {
            "technical": cls.TECHNICAL_SCHEMA,
            "compliance": cls.COMPLIANCE_SCHEMA,
            "process": cls.PROCESS_SCHEMA,
            "architecture": cls.ARCHITECTURE_SCHEMA,
        }

        return schema_map.get(standard_type, cls.BASE_SCHEMA)

    @classmethod
    def validate_against_schema(
        cls, data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate data against schema."""
        try:
            import jsonschema

            jsonschema.validate(data, schema)
            return {"valid": True, "errors": []}
        except ImportError:
            # Fallback validation if jsonschema not available
            return cls._basic_validation(data, schema)
        except jsonschema.ValidationError as e:
            return {"valid": False, "errors": [str(e)]}

    @classmethod
    def _basic_validation(
        cls, data: dict[str, Any], schema: dict[str, Any]
    ) -> dict[str, Any]:
        """Basic validation without jsonschema."""
        errors = []

        # Check required fields
        required = schema.get("required", [])
        for field_name in required:
            if field_name not in data:
                errors.append(f"Required field '{field_name}' is missing")

        # Check field types
        properties = schema.get("properties", {})
        for field_name, value in data.items():
            if field_name in properties:
                field_schema = properties[field_name]
                if not cls._validate_field_type(value, field_schema):
                    errors.append(f"Field '{field_name}' has invalid type")

        return {"valid": len(errors) == 0, "errors": errors}

    @classmethod
    def _validate_field_type(cls, value: Any, field_schema: dict[str, Any]) -> bool:
        """Validate field type."""
        expected_type = field_schema.get("type")

        if expected_type == "string":
            return isinstance(value, str)
        elif expected_type == "array":
            return isinstance(value, list)
        elif expected_type == "object":
            return isinstance(value, dict)
        elif expected_type == "integer":
            return isinstance(value, int)
        elif expected_type == "number":
            return isinstance(value, int | float)
        elif expected_type == "boolean":
            return isinstance(value, bool)

        return True
