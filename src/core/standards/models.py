"""
Core data models for the MCP Standards Server.

This module defines the primary data structures used throughout the system
for representing standards, requirements, evidence, and compliance mappings.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Priority(str, Enum):
    """Priority levels for standards and requirements."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceType(str, Enum):
    """Types of evidence for compliance."""

    DOCUMENT = "document"
    CODE = "code"
    TEST = "test"
    REVIEW = "review"
    AUDIT = "audit"
    CERTIFICATE = "certificate"
    OTHER = "other"


class ValidationStatus(str, Enum):
    """Status of validation or evidence verification."""

    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    EXPIRED = "expired"


class ImplementationStatus(str, Enum):
    """Implementation status for compliance mappings."""

    NOT_IMPLEMENTED = "not_implemented"
    PLANNED = "planned"
    PARTIALLY_IMPLEMENTED = "partially_implemented"
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"


@dataclass
class StandardMetadata:
    """Metadata associated with a standard."""

    version: str = "1.0.0"
    last_updated: datetime | None = None
    authors: list[str] = field(default_factory=list)
    source: str | None = None
    compliance_frameworks: list[str] = field(default_factory=list)
    nist_controls: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    language: str = "en"
    scope: str | None = None
    applicability: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            elif isinstance(value, list):
                result[key] = list(value)  # type: ignore[assignment]
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StandardMetadata":
        """Create from dictionary."""
        # Handle backward compatibility: convert 'author' to 'authors'
        if "author" in data and "authors" not in data:
            authors_value = data.pop("author")
            if isinstance(authors_value, str):
                data["authors"] = [authors_value]
            elif isinstance(authors_value, list):
                data["authors"] = authors_value

        if "last_updated" in data and data["last_updated"]:
            if isinstance(data["last_updated"], str):
                data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


@dataclass
class Requirement:
    """Individual requirement within a standard."""

    id: str
    title: str
    description: str
    priority: Priority = Priority.MEDIUM
    category: str = "general"
    mandatory: bool = True
    evidence_required: bool = False
    validation_criteria: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    related_controls: list[str] = field(default_factory=list)
    implementation_notes: str | None = None
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.value,
            "category": self.category,
            "mandatory": self.mandatory,
            "evidence_required": self.evidence_required,
            "validation_criteria": self.validation_criteria,
            "tags": self.tags,
            "related_controls": self.related_controls,
            "implementation_notes": self.implementation_notes,
            "examples": self.examples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Requirement":
        """Create from dictionary."""
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = Priority(data["priority"])
        return cls(**data)


@dataclass
class Evidence:
    """Evidence for compliance tracking and validation."""

    id: str
    requirement_id: str
    type: EvidenceType
    description: str
    location: str | None = None
    status: ValidationStatus = ValidationStatus.PENDING
    created_at: datetime | None = None
    verified_at: datetime | None = None
    verifier: str | None = None
    notes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set default created_at if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "requirement_id": self.requirement_id,
            "type": self.type.value,
            "description": self.description,
            "location": self.location,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            "verifier": self.verifier,
            "notes": self.notes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        """Create from dictionary."""
        if "type" in data and isinstance(data["type"], str):
            data["type"] = EvidenceType(data["type"])
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ValidationStatus(data["status"])
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "verified_at" in data and isinstance(data["verified_at"], str):
            data["verified_at"] = datetime.fromisoformat(data["verified_at"])
        return cls(**data)


@dataclass
class ComplianceMapping:
    """Mapping between standards and compliance frameworks like NIST 800-53 r5."""

    standard_id: str
    control_id: str
    control_family: str
    implementation_status: ImplementationStatus = ImplementationStatus.NOT_IMPLEMENTED
    assessment_methods: list[str] = field(default_factory=list)
    responsible_entity: str | None = None
    implementation_guidance: str | None = None
    last_assessed: datetime | None = None
    next_assessment: datetime | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard_id": self.standard_id,
            "control_id": self.control_id,
            "control_family": self.control_family,
            "implementation_status": self.implementation_status.value,
            "assessment_methods": self.assessment_methods,
            "responsible_entity": self.responsible_entity,
            "implementation_guidance": self.implementation_guidance,
            "last_assessed": (
                self.last_assessed.isoformat() if self.last_assessed else None
            ),
            "next_assessment": (
                self.next_assessment.isoformat() if self.next_assessment else None
            ),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComplianceMapping":
        """Create from dictionary."""
        if "implementation_status" in data and isinstance(
            data["implementation_status"], str
        ):
            data["implementation_status"] = ImplementationStatus(
                data["implementation_status"]
            )
        if "last_assessed" in data and isinstance(data["last_assessed"], str):
            data["last_assessed"] = datetime.fromisoformat(data["last_assessed"])
        if "next_assessment" in data and isinstance(data["next_assessment"], str):
            data["next_assessment"] = datetime.fromisoformat(data["next_assessment"])
        return cls(**data)


@dataclass
class RuleCondition:
    """Condition for rule engine integration."""

    field: str
    operator: str
    value: Any
    description: str | None = None

    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate the condition against a context."""
        if self.field not in context:
            return False

        actual_value = context[self.field]

        if self.operator == "equals":
            return bool(actual_value == self.value)
        elif self.operator == "contains":
            return (
                self.value in actual_value
                if isinstance(actual_value, list | str)
                else False
            )
        elif self.operator == "in":
            return (
                actual_value in self.value
                if isinstance(self.value, list | tuple)
                else False
            )
        elif self.operator == "greater_than":
            return bool(actual_value > self.value)
        elif self.operator == "less_than":
            return bool(actual_value < self.value)
        elif self.operator == "matches":
            import re

            return bool(re.match(self.value, str(actual_value)))
        else:
            return False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RuleCondition":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CrossReference:
    """Cross-reference between standards."""

    source_standard: str
    target_standard: str
    relationship_type: str
    description: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_standard": self.source_standard,
            "target_standard": self.target_standard,
            "relationship_type": self.relationship_type,
            "description": self.description,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CrossReference":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SearchResult:
    """Result from semantic search operations."""

    standard_id: str
    title: str
    score: float
    excerpt: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard_id": self.standard_id,
            "title": self.title,
            "score": self.score,
            "excerpt": self.excerpt,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ValidationResult:
    """Result from validation operations."""

    standard_id: str
    requirement_id: str | None = None
    status: ValidationStatus = ValidationStatus.PENDING
    score: float | None = None
    messages: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "standard_id": self.standard_id,
            "requirement_id": self.requirement_id,
            "status": self.status.value,
            "score": self.score,
            "messages": self.messages,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        """Create from dictionary."""
        if "status" in data and isinstance(data["status"], str):
            data["status"] = ValidationStatus(data["status"])
        return cls(**data)


@dataclass
class Standard:
    """Main standard document model."""

    id: str
    title: str
    description: str
    content: str
    category: str
    subcategory: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: StandardMetadata = field(default_factory=StandardMetadata)
    requirements: list[Requirement] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    compliance_mappings: list[ComplianceMapping] = field(default_factory=list)
    cross_references: list[CrossReference] = field(default_factory=list)
    rules: list[RuleCondition] = field(default_factory=list)
    examples: list[str] = field(default_factory=list)
    priority: Priority = Priority.MEDIUM
    version: str = "1.0.0"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    author: str | None = None
    source: str | None = None

    def __post_init__(self) -> None:
        """Set default timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()

    def add_requirement(self, requirement: Requirement) -> None:
        """Add a requirement to the standard."""
        self.requirements.append(requirement)

    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence to the standard."""
        self.evidence.append(evidence)

    def add_compliance_mapping(self, mapping: ComplianceMapping) -> None:
        """Add a compliance mapping to the standard."""
        self.compliance_mappings.append(mapping)

    def get_mandatory_requirements(self) -> list[Requirement]:
        """Get all mandatory requirements."""
        return [req for req in self.requirements if req.mandatory]

    def get_requirements_by_category(self, category: str) -> list[Requirement]:
        """Get requirements by category."""
        return [req for req in self.requirements if req.category == category]

    def get_evidence_by_status(self, status: ValidationStatus) -> list[Evidence]:
        """Get evidence by status."""
        return [ev for ev in self.evidence if ev.status == status]

    def calculate_compliance_score(self) -> float:
        """Calculate overall compliance score."""
        if not self.requirements:
            return 0.0

        total_requirements = len(self.requirements)
        verified_evidence = len(self.get_evidence_by_status(ValidationStatus.VERIFIED))

        return min(verified_evidence / total_requirements, 1.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "content": self.content,
            "category": self.category,
            "subcategory": self.subcategory,
            "tags": self.tags,
            "metadata": self.metadata.to_dict(),
            "requirements": [req.to_dict() for req in self.requirements],
            "evidence": [ev.to_dict() for ev in self.evidence],
            "compliance_mappings": [
                mapping.to_dict() for mapping in self.compliance_mappings
            ],
            "cross_references": [ref.to_dict() for ref in self.cross_references],
            "rules": [rule.to_dict() for rule in self.rules],
            "examples": self.examples,
            "priority": self.priority.value,
            "version": self.version,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "author": self.author,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Standard":
        """Create from dictionary."""
        # Handle enum conversion
        if "priority" in data and isinstance(data["priority"], str):
            data["priority"] = Priority(data["priority"])

        # Handle datetime conversion
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])

        # Handle nested objects
        if "metadata" in data and isinstance(data["metadata"], dict):
            data["metadata"] = StandardMetadata.from_dict(data["metadata"])

        if "requirements" in data and isinstance(data["requirements"], list):
            data["requirements"] = [
                Requirement.from_dict(req) for req in data["requirements"]
            ]

        if "evidence" in data and isinstance(data["evidence"], list):
            data["evidence"] = [Evidence.from_dict(ev) for ev in data["evidence"]]

        if "compliance_mappings" in data and isinstance(
            data["compliance_mappings"], list
        ):
            data["compliance_mappings"] = [
                ComplianceMapping.from_dict(mapping)
                for mapping in data["compliance_mappings"]
            ]

        if "cross_references" in data and isinstance(data["cross_references"], list):
            data["cross_references"] = [
                CrossReference.from_dict(ref) for ref in data["cross_references"]
            ]

        if "rules" in data and isinstance(data["rules"], list):
            data["rules"] = [RuleCondition.from_dict(rule) for rule in data["rules"]]

        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Standard":
        """Create from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self) -> str:
        """String representation."""
        return f"Standard(id={self.id}, title={self.title}, category={self.category})"

    def __repr__(self) -> str:
        """Detailed representation."""
        return (
            f"Standard(id={self.id}, title={self.title}, category={self.category}, "
            f"requirements={len(self.requirements)}, evidence={len(self.evidence)})"
        )
