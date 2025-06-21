"""
Standards Engine Data Models
@nist-controls: AC-4, SC-28, SI-12
@evidence: Data models for standards management
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class StandardType(Enum):
    """
    Standard document types from standards repository
    @nist-controls: CM-7
    @evidence: Enumerated standard types
    """
    CS = "coding"
    TS = "testing"
    SEC = "security"
    FE = "frontend"
    DE = "data_engineering"
    CN = "cloud_native"
    OBS = "observability"
    LEG = "legal_compliance"
    PM = "project_management"
    KM = "knowledge_management"
    WD = "web_design"
    EVT = "event_driven"
    DOP = "devops"
    CONT = "content"
    SEO = "seo"
    COST = "cost_optimization"
    GH = "github"
    TOOL = "tools"
    UNIFIED = "unified"

    @classmethod
    def from_string(cls, value: str) -> 'StandardType':
        """Convert string to StandardType"""
        value_upper = value.upper()
        for member in cls:
            if member.name == value_upper:
                return member
        raise ValueError(f"Unknown standard type: {value}")


@dataclass
class StandardSection:
    """
    Represents a section of a standard
    @nist-controls: AC-4
    @evidence: Granular standard content management
    """
    id: str
    type: StandardType
    section: str
    content: str
    tokens: int
    version: str
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    last_updated: datetime | None = None
    dependencies: list[str] = field(default_factory=list)
    nist_controls: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "section": self.section,
            "title": self.title,
            "content": self.content,
            "tokens": self.tokens,
            "version": self.version,
            "tags": self.tags,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "dependencies": self.dependencies,
            "nist_controls": list(self.nist_controls),
            "metadata": self.metadata
        }


@dataclass
class Standard:
    """
    Complete standard document
    @nist-controls: AC-4, CM-7
    @evidence: Full standard document representation
    """
    id: str
    title: str
    description: str | None = None
    category: str = ""
    version: str = "latest"
    sections: list[StandardSection] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "sections": [s.to_dict() for s in self.sections],
            "tags": self.tags,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class StandardQuery(BaseModel):
    """
    Query for loading standards
    @nist-controls: SI-10
    @evidence: Validated standard query format
    """
    query: str = Field(..., description="Standard query string")
    context: str | None = Field(None, description="Additional context for query")
    version: str = Field("latest", description="Standard version")
    token_limit: int | None = Field(10000, description="Maximum tokens to return")
    include_examples: bool = Field(True, description="Include code examples")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query format"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        # Basic injection prevention
        if any(char in v for char in ['<', '>', '\\', '\0']):
            raise ValueError("Query contains invalid characters")
        return v.strip()

    @field_validator('token_limit')
    @classmethod
    def validate_token_limit(cls, v: int | None) -> int | None:
        """Ensure reasonable token limits"""
        if v is not None:
            if v < 100:
                raise ValueError("Token limit must be at least 100")
            if v > 100000:
                raise ValueError("Token limit cannot exceed 100000")
        return v


class StandardLoadResult(BaseModel):
    """
    Result of loading standards
    @nist-controls: AU-10
    @evidence: Auditable standard loading results
    """
    standards: list[dict[str, Any]] = Field(..., description="Loaded standard sections")
    metadata: dict[str, Any] = Field(..., description="Load metadata")
    query_info: dict[str, Any] = Field(default_factory=dict, description="Query processing info")

    model_config = {
        "json_schema_extra": {
            "example": {
                "standards": [
                    {
                        "id": "CS:api:rest",
                        "type": "coding",
                        "section": "rest",
                        "content": "REST API best practices...",
                        "tokens": 1500
                    }
                ],
                "metadata": {
                    "version": "latest",
                    "token_count": 1500,
                    "refs_loaded": ["CS:api"]
                },
                "query_info": {
                    "original_query": "secure api design",
                    "mapped_refs": ["CS:api", "SEC:api"],
                    "processing_time_ms": 45
                }
            }
        }
    }


@dataclass
class NaturalLanguageMapping:
    """
    Maps natural language to standard references
    @nist-controls: AC-4
    @evidence: Controlled mapping of queries to standards
    """
    query_pattern: str
    standard_refs: list[str]
    confidence: float
    keywords: list[str]

    def matches(self, query: str) -> bool:
        """Check if query matches this mapping"""
        query_lower = query.lower()

        # Check for exact pattern match
        if self.query_pattern in query_lower:
            return True

        # Check for keyword matches
        matching_keywords = sum(1 for kw in self.keywords if kw in query_lower)
        return matching_keywords >= len(self.keywords) * 0.6  # 60% threshold


class StandardCache(BaseModel):
    """
    Cache entry for standards
    @nist-controls: SC-28
    @evidence: Secure caching with TTL
    """
    key: str
    value: dict[str, Any] | list[dict[str, Any]]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    hit_count: int | None = None  # Alias for access_count

    @model_validator(mode='after')
    def sync_counts(self) -> 'StandardCache':
        """Sync hit_count with access_count"""
        if self.hit_count is not None and self.access_count == 0:
            self.access_count = self.hit_count
        elif self.hit_count is None:
            self.hit_count = self.access_count
        return self

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        # If expires_at is naive, compare with naive datetime
        if self.expires_at.tzinfo is None:
            return datetime.now() > self.expires_at
        else:
            # Use timezone-aware comparison
            return datetime.now(timezone.utc) > self.expires_at

    def increment_access(self) -> None:
        """Track cache hits"""
        self.access_count += 1
        if self.hit_count is not None:
            self.hit_count += 1


class TokenOptimizationStrategy(Enum):
    """
    Strategies for token optimization
    @nist-controls: SA-8
    @evidence: Defined optimization strategies
    """
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"
    ESSENTIAL_ONLY = "essential_only"
    HIERARCHICAL = "hierarchical"


@dataclass
class TokenBudget:
    """
    Token budget management
    @nist-controls: SA-8
    @evidence: Resource management for LLM contexts
    """
    total: int
    used: int = 0
    available: int | None = None
    reserved: int = 0

    def __post_init__(self):
        """Initialize calculated fields"""
        # Validate non-negative values
        if self.total < 0:
            raise ValueError("Total tokens cannot be negative")
        if self.used < 0:
            raise ValueError("Used tokens cannot be negative")
        if self.reserved < 0:
            raise ValueError("Reserved tokens cannot be negative")
            
        if self.available is None:
            self.available = self.total - self.used - self.reserved
        elif self.available < 0:
            raise ValueError("Available tokens cannot be negative")

    @property
    def total_limit(self) -> int:
        """Alias for total"""
        return self.total

    def get_net_available(self) -> int:
        """Get available minus reserved"""
        return self.available - self.reserved if self.available else 0

    def can_consume(self, tokens: int) -> bool:
        """Check if tokens can be consumed"""
        return tokens <= self.get_net_available()

    def consume(self, tokens: int) -> None:
        """Consume tokens from budget"""
        if not self.can_consume(tokens):
            raise ValueError(f"Cannot consume {tokens} tokens, only {self.get_net_available()} available")
        self.used += tokens
        if self.available is not None:
            self.available -= tokens

    def can_fit(self, tokens: int) -> bool:
        """Check if tokens fit in budget"""
        return tokens <= self.available if self.available else False

    def allocate(self, tokens: int) -> bool:
        """Allocate tokens from budget"""
        if self.can_fit(tokens):
            self.used += tokens
            self.available = self.total - self.used - self.reserved
            return True
        return False

    def reserve(self, tokens: int) -> bool:
        """Reserve tokens for future use"""
        if tokens <= (self.total - self.used - self.reserved):
            self.reserved += tokens
            self.available = self.total - self.used - self.reserved
            return True
        return False
