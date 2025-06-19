"""
Standards Engine Data Models
@nist-controls: AC-4, SC-28, SI-12
@evidence: Data models for standards management
"""
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


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
    last_updated: datetime
    dependencies: List[str] = field(default_factory=list)
    nist_controls: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "type": self.type.value,
            "section": self.section,
            "content": self.content,
            "tokens": self.tokens,
            "version": self.version,
            "last_updated": self.last_updated.isoformat(),
            "dependencies": self.dependencies,
            "nist_controls": list(self.nist_controls),
            "metadata": self.metadata
        }


class StandardQuery(BaseModel):
    """
    Query for loading standards
    @nist-controls: SI-10
    @evidence: Validated standard query format
    """
    query: str = Field(..., description="Standard query string")
    context: Optional[str] = Field(None, description="Additional context for query")
    version: str = Field("latest", description="Standard version")
    token_limit: Optional[int] = Field(None, description="Maximum tokens to return")
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query format"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        # Basic injection prevention
        if any(char in v for char in ['<', '>', '\\', '\0']):
            raise ValueError("Query contains invalid characters")
        return v.strip()
    
    @validator('token_limit')
    def validate_token_limit(cls, v):
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
    standards: List[Dict[str, Any]] = Field(..., description="Loaded standard sections")
    metadata: Dict[str, Any] = Field(..., description="Load metadata")
    query_info: Dict[str, Any] = Field(default_factory=dict, description="Query processing info")
    
    class Config:
        json_schema_extra = {
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


@dataclass
class NaturalLanguageMapping:
    """
    Maps natural language to standard references
    @nist-controls: AC-4
    @evidence: Controlled mapping of queries to standards
    """
    query_pattern: str
    standard_refs: List[str]
    confidence: float
    keywords: List[str]
    
    def matches(self, query: str) -> bool:
        """Check if query matches this mapping"""
        query_lower = query.lower()
        
        # Check for exact pattern match
        if self.query_pattern in query_lower:
            return True
            
        # Check for keyword matches
        matching_keywords = sum(1 for kw in self.keywords if kw in query_lower)
        if matching_keywords >= len(self.keywords) * 0.6:  # 60% threshold
            return True
            
        return False


class StandardCache(BaseModel):
    """
    Cache entry for standards
    @nist-controls: SC-28
    @evidence: Secure caching with TTL
    """
    key: str
    value: List[Dict[str, Any]]
    created_at: datetime
    expires_at: datetime
    access_count: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return datetime.utcnow() > self.expires_at
    
    def increment_access(self):
        """Track cache hits"""
        self.access_count += 1


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
    total_limit: int
    used: int = 0
    reserved: int = 0
    
    @property
    def available(self) -> int:
        """Calculate available tokens"""
        return self.total_limit - self.used - self.reserved
    
    def can_fit(self, tokens: int) -> bool:
        """Check if tokens fit in budget"""
        return tokens <= self.available
    
    def allocate(self, tokens: int) -> bool:
        """Allocate tokens from budget"""
        if self.can_fit(tokens):
            self.used += tokens
            return True
        return False
    
    def reserve(self, tokens: int) -> bool:
        """Reserve tokens for future use"""
        if tokens <= (self.total_limit - self.used - self.reserved):
            self.reserved += tokens
            return True
        return False