"""
Pydantic models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class ProjectAnalysisRequest(BaseModel):
    languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    project_type: str = Field(default="")
    deployment_target: str = Field(default="")
    team_size: str = Field(default="")
    compliance_requirements: List[str] = Field(default_factory=list)
    existing_tools: List[str] = Field(default_factory=list)
    performance_requirements: Dict[str, Any] = Field(default_factory=dict)
    security_requirements: Dict[str, Any] = Field(default_factory=dict)
    scalability_requirements: Dict[str, Any] = Field(default_factory=dict)

class StandardSummary(BaseModel):
    id: str
    title: str
    description: str
    category: str
    tags: List[str]
    priority: str
    version: str

class StandardDetail(StandardSummary):
    subcategory: str
    examples: List[Dict[str, Any]]
    rules: Dict[str, Any]
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    metadata: Dict[str, Any]

class SearchResult(BaseModel):
    standard: StandardSummary
    score: float
    highlights: Dict[str, List[str]]

class Recommendation(BaseModel):
    standard: StandardSummary
    relevance_score: float
    confidence: float
    reasoning: str
    implementation_notes: str

class WebSocketMessage(BaseModel):
    type: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)