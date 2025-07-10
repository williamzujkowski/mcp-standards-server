"""
Pydantic models for API requests and responses
"""
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    filters: dict[str, Any] | None = Field(default_factory=dict)
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

class ProjectAnalysisRequest(BaseModel):
    languages: list[str] = Field(default_factory=list)
    frameworks: list[str] = Field(default_factory=list)
    project_type: str = Field(default="")
    deployment_target: str = Field(default="")
    team_size: str = Field(default="")
    compliance_requirements: list[str] = Field(default_factory=list)
    existing_tools: list[str] = Field(default_factory=list)
    performance_requirements: dict[str, Any] = Field(default_factory=dict)
    security_requirements: dict[str, Any] = Field(default_factory=dict)
    scalability_requirements: dict[str, Any] = Field(default_factory=dict)

class StandardSummary(BaseModel):
    id: str
    title: str
    description: str
    category: str
    tags: list[str]
    priority: str
    version: str

class StandardDetail(StandardSummary):
    subcategory: str
    examples: list[dict[str, Any]]
    rules: dict[str, Any]
    created_at: datetime | None
    updated_at: datetime | None
    metadata: dict[str, Any]

class SearchResult(BaseModel):
    standard: StandardSummary
    score: float
    highlights: dict[str, list[str]]

class Recommendation(BaseModel):
    standard: StandardSummary
    relevance_score: float
    confidence: float
    reasoning: str
    implementation_notes: str

class WebSocketMessage(BaseModel):
    type: str
    data: dict[str, Any] | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
