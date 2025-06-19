"""
Standards Engine - Intelligent loading and caching
@nist-controls: AC-4, SC-28, SI-12
@evidence: Information flow control and secure caching
"""
from typing import Dict, List, Optional, Set, Any, Tuple
from pathlib import Path
import json
import yaml
import re
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

import redis
from redis.exceptions import RedisError

from .models import (
    StandardType, StandardSection, StandardQuery, StandardLoadResult,
    NaturalLanguageMapping, StandardCache, TokenBudget, TokenOptimizationStrategy
)
from ..logging import get_logger, audit_log


logger = get_logger(__name__)


class NaturalLanguageMapper:
    """
    Maps natural language queries to standards
    Based on mappings in CLAUDE.md
    @nist-controls: AC-4
    @evidence: Controlled query mapping
    """
    
    def __init__(self):
        self.mappings = self._initialize_mappings()
        
    def _initialize_mappings(self) -> List[NaturalLanguageMapping]:
        """Initialize standard mappings"""
        return [
            NaturalLanguageMapping(
                query_pattern="secure api",
                standard_refs=["CS:api", "SEC:api", "TS:integration"],
                confidence=0.95,
                keywords=["secure", "api", "security", "endpoint"]
            ),
            NaturalLanguageMapping(
                query_pattern="react app",
                standard_refs=["FE:react", "WD:*", "TS:jest", "CS:javascript"],
                confidence=0.90,
                keywords=["react", "frontend", "component", "jsx"]
            ),
            NaturalLanguageMapping(
                query_pattern="microservices",
                standard_refs=["CN:microservices", "EVT:*", "OBS:distributed"],
                confidence=0.92,
                keywords=["microservice", "distributed", "service", "mesh"]
            ),
            NaturalLanguageMapping(
                query_pattern="ci/cd pipeline",
                standard_refs=["DOP:cicd", "GH:actions", "TS:*"],
                confidence=0.88,
                keywords=["ci", "cd", "pipeline", "deployment", "automation"]
            ),
            NaturalLanguageMapping(
                query_pattern="database optimization",
                standard_refs=["CS:performance", "DE:optimization", "OBS:metrics"],
                confidence=0.85,
                keywords=["database", "optimization", "performance", "query"]
            ),
            NaturalLanguageMapping(
                query_pattern="documentation system",
                standard_refs=["KM:*", "CS:documentation", "DOP:automation"],
                confidence=0.87,
                keywords=["documentation", "docs", "knowledge", "wiki"]
            ),
            NaturalLanguageMapping(
                query_pattern="nist compliance",
                standard_refs=["SEC:*", "LEG:compliance", "OBS:logging", "TS:security"],
                confidence=0.95,
                keywords=["nist", "compliance", "controls", "800-53"]
            ),
            NaturalLanguageMapping(
                query_pattern="fedramp",
                standard_refs=["SEC:fedramp", "LEG:fedramp", "CN:govcloud", "OBS:continuous-monitoring"],
                confidence=0.93,
                keywords=["fedramp", "federal", "government", "ato"]
            ),
            NaturalLanguageMapping(
                query_pattern="authentication",
                standard_refs=["SEC:authentication", "CS:auth", "TS:auth"],
                confidence=0.91,
                keywords=["auth", "authentication", "login", "sso", "oauth"]
            ),
            NaturalLanguageMapping(
                query_pattern="kubernetes",
                standard_refs=["CN:kubernetes", "DOP:kubernetes", "OBS:kubernetes"],
                confidence=0.89,
                keywords=["kubernetes", "k8s", "container", "orchestration"]
            )
        ]
    
    def map_query(self, query: str) -> Tuple[List[str], float]:
        """
        Map natural language query to standard notations
        Returns (standard_refs, confidence)
        """
        matched_standards = []
        total_confidence = 0.0
        match_count = 0
        
        for mapping in self.mappings:
            if mapping.matches(query):
                matched_standards.extend(mapping.standard_refs)
                total_confidence += mapping.confidence
                match_count += 1
                
        # Remove duplicates while preserving order
        seen = set()
        unique_standards = []
        for std in matched_standards:
            if std not in seen:
                seen.add(std)
                unique_standards.append(std)
                
        # Calculate average confidence
        avg_confidence = total_confidence / match_count if match_count > 0 else 0.0
        
        return unique_standards, avg_confidence


class StandardsEngine:
    """
    Core engine for loading and managing standards
    @nist-controls: AC-3, AC-4, CM-7
    @evidence: Access control and least functionality
    """
    
    def __init__(
        self,
        standards_path: Path,
        redis_client: Optional[redis.Redis] = None,
        cache_ttl: int = 3600
    ):
        self.standards_path = standards_path
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
        self.nl_mapper = NaturalLanguageMapper()
        self.loaded_standards: Dict[str, StandardSection] = {}
        
        # Initialize schema if available
        self.schema = self._load_schema()
        
    def _load_schema(self) -> Optional[Dict[str, Any]]:
        """Load standards schema"""
        schema_path = self.standards_path / "standards-schema.yaml"
        if schema_path.exists():
            with open(schema_path) as f:
                return yaml.safe_load(f)
        return None
    
    @audit_log(["AC-4", "SI-10"])
    async def parse_query(self, query: str) -> Tuple[List[str], Dict[str, Any]]:
        """
        Parse various query formats
        Returns (standard_refs, query_info)
        @nist-controls: SI-10
        @evidence: Input validation for queries
        """
        # Remove @ prefix if present
        query = query.strip().lstrip('@')
        query_info = {
            "original_query": query,
            "query_type": "unknown",
            "confidence": 1.0
        }
        
        # Natural language query
        if ':' not in query and not query.startswith('load'):
            refs, confidence = self.nl_mapper.map_query(query)
            query_info["query_type"] = "natural_language"
            query_info["confidence"] = confidence
            query_info["mapped_refs"] = refs
            return refs, query_info
            
        # Direct notation (CS:api, SEC:*, etc.)
        if ':' in query:
            parts = []
            for part in query.split('+'):
                part = part.strip()
                if part and self._validate_standard_ref(part):
                    parts.append(part)
            query_info["query_type"] = "direct_notation"
            query_info["refs"] = parts
            return parts, query_info
            
        # Load command
        if query.startswith('load'):
            # Extract the query part
            query = query[4:].strip()
            return await self.parse_query(query)
            
        return [], query_info
    
    def _validate_standard_ref(self, ref: str) -> bool:
        """Validate standard reference format"""
        pattern = r'^[A-Z]+:[a-zA-Z0-9_\-\*]+$'
        return bool(re.match(pattern, ref))
    
    @audit_log(["AC-4", "SC-28"])
    async def load_standards(
        self,
        query_obj: StandardQuery
    ) -> StandardLoadResult:
        """
        Load standards based on query
        @nist-controls: AC-4, SC-28
        @evidence: Information flow control with caching
        """
        start_time = datetime.now()
        
        # Parse the query
        standard_refs, query_info = await self.parse_query(query_obj.query)
        
        # Add context-based standards
        if query_obj.context:
            context_refs = await self._analyze_context(query_obj.context)
            standard_refs.extend(context_refs)
            query_info["context_refs"] = context_refs
            
        # Initialize token budget
        budget = TokenBudget(total_limit=query_obj.token_limit or 50000)
        
        # Load each standard
        loaded = []
        total_tokens = 0
        
        for ref in standard_refs:
            # Check cache first
            cached_sections = await self._get_from_cache(ref, query_obj.version)
            if cached_sections:
                sections = cached_sections
            else:
                sections = await self._load_standard_sections(ref, query_obj.version)
                await self._save_to_cache(ref, query_obj.version, sections)
            
            for section in sections:
                if budget.can_fit(section.tokens):
                    budget.allocate(section.tokens)
                    loaded.append(section)
                    total_tokens += section.tokens
                else:
                    # Apply token optimization
                    optimized = await self._optimize_for_tokens(
                        section,
                        budget.available,
                        TokenOptimizationStrategy.SUMMARIZE
                    )
                    if optimized and budget.can_fit(optimized.tokens):
                        budget.allocate(optimized.tokens)
                        loaded.append(optimized)
                        total_tokens += optimized.tokens
                    else:
                        # Can't fit even optimized version
                        break
                        
                if not budget.available:
                    break
                    
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        query_info["processing_time_ms"] = int(processing_time)
        
        # Build result
        result = StandardLoadResult(
            standards=[s.to_dict() for s in loaded],
            metadata={
                "version": query_obj.version,
                "token_count": total_tokens,
                "refs_loaded": list(set(s.id for s in loaded)),
                "total_sections": len(loaded)
            },
            query_info=query_info
        )
        
        logger.info(
            f"Loaded {len(loaded)} standard sections ({total_tokens} tokens)",
            extra={"query": query_obj.query, "refs": standard_refs}
        )
        
        return result
    
    async def _get_from_cache(
        self,
        ref: str,
        version: str
    ) -> Optional[List[StandardSection]]:
        """Get sections from cache"""
        if not self.redis_client:
            return None
            
        cache_key = f"standard:{ref}:{version}"
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                data = json.loads(cached_data)
                sections = []
                for item in data:
                    # Reconstruct StandardSection
                    section = StandardSection(
                        id=item["id"],
                        type=StandardType.from_string(item["type"]),
                        section=item["section"],
                        content=item["content"],
                        tokens=item["tokens"],
                        version=item["version"],
                        last_updated=datetime.fromisoformat(item["last_updated"]),
                        dependencies=item.get("dependencies", []),
                        nist_controls=set(item.get("nist_controls", [])),
                        metadata=item.get("metadata", {})
                    )
                    sections.append(section)
                return sections
        except (RedisError, json.JSONDecodeError) as e:
            logger.error(f"Cache retrieval error: {e}")
            
        return None
    
    async def _save_to_cache(
        self,
        ref: str,
        version: str,
        sections: List[StandardSection]
    ) -> None:
        """Save sections to cache"""
        if not self.redis_client or not sections:
            return
            
        cache_key = f"standard:{ref}:{version}"
        cache_data = [s.to_dict() for s in sections]
        
        try:
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(cache_data)
            )
        except RedisError as e:
            logger.error(f"Cache save error: {e}")
    
    async def _load_standard_sections(
        self,
        ref: str,
        version: str
    ) -> List[StandardSection]:
        """Load sections for a standard reference"""
        # Parse reference (e.g., "CS:api" or "SEC:*")
        parts = ref.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid standard reference: {ref}")
            
        std_type = parts[0]
        section = parts[1]
        
        sections = []
        
        if section == '*':
            # Load all sections
            sections = await self._load_all_sections(std_type, version)
        else:
            # Load specific section
            section_data = await self._load_section(std_type, section, version)
            if section_data:
                sections.append(section_data)
                
        return sections
    
    async def _load_section(
        self,
        std_type: str,
        section: str,
        version: str
    ) -> Optional[StandardSection]:
        """Load a specific section from file"""
        # This is a placeholder - in real implementation, would load from actual files
        # For now, return mock data
        return StandardSection(
            id=f"{std_type}:{section}",
            type=StandardType.from_string(std_type),
            section=section,
            content=f"Content for {std_type}:{section} standard",
            tokens=1000,
            version=version,
            last_updated=datetime.now(),
            dependencies=[],
            nist_controls=set(),
            metadata={}
        )
    
    async def _load_all_sections(
        self,
        std_type: str,
        version: str
    ) -> List[StandardSection]:
        """Load all sections for a standard type"""
        # Placeholder - would load from actual files
        sections = []
        for section in ["intro", "best-practices", "examples"]:
            section_data = await self._load_section(std_type, section, version)
            if section_data:
                sections.append(section_data)
        return sections
    
    async def _analyze_context(self, context: str) -> List[str]:
        """Analyze context to suggest additional standards"""
        # Simple keyword analysis
        suggested = []
        
        context_lower = context.lower()
        if "test" in context_lower:
            suggested.append("TS:*")
        if "security" in context_lower:
            suggested.append("SEC:*")
        if "api" in context_lower:
            suggested.append("CS:api")
            
        return suggested
    
    async def _optimize_for_tokens(
        self,
        section: StandardSection,
        max_tokens: int,
        strategy: TokenOptimizationStrategy
    ) -> Optional[StandardSection]:
        """
        Optimize section for token limit
        @nist-controls: SA-8
        @evidence: Resource optimization for LLM contexts
        """
        if section.tokens <= max_tokens:
            return section
            
        if strategy == TokenOptimizationStrategy.TRUNCATE:
            # Simple truncation
            ratio = max_tokens / section.tokens
            truncated_content = section.content[:int(len(section.content) * ratio)]
            
            return StandardSection(
                id=section.id,
                type=section.type,
                section=section.section,
                content=truncated_content + "\n... [truncated]",
                tokens=max_tokens,
                version=section.version,
                last_updated=section.last_updated,
                dependencies=section.dependencies,
                nist_controls=section.nist_controls,
                metadata={**section.metadata, "optimized": True}
            )
            
        # Other strategies would be implemented here
        return None
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count (simplified)"""
        # Rough estimate: ~1 token per 4 characters
        return len(text) // 4