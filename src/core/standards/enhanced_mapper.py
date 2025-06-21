"""
Enhanced Natural Language Mapper with Semantic Search
@nist-controls: SI-10, AC-4
@evidence: ML-enhanced query understanding
@oscal-component: standards-engine
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import NaturalLanguageMapping
from .semantic_search import (
    QueryExpander,
    SearchResult,
    SemanticSearchEngine,
    create_semantic_search_engine,
)

logger = logging.getLogger(__name__)


@dataclass
class MappingResult:
    """Result from enhanced mapping"""
    standard_refs: list[str]
    confidence: float
    method: str  # 'static', 'semantic', 'hybrid'
    keywords: list[str]
    semantic_matches: list[SearchResult] | None = None


class EnhancedNaturalLanguageMapper:
    """
    Enhanced mapper combining static rules and semantic search
    @nist-controls: SI-10, AC-4
    @evidence: Hybrid query mapping with ML
    """

    def __init__(
        self,
        static_mappings: list[NaturalLanguageMapping] | None = None,
        semantic_engine: SemanticSearchEngine | None = None,
        index_path: Path | None = None
    ):
        # Initialize static mappings (backward compatible)
        self.static_mappings = static_mappings or self._initialize_default_mappings()

        # Initialize semantic search
        self.semantic_engine = semantic_engine
        if not self.semantic_engine and index_path:
            try:
                self.semantic_engine = create_semantic_search_engine(index_path=index_path)
            except Exception as e:
                logger.warning(f"Failed to initialize semantic search: {e}")

        # Query expander for better matching
        self.query_expander = QueryExpander()

        # Cache for performance
        self._cache: dict[str, MappingResult] = {}

    def _initialize_default_mappings(self) -> list[NaturalLanguageMapping]:
        """Initialize default static mappings"""
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
                query_pattern="nist compliance",
                standard_refs=["SEC:*", "LEG:compliance", "OBS:logging", "TS:security"],
                confidence=0.95,
                keywords=["nist", "compliance", "controls", "800-53"]
            ),
            NaturalLanguageMapping(
                query_pattern="authentication",
                standard_refs=["SEC:authentication", "CS:auth", "TS:auth"],
                confidence=0.91,
                keywords=["auth", "authentication", "login", "sso", "oauth"]
            ),
            NaturalLanguageMapping(
                query_pattern="kubernetes",
                standard_refs=["CN:kubernetes", "CN:containers", "OBS:k8s"],
                confidence=0.93,
                keywords=["k8s", "kubernetes", "container", "orchestration"]
            ),
            NaturalLanguageMapping(
                query_pattern="machine learning",
                standard_refs=["AI:*", "DE:ml-pipeline", "CS:python", "TS:ml"],
                confidence=0.89,
                keywords=["ml", "ai", "machine learning", "model", "training"]
            ),
            NaturalLanguageMapping(
                query_pattern="data privacy",
                standard_refs=["LEG:privacy", "SEC:data-protection", "DE:anonymization"],
                confidence=0.92,
                keywords=["privacy", "gdpr", "pii", "data protection", "anonymization"]
            )
        ]

    def map_query(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        use_semantic: bool = True,
        confidence_threshold: float = 0.6
    ) -> MappingResult:
        """
        Map natural language query to standard references

        Args:
            query: Natural language query
            context: Optional context (project type, language, etc.)
            use_semantic: Whether to use semantic search
            confidence_threshold: Minimum confidence for results

        Returns:
            MappingResult with standard references and metadata
        """
        # Check cache
        cache_key = f"{query}:{str(context)}:{use_semantic}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Try static mapping first
        static_result = self._map_static(query, confidence_threshold)

        # If semantic search is available and enabled
        if use_semantic and self.semantic_engine:
            semantic_result = self._map_semantic(query, context, confidence_threshold)

            # Combine results
            result = self._combine_results(static_result, semantic_result, query)
        else:
            result = static_result if static_result else MappingResult(
                standard_refs=[],
                confidence=0.0,
                method='none',
                keywords=[],
                semantic_matches=None
            )

        # Cache result
        self._cache[cache_key] = result

        return result

    def _map_static(self, query: str, confidence_threshold: float) -> MappingResult | None:
        """Map using static rules"""
        query_lower = query.lower()
        best_match = None
        best_score = 0.0

        for mapping in self.static_mappings:
            # Check pattern match
            if mapping.query_pattern.lower() in query_lower:
                score = mapping.confidence
            else:
                # Check keyword overlap
                keyword_matches = sum(
                    1 for keyword in mapping.keywords
                    if keyword.lower() in query_lower
                )
                if keyword_matches == 0:
                    continue

                score = (keyword_matches / len(mapping.keywords)) * mapping.confidence * 0.8

            if score > best_score and score >= confidence_threshold:
                best_score = score
                best_match = mapping

        if best_match:
            return MappingResult(
                standard_refs=best_match.standard_refs,
                confidence=best_score,
                method='static',
                keywords=best_match.keywords
            )

        return None

    def _map_semantic(
        self,
        query: str,
        context: dict[str, Any] | None,
        confidence_threshold: float
    ) -> MappingResult | None:
        """Map using semantic search"""
        try:
            # Expand query
            expanded_query = self.query_expander.expand_query(query)

            # Search
            results = self.semantic_engine.search(
                expanded_query,
                k=10,
                min_score=confidence_threshold
            )

            if not results:
                return None

            # Extract standard references from results
            standard_refs = []
            keywords = set()

            for result in results[:5]:  # Top 5 results
                # Extract standard ID from metadata
                if 'standard_id' in result.metadata:
                    standard_refs.append(result.metadata['standard_id'])

                # Extract keywords from content
                content_lower = result.content.lower()
                for word in query.lower().split():
                    if len(word) > 3 and word in content_lower:
                        keywords.add(word)

            # Calculate confidence based on top result score
            confidence = results[0].score if results else 0.0

            # Rerank if context provided
            if context and self.semantic_engine:
                results = self.semantic_engine.rerank_results(query, results, context)

            return MappingResult(
                standard_refs=list(set(standard_refs)),
                confidence=confidence,
                method='semantic',
                keywords=list(keywords),
                semantic_matches=results[:5]
            )

        except Exception as e:
            logger.error(f"Semantic mapping failed: {e}")
            return None

    def _combine_results(
        self,
        static_result: MappingResult | None,
        semantic_result: MappingResult | None,
        query: str
    ) -> MappingResult:
        """Combine static and semantic results"""
        # If only one result exists, return it
        if not static_result:
            return semantic_result or MappingResult([], 0.0, 'none', [])
        if not semantic_result:
            return static_result

        # Combine both results
        combined_refs = list(set(static_result.standard_refs + semantic_result.standard_refs))
        combined_keywords = list(set(static_result.keywords + semantic_result.keywords))

        # Weight confidence scores
        # Static gets higher weight for exact matches
        if static_result.confidence > 0.9:
            combined_confidence = static_result.confidence * 0.7 + semantic_result.confidence * 0.3
        else:
            combined_confidence = static_result.confidence * 0.4 + semantic_result.confidence * 0.6

        return MappingResult(
            standard_refs=combined_refs,
            confidence=min(combined_confidence, 1.0),
            method='hybrid',
            keywords=combined_keywords,
            semantic_matches=semantic_result.semantic_matches
        )

    def add_mapping(self, mapping: NaturalLanguageMapping):
        """Add a new static mapping"""
        self.static_mappings.append(mapping)
        # Clear cache as mappings changed
        self._cache.clear()

    def update_semantic_index(self, standards: list[dict[str, Any]]):
        """Update semantic search index with new standards"""
        if not self.semantic_engine:
            logger.warning("No semantic engine available for indexing")
            return

        try:
            self.semantic_engine.index_standards(standards)
            # Clear cache as index changed
            self._cache.clear()
            logger.info(f"Updated semantic index with {len(standards)} standards")
        except Exception as e:
            logger.error(f"Failed to update semantic index: {e}")

    def get_suggestions(self, partial_query: str, limit: int = 5) -> list[str]:
        """Get query suggestions based on partial input"""
        suggestions = set()
        partial_lower = partial_query.lower()

        # Check static mappings
        for mapping in self.static_mappings:
            if partial_lower in mapping.query_pattern.lower():
                suggestions.add(mapping.query_pattern)

            # Check keywords
            for keyword in mapping.keywords:
                if keyword.lower().startswith(partial_lower):
                    suggestions.add(keyword)

        # Add from query expander
        for term, expansions in self.query_expander.expansions.items():
            if term.startswith(partial_lower):
                suggestions.add(term)
            for expansion in expansions:
                if expansion.startswith(partial_lower):
                    suggestions.add(expansion)

        return sorted(suggestions)[:limit]

    def explain_mapping(self, query: str) -> dict[str, Any]:
        """Explain how a query was mapped"""
        result = self.map_query(query)

        explanation = {
            "query": query,
            "method": result.method,
            "confidence": result.confidence,
            "standard_refs": result.standard_refs,
            "keywords_matched": result.keywords
        }

        if result.method in ['semantic', 'hybrid'] and result.semantic_matches:
            explanation["semantic_matches"] = [
                {
                    "content": match.content[:200] + "..." if len(match.content) > 200 else match.content,
                    "score": match.score,
                    "type": match.metadata.get('type', 'unknown')
                }
                for match in result.semantic_matches[:3]
            ]

        return explanation

    def get_mapping_stats(self) -> dict[str, Any]:
        """Get statistics about mappings"""
        stats = {
            "static_mappings": len(self.static_mappings),
            "cache_size": len(self._cache),
            "has_semantic": self.semantic_engine is not None
        }

        if self.semantic_engine:
            stats["semantic_index"] = self.semantic_engine.get_index_stats()

        return stats


# Factory function for backward compatibility
def create_enhanced_mapper(
    use_semantic: bool = True,
    index_path: Path | None = None
) -> EnhancedNaturalLanguageMapper:
    """Create an enhanced mapper with optional semantic search"""
    mapper = EnhancedNaturalLanguageMapper(index_path=index_path)

    # Try to load existing index if available
    if use_semantic and index_path and index_path.exists():
        try:
            mapper.semantic_engine = create_semantic_search_engine(index_path=index_path)
            logger.info("Loaded existing semantic index")
        except Exception as e:
            logger.warning(f"Could not load semantic index: {e}")

    return mapper
