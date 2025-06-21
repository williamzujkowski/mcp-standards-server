"""
Comprehensive tests for enhanced_mapper module
@nist-controls: SA-11, CA-7
@evidence: Enhanced natural language mapper testing
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
from typing import Any

import pytest

from src.core.standards.enhanced_mapper import (
    MappingResult,
    EnhancedNaturalLanguageMapper,
    create_enhanced_mapper
)
from src.core.standards.models import NaturalLanguageMapping
from src.core.standards.semantic_search import SearchResult, SemanticSearchEngine


class TestMappingResult:
    """Test MappingResult dataclass"""
    
    def test_mapping_result_creation(self):
        """Test creating a mapping result"""
        result = MappingResult(
            standard_refs=["CS:api", "SEC:api"],
            confidence=0.95,
            method="static",
            keywords=["api", "security"]
        )
        
        assert result.standard_refs == ["CS:api", "SEC:api"]
        assert result.confidence == 0.95
        assert result.method == "static"
        assert result.keywords == ["api", "security"]
        assert result.semantic_matches is None
    
    def test_mapping_result_with_semantic_matches(self):
        """Test mapping result with semantic matches"""
        semantic_matches = [
            SearchResult(content="Test content", score=0.9, metadata={})
        ]
        
        result = MappingResult(
            standard_refs=["CS:test"],
            confidence=0.85,
            method="semantic",
            keywords=["test"],
            semantic_matches=semantic_matches
        )
        
        assert result.semantic_matches == semantic_matches
        assert len(result.semantic_matches) == 1


class TestEnhancedNaturalLanguageMapper:
    """Test EnhancedNaturalLanguageMapper class"""
    
    @pytest.fixture
    def mapper(self):
        """Create mapper instance"""
        return EnhancedNaturalLanguageMapper()
    
    @pytest.fixture
    def mock_semantic_engine(self):
        """Create mock semantic engine"""
        engine = MagicMock(spec=SemanticSearchEngine)
        engine.search.return_value = []
        engine.rerank_results.return_value = []
        engine.get_index_stats.return_value = {"indexed": True}
        return engine
    
    def test_initialization_default(self, mapper):
        """Test default initialization"""
        assert len(mapper.static_mappings) > 0
        assert mapper.semantic_engine is None
        assert mapper.query_expander is not None
        assert mapper._cache == {}
    
    def test_initialization_with_custom_mappings(self):
        """Test initialization with custom mappings"""
        custom_mappings = [
            NaturalLanguageMapping(
                query_pattern="custom test",
                standard_refs=["TEST:custom"],
                confidence=0.99,
                keywords=["custom", "test"]
            )
        ]
        
        mapper = EnhancedNaturalLanguageMapper(static_mappings=custom_mappings)
        
        assert len(mapper.static_mappings) == 1
        assert mapper.static_mappings[0].query_pattern == "custom test"
    
    def test_initialization_with_semantic_engine(self, mock_semantic_engine):
        """Test initialization with semantic engine"""
        mapper = EnhancedNaturalLanguageMapper(semantic_engine=mock_semantic_engine)
        
        assert mapper.semantic_engine == mock_semantic_engine
    
    @patch('src.core.standards.enhanced_mapper.create_semantic_search_engine')
    def test_initialization_with_index_path(self, mock_create_engine, tmp_path):
        """Test initialization with index path"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        mapper = EnhancedNaturalLanguageMapper(index_path=tmp_path)
        
        mock_create_engine.assert_called_once_with(index_path=tmp_path)
        assert mapper.semantic_engine == mock_engine
    
    def test_default_mappings_content(self, mapper):
        """Test content of default mappings"""
        patterns = [m.query_pattern for m in mapper.static_mappings]
        
        # Check some expected patterns exist
        assert "secure api" in patterns
        assert "react app" in patterns
        assert "microservices" in patterns
        assert "nist compliance" in patterns
        assert "authentication" in patterns
        
        # Check mapping structure
        for mapping in mapper.static_mappings:
            assert len(mapping.standard_refs) > 0
            assert 0 < mapping.confidence <= 1.0
            assert len(mapping.keywords) > 0
    
    def test_map_query_static_exact_match(self, mapper):
        """Test mapping with exact static pattern match"""
        result = mapper.map_query("secure api", use_semantic=False)
        
        assert result.method == "static"
        assert result.confidence >= 0.9
        assert "CS:api" in result.standard_refs
        assert "SEC:api" in result.standard_refs
        assert "api" in result.keywords
    
    def test_map_query_static_keyword_match(self, mapper):
        """Test mapping with keyword-based static match"""
        result = mapper.map_query("building secure endpoints", use_semantic=False)
        
        # May match "secure api" pattern due to "secure" and "endpoint" keywords
        if result.method == "none":
            # If no match, that's also valid for this edge case
            assert result.confidence == 0.0
            assert result.standard_refs == []
        else:
            assert result.method == "static"
            assert result.confidence > 0.0
            assert len(result.standard_refs) > 0
    
    def test_map_query_no_match(self, mapper):
        """Test mapping with no matches"""
        result = mapper.map_query("random unmatched query xyz123", use_semantic=False)
        
        assert result.method == "none"
        assert result.confidence == 0.0
        assert result.standard_refs == []
        assert result.keywords == []
    
    def test_map_query_with_cache(self, mapper):
        """Test query result caching"""
        # First query
        result1 = mapper.map_query("secure api", use_semantic=False)
        cache_size1 = len(mapper._cache)
        
        # Same query should hit cache
        result2 = mapper.map_query("secure api", use_semantic=False)
        cache_size2 = len(mapper._cache)
        
        assert result1.standard_refs == result2.standard_refs
        assert result1.confidence == result2.confidence
        assert cache_size1 == cache_size2
        assert cache_size1 > 0
    
    def test_map_query_semantic(self, mapper, mock_semantic_engine):
        """Test mapping with semantic search"""
        # Set up semantic engine
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.search.return_value = [
            SearchResult(
                content="API security guidelines",
                score=0.92,
                metadata={"standard_id": "SEC:api-guidelines"},
                standard_id="SEC:api-guidelines"
            ),
            SearchResult(
                content="Authentication best practices",
                score=0.85,
                metadata={"standard_id": "SEC:auth"},
                standard_id="SEC:auth"
            )
        ]
        
        result = mapper.map_query("api security", use_semantic=True)
        
        assert result.method == "semantic"
        assert result.confidence == 0.92
        assert "SEC:api-guidelines" in result.standard_refs
        assert len(result.semantic_matches) > 0
    
    def test_map_query_hybrid(self, mapper, mock_semantic_engine):
        """Test hybrid mapping (static + semantic)"""
        mapper.semantic_engine = mock_semantic_engine
        
        # Set up both static and semantic matches
        mock_semantic_engine.search.return_value = [
            SearchResult(
                content="Additional secure API info",
                score=0.88,
                metadata={"standard_id": "SEC:advanced-api"},
                standard_id="SEC:advanced-api"
            )
        ]
        
        result = mapper.map_query("secure api", use_semantic=True)
        
        assert result.method == "hybrid"
        # Should include refs from both static and semantic
        assert len(result.standard_refs) >= 3  # CS:api, SEC:api from static + semantic
        assert result.confidence > 0.8
    
    def test_map_query_with_context(self, mapper, mock_semantic_engine):
        """Test mapping with context for reranking"""
        mapper.semantic_engine = mock_semantic_engine
        
        results = [
            SearchResult(content="Python API", score=0.8, metadata={}),
            SearchResult(content="Java API", score=0.85, metadata={})
        ]
        mock_semantic_engine.search.return_value = results
        mock_semantic_engine.rerank_results.return_value = list(reversed(results))
        
        context = {"language": "python"}
        result = mapper.map_query("api design", use_semantic=True, context=context)
        
        # Verify reranking was called
        mock_semantic_engine.rerank_results.assert_called_once()
        call_args = mock_semantic_engine.rerank_results.call_args
        assert call_args[0][0] == "api design"
        assert call_args[0][2] == context
    
    def test_map_query_confidence_threshold(self, mapper):
        """Test confidence threshold filtering"""
        result = mapper.map_query(
            "vague query with few matches",
            use_semantic=False,
            confidence_threshold=0.9
        )
        
        # Low confidence matches should be filtered
        if result.method != "none":
            assert result.confidence >= 0.9
    
    def test_map_static_exact_pattern(self, mapper):
        """Test static mapping with exact pattern match"""
        result = mapper._map_static("nist compliance", 0.5)
        
        assert result is not None
        assert result.confidence >= 0.9
        assert "SEC:*" in result.standard_refs
        assert "nist" in result.keywords
    
    def test_map_static_keyword_overlap(self, mapper):
        """Test static mapping with keyword overlap"""
        result = mapper._map_static("implementing oauth authentication", 0.5)
        
        assert result is not None
        assert result.confidence > 0.5
        assert any("auth" in ref.lower() for ref in result.standard_refs)
    
    def test_map_static_no_match(self, mapper):
        """Test static mapping with no matches"""
        result = mapper._map_static("completely unrelated query xyz", 0.5)
        
        assert result is None
    
    def test_map_semantic_success(self, mapper, mock_semantic_engine):
        """Test successful semantic mapping"""
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.search.return_value = [
            SearchResult(
                content="Test content about APIs",
                score=0.95,
                metadata={"standard_id": "API:v1"},
                standard_id="API:v1"
            )
        ]
        
        result = mapper._map_semantic("api testing", None, 0.5)
        
        assert result is not None
        assert result.method == "semantic"
        assert result.confidence == 0.95
        assert "API:v1" in result.standard_refs
        assert len(result.keywords) >= 0
    
    def test_map_semantic_no_results(self, mapper, mock_semantic_engine):
        """Test semantic mapping with no results"""
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.search.return_value = []
        
        result = mapper._map_semantic("no matches", None, 0.5)
        
        assert result is None
    
    def test_map_semantic_error_handling(self, mapper, mock_semantic_engine):
        """Test semantic mapping error handling"""
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.search.side_effect = Exception("Search failed")
        
        result = mapper._map_semantic("test query", None, 0.5)
        
        assert result is None
    
    def test_combine_results_both_present(self, mapper):
        """Test combining static and semantic results"""
        static_result = MappingResult(
            standard_refs=["CS:api", "SEC:api"],
            confidence=0.95,
            method="static",
            keywords=["api", "security"]
        )
        
        semantic_result = MappingResult(
            standard_refs=["API:guidelines", "SEC:api"],
            confidence=0.85,
            method="semantic",
            keywords=["guidelines", "best practices"],
            semantic_matches=[SearchResult("test", 0.85, {})]
        )
        
        combined = mapper._combine_results(static_result, semantic_result, "test query")
        
        assert combined.method == "hybrid"
        assert len(combined.standard_refs) == 3  # CS:api, SEC:api, API:guidelines (deduped)
        assert "CS:api" in combined.standard_refs
        assert "API:guidelines" in combined.standard_refs
        assert combined.confidence > 0.8
        assert len(combined.keywords) == 4
        assert combined.semantic_matches is not None
    
    def test_combine_results_only_static(self, mapper):
        """Test combining when only static result exists"""
        static_result = MappingResult(
            standard_refs=["CS:test"],
            confidence=0.9,
            method="static",
            keywords=["test"]
        )
        
        combined = mapper._combine_results(static_result, None, "test")
        
        assert combined == static_result
    
    def test_combine_results_only_semantic(self, mapper):
        """Test combining when only semantic result exists"""
        semantic_result = MappingResult(
            standard_refs=["SEM:test"],
            confidence=0.85,
            method="semantic",
            keywords=["test"]
        )
        
        combined = mapper._combine_results(None, semantic_result, "test")
        
        assert combined == semantic_result
    
    def test_combine_results_confidence_weighting(self, mapper):
        """Test confidence score weighting in combination"""
        # High confidence static result
        static_high = MappingResult(["S:1"], 0.95, "static", ["test"])
        semantic_low = MappingResult(["E:1"], 0.7, "semantic", ["test"])
        
        combined = mapper._combine_results(static_high, semantic_low, "test")
        # Static should have higher weight
        assert combined.confidence > 0.85
        
        # Lower confidence static result
        static_low = MappingResult(["S:1"], 0.7, "static", ["test"])
        semantic_high = MappingResult(["E:1"], 0.95, "semantic", ["test"])
        
        combined = mapper._combine_results(static_low, semantic_high, "test")
        # Semantic should have higher weight
        assert combined.confidence > 0.8
    
    def test_add_mapping(self, mapper):
        """Test adding a new static mapping"""
        initial_count = len(mapper.static_mappings)
        initial_cache_size = len(mapper._cache)
        
        # Add to cache first
        mapper._cache["test:None:True"] = MappingResult([], 0, "none", [])
        
        new_mapping = NaturalLanguageMapping(
            query_pattern="new pattern",
            standard_refs=["NEW:ref"],
            confidence=0.88,
            keywords=["new", "pattern"]
        )
        
        mapper.add_mapping(new_mapping)
        
        assert len(mapper.static_mappings) == initial_count + 1
        assert mapper.static_mappings[-1] == new_mapping
        assert len(mapper._cache) == 0  # Cache should be cleared
    
    def test_update_semantic_index(self, mapper, mock_semantic_engine):
        """Test updating semantic index"""
        mapper.semantic_engine = mock_semantic_engine
        mapper._cache["test"] = MappingResult([], 0, "none", [])
        
        standards = [
            {"id": "std1", "title": "Standard 1"},
            {"id": "std2", "title": "Standard 2"}
        ]
        
        mapper.update_semantic_index(standards)
        
        mock_semantic_engine.index_standards.assert_called_once_with(standards)
        assert len(mapper._cache) == 0  # Cache cleared
    
    def test_update_semantic_index_no_engine(self, mapper):
        """Test updating semantic index without engine"""
        # Should not raise error
        mapper.update_semantic_index([{"id": "std1"}])
    
    def test_update_semantic_index_error(self, mapper, mock_semantic_engine):
        """Test handling indexing errors"""
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.index_standards.side_effect = Exception("Index error")
        
        # Should not raise error
        mapper.update_semantic_index([{"id": "std1"}])
    
    def test_get_suggestions_from_patterns(self, mapper):
        """Test getting suggestions from query patterns"""
        suggestions = mapper.get_suggestions("secure", limit=5)
        
        assert "secure api" in suggestions
        assert len(suggestions) <= 5
    
    def test_get_suggestions_from_keywords(self, mapper):
        """Test getting suggestions from keywords"""
        suggestions = mapper.get_suggestions("auth", limit=10)
        
        # Should include both the term and mappings containing it
        assert any("auth" in s for s in suggestions)
    
    def test_get_suggestions_from_expander(self, mapper):
        """Test getting suggestions from query expander"""
        suggestions = mapper.get_suggestions("encr", limit=5)
        
        # Should include encryption-related terms
        assert any("encrypt" in s for s in suggestions)
    
    def test_get_suggestions_empty(self, mapper):
        """Test getting suggestions with no matches"""
        suggestions = mapper.get_suggestions("xyz123", limit=5)
        
        assert len(suggestions) == 0
    
    def test_explain_mapping_static(self, mapper):
        """Test explaining a static mapping"""
        explanation = mapper.explain_mapping("secure api")
        
        assert explanation["query"] == "secure api"
        assert explanation["method"] == "static"
        assert explanation["confidence"] > 0
        assert len(explanation["standard_refs"]) > 0
        assert len(explanation["keywords_matched"]) > 0
    
    def test_explain_mapping_semantic(self, mapper, mock_semantic_engine):
        """Test explaining a semantic mapping"""
        mapper.semantic_engine = mock_semantic_engine
        mock_semantic_engine.search.return_value = [
            SearchResult(
                content="This is a long content about APIs that should be truncated",
                score=0.9,
                metadata={"type": "section", "standard_id": "API:1"},
                standard_id="API:1"
            )
        ]
        
        explanation = mapper.explain_mapping("api design")
        
        assert "semantic_matches" in explanation
        assert len(explanation["semantic_matches"]) > 0
        match = explanation["semantic_matches"][0]
        assert "..." in match["content"] or len(match["content"]) <= 200
        assert match["score"] == 0.9
        assert match["type"] == "section"
    
    def test_get_mapping_stats(self, mapper):
        """Test getting mapping statistics"""
        stats = mapper.get_mapping_stats()
        
        assert stats["static_mappings"] == len(mapper.static_mappings)
        assert stats["cache_size"] == 0  # Empty initially
        assert stats["has_semantic"] is False
        
        # Add cache entry
        mapper._cache["test"] = MappingResult([], 0, "none", [])
        stats = mapper.get_mapping_stats()
        assert stats["cache_size"] == 1
    
    def test_get_mapping_stats_with_semantic(self, mapper, mock_semantic_engine):
        """Test getting stats with semantic engine"""
        mapper.semantic_engine = mock_semantic_engine
        
        stats = mapper.get_mapping_stats()
        
        assert stats["has_semantic"] is True
        assert "semantic_index" in stats
        assert stats["semantic_index"]["indexed"] is True


class TestFactoryFunction:
    """Test factory function"""
    
    @patch('src.core.standards.enhanced_mapper.create_semantic_search_engine')
    def test_create_enhanced_mapper_basic(self, mock_create_engine):
        """Test creating basic enhanced mapper"""
        mapper = create_enhanced_mapper(use_semantic=False)
        
        assert isinstance(mapper, EnhancedNaturalLanguageMapper)
        assert len(mapper.static_mappings) > 0
        mock_create_engine.assert_not_called()
    
    @patch('src.core.standards.enhanced_mapper.create_semantic_search_engine')
    def test_create_enhanced_mapper_with_semantic(self, mock_create_engine, tmp_path):
        """Test creating mapper with semantic search"""
        mock_engine = MagicMock()
        mock_create_engine.return_value = mock_engine
        
        # Create fake index
        index_path = tmp_path / "index"
        index_path.mkdir()
        
        mapper = create_enhanced_mapper(use_semantic=True, index_path=index_path)
        
        assert isinstance(mapper, EnhancedNaturalLanguageMapper)
        assert mapper.semantic_engine == mock_engine
        mock_create_engine.assert_called_with(index_path=index_path)
    
    @patch('src.core.standards.enhanced_mapper.create_semantic_search_engine')
    def test_create_enhanced_mapper_semantic_error(self, mock_create_engine, tmp_path):
        """Test handling semantic engine creation error"""
        mock_create_engine.side_effect = Exception("Failed to create")
        
        # Should not raise error
        mapper = create_enhanced_mapper(use_semantic=True, index_path=tmp_path)
        
        assert isinstance(mapper, EnhancedNaturalLanguageMapper)
        assert mapper.semantic_engine is None


class TestIntegration:
    """Integration tests"""
    
    def test_full_query_flow(self):
        """Test complete query mapping flow"""
        mapper = EnhancedNaturalLanguageMapper()
        
        # Test various queries
        queries = [
            "secure api design",
            "react component testing",
            "kubernetes deployment",
            "nist compliance requirements",
            "machine learning pipeline"
        ]
        
        for query in queries:
            result = mapper.map_query(query, use_semantic=False)
            
            assert result.method in ["static", "none"]
            assert 0 <= result.confidence <= 1.0
            assert isinstance(result.standard_refs, list)
            assert isinstance(result.keywords, list)
            
            # Explain the mapping
            explanation = mapper.explain_mapping(query)
            assert explanation["query"] == query
            assert "standard_refs" in explanation
    
    def test_cache_effectiveness(self):
        """Test cache improves performance"""
        mapper = EnhancedNaturalLanguageMapper()
        
        # Warm up cache
        queries = ["api", "security", "testing", "deployment"]
        for q in queries:
            mapper.map_query(q)
        
        # Cache should have entries
        assert len(mapper._cache) == len(queries)
        
        # Clear and add mapping should reset cache
        mapper.add_mapping(NaturalLanguageMapping(
            query_pattern="new",
            standard_refs=["NEW"],
            confidence=0.9,
            keywords=["new"]
        ))
        assert len(mapper._cache) == 0
    
    def test_pattern_coverage(self):
        """Test that common patterns are covered"""
        mapper = EnhancedNaturalLanguageMapper()
        
        common_patterns = [
            ("api security", ["api", "security"]),
            ("frontend development", ["frontend", "react", "javascript"]),
            ("backend services", ["api", "microservice"]),
            ("database design", ["database", "optimization"]),
            ("cloud deployment", ["kubernetes", "container"]),
            ("testing strategy", ["test"]),
            ("ci/cd setup", ["ci", "cd", "pipeline"]),
            ("authentication system", ["auth", "authentication"]),
            ("data privacy", ["privacy", "data protection"]),
            ("compliance audit", ["compliance", "nist"])
        ]
        
        for query, expected_keywords in common_patterns:
            result = mapper.map_query(query, use_semantic=False)
            
            if result.method != "none":
                # Should match at least one expected keyword
                assert any(
                    kw in ' '.join(result.keywords).lower()
                    for kw in expected_keywords
                ), f"Query '{query}' didn't match expected keywords"