"""
Comprehensive Tests for Standards Engine
@nist-controls: SA-11, CA-7
@evidence: Unit tests for standards engine functionality
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import redis
import yaml
from redis.exceptions import RedisError

from src.core.standards.engine import NaturalLanguageMapper, StandardsEngine
from src.core.standards.models import (
    NaturalLanguageMapping,
    StandardLoadResult,
    StandardQuery,
    StandardSection,
    StandardType,
    TokenOptimizationStrategy,
)


@pytest.fixture
def test_standards_path(tmp_path):
    """Create test standards directory with schema"""
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()

    # Create schema file
    schema = {
        "version": "1.0",
        "types": ["CS", "SEC", "TS", "FE", "CN"]
    }
    schema_file = standards_dir / "standards-schema.yaml"
    schema_file.write_text(yaml.dump(schema))

    # Create test standard files
    cs_dir = standards_dir / "CS"
    cs_dir.mkdir()

    # Create api.yaml
    api_content = {
        "id": "CS:api",
        "title": "API Standards",
        "content": "API design best practices.",
        "tags": ["api", "rest"]
    }
    api_file = cs_dir / "api.yaml"
    api_file.write_text(yaml.dump(api_content))

    # Create security.yaml
    security_content = {
        "id": "CS:security",
        "title": "Security Standards",
        "content": "Security best practices.",
        "tags": ["security", "auth"]
    }
    security_file = cs_dir / "security.yaml"
    security_file.write_text(yaml.dump(security_content))

    # Create standards index
    index_content = {
        "standards": {
            "CS:api": {
                "file": "CS/api.yaml",
                "type": "CS",
                "section": "api"
            },
            "CS:security": {
                "file": "CS/security.yaml",
                "type": "CS",
                "section": "security"
            }
        }
    }
    index_file = standards_dir / "standards_index.json"
    index_file.write_text(json.dumps(index_content))

    return standards_dir


class TestNaturalLanguageMapper:
    """Test NaturalLanguageMapper functionality"""

    def test_initialize_mappings(self):
        """Test mappings initialization"""
        mapper = NaturalLanguageMapper()

        assert len(mapper.mappings) > 0
        assert isinstance(mapper.mappings[0], NaturalLanguageMapping)
        assert mapper.mappings[0].query_pattern == "secure api"

    def test_map_query_exact_match(self):
        """Test mapping with exact pattern match"""
        mapper = NaturalLanguageMapper()

        refs, confidence = mapper.map_query("secure api implementation")

        assert "CS:api" in refs
        assert "SEC:api" in refs
        assert confidence > 0.9

    def test_map_query_keyword_match(self):
        """Test mapping with keyword matches"""
        mapper = NaturalLanguageMapper()

        refs, confidence = mapper.map_query("building microservice architecture with distributed systems")

        assert "CN:microservices" in refs
        assert confidence > 0.8

    def test_map_query_multiple_matches(self):
        """Test query matching multiple mappings"""
        mapper = NaturalLanguageMapper()

        refs, confidence = mapper.map_query("secure kubernetes authentication for microservices")

        # Should match multiple mappings
        assert "CN:kubernetes" in refs
        assert "SEC:authentication" in refs
        assert "CN:microservices" in refs
        assert confidence > 0

    def test_map_query_no_match(self):
        """Test query with no matches"""
        mapper = NaturalLanguageMapper()

        refs, confidence = mapper.map_query("random unrelated query xyz123")

        assert refs == []
        assert confidence == 0.0

    def test_map_query_deduplication(self):
        """Test that duplicate refs are removed"""
        mapper = NaturalLanguageMapper()

        # Add a duplicate mapping for testing
        mapper.mappings.append(
            NaturalLanguageMapping(
                query_pattern="test duplicate",
                standard_refs=["CS:api", "CS:api", "SEC:api"],
                confidence=0.9,
                keywords=["test", "duplicate"]
            )
        )

        refs, confidence = mapper.map_query("test duplicate pattern")

        # Should have unique refs only
        assert refs.count("CS:api") == 1


class TestStandardsEngine:
    """Test StandardsEngine functionality"""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client"""
        mock = MagicMock(spec=redis.Redis)
        mock.get.return_value = None
        mock.setex.return_value = True
        return mock


    def test_engine_initialization(self, test_standards_path, mock_redis):
        """Test engine initialization with all components"""
        engine = StandardsEngine(
            test_standards_path,
            redis_client=mock_redis,
            cache_ttl=1800
        )

        assert engine.standards_path == test_standards_path
        assert engine.redis_client == mock_redis
        assert engine.cache_ttl == 1800
        assert engine.nl_mapper is not None
        assert engine.schema is not None
        assert engine.loaded_standards == {}

    def test_engine_without_redis(self, test_standards_path):
        """Test engine initialization without Redis"""
        engine = StandardsEngine(test_standards_path)

        assert engine.redis_client is None
        assert engine.cache_ttl == 3600  # Default

    def test_load_schema_missing(self, tmp_path):
        """Test engine with missing schema file"""
        engine = StandardsEngine(tmp_path)

        assert engine.schema is None

    def test_validate_standard_ref(self, test_standards_path):
        """Test standard reference validation"""
        engine = StandardsEngine(test_standards_path)

        # Valid refs
        assert engine._validate_standard_ref("CS:api")
        assert engine._validate_standard_ref("SEC:auth-module")
        assert engine._validate_standard_ref("TS:*")

        # Invalid refs
        assert not engine._validate_standard_ref("cs:api")  # Lowercase type
        assert not engine._validate_standard_ref("CS")  # Missing section
        assert not engine._validate_standard_ref("CS:api:extra")  # Too many parts
        assert not engine._validate_standard_ref("CS:api space")  # Space in section

    @pytest.mark.asyncio
    async def test_parse_query_natural_language(self, test_standards_path):
        """Test parsing natural language queries"""
        engine = StandardsEngine(test_standards_path)

        refs, query_info = await engine.parse_query("secure api design")

        assert "CS:api" in refs
        assert "SEC:api" in refs
        assert query_info["query_type"] == "natural_language"
        assert query_info["confidence"] > 0
        assert query_info["mapped_refs"] == refs

    @pytest.mark.asyncio
    async def test_parse_query_direct_notation(self, test_standards_path):
        """Test parsing direct notation queries"""
        engine = StandardsEngine(test_standards_path)

        refs, query_info = await engine.parse_query("CS:api + SEC:auth")

        assert refs == ["CS:api", "SEC:auth"]
        assert query_info["query_type"] == "direct_notation"
        assert query_info["refs"] == refs

    @pytest.mark.asyncio
    async def test_parse_query_with_at_prefix(self, test_standards_path):
        """Test parsing queries with @ prefix"""
        engine = StandardsEngine(test_standards_path)

        refs, query_info = await engine.parse_query("@CS:api")

        assert refs == ["CS:api"]
        assert query_info["query_type"] == "direct_notation"

    @pytest.mark.asyncio
    async def test_parse_query_load_command(self, test_standards_path):
        """Test parsing load command"""
        engine = StandardsEngine(test_standards_path)

        refs, query_info = await engine.parse_query("load CS:api")

        assert refs == ["CS:api"]
        assert query_info["query_type"] == "direct_notation"

    @pytest.mark.asyncio
    async def test_parse_query_empty(self, test_standards_path):
        """Test parsing empty query"""
        engine = StandardsEngine(test_standards_path)

        refs, query_info = await engine.parse_query("")

        assert refs == []
        assert query_info["query_type"] == "natural_language"
        assert query_info["confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_analyze_context(self, test_standards_path):
        """Test context analysis"""
        engine = StandardsEngine(test_standards_path)

        # Test context
        refs = await engine._analyze_context("Building API with security testing")

        assert "TS:*" in refs  # Contains "test"
        assert "SEC:*" in refs  # Contains "security"
        assert "CS:api" in refs  # Contains "api"

    @pytest.mark.asyncio
    async def test_load_section_mock(self, test_standards_path):
        """Test loading a specific section"""
        engine = StandardsEngine(test_standards_path)

        section = await engine._load_section("CS", "api", "latest")

        assert section is not None
        assert section.id == "CS:api"
        assert section.type == StandardType.CS
        assert section.section == "api"
        assert section.version == "latest"
        assert section.tokens > 0

    @pytest.mark.asyncio
    async def test_load_all_sections(self, test_standards_path):
        """Test loading all sections for a type"""
        engine = StandardsEngine(test_standards_path)

        sections = await engine._load_all_sections("CS", "latest")

        assert len(sections) > 0
        assert all(s.type == StandardType.CS for s in sections)
        assert all(s.version == "latest" for s in sections)

    @pytest.mark.asyncio
    async def test_load_standard_sections_specific(self, test_standards_path):
        """Test loading specific standard section"""
        engine = StandardsEngine(test_standards_path)

        sections = await engine._load_standard_sections("CS:api", "latest")

        assert len(sections) == 1
        assert sections[0].id == "CS:api"

    @pytest.mark.asyncio
    async def test_load_standard_sections_wildcard(self, test_standards_path):
        """Test loading all sections with wildcard"""
        engine = StandardsEngine(test_standards_path)

        sections = await engine._load_standard_sections("CS:*", "latest")

        assert len(sections) >= 1  # At least one section
        assert all(s.type == StandardType.CS for s in sections)

    @pytest.mark.asyncio
    async def test_load_standard_sections_invalid_ref(self, test_standards_path):
        """Test loading with invalid reference"""
        engine = StandardsEngine(test_standards_path)

        with pytest.raises(ValueError) as exc_info:
            await engine._load_standard_sections("INVALID", "latest")

        assert "Invalid standard reference" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_optimize_for_tokens_truncate(self, test_standards_path):
        """Test token optimization with truncation"""
        engine = StandardsEngine(test_standards_path)

        # Create a large section
        section = StandardSection(
            id="TEST:large",
            type=StandardType.CS,
            section="large",
            content="X" * 1000,  # Large content
            tokens=1000,
            version="latest",
            last_updated=datetime.now(),
            dependencies=[],
            nist_controls=set(),
            metadata={}
        )

        # Optimize to 500 tokens
        optimized = await engine._optimize_for_tokens(
            section,
            500,
            TokenOptimizationStrategy.TRUNCATE
        )

        assert optimized is not None
        assert optimized.tokens == 500
        assert "[truncated]" in optimized.content
        assert optimized.metadata["optimized"] is True

    @pytest.mark.asyncio
    async def test_optimize_for_tokens_fits(self, test_standards_path):
        """Test optimization when content already fits"""
        engine = StandardsEngine(test_standards_path)

        section = StandardSection(
            id="TEST:small",
            type=StandardType.CS,
            section="small",
            content="Small content",
            tokens=100,
            version="latest",
            last_updated=datetime.now(),
            dependencies=[],
            nist_controls=set(),
            metadata={}
        )

        # Try to optimize to 200 tokens (already fits)
        optimized = await engine._optimize_for_tokens(
            section,
            200,
            TokenOptimizationStrategy.TRUNCATE
        )

        assert optimized == section  # Should return original

    def test_count_tokens(self, test_standards_path):
        """Test token counting"""
        engine = StandardsEngine(test_standards_path)

        # Approximately 1 token per 4 characters
        assert engine._count_tokens("test") == 1
        assert engine._count_tokens("test" * 10) == 10
        assert engine._count_tokens("") == 0


class TestStandardsEngineWithCache:
    """Test StandardsEngine caching functionality"""

    @pytest.fixture
    def mock_redis_with_data(self):
        """Create mock Redis with cached data"""
        mock = MagicMock(spec=redis.Redis)

        # Mock cached data
        cached_sections = [
            {
                "id": "CS:api",
                "type": "CS",
                "section": "api",
                "content": "Cached API standards",
                "tokens": 500,
                "version": "latest",
                "last_updated": datetime.now().isoformat(),
                "dependencies": [],
                "nist_controls": ["AC-3", "AU-2"],
                "metadata": {"cached": True}
            }
        ]

        mock.get.return_value = json.dumps(cached_sections)
        mock.setex.return_value = True
        return mock

    @pytest.mark.asyncio
    async def test_get_from_cache_hit(self, test_standards_path, mock_redis_with_data):
        """Test cache hit"""
        engine = StandardsEngine(test_standards_path, redis_client=mock_redis_with_data)

        sections = await engine._get_from_cache("CS:api", "latest")

        assert sections is not None
        assert len(sections) == 1
        assert sections[0].id == "CS:api"
        assert sections[0].metadata["cached"] is True

        mock_redis_with_data.get.assert_called_once_with("standard:CS:api:latest")

    @pytest.mark.asyncio
    async def test_get_from_cache_miss(self, test_standards_path, mock_redis_client):
        """Test cache miss"""
        engine = StandardsEngine(test_standards_path, redis_client=mock_redis_client)

        sections = await engine._get_from_cache("CS:api", "latest")

        assert sections is None
        mock_redis_client.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_from_cache_no_redis(self, test_standards_path):
        """Test cache with no Redis client"""
        engine = StandardsEngine(test_standards_path)

        sections = await engine._get_from_cache("CS:api", "latest")

        assert sections is None

    @pytest.mark.asyncio
    async def test_get_from_cache_redis_error(self, test_standards_path):
        """Test cache with Redis error"""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.get.side_effect = RedisError("Connection failed")

        engine = StandardsEngine(test_standards_path, redis_client=mock_redis)

        with patch('src.core.standards.engine.logger') as mock_logger:
            sections = await engine._get_from_cache("CS:api", "latest")

            assert sections is None
            mock_logger.error.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_to_cache(self, test_standards_path, mock_redis_client):
        """Test saving to cache"""
        engine = StandardsEngine(test_standards_path, redis_client=mock_redis_client)

        sections = [
            StandardSection(
                id="CS:api",
                type=StandardType.CS,
                section="api",
                content="Test content",
                tokens=100,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
        ]

        await engine._save_to_cache("CS:api", "latest", sections)

        mock_redis_client.setex.assert_called_once()
        call_args = mock_redis_client.setex.call_args
        assert call_args[0][0] == "standard:CS:api:latest"
        assert call_args[0][1] == 3600  # Default TTL

    @pytest.mark.asyncio
    async def test_save_to_cache_no_redis(self, test_standards_path):
        """Test save to cache with no Redis"""
        engine = StandardsEngine(test_standards_path)

        sections = [MagicMock()]

        # Should not raise error
        await engine._save_to_cache("CS:api", "latest", sections)

    @pytest.mark.asyncio
    async def test_save_to_cache_redis_error(self, test_standards_path):
        """Test save to cache with Redis error"""
        mock_redis = MagicMock(spec=redis.Redis)
        mock_redis.setex.side_effect = RedisError("Write failed")

        engine = StandardsEngine(test_standards_path, redis_client=mock_redis)

        sections = [
            StandardSection(
                id="CS:api",
                type=StandardType.CS,
                section="api",
                content="Test",
                tokens=10,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
        ]

        with patch('src.core.standards.engine.logger') as mock_logger:
            await engine._save_to_cache("CS:api", "latest", sections)

            mock_logger.error.assert_called_once()


class TestStandardsEngineLoadStandards:
    """Test the main load_standards functionality"""

    @pytest.fixture
    def mock_engine(self, test_standards_path):
        """Create engine with mocked methods"""
        engine = StandardsEngine(test_standards_path)

        # Mock internal methods
        engine._get_from_cache = AsyncMock(return_value=None)
        engine._save_to_cache = AsyncMock()
        engine._load_standard_sections = AsyncMock(return_value=[
            StandardSection(
                id="CS:api",
                type=StandardType.CS,
                section="api",
                content="API standards content",
                tokens=1000,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls={"AC-3", "AU-2"},
                metadata={}
            )
        ])

        return engine

    @pytest.mark.asyncio
    async def test_load_standards_basic(self, mock_engine):
        """Test basic standards loading"""
        query = StandardQuery(query="CS:api", token_limit=5000)

        result = await mock_engine.load_standards(query)

        assert isinstance(result, StandardLoadResult)
        assert len(result.standards) == 1
        assert result.standards[0]["id"] == "CS:api"
        assert result.metadata["token_count"] == 1000
        assert result.metadata["refs_loaded"] == ["CS:api"]
        assert result.query_info["query_type"] == "direct_notation"

    @pytest.mark.asyncio
    async def test_load_standards_with_context(self, mock_engine):
        """Test loading with context analysis"""
        query = StandardQuery(
            query="authentication",
            context="Building secure API with testing",
            token_limit=10000
        )

        result = await mock_engine.load_standards(query)

        # Should have added context-based standards
        assert "context_refs" in result.query_info
        assert "TS:*" in result.query_info["context_refs"]
        assert "SEC:*" in result.query_info["context_refs"]

    @pytest.mark.asyncio
    async def test_load_standards_token_limit(self, mock_engine):
        """Test token limit enforcement"""
        # Create multiple large sections
        mock_engine._load_standard_sections = AsyncMock(return_value=[
            StandardSection(
                id=f"CS:section{i}",
                type=StandardType.CS,
                section=f"section{i}",
                content=f"Content {i}",
                tokens=2000,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
            for i in range(5)
        ])

        query = StandardQuery(query="CS:*", token_limit=5000)

        result = await mock_engine.load_standards(query)

        # Should only load sections that fit in token budget
        assert len(result.standards) <= 3  # 5000 / 2000 = 2.5
        assert result.metadata["token_count"] <= 5000

    @pytest.mark.asyncio
    async def test_load_standards_with_optimization(self, mock_engine):
        """Test loading with token optimization"""
        # Large section that needs optimization
        large_section = StandardSection(
            id="CS:large",
            type=StandardType.CS,
            section="large",
            content="X" * 10000,
            tokens=10000,
            version="latest",
            last_updated=datetime.now(),
            dependencies=[],
            nist_controls=set(),
            metadata={}
        )

        mock_engine._load_standard_sections = AsyncMock(return_value=[large_section])

        query = StandardQuery(query="CS:large", token_limit=5000)

        result = await mock_engine.load_standards(query)

        # Should have optimized the content
        assert len(result.standards) == 1
        assert result.metadata["token_count"] <= 5000
        assert result.standards[0]["metadata"].get("optimized") is True

    @pytest.mark.asyncio
    async def test_load_standards_with_cache_hit(self, test_standards_path):
        """Test loading with cache hit"""
        mock_redis = MagicMock(spec=redis.Redis)

        # Setup cached data
        cached_sections = [
            {
                "id": "CS:api",
                "type": "CS",
                "section": "api",
                "content": "Cached content",
                "tokens": 500,
                "version": "latest",
                "last_updated": datetime.now().isoformat(),
                "dependencies": [],
                "nist_controls": [],
                "metadata": {"from_cache": True}
            }
        ]
        mock_redis.get.return_value = json.dumps(cached_sections)

        engine = StandardsEngine(test_standards_path, redis_client=mock_redis)
        engine._save_to_cache = AsyncMock()

        query = StandardQuery(query="CS:api")

        result = await engine.load_standards(query)

        # Should use cached data
        assert len(result.standards) == 1
        assert result.standards[0]["metadata"]["from_cache"] is True

        # Should not save to cache (already cached)
        engine._save_to_cache.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_standards_processing_time(self, mock_engine):
        """Test processing time measurement"""
        query = StandardQuery(query="CS:api")

        result = await mock_engine.load_standards(query)

        assert "processing_time_ms" in result.query_info
        assert isinstance(result.query_info["processing_time_ms"], int)
        assert result.query_info["processing_time_ms"] >= 0
