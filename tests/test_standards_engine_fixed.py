"""
Fixed Tests for Standards Engine
@nist-controls: SA-11, CA-7
@evidence: Comprehensive unit tests for standards engine
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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
    TokenBudget,
    TokenOptimizationStrategy,
)


# Mock the audit_log decorator to avoid issues
def mock_audit_log(controls):
    """Mock audit log decorator"""
    def decorator(func):
        return func
    return decorator


class TestStandardsEngineCore:
    """Test core StandardsEngine functionality with mocked decorators"""
    
    @pytest.fixture
    def test_standards_path(self, tmp_path):
        """Create test standards directory"""
        standards_dir = tmp_path / "standards"
        standards_dir.mkdir()
        
        # Create basic structure
        cs_dir = standards_dir / "CS"
        cs_dir.mkdir()
        
        api_file = cs_dir / "api.yaml"
        api_file.write_text("""
id: CS.api
title: API Standards
content: |
  REST API design best practices.
  Always use versioning.
  Implement proper error handling.
tags:
  - api
  - rest
  - design
""")
        
        return standards_dir
    
    @pytest.mark.asyncio
    async def test_parse_query_load_command_fixed(self, test_standards_path):
        """Test parsing load command correctly"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            refs, query_info = await engine.parse_query("load CS:api")
            
            assert refs == ["CS:api"]
            assert query_info["query_type"] == "direct_notation"
    
    @pytest.mark.asyncio
    async def test_parse_query_empty_fixed(self, test_standards_path):
        """Test parsing empty query correctly"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            refs, query_info = await engine.parse_query("")
            
            assert refs == []
            assert query_info["query_type"] == "natural_language"
            assert query_info["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_load_standards_complete_flow(self, test_standards_path):
        """Test complete load_standards flow"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            # Mock the file loading since we don't have real YAML files
            async def mock_load_section(std_type, section, version):
                return StandardSection(
                    id=f"{std_type}:{section}",
                    type=StandardType.from_string(std_type),
                    section=section,
                    content=f"Mock content for {std_type}:{section}",
                    tokens=100,
                    version=version,
                    last_updated=datetime.now(),
                    dependencies=[],
                    nist_controls={"AC-3", "AU-2"},
                    metadata={"mocked": True}
                )
            
            engine._load_section = mock_load_section
            
            query = StandardQuery(query="CS:api", token_limit=5000)
            result = await engine.load_standards(query)
            
            assert isinstance(result, StandardLoadResult)
            assert len(result.standards) == 1
            assert result.standards[0]["id"] == "CS:api"
            assert result.metadata["token_count"] == 100
    
    @pytest.mark.asyncio
    async def test_cache_operations_with_redis(self, test_standards_path):
        """Test cache operations with real Redis mock"""
        mock_redis = MagicMock(spec=redis.Redis)
        
        # Test cache miss
        mock_redis.get.return_value = None
        
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path, redis_client=mock_redis)
            
            # Test get from cache
            sections = await engine._get_from_cache("CS:api", "latest")
            assert sections is None
            
            # Test save to cache
            test_sections = [
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
            
            await engine._save_to_cache("CS:api", "latest", test_sections)
            
            mock_redis.setex.assert_called_once()
            call_args = mock_redis.setex.call_args[0]
            assert call_args[0] == "standard:CS:api:latest"
            assert call_args[1] == 3600  # Default TTL
            
            # Verify JSON serialization
            saved_data = json.loads(call_args[2])
            assert saved_data[0]["id"] == "CS:api"
    
    @pytest.mark.asyncio
    async def test_cache_hit_flow(self, test_standards_path):
        """Test cache hit scenario"""
        mock_redis = MagicMock(spec=redis.Redis)
        
        # Setup cached data
        cached_data = [
            {
                "id": "CS:api",
                "type": "coding",
                "section": "api",
                "content": "Cached API content",
                "tokens": 150,
                "version": "latest",
                "last_updated": datetime.now().isoformat(),
                "dependencies": ["CS:general"],
                "nist_controls": ["AC-3"],
                "metadata": {"cached": True}
            }
        ]
        mock_redis.get.return_value = json.dumps(cached_data)
        
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path, redis_client=mock_redis)
            
            sections = await engine._get_from_cache("CS:api", "latest")
            
            assert sections is not None
            assert len(sections) == 1
            assert sections[0].id == "CS:api"
            assert sections[0].content == "Cached API content"
            assert sections[0].tokens == 150
            assert sections[0].dependencies == ["CS:general"]
            assert "AC-3" in sections[0].nist_controls
    
    @pytest.mark.asyncio
    async def test_cache_error_handling(self, test_standards_path):
        """Test cache error scenarios"""
        mock_redis = MagicMock(spec=redis.Redis)
        
        # Test Redis connection error on get
        mock_redis.get.side_effect = RedisError("Connection lost")
        
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            with patch('src.core.standards.engine.logger') as mock_logger:
                engine = StandardsEngine(test_standards_path, redis_client=mock_redis)
                
                sections = await engine._get_from_cache("CS:api", "latest")
                
                assert sections is None
                mock_logger.error.assert_called_once()
                assert "Cache retrieval error" in mock_logger.error.call_args[0][0]
        
        # Test Redis error on save
        mock_redis.get.side_effect = None
        mock_redis.setex.side_effect = RedisError("Write failed")
        
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            with patch('src.core.standards.engine.logger') as mock_logger:
                engine = StandardsEngine(test_standards_path, redis_client=mock_redis)
                
                test_sections = [
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
                
                await engine._save_to_cache("CS:api", "latest", test_sections)
                
                mock_logger.error.assert_called_once()
                assert "Cache save error" in mock_logger.error.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_load_standards_with_token_optimization(self, test_standards_path):
        """Test token optimization in load_standards"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            # Create a large section that exceeds token limit
            large_section = StandardSection(
                id="CS:large",
                type=StandardType.CS,
                section="large",
                content="X" * 10000,  # Very large content
                tokens=2500,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
            
            # Mock to return large section
            engine._load_standard_sections = AsyncMock(return_value=[large_section])
            engine._get_from_cache = AsyncMock(return_value=None)
            engine._save_to_cache = AsyncMock()
            
            # Query with small token limit
            query = StandardQuery(query="CS:large", token_limit=1000)
            
            result = await engine.load_standards(query)
            
            # Should have optimized the content
            assert len(result.standards) == 1
            assert result.standards[0]["tokens"] <= 1000
            assert result.standards[0]["metadata"]["optimized"] is True
            assert "[truncated]" in result.standards[0]["content"]
    
    @pytest.mark.asyncio
    async def test_load_standards_token_budget_exhaustion(self, test_standards_path):
        """Test behavior when token budget is exhausted"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            # Create multiple sections
            sections = [
                StandardSection(
                    id=f"CS:section{i}",
                    type=StandardType.CS,
                    section=f"section{i}",
                    content=f"Content for section {i}",
                    tokens=1000,
                    version="latest",
                    last_updated=datetime.now(),
                    dependencies=[],
                    nist_controls=set(),
                    metadata={}
                )
                for i in range(10)
            ]
            
            engine._load_standard_sections = AsyncMock(return_value=sections)
            engine._get_from_cache = AsyncMock(return_value=None)
            engine._save_to_cache = AsyncMock()
            
            # Query with limited tokens
            query = StandardQuery(query="CS:*", token_limit=2500)
            
            result = await engine.load_standards(query)
            
            # Should only load 2 sections (2000 tokens)
            assert len(result.standards) == 2
            assert result.metadata["token_count"] == 2000
    
    @pytest.mark.asyncio
    async def test_load_standards_with_context_analysis(self, test_standards_path):
        """Test context-based standard loading"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            # Mock methods
            engine._load_standard_sections = AsyncMock(return_value=[
                StandardSection(
                    id="CS:api",
                    type=StandardType.CS,
                    section="api",
                    content="API content",
                    tokens=100,
                    version="latest",
                    last_updated=datetime.now(),
                    dependencies=[],
                    nist_controls=set(),
                    metadata={}
                )
            ])
            engine._get_from_cache = AsyncMock(return_value=None)
            engine._save_to_cache = AsyncMock()
            
            query = StandardQuery(
                query="authentication",
                context="Building secure API with comprehensive testing",
                token_limit=5000
            )
            
            result = await engine.load_standards(query)
            
            # Should have context refs
            assert "context_refs" in result.query_info
            assert "TS:*" in result.query_info["context_refs"]  # "testing" in context
            assert "SEC:*" in result.query_info["context_refs"]  # "secure" in context
            assert "CS:api" in result.query_info["context_refs"]  # "API" in context
    
    def test_count_tokens_method(self, test_standards_path):
        """Test token counting"""
        engine = StandardsEngine(test_standards_path)
        
        # Test various strings
        assert engine._count_tokens("") == 0
        assert engine._count_tokens("test") == 1  # 4 chars = 1 token
        assert engine._count_tokens("hello world") == 2  # 11 chars = 2 tokens
        assert engine._count_tokens("X" * 100) == 25  # 100 chars = 25 tokens
    
    @pytest.mark.asyncio
    async def test_optimize_for_tokens_other_strategies(self, test_standards_path):
        """Test token optimization with no valid strategy"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(test_standards_path)
            
            section = StandardSection(
                id="TEST:section",
                type=StandardType.CS,
                section="test",
                content="Test content",
                tokens=1000,
                version="latest",
                last_updated=datetime.now(),
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
            
            # Test with unsupported strategy
            result = await engine._optimize_for_tokens(
                section,
                500,
                TokenOptimizationStrategy.SUMMARIZE  # Not implemented
            )
            
            assert result is None


class TestStandardsLoader:
    """Test standards loading functionality"""
    
    @pytest.fixture
    def standards_path_with_files(self, tmp_path):
        """Create standards directory with actual YAML files"""
        standards_dir = tmp_path / "standards"
        standards_dir.mkdir()
        
        # Create schema
        schema = {
            "version": "1.0",
            "types": {
                "CS": "Coding Standards",
                "SEC": "Security Standards"
            }
        }
        schema_file = standards_dir / "standards-schema.yaml"
        schema_file.write_text(yaml.dump(schema))
        
        # Create CS directory with files
        cs_dir = standards_dir / "CS"
        cs_dir.mkdir()
        
        # API standards
        api_yaml = {
            "id": "CS.api",
            "title": "API Design Standards",
            "sections": {
                "rest": {
                    "content": "RESTful API design principles",
                    "tokens": 150
                },
                "graphql": {
                    "content": "GraphQL best practices",
                    "tokens": 200
                }
            }
        }
        (cs_dir / "api.yaml").write_text(yaml.dump(api_yaml))
        
        return standards_dir
    
    @pytest.mark.asyncio
    async def test_load_section_file_based(self, standards_path_with_files):
        """Test loading section from actual files"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(standards_path_with_files)
            
            # Override the mock implementation to actually load files
            # This would require implementing the actual file loading logic
            # For now, we'll continue with the mock
            section = await engine._load_section("CS", "api", "latest")
            
            assert section is not None
            assert section.id == "CS:api"
            assert section.type == StandardType.CS