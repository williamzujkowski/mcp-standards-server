"""
Final Tests to Complete Standards Engine Coverage
@nist-controls: SA-11, CA-7
@evidence: Remaining tests for full coverage
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.core.standards.engine import StandardsEngine
from src.core.standards.models import (
    StandardQuery,
    StandardSection,
    StandardType,
    TokenOptimizationStrategy,
)


# Mock the audit_log decorator
def mock_audit_log(controls):
    """Mock audit log decorator"""
    def decorator(func):
        return func
    return decorator


class TestStandardsEngineRemainingCoverage:
    """Test remaining uncovered lines in standards engine"""
    
    @pytest.mark.asyncio
    async def test_parse_query_recursive_load(self, tmp_path):
        """Test recursive load command parsing"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(tmp_path)
            
            # Test the recursive call in parse_query for load command
            refs, query_info = await engine.parse_query("load load CS:api")
            
            assert refs == ["CS:api"]
            assert query_info["query_type"] == "direct_notation"
    
    @pytest.mark.asyncio
    async def test_load_standards_cache_miss_then_save(self, tmp_path):
        """Test cache miss followed by save operation"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(tmp_path)
            
            # Mock cache miss (returns None)
            engine._get_from_cache = AsyncMock(return_value=None)
            
            # Mock loading sections
            test_section = StandardSection(
                id="CS:api",
                type=StandardType.CS,
                section="api",
                content="Test content",
                tokens=100,
                version="latest",
                last_updated=None,  # Will use datetime.now()
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
            engine._load_standard_sections = AsyncMock(return_value=[test_section])
            
            # Mock save to cache
            engine._save_to_cache = AsyncMock()
            
            query = StandardQuery(query="CS:api")
            result = await engine.load_standards(query)
            
            # Verify cache was checked
            engine._get_from_cache.assert_called_once_with("CS:api", "latest")
            
            # Verify sections were loaded
            engine._load_standard_sections.assert_called_once_with("CS:api", "latest")
            
            # Verify cache was saved
            engine._save_to_cache.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_standards_optimize_cannot_fit(self, tmp_path):
        """Test when even optimized content cannot fit"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(tmp_path)
            
            # Create very large section
            large_section = StandardSection(
                id="CS:huge",
                type=StandardType.CS,
                section="huge",
                content="X" * 100000,
                tokens=25000,
                version="latest",
                last_updated=None,
                dependencies=[],
                nist_controls=set(),
                metadata={}
            )
            
            engine._get_from_cache = AsyncMock(return_value=None)
            engine._load_standard_sections = AsyncMock(return_value=[large_section])
            engine._save_to_cache = AsyncMock()
            
            # Override optimize method to return None (cannot optimize enough)
            engine._optimize_for_tokens = AsyncMock(return_value=None)
            
            # Very small token limit
            query = StandardQuery(query="CS:huge", token_limit=100)
            
            result = await engine.load_standards(query)
            
            # Should have tried to optimize but failed to include section
            engine._optimize_for_tokens.assert_called_once()
            assert len(result.standards) == 0
            assert result.metadata["token_count"] == 0
    
    @pytest.mark.asyncio
    async def test_load_standards_token_budget_exactly_exhausted(self, tmp_path):
        """Test when token budget is exactly exhausted"""
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            engine = StandardsEngine(tmp_path)
            
            # Create sections that exactly match budget
            sections = [
                StandardSection(
                    id="CS:s1",
                    type=StandardType.CS,
                    section="s1",
                    content="Content 1",
                    tokens=1000,
                    version="latest",
                    last_updated=None,
                    dependencies=[],
                    nist_controls=set(),
                    metadata={}
                ),
                StandardSection(
                    id="CS:s2",
                    type=StandardType.CS,
                    section="s2",
                    content="Content 2",
                    tokens=1000,
                    version="latest",
                    last_updated=None,
                    dependencies=[],
                    nist_controls=set(),
                    metadata={}
                )
            ]
            
            engine._get_from_cache = AsyncMock(return_value=None)
            engine._load_standard_sections = AsyncMock(return_value=sections)
            engine._save_to_cache = AsyncMock()
            
            # Exactly 2000 tokens
            query = StandardQuery(query="CS:*", token_limit=2000)
            
            result = await engine.load_standards(query)
            
            # Should load exactly 2 sections
            assert len(result.standards) == 2
            assert result.metadata["token_count"] == 2000
    
    @pytest.mark.asyncio
    async def test_get_from_cache_json_decode_error(self, tmp_path):
        """Test cache retrieval with JSON decode error"""
        import redis
        from redis.exceptions import RedisError
        
        mock_redis = AsyncMock(spec=redis.Redis)
        mock_redis.get.return_value = "invalid json {"  # Invalid JSON
        
        with patch('src.core.standards.engine.audit_log', mock_audit_log):
            with patch('src.core.standards.engine.logger') as mock_logger:
                engine = StandardsEngine(tmp_path, redis_client=mock_redis)
                
                sections = await engine._get_from_cache("CS:api", "latest")
                
                assert sections is None
                mock_logger.error.assert_called_once()
                assert "Cache retrieval error" in mock_logger.error.call_args[0][0]