"""
Additional Model Tests
@nist-controls: SA-11, CA-7
@evidence: Additional unit tests for data models
"""

from datetime import datetime, timedelta

import pytest

from src.core.mcp.models import AuthenticationLevel, SessionInfo
from src.core.standards.models import (
    StandardCache,
    StandardQuery,
    StandardSection,
    StandardType,
    TokenBudget,
)


class TestStandardModels:
    """Test standard models edge cases"""
    
    def test_standard_type_from_string(self):
        """Test StandardType.from_string method"""
        # Valid conversions
        assert StandardType.from_string("CS") == StandardType.CS
        assert StandardType.from_string("cs") == StandardType.CS
        assert StandardType.from_string("SEC") == StandardType.SEC
        
        # Invalid conversion
        with pytest.raises(ValueError) as exc_info:
            StandardType.from_string("INVALID")
        assert "Unknown standard type: INVALID" in str(exc_info.value)
    
    def test_standard_section_to_dict(self):
        """Test StandardSection.to_dict method"""
        section = StandardSection(
            id="CS.api.rest",
            type=StandardType.CS,
            section="rest",
            content="REST API standards",
            tokens=100,
            version="1.0",
            last_updated=datetime(2024, 1, 1),
            dependencies=["CS.api"],
            nist_controls={"AC-3", "AU-2"},
            metadata={"author": "test"}
        )
        
        result = section.to_dict()
        
        assert result["id"] == "CS.api.rest"
        assert result["type"] == "coding"
        assert result["content"] == "REST API standards"
        assert result["tokens"] == 100
        assert result["dependencies"] == ["CS.api"]
        assert set(result["nist_controls"]) == {"AC-3", "AU-2"}
        assert result["metadata"]["author"] == "test"
    
    def test_standard_query_validation(self):
        """Test StandardQuery validation"""
        # Valid query
        query = StandardQuery(query="test query", token_limit=5000)
        assert query.query == "test query"
        assert query.token_limit == 5000
        
        # Empty query
        with pytest.raises(ValueError):
            StandardQuery(query="   ")
        
        # Query with invalid characters
        with pytest.raises(ValueError):
            StandardQuery(query="test<script>alert('xss')</script>")
        
        # Token limit too low
        with pytest.raises(ValueError):
            StandardQuery(query="test", token_limit=50)
        
        # Token limit too high
        with pytest.raises(ValueError):
            StandardQuery(query="test", token_limit=200000)
    
    def test_standard_cache_expiry(self):
        """Test StandardCache expiry checking"""
        # Not expired
        cache = StandardCache(
            key="test",
            value=[],
            created_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            access_count=0
        )
        assert not cache.is_expired()
        
        # Expired
        expired_cache = StandardCache(
            key="test",
            value=[],
            created_at=datetime.utcnow() - timedelta(hours=2),
            expires_at=datetime.utcnow() - timedelta(hours=1),
            access_count=0
        )
        assert expired_cache.is_expired()
    
    def test_token_budget_operations(self):
        """Test TokenBudget operations"""
        budget = TokenBudget(total_limit=1000)
        
        # Initial state
        assert budget.available == 1000
        assert budget.can_fit(500)
        
        # Allocate tokens
        assert budget.allocate(300)
        assert budget.used == 300
        assert budget.available == 700
        
        # Reserve tokens
        assert budget.reserve(200)
        assert budget.reserved == 200
        assert budget.available == 500
        
        # Cannot fit
        assert not budget.can_fit(600)
        assert not budget.allocate(600)
        
        # Cannot reserve beyond limit
        assert not budget.reserve(600)


class TestMCPModels:
    """Test MCP model edge cases"""
    
    def test_session_info_idle_timeout(self):
        """Test SessionInfo idle timeout checking"""
        session = SessionInfo(
            session_id="test",
            user_id="user",
            created_at=datetime.now(),
            last_activity=datetime.now() - timedelta(minutes=31),
            expires_at=datetime.now() + timedelta(hours=1),
            auth_level=AuthenticationLevel.BASIC,
            permissions=[],
            metadata={}
        )
        
        # Should be idle (default timeout is 30 minutes)
        assert session.is_idle_timeout()