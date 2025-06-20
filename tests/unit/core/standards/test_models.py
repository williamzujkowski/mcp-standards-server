"""
Test Standards Models
@nist-controls: SA-11, CA-7
@evidence: Unit tests for standards data models
"""

from datetime import datetime, timedelta

import pytest

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
        with pytest.raises(ValueError):
            StandardType.from_string("INVALID")

    def test_standard_section_with_metadata(self):
        """Test StandardSection with full metadata"""
        section = StandardSection(
            id="CS:api",
            type=StandardType.CS,
            section="api",
            title="API Standards",
            content="API design guidelines",
            tags=["api", "rest", "design"],
            metadata={
                "author": "Test Author",
                "version": "1.0.0",
                "custom_field": "value"
            },
            version="latest",
            tokens=100,
            last_updated=datetime.now()
        )

        assert section.id == "CS:api"
        assert section.metadata["author"] == "Test Author"
        assert "api" in section.tags

    def test_standard_query_defaults(self):
        """Test StandardQuery default values"""
        query = StandardQuery(query="test query")

        assert query.query == "test query"
        assert query.context is None
        assert query.version == "latest"
        assert query.token_limit == 10000
        assert query.include_examples is True

    def test_token_budget_operations(self):
        """Test TokenBudget calculations"""
        budget = TokenBudget(
            total=1000,
            used=300,
            available=700,
            reserved=100
        )

        assert budget.available == 700
        assert budget.get_net_available() == 600  # available - reserved

        # Test consuming tokens
        assert budget.can_consume(500) is True
        assert budget.can_consume(700) is False  # exceeds net available

        budget.consume(200)
        assert budget.used == 500
        assert budget.available == 500

    def test_standard_cache_entry(self):
        """Test StandardCache model"""
        cache_entry = StandardCache(
            key="test_key",
            value={"data": "test"},
            expires_at=datetime.now() + timedelta(hours=1),
            hit_count=5,
            created_at=datetime.now()
        )

        assert cache_entry.key == "test_key"
        assert cache_entry.is_expired() is False
        assert cache_entry.hit_count == 5


# ============================================================
# Merged from tests/test_models_coverage.py
# ============================================================


class TestModelsCoverage:
    """Additional tests for model coverage"""

    def test_standard_section_equality(self):
        """Test StandardSection equality comparison"""
        section1 = StandardSection(
            id="CS:api",
            type=StandardType.CS,
            section="api",
            title="API Standards",
            content="content",
            version="1.0.0",
            tokens=100
        )

        section2 = StandardSection(
            id="CS:api",
            type=StandardType.CS,
            section="api",
            title="API Standards",
            content="content",
            version="1.0.0",
            tokens=100
        )

        assert section1.id == section2.id
        assert section1.type == section2.type

    def test_token_budget_edge_cases(self):
        """Test TokenBudget edge cases"""
        # Zero budget
        budget = TokenBudget(total=0, used=0, available=0, reserved=0)
        assert budget.can_consume(1) is False

        # Negative values should raise error
        with pytest.raises(ValueError):
            TokenBudget(total=-100, used=0, available=-100, reserved=0)

    def test_standard_cache_expiration(self):
        """Test cache expiration logic"""
        # Already expired
        expired_cache = StandardCache(
            key="expired",
            value={},
            expires_at=datetime.now() - timedelta(hours=1),
            created_at=datetime.now() - timedelta(hours=2)
        )
        assert expired_cache.is_expired() is True

        # Not expired
        valid_cache = StandardCache(
            key="valid",
            value={},
            expires_at=datetime.now() + timedelta(hours=1),
            created_at=datetime.now()
        )
        assert valid_cache.is_expired() is False
