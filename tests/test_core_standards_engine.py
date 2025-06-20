"""
Test Standards Engine
@nist-controls: SA-11, CA-7
@evidence: Unit tests for standards engine
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.standards.engine import StandardsEngine


class TestStandardsEngine:
    """Test StandardsEngine initialization and basic functionality"""
    
    @pytest.fixture
    def test_standards_path(self, tmp_path):
        """Create test standards directory"""
        standards_dir = tmp_path / "standards"
        standards_dir.mkdir()
        
        # Create a test standard file
        cs_dir = standards_dir / "CS"
        cs_dir.mkdir()
        
        api_file = cs_dir / "api.yaml"
        api_file.write_text("""
id: CS.api
title: API Standards
content: |
  API design best practices.
tags:
  - api
  - rest
""")
        
        # Create schema file
        schema_file = standards_dir / "standards-schema.yaml"
        schema_file.write_text("""
version: "1.0"
types:
  - CS
  - SEC
  - TS
  - FE
  - CN
""")
        
        return standards_dir
    
    def test_engine_initialization(self, test_standards_path):
        """Test engine initializes correctly"""
        engine = StandardsEngine(test_standards_path)
        
        assert engine.standards_path == test_standards_path
        assert engine.redis_client is None  # No redis provided
        assert engine.nl_mapper is not None
        assert engine.cache_ttl == 3600  # Default TTL
        assert engine.loaded_standards == {}
    
    def test_engine_with_redis_url(self, test_standards_path):
        """Test engine initialization with Redis URL"""
        with patch.dict('os.environ', {'REDIS_URL': 'redis://localhost:6379'}):
            engine = StandardsEngine(test_standards_path)
            
            assert engine.standards_path == test_standards_path
            # Redis client would be initialized if Redis was available
    
    def test_engine_without_standards_path(self):
        """Test engine initialization without valid path"""
        # Engine should initialize even with non-existent path
        engine = StandardsEngine(Path("/nonexistent/path"))
        assert engine.standards_path == Path("/nonexistent/path")
        assert engine.schema is None  # No schema file exists
    
    @pytest.mark.asyncio
    async def test_get_catalog(self, test_standards_path):
        """Test loading schema which contains catalog of types"""
        engine = StandardsEngine(test_standards_path)
        
        # The schema contains the catalog of standard types
        assert engine.schema is not None
        assert "types" in engine.schema
        assert "CS" in engine.schema["types"]
        assert "SEC" in engine.schema["types"]
        assert "TS" in engine.schema["types"]