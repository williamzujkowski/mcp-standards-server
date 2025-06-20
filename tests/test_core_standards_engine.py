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
        
        return standards_dir
    
    def test_engine_initialization(self, test_standards_path):
        """Test engine initializes correctly"""
        engine = StandardsEngine(test_standards_path)
        
        assert engine.standards_path == test_standards_path
        assert engine.cache is not None
        assert engine.natural_mapper is not None
        assert engine.loader is not None
        assert engine.optimizer is not None
    
    def test_engine_with_redis_url(self, test_standards_path):
        """Test engine initialization with Redis URL"""
        with patch.dict('os.environ', {'REDIS_URL': 'redis://localhost:6379'}):
            engine = StandardsEngine(test_standards_path)
            
            assert engine.standards_path == test_standards_path
            # Redis client would be initialized if Redis was available
    
    def test_engine_without_standards_path(self):
        """Test engine initialization without valid path"""
        with pytest.raises(Exception):
            StandardsEngine(Path("/nonexistent/path"))
    
    @pytest.mark.asyncio
    async def test_get_catalog(self, test_standards_path):
        """Test getting standards catalog"""
        engine = StandardsEngine(test_standards_path)
        
        # Mock the loader's get_catalog method
        engine.loader.get_catalog = MagicMock(return_value=["CS", "SEC", "TS"])
        
        catalog = await engine.get_catalog()
        
        assert isinstance(catalog, list)
        assert "CS" in catalog
        assert "SEC" in catalog
        assert "TS" in catalog