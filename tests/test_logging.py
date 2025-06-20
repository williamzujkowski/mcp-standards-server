"""
Test Logging Module
@nist-controls: SA-11, CA-7
@evidence: Unit tests for logging functionality
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.core.logging import get_logger


class TestLogging:
    """Test logging functionality"""
    
    def test_get_logger(self):
        """Test getting a logger instance"""
        logger = get_logger("test.module")
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"
    
    def test_get_logger_with_none(self):
        """Test getting logger with None name"""
        logger = get_logger(None)
        
        assert isinstance(logger, logging.Logger)
        assert logger.name == "mcp-standards"
    
    def test_structured_formatter(self):
        """Test StructuredFormatter"""
        from src.core.logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "INFO"
        assert data["message"] == "Test message"
        assert data["logger"] == "test.logger"
        assert "timestamp" in data
    
    def test_structured_formatter_with_extras(self):
        """Test StructuredFormatter with extra fields"""
        from src.core.logging import StructuredFormatter
        
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        
        # Add extra fields
        record.context = {"user_id": "test-user"}
        record.event = "test.event"
        record.nist_controls = ["AC-3", "AU-2"]
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["context"] == {"user_id": "test-user"}
        assert data["event"] == "test.event"
        assert data["nist_controls"] == ["AC-3", "AU-2"]