"""
Test Logging Module
@nist-controls: SA-11, CA-7
@evidence: Unit tests for logging functionality
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from src.core.logging import (
    SecurityEventLogger,
    StructuredFormatter,
    audit_log,
    get_logger,
    log_security_event,
)
from src.core.mcp.models import ComplianceContext


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

    def test_structured_formatter_with_exception(self):
        """Test StructuredFormatter with exception info"""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except Exception:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test exception" in data["exception"]

    @pytest.mark.asyncio
    async def test_log_security_event(self):
        """Test log_security_event function"""
        logger = MagicMock()
        context = ComplianceContext(user_id="test-user", roles=["admin"])

        await log_security_event(
            logger,
            "test.event",
            context,
            {"action": "test_action"}
        )

        logger.info.assert_called_once()
        call_args = logger.info.call_args
        assert "Security event: test.event" in call_args[0]
        extra = call_args[1]["extra"]
        assert extra["event"] == "test.event"
        assert extra["context"]["user_id"] == "test-user"
        assert extra["details"]["action"] == "test_action"

    @pytest.mark.asyncio
    async def test_log_security_event_no_context(self):
        """Test log_security_event without context"""
        logger = MagicMock()

        await log_security_event(
            logger,
            "test.event",
            None,
            {"action": "test_action"}
        )

        logger.info.assert_called_once()
        extra = logger.info.call_args[1]["extra"]
        assert extra["context"] == {}

    @pytest.mark.asyncio
    async def test_audit_log_decorator_async(self):
        """Test audit_log decorator with async function"""
        @audit_log(["AC-3", "AU-2"])
        async def test_function(value: int) -> int:
            return value * 2

        with patch('src.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = await test_function(5)

            assert result == 10
            assert mock_logger.info.call_count == 2

            # Check entry log
            entry_call = mock_logger.info.call_args_list[0]
            assert "Entering test_function" in entry_call[0]
            assert entry_call[1]["extra"]["event"] == "test_function.entry"

            # Check success log
            success_call = mock_logger.info.call_args_list[1]
            assert "Completed test_function" in success_call[0]
            assert success_call[1]["extra"]["event"] == "test_function.success"

    @pytest.mark.asyncio
    async def test_audit_log_decorator_async_with_exception(self):
        """Test audit_log decorator with async function that raises exception"""
        @audit_log(["AC-3", "AU-2"])
        async def test_function():
            raise ValueError("Test error")

        with patch('src.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError):
                await test_function()

            # Check that error was logged
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args
            assert "Failed test_function" in error_call[0][0]
            assert error_call[1]["extra"]["event"] == "test_function.failure"

    def test_audit_log_decorator_sync(self):
        """Test audit_log decorator with sync function"""
        @audit_log(["AC-3", "AU-2"])
        def test_function(value: int) -> int:
            return value * 2

        with patch('src.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = test_function(5)

            assert result == 10
            assert mock_logger.info.call_count == 2

    def test_audit_log_decorator_sync_with_exception(self):
        """Test audit_log decorator with sync function that raises exception"""
        @audit_log(["AC-3", "AU-2"])
        def test_function():
            raise ValueError("Test error")

        with patch('src.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with pytest.raises(ValueError):
                test_function()

            mock_logger.error.assert_called_once()

    def test_audit_log_decorator_with_context(self):
        """Test audit_log decorator with ComplianceContext argument"""
        @audit_log(["AC-3", "AU-2"])
        def test_function(context: ComplianceContext, value: int) -> int:
            return value * 2

        with patch('src.core.logging.get_logger') as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            context = ComplianceContext(user_id="test-user", roles=["admin"])
            result = test_function(context, 5)

            assert result == 10

            # Check that context was included in logs
            entry_call = mock_logger.info.call_args_list[0]
            assert entry_call[1]["extra"]["context"]["user_id"] == "test-user"


class TestSecurityEventLogger:
    """Test SecurityEventLogger class"""

    @pytest.mark.asyncio
    async def test_log_authentication_attempt_success(self):
        """Test logging successful authentication"""
        mock_logger = MagicMock()
        event_logger = SecurityEventLogger(mock_logger)

        with patch('src.core.logging.log_security_event') as mock_log_event:
            await event_logger.log_authentication_attempt(
                user_id="test-user",
                success=True,
                method="password",
                ip_address="192.168.1.1"
            )

            mock_log_event.assert_called_once_with(
                mock_logger,
                "authentication.success",
                None,
                {
                    "user_id": "test-user",
                    "method": "password",
                    "ip_address": "192.168.1.1"
                }
            )

    @pytest.mark.asyncio
    async def test_log_authentication_attempt_failure(self):
        """Test logging failed authentication"""
        mock_logger = MagicMock()
        event_logger = SecurityEventLogger(mock_logger)

        with patch('src.core.logging.log_security_event') as mock_log_event:
            await event_logger.log_authentication_attempt(
                user_id="test-user",
                success=False,
                method="password",
                ip_address="192.168.1.1"
            )

            mock_log_event.assert_called_once()
            call_args = mock_log_event.call_args[0]
            assert call_args[1] == "authentication.failure"

    @pytest.mark.asyncio
    async def test_log_authorization_decision_granted(self):
        """Test logging granted authorization"""
        mock_logger = MagicMock()
        event_logger = SecurityEventLogger(mock_logger)
        context = ComplianceContext(user_id="test-user", roles=["admin"])

        with patch('src.core.logging.log_security_event') as mock_log_event:
            await event_logger.log_authorization_decision(
                context=context,
                resource="/api/users",
                action="DELETE",
                granted=True
            )

            mock_log_event.assert_called_once_with(
                mock_logger,
                "authorization.granted",
                context,
                {
                    "resource": "/api/users",
                    "action": "DELETE"
                }
            )

    @pytest.mark.asyncio
    async def test_log_authorization_decision_denied(self):
        """Test logging denied authorization"""
        mock_logger = MagicMock()
        event_logger = SecurityEventLogger(mock_logger)
        context = ComplianceContext(user_id="test-user", roles=["user"])

        with patch('src.core.logging.log_security_event') as mock_log_event:
            await event_logger.log_authorization_decision(
                context=context,
                resource="/api/admin",
                action="GET",
                granted=False
            )

            call_args = mock_log_event.call_args[0]
            assert call_args[1] == "authorization.denied"

    @pytest.mark.asyncio
    async def test_log_data_access(self):
        """Test logging data access"""
        mock_logger = MagicMock()
        event_logger = SecurityEventLogger(mock_logger)
        context = ComplianceContext(user_id="test-user", roles=["analyst"])

        with patch('src.core.logging.log_security_event') as mock_log_event:
            await event_logger.log_data_access(
                context=context,
                data_type="user_records",
                operation="READ",
                record_count=100
            )

            mock_log_event.assert_called_once_with(
                mock_logger,
                "data.access",
                context,
                {
                    "data_type": "user_records",
                    "operation": "READ",
                    "record_count": 100
                }
            )
