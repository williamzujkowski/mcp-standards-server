"""
Test Standards MCP Handlers
@nist-controls: SA-11, CA-7
@evidence: Unit tests for standards-specific handlers
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.mcp.handlers import HandlerRegistry
from src.core.mcp.models import ComplianceContext, MCPMessage
from src.core.standards.engine import StandardsEngine
from src.core.standards.handlers import (
    AnalyzeCodeHandler,
    GenerateCodeHandler,
    ListMethodsHandler,
    LoadStandardsHandler,
)
from src.core.standards.models import StandardLoadResult, StandardQuery


class TestLoadStandardsHandler:
    """Test LoadStandardsHandler"""

    @pytest.fixture
    def mock_engine(self):
        """Create mock standards engine"""
        engine = MagicMock(spec=StandardsEngine)
        engine.load_standards = AsyncMock()
        return engine

    @pytest.fixture
    def handler(self, mock_engine):
        """Create handler instance"""
        return LoadStandardsHandler(mock_engine)

    @pytest.fixture
    def context(self):
        """Create test context"""
        return ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

    def test_handler_initialization(self, handler, mock_engine):
        """Test handler initializes with engine"""
        assert handler.standards_engine == mock_engine
        assert handler.required_permissions == ["standards.read"]

    @pytest.mark.asyncio
    async def test_handle_basic_query(self, handler, mock_engine, context):
        """Test handling basic standards query"""
        # Setup mock response
        mock_result = StandardLoadResult(
            standards=[
                {"id": "CS.api", "content": "API standard content"},
                {"id": "SEC.auth", "content": "Auth standard content"}
            ],
            metadata={
                "query": "api security",
                "token_count": 500,
                "version": "latest"
            }
        )
        mock_engine.load_standards.return_value = mock_result

        # Create message
        message = MCPMessage(
            id="test-123",
            method="load_standards",
            params={
                "query": "api security",
                "version": "latest"
            },
            timestamp=time.time()
        )

        # Handle message
        with patch('src.core.standards.handlers.audit_log'):
            result = await handler.handle(message, context)

        # Verify engine was called correctly
        mock_engine.load_standards.assert_called_once()
        call_args = mock_engine.load_standards.call_args[0][0]
        assert isinstance(call_args, StandardQuery)
        assert call_args.query == "api security"
        assert call_args.version == "latest"

        # Verify result
        assert result["standards"] == mock_result.standards
        assert result["metadata"] == mock_result.metadata

    @pytest.mark.asyncio
    async def test_handle_with_context_and_token_limit(self, handler, mock_engine, context):
        """Test handling query with context and token limit"""
        mock_result = StandardLoadResult(
            standards=[{"id": "CS.api", "content": "API content"}],
            metadata={"token_count": 1000}
        )
        mock_engine.load_standards.return_value = mock_result

        message = MCPMessage(
            id="test-456",
            method="load_standards",
            params={
                "query": "authentication",
                "context": "Building OAuth2 implementation",
                "token_limit": 5000
            },
            timestamp=time.time()
        )

        with patch('src.core.standards.handlers.audit_log'):
            await handler.handle(message, context)

        # Verify query parameters
        call_args = mock_engine.load_standards.call_args[0][0]
        assert call_args.query == "authentication"
        assert call_args.context == "Building OAuth2 implementation"
        assert call_args.token_limit == 5000

    @pytest.mark.asyncio
    async def test_handle_default_parameters(self, handler, mock_engine, context):
        """Test handling with default parameters"""
        mock_result = StandardLoadResult(standards=[], metadata={})
        mock_engine.load_standards.return_value = mock_result

        message = MCPMessage(
            id="test-789",
            method="load_standards",
            params={"query": "default"},  # Provide a minimal query
            timestamp=time.time()
        )

        with patch('src.core.standards.handlers.audit_log'):
            await handler.handle(message, context)

        # Verify defaults for other parameters
        call_args = mock_engine.load_standards.call_args[0][0]
        assert call_args.query == "default"
        assert call_args.context is None
        assert call_args.version == "latest"
        assert call_args.token_limit is None

    @pytest.mark.asyncio
    async def test_audit_logging(self, handler, mock_engine, context):
        """Test handle method completes successfully with audit logging decorator"""
        mock_result = StandardLoadResult(
            standards=[{"id": "test", "content": "test"}],
            metadata={"count": 1},
            query_info={"type": "test"}
        )
        mock_engine.load_standards.return_value = mock_result

        message = MCPMessage(
            id="test-audit",
            method="load_standards",
            params={"query": "test"},
            timestamp=time.time()
        )

        # Just verify the method works correctly with the decorator applied
        result = await handler.handle(message, context)

        assert "standards" in result
        assert "metadata" in result
        assert "query_info" in result
        assert result["standards"] == [{"id": "test", "content": "test"}]


class TestAnalyzeCodeHandler:
    """Test AnalyzeCodeHandler"""

    @pytest.fixture
    def handler(self):
        """Create handler instance"""
        return AnalyzeCodeHandler()

    @pytest.fixture
    def context(self):
        """Create test context"""
        return ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

    def test_handler_initialization(self, handler):
        """Test handler initialization"""
        assert handler.required_permissions == ["code.analyze"]

    @pytest.mark.asyncio
    async def test_handle_not_implemented(self, handler, context):
        """Test handler returns not implemented"""
        message = MCPMessage(
            id="test-analyze",
            method="analyze_code",
            params={
                "code": "def test(): pass",
                "language": "python"
            },
            timestamp=time.time()
        )

        result = await handler.handle(message, context)

        assert result["status"] == "not_implemented"
        assert "Phase 1" in result["message"]


class TestGenerateCodeHandler:
    """Test GenerateCodeHandler"""

    @pytest.fixture
    def handler(self):
        """Create handler instance"""
        return GenerateCodeHandler()

    @pytest.fixture
    def context(self):
        """Create test context"""
        return ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

    def test_handler_initialization(self, handler):
        """Test handler initialization"""
        assert handler.required_permissions == ["code.generate"]

    @pytest.mark.asyncio
    async def test_handle_with_parameters(self, handler, context):
        """Test handler with template and controls"""
        message = MCPMessage(
            id="test-generate",
            method="generate_code",
            params={
                "template": "api-endpoint",
                "controls": ["AC-3", "AU-2"]
            },
            timestamp=time.time()
        )

        result = await handler.handle(message, context)

        assert result["status"] == "not_implemented"
        assert "Phase 2" in result["message"]
        assert result["template"] == "api-endpoint"
        assert result["controls"] == ["AC-3", "AU-2"]

    @pytest.mark.asyncio
    async def test_handle_without_parameters(self, handler, context):
        """Test handler without parameters"""
        message = MCPMessage(
            id="test-generate-empty",
            method="generate_code",
            params={},
            timestamp=time.time()
        )

        result = await handler.handle(message, context)

        assert result["status"] == "not_implemented"
        assert result["template"] == ""
        assert result["controls"] == []


class TestListMethodsHandler:
    """Test ListMethodsHandler"""

    @pytest.fixture
    def mock_registry(self):
        """Create mock handler registry"""
        registry = MagicMock(spec=HandlerRegistry)
        registry.list_methods.return_value = {
            "method1": {
                "description": "Test method 1",
                "permissions": ["read"],
                "deprecated": False
            },
            "method2": {
                "description": "Test method 2",
                "permissions": ["write"],
                "deprecated": True
            }
        }
        return registry

    @pytest.fixture
    def handler(self, mock_registry):
        """Create handler instance"""
        return ListMethodsHandler(mock_registry)

    @pytest.fixture
    def context(self):
        """Create test context"""
        return ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

    def test_handler_initialization(self, handler, mock_registry):
        """Test handler initialization"""
        assert handler.handler_registry == mock_registry
        assert handler.required_permissions == []  # Public method

    @pytest.mark.asyncio
    async def test_handle_list_methods(self, handler, mock_registry, context):
        """Test listing available methods"""
        message = MCPMessage(
            id="test-list",
            method="list_methods",
            params={},
            timestamp=time.time()
        )

        result = await handler.handle(message, context)

        # Verify registry was called
        mock_registry.list_methods.assert_called_once()

        # Verify result structure
        assert "methods" in result
        assert "version" in result
        assert result["version"] == "0.1.0"

        # Verify methods data
        methods = result["methods"]
        assert len(methods) == 2
        assert "method1" in methods
        assert "method2" in methods
        assert methods["method1"]["description"] == "Test method 1"
        assert methods["method2"]["deprecated"] is True


class TestHandlerIntegration:
    """Test handler integration scenarios"""

    @pytest.mark.asyncio
    async def test_handler_registration_and_usage(self):
        """Test registering and using standards handlers"""
        # Create components
        registry = HandlerRegistry()
        engine = MagicMock(spec=StandardsEngine)
        engine.load_standards = AsyncMock(
            return_value=StandardLoadResult(
                standards=[{"id": "test", "content": "test content"}],
                metadata={"token_count": 100}
            )
        )

        # Register handlers
        load_handler = LoadStandardsHandler(engine)
        analyze_handler = AnalyzeCodeHandler()
        generate_handler = GenerateCodeHandler()
        list_handler = ListMethodsHandler(registry)

        registry.register("load_standards", load_handler, description="Load standards")
        registry.register("analyze_code", analyze_handler, description="Analyze code")
        registry.register("generate_code", generate_handler, description="Generate code")
        registry.register("list_methods", list_handler, description="List methods")

        # Verify all registered
        methods = registry.list_methods()
        assert len(methods) == 4

        # Test each handler through registry
        context = ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )

        # Test load_standards
        message = MCPMessage(
            id="test-load",
            method="load_standards",
            params={"query": "test query"},
            timestamp=time.time()
        )

        handler = registry.get_handler("load_standards")
        assert handler is not None

        with patch('src.core.standards.handlers.audit_log'):
            result = await handler.handle(message, context)
            assert "standards" in result
            assert len(result["standards"]) == 1
