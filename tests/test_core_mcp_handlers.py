"""
Test MCP Protocol Handlers
@nist-controls: SA-11, CA-7
@evidence: Unit tests for MCP handler framework
"""

import inspect
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.core.mcp.handlers import HandlerRegistry, MCPHandler
from src.core.mcp.models import ComplianceContext, MCPMessage


class SimpleHandler(MCPHandler):
    """Simple handler for testing"""
    
    async def handle(self, message: MCPMessage, context: ComplianceContext) -> dict[str, Any]:
        return {"result": "simple", "method": message.method}


class PermissionHandler(MCPHandler):
    """Handler with permissions for testing"""
    
    required_permissions = ["read", "write"]
    
    async def handle(self, message: MCPMessage, context: ComplianceContext) -> dict[str, Any]:
        return {"result": "permitted", "method": message.method}


class ParameterHandler(MCPHandler):
    """Handler with parameters for testing"""
    
    async def handle(
        self, 
        message: MCPMessage, 
        context: ComplianceContext,
        param1: str = "",
        param2: int = 0
    ) -> dict[str, Any]:
        return {
            "result": "parameterized",
            "method": message.method,
            "param1": param1,
            "param2": param2
        }


class TestMCPHandler:
    """Test MCPHandler base class"""
    
    @pytest.fixture
    def simple_handler(self):
        """Create simple handler instance"""
        return SimpleHandler()
    
    @pytest.fixture
    def permission_handler(self):
        """Create handler with permissions"""
        return PermissionHandler()
    
    @pytest.fixture
    def context(self):
        """Create test context"""
        return ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=1234567890.0,
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )
    
    @pytest.fixture
    def message(self):
        """Create test message"""
        return MCPMessage(
            id="test-123",
            method="test.method",
            params={},
            timestamp=1234567890.0
        )
    
    def test_handler_has_required_methods(self, simple_handler):
        """Test handler has required abstract methods"""
        assert hasattr(simple_handler, 'handle')
        assert hasattr(simple_handler, 'check_permissions')
        assert hasattr(simple_handler, 'validate_params')
        assert hasattr(simple_handler, 'required_permissions')
    
    @pytest.mark.asyncio
    async def test_simple_handler_handle(self, simple_handler, message, context):
        """Test simple handler handle method"""
        result = await simple_handler.handle(message, context)
        
        assert result["result"] == "simple"
        assert result["method"] == "test.method"
    
    def test_check_permissions_no_requirements(self, simple_handler, context):
        """Test permission check with no requirements"""
        assert simple_handler.required_permissions == []
        assert simple_handler.check_permissions(context) is True
    
    def test_check_permissions_with_requirements(self, permission_handler, context):
        """Test permission check with requirements"""
        assert permission_handler.required_permissions == ["read", "write"]
        # Default implementation returns True (placeholder)
        assert permission_handler.check_permissions(context) is True
    
    @pytest.mark.asyncio
    async def test_validate_params_empty(self, simple_handler):
        """Test parameter validation with no params"""
        params = {}
        validated = await simple_handler.validate_params(params)
        assert validated == params
    
    @pytest.mark.asyncio
    async def test_validate_params_extra_params(self, simple_handler):
        """Test parameter validation with extra params"""
        params = {"unexpected": "value"}
        
        with pytest.raises(ValueError) as exc_info:
            await simple_handler.validate_params(params)
        
        assert "Unexpected parameters: {'unexpected'}" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_params_with_handler_params(self):
        """Test parameter validation with handler that has params"""
        handler = ParameterHandler()
        
        # Valid params
        params = {"param1": "test", "param2": 42}
        validated = await handler.validate_params(params)
        assert validated == params
        
        # Extra params should raise error
        params = {"param1": "test", "param2": 42, "extra": "bad"}
        with pytest.raises(ValueError) as exc_info:
            await handler.validate_params(params)
        
        assert "Unexpected parameters: {'extra'}" in str(exc_info.value)


class TestHandlerRegistry:
    """Test HandlerRegistry"""
    
    @pytest.fixture
    def registry(self):
        """Create registry instance"""
        return HandlerRegistry()
    
    @pytest.fixture
    def handler(self):
        """Create test handler"""
        return SimpleHandler()
    
    def test_registry_initialization(self, registry):
        """Test registry initializes empty"""
        assert registry._handlers == {}
        assert registry._handler_metadata == {}
    
    def test_register_handler(self, registry, handler):
        """Test handler registration"""
        registry.register(
            "test.method",
            handler,
            description="Test method",
            deprecated=False
        )
        
        assert "test.method" in registry._handlers
        assert registry._handlers["test.method"] == handler
        
        metadata = registry._handler_metadata["test.method"]
        assert metadata["description"] == "Test method"
        assert metadata["deprecated"] is False
        assert metadata["permissions"] == []
    
    def test_register_handler_with_permissions(self, registry):
        """Test registering handler with permissions"""
        handler = PermissionHandler()
        registry.register("perm.method", handler, description="Permission test")
        
        metadata = registry._handler_metadata["perm.method"]
        assert metadata["permissions"] == ["read", "write"]
    
    def test_register_duplicate_handler(self, registry, handler):
        """Test registering duplicate method raises error"""
        registry.register("test.method", handler)
        
        with pytest.raises(ValueError) as exc_info:
            registry.register("test.method", handler)
        
        assert "Handler already registered for method: test.method" in str(exc_info.value)
    
    def test_get_handler(self, registry, handler):
        """Test getting registered handler"""
        registry.register("test.method", handler)
        
        retrieved = registry.get_handler("test.method")
        assert retrieved == handler
    
    def test_get_handler_not_found(self, registry):
        """Test getting non-existent handler"""
        assert registry.get_handler("unknown.method") is None
    
    def test_list_methods(self, registry):
        """Test listing all methods"""
        handler1 = SimpleHandler()
        handler2 = PermissionHandler()
        
        registry.register("method1", handler1, description="Method 1")
        registry.register("method2", handler2, description="Method 2", deprecated=True)
        
        methods = registry.list_methods()
        
        assert len(methods) == 2
        assert "method1" in methods
        assert "method2" in methods
        
        assert methods["method1"]["description"] == "Method 1"
        assert methods["method1"]["deprecated"] is False
        assert methods["method1"]["permissions"] == []
        
        assert methods["method2"]["description"] == "Method 2"
        assert methods["method2"]["deprecated"] is True
        assert methods["method2"]["permissions"] == ["read", "write"]
    
    def test_unregister_handler(self, registry, handler):
        """Test unregistering handler"""
        registry.register("test.method", handler)
        assert "test.method" in registry._handlers
        
        registry.unregister("test.method")
        assert "test.method" not in registry._handlers
        assert "test.method" not in registry._handler_metadata
    
    def test_unregister_nonexistent(self, registry):
        """Test unregistering non-existent handler is safe"""
        # Should not raise error
        registry.unregister("unknown.method")


class TestHandlerIntegration:
    """Test handler integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_handler_lifecycle(self):
        """Test complete handler lifecycle"""
        # Create registry and handler
        registry = HandlerRegistry()
        handler = PermissionHandler()
        
        # Register handler
        registry.register(
            "lifecycle.test",
            handler,
            description="Lifecycle test handler"
        )
        
        # Create message and context
        message = MCPMessage(
            id="lifecycle-123",
            method="lifecycle.test",
            params={},
            timestamp=1234567890.0
        )
        
        context = ComplianceContext(
            user_id="test-user",
            organization_id="test-org",
            session_id="test-session",
            request_id="test-request",
            timestamp=1234567890.0,
            ip_address="127.0.0.1",
            user_agent="test-client",
            auth_method="jwt",
            risk_score=0.0
        )
        
        # Get handler from registry
        retrieved_handler = registry.get_handler("lifecycle.test")
        assert retrieved_handler is not None
        
        # Check permissions
        assert retrieved_handler.check_permissions(context) is True
        
        # Validate params
        validated_params = await retrieved_handler.validate_params(message.params)
        assert validated_params == {}
        
        # Handle message
        result = await retrieved_handler.handle(message, context)
        assert result["result"] == "permitted"
        assert result["method"] == "lifecycle.test"
    
    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        """Test multiple handlers in registry"""
        registry = HandlerRegistry()
        
        # Register multiple handlers
        handlers = {
            "handler.simple": SimpleHandler(),
            "handler.permission": PermissionHandler(),
            "handler.parameter": ParameterHandler()
        }
        
        for method, handler in handlers.items():
            registry.register(method, handler)
        
        # Verify all registered
        methods = registry.list_methods()
        assert len(methods) == 3
        
        # Test each handler
        for method, handler in handlers.items():
            retrieved = registry.get_handler(method)
            assert retrieved == handler
            
            # Create test message
            message = MCPMessage(
                id=f"{method}-123",
                method=method,
                params={},
                timestamp=1234567890.0
            )
            
            context = ComplianceContext(
                user_id="test-user",
                organization_id="test-org",
                session_id="test-session",
                request_id="test-request",
                timestamp=1234567890.0,
                ip_address="127.0.0.1",
                user_agent="test-client",
                auth_method="jwt",
                risk_score=0.0
            )
            
            # Handle message
            result = await retrieved.handle(message, context)
            assert "result" in result
            assert result["method"] == method