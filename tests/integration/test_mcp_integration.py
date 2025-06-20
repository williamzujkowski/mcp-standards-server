"""
MCP Protocol Integration Tests
@nist-controls: SA-11, CA-7
@evidence: Integration tests for complete MCP workflows
"""

import asyncio
import json
import time
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from mcp.types import TextContent

from src.compliance.scanner import ComplianceScanner
from src.core.mcp.handlers import HandlerRegistry
from src.core.mcp.models import ComplianceContext, MCPMessage
from src.core.mcp.server import MCPServer, create_app
from src.core.standards.engine import StandardsEngine
from src.core.standards.handlers import (
    AnalyzeCodeHandler,
    GenerateCodeHandler,
    ListMethodsHandler,
    LoadStandardsHandler,
)
from src.server import app as main_app
from src.server import call_tool, initialize_server


@pytest.mark.integration
class TestMCPEndToEnd:
    """Test complete MCP workflows"""
    
    @pytest.fixture
    async def initialized_app(self, test_data_dir, mock_compliance_scanner):
        """Create fully initialized app"""
        with patch('src.server.Path') as mock_path:
            mock_path.return_value.parent.parent = test_data_dir.parent
            
            # Initialize server components
            await initialize_server()
            
            # Patch compliance scanner
            with patch('src.server.compliance_scanner', mock_compliance_scanner):
                yield main_app
    
    @pytest.mark.asyncio
    async def test_complete_tool_workflow(self, initialized_app):
        """Test complete tool workflow from list to execution"""
        # Step 1: List available tools
        from src.server import list_tools
        tools = await list_tools()
        
        assert len(tools) > 0
        tool_names = [tool.name for tool in tools]
        assert "load_standards" in tool_names
        assert "analyze_code" in tool_names
        
        # Step 2: Call load_standards tool
        result = await call_tool("load_standards", {"query": "api security"})
        
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        assert "Standards Loaded" in result[0].text
        
        # Step 3: Call suggest_controls tool
        result = await call_tool("suggest_controls", {
            "description": "Building user authentication system"
        })
        
        assert len(result) == 1
        assert "NIST Control Suggestions" in result[0].text
        assert "IA-2" in result[0].text  # Authentication control
    
    @pytest.mark.asyncio
    async def test_resource_workflow(self, initialized_app):
        """Test resource listing and reading"""
        # List resources
        from src.server import list_resources, read_resource
        resources = await list_resources()
        
        assert len(resources) > 0
        resource_uris = [r.uri for r in resources]
        assert "standards://catalog" in resource_uris
        
        # Read a resource
        from pydantic import AnyUrl
        
        resource = await read_resource(AnyUrl("standards://catalog"))
        
        assert resource.uri == "standards://catalog"
        assert resource.mimeType == "application/json"
        assert "standards" in resource.text
    
    @pytest.mark.asyncio
    async def test_prompt_workflow(self, initialized_app):
        """Test prompt listing and generation"""
        # List prompts
        from src.server import list_prompts, get_prompt
        prompts = await list_prompts()
        
        assert len(prompts) > 0
        prompt_names = [p.name for p in prompts]
        assert "secure-api-design" in prompt_names
        
        # Get a prompt
        prompt = await get_prompt("secure-api-design", {"api_type": "GraphQL"})
        
        assert prompt.description == "Design a secure GraphQL API"
        assert len(prompt.messages) > 0
        assert "GraphQL" in prompt.messages[0].content


@pytest.mark.integration
class TestMCPServerIntegration:
    """Test MCP server integration with all components"""
    
    @pytest.fixture
    def integrated_server(self, test_config, mock_standards_engine, mock_compliance_scanner):
        """Create server with all components integrated"""
        server = MCPServer(test_config)
        
        # Register all handlers
        registry = server.handler_registry
        
        # Standards handlers
        registry.register(
            "load_standards",
            LoadStandardsHandler(mock_standards_engine),
            description="Load standards based on query"
        )
        
        registry.register(
            "analyze_code",
            AnalyzeCodeHandler(),
            description="Analyze code for NIST controls"
        )
        
        registry.register(
            "generate_code",
            GenerateCodeHandler(),
            description="Generate compliant code"
        )
        
        registry.register(
            "list_methods",
            ListMethodsHandler(registry),
            description="List available methods"
        )
        
        return server
    
    @pytest.mark.asyncio
    @pytest.mark.skip(reason="WebSocket implementation needs to be aligned with MCP SDK")
    async def test_websocket_full_workflow(self, integrated_server, valid_jwt_token):
        """Test complete WebSocket workflow"""
        client = TestClient(integrated_server.app)
        
        with patch('src.core.mcp.server.log_security_event'):
            with client.websocket_connect(f"/mcp?token={valid_jwt_token}") as websocket:
                # Test 1: List methods
                list_msg = MCPMessage(
                    id="list-123",
                    method="list_methods",
                    params={},
                    timestamp=time.time()
                )
                
                websocket.send_text(list_msg.json())
                response = websocket.receive_json()
                
                assert response["id"] == "list-123"
                assert response["error"] is None
                assert "methods" in response["result"]
                assert "load_standards" in response["result"]["methods"]
                
                # Test 2: Load standards
                load_msg = MCPMessage(
                    id="load-456",
                    method="load_standards",
                    params={"query": "authentication"},
                    timestamp=time.time()
                )
                
                websocket.send_text(load_msg.json())
                response = websocket.receive_json()
                
                assert response["id"] == "load-456"
                assert response["error"] is None
                assert "standards" in response["result"]
                
                # Test 3: Invalid method
                invalid_msg = MCPMessage(
                    id="invalid-789",
                    method="non_existent_method",
                    params={},
                    timestamp=time.time()
                )
                
                websocket.send_text(invalid_msg.json())
                response = websocket.receive_json()
                
                assert response["id"] == "invalid-789"
                assert response["error"] is not None
                assert "Unknown method" in response["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, integrated_server, valid_jwt_token):
        """Test handling concurrent WebSocket requests"""
        client = TestClient(integrated_server.app)
        
        with patch('src.core.mcp.server.log_security_event'):
            with client.websocket_connect(f"/mcp?token={valid_jwt_token}") as websocket:
                # Send multiple requests without waiting for responses
                messages = []
                for i in range(5):
                    msg = MCPMessage(
                        id=f"concurrent-{i}",
                        method="list_methods",
                        params={},
                        timestamp=time.time()
                    )
                    messages.append(msg)
                    websocket.send_text(msg.json())
                
                # Receive all responses
                responses = []
                for _ in range(5):
                    response = websocket.receive_json()
                    responses.append(response)
                
                # Verify all responses
                response_ids = [r["id"] for r in responses]
                expected_ids = [f"concurrent-{i}" for i in range(5)]
                
                assert set(response_ids) == set(expected_ids)
                assert all(r["error"] is None for r in responses)
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, integrated_server, valid_jwt_token):
        """Test session creation and cleanup"""
        client = TestClient(integrated_server.app)
        
        # Track session creation
        initial_sessions = len(integrated_server.sessions)
        
        with patch('src.core.mcp.server.log_security_event'):
            with client.websocket_connect(f"/mcp?token={valid_jwt_token}") as websocket:
                # Verify session was created
                assert len(integrated_server.sessions) == initial_sessions + 1
                
                # Find the new session
                session_id = None
                for sid in integrated_server.sessions:
                    if sid not in integrated_server.sessions:
                        session_id = sid
                        break
                
                # Send a message to keep session active
                msg = MCPMessage(
                    id="session-test",
                    method="list_methods",
                    params={},
                    timestamp=time.time()
                )
                websocket.send_text(msg.json())
                websocket.receive_json()
        
        # Verify session was cleaned up after disconnect
        assert len(integrated_server.sessions) == initial_sessions


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across components"""
    
    @pytest.mark.asyncio
    async def test_handler_error_propagation(self):
        """Test error propagation from handler to client"""
        # Create server with failing handler
        server = MCPServer({"jwt_secret": "test", "cors_origins": ["*"]})
        
        class FailingHandler:
            required_permissions = []
            
            async def handle(self, message: MCPMessage, context: ComplianceContext) -> dict[str, Any]:
                raise ValueError("Intentional test error")
            
            def check_permissions(self, context: ComplianceContext) -> bool:
                return True
            
            async def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
                return params
        
        server.register_handler("failing_method", FailingHandler())
        
        # Create message and context
        message = MCPMessage(
            id="fail-test",
            method="failing_method",
            params={},
            timestamp=time.time()
        )
        
        context = ComplianceContext(
            user_id="test",
            organization_id="test",
            session_id="test",
            request_id="test",
            timestamp=time.time(),
            ip_address="127.0.0.1",
            user_agent="test",
            auth_method="jwt",
            risk_score=0.0
        )
        
        # Test error handling
        with patch('src.core.mcp.server.log_security_event'):
            with pytest.raises(ValueError) as exc_info:
                await server.handle_message(message, context)
            
            assert "Intentional test error" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_authentication_error_handling(self):
        """Test authentication error scenarios"""
        server = MCPServer({"jwt_secret": "test", "cors_origins": ["*"]})
        client = TestClient(server.app)
        
        # Test 1: No token
        with pytest.raises(Exception):
            with client.websocket_connect("/mcp") as websocket:
                pass
        
        # Test 2: Invalid token
        with pytest.raises(Exception):
            with client.websocket_connect("/mcp?token=invalid-token") as websocket:
                pass
        
        # Test 3: Expired token
        expired_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjoxNjAwMDAwMDAwfQ.test"
        with pytest.raises(Exception):
            with client.websocket_connect(f"/mcp?token={expired_token}") as websocket:
                pass