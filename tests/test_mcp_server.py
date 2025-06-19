"""
Test MCP Server
@nist-controls: SA-11, CA-7
@evidence: Unit tests for MCP server implementation
"""
import pytest
from typing import Dict, Any, List

from src.server import (
    list_tools, handle_load_standards, handle_suggest_controls,
    handle_analyze_code, handle_generate_template
)
from src.standards.models import StandardQuery


class TestMCPTools:
    """Test MCP tool definitions"""
    
    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test that tools are properly defined"""
        tools = await list_tools()
        
        assert len(tools) > 0
        
        # Check for essential tools
        tool_names = [tool.name for tool in tools]
        assert "load_standards" in tool_names
        assert "analyze_code" in tool_names
        assert "suggest_controls" in tool_names
        assert "generate_template" in tool_names
        assert "validate_compliance" in tool_names
        
        # Check tool structure
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')
            assert isinstance(tool.inputSchema, dict)
            assert tool.inputSchema.get('type') == 'object'
    
    @pytest.mark.asyncio
    async def test_handle_suggest_controls(self):
        """Test suggest_controls handler"""
        arguments = {
            "description": "Building user authentication system",
            "code_snippet": ""
        }
        
        result = await handle_suggest_controls(arguments)
        
        assert isinstance(result, str)
        assert "NIST Control Suggestions" in result
        assert "IA-2" in result  # Should suggest authentication controls
        assert "IA-5" in result  # Should suggest authenticator management
    
    @pytest.mark.asyncio
    async def test_handle_analyze_code(self):
        """Test analyze_code handler"""
        arguments = {
            "code": """
def authenticate_user(username, password):
    # Check user credentials
    user = db.get_user(username)
    if user and user.check_password(password):
        log_event('user_login', username)
        return generate_token(user)
    return None
            """,
            "language": "python"
        }
        
        result = await handle_analyze_code(arguments)
        
        assert isinstance(result, str)
        assert "Code Analysis Results" in result
        assert "python" in result.lower()
        assert "Detected NIST Controls" in result
    
    @pytest.mark.asyncio
    async def test_handle_generate_template(self):
        """Test generate_template handler"""
        arguments = {
            "template_type": "api-endpoint",
            "language": "python",
            "controls": ["AC-3", "AU-2"]
        }
        
        result = await handle_generate_template(arguments)
        
        assert isinstance(result, str)
        assert "@nist-controls:" in result
        assert "AC-3" in result
        assert "def" in result  # Python function
        assert "async" in result  # Async endpoint


class TestMCPResources:
    """Test MCP resource handling"""
    
    @pytest.mark.asyncio
    async def test_list_resources(self):
        """Test resource listing"""
        from src.server import list_resources
        
        resources = await list_resources()
        
        assert len(resources) > 0
        
        # Check resource structure
        for resource in resources:
            assert 'uri' in resource
            assert 'name' in resource
            assert 'description' in resource
            assert 'mimeType' in resource
            
        # Check specific resources
        uris = [r['uri'] for r in resources]
        assert "standards://catalog" in uris
        assert "standards://nist-controls" in uris


class TestMCPPrompts:
    """Test MCP prompt handling"""
    
    @pytest.mark.asyncio
    async def test_list_prompts(self):
        """Test prompt listing"""
        from src.server import list_prompts
        
        prompts = await list_prompts()
        
        assert len(prompts) > 0
        
        # Check prompt structure
        for prompt in prompts:
            assert 'name' in prompt
            assert 'description' in prompt
            assert 'arguments' in prompt
            
        # Check specific prompts
        names = [p['name'] for p in prompts]
        assert "secure-api-design" in names
        assert "compliance-checklist" in names
    
    @pytest.mark.asyncio
    async def test_get_prompt(self):
        """Test prompt generation"""
        from src.server import get_prompt
        
        # Test secure-api-design prompt
        prompt = await get_prompt("secure-api-design", {"api_type": "REST"})
        
        assert 'description' in prompt
        assert 'messages' in prompt
        assert len(prompt['messages']) > 0
        assert prompt['messages'][0]['role'] == 'user'
        assert 'REST' in prompt['messages'][0]['content']
        assert 'NIST' in prompt['messages'][0]['content']