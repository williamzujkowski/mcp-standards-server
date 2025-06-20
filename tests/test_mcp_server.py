"""
Test MCP Server
@nist-controls: SA-11, CA-7
@evidence: Unit tests for MCP server implementation
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent

from src.server import (
    app,
    call_tool,
    get_prompt,
    handle_analyze_code,
    handle_generate_template,
    handle_load_standards,
    handle_suggest_controls,
    handle_validate_compliance,
    initialize_server,
    list_prompts,
    list_resources,
    list_tools,
    main,
    read_resource,
)


@pytest.fixture
def mock_standards_engine():
    """Mock standards engine"""
    from src.core.standards.models import StandardLoadResult
    
    engine = MagicMock()
    engine.load_standards = AsyncMock(
        return_value=StandardLoadResult(
            standards=[
                {"id": "CS.api", "content": "API standards content..." * 50},
                {"id": "SEC.auth", "content": "Authentication standards..." * 50}
            ],
            metadata={"token_count": 1000, "version": "latest"}
        )
    )
    return engine


@pytest.fixture
def mock_compliance_scanner():
    """Mock compliance scanner"""
    scanner = MagicMock()
    return scanner


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


class TestToolHandlers:
    """Test individual tool handlers"""
    
    @pytest.mark.asyncio
    async def test_handle_load_standards_success(self, mock_standards_engine):
        """Test successful standards loading"""
        with patch('src.server.standards_engine', mock_standards_engine):
            arguments = {
                "query": "api security",
                "context": "Building REST API",
                "token_limit": 5000
            }
            
            result = await handle_load_standards(arguments)
            
            assert "Standards Loaded" in result
            assert "Query: api security" in result
            assert "Tokens: 1000" in result
            assert "CS.api" in result
            assert "SEC.auth" in result
    
    @pytest.mark.asyncio
    async def test_handle_load_standards_no_engine(self):
        """Test loading standards without engine"""
        with patch('src.server.standards_engine', None):
            result = await handle_load_standards({"query": "test"})
            assert result == "Standards engine not initialized"
    
    @pytest.mark.asyncio
    async def test_handle_validate_compliance(self):
        """Test validate compliance handler"""
        arguments = {
            "file_path": "/test/path",
            "profile": "high"
        }
        
        result = await handle_validate_compliance(arguments)
        
        assert "Compliance Validation Report" in result
        assert "/test/path" in result
        assert "HIGH" in result
        assert "Critical Findings" in result
        assert "Recommendations" in result
    
    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_standards_engine):
        """Test successful tool call"""
        with patch('src.server.standards_engine', mock_standards_engine):
            result = await call_tool("load_standards", {"query": "test"})
            
            assert len(result) == 1
            assert isinstance(result[0], TextContent)
            assert result[0].type == "text"
            assert "Standards Loaded" in result[0].text
    
    @pytest.mark.asyncio
    async def test_call_tool_unknown(self):
        """Test calling unknown tool"""
        result = await call_tool("unknown_tool", {})
        
        assert len(result) == 1
        assert result[0].text == "Unknown tool: unknown_tool"
    
    @pytest.mark.asyncio
    async def test_call_tool_error_handling(self):
        """Test tool error handling"""
        with patch('src.server.handle_load_standards', side_effect=Exception("Test error")):
            result = await call_tool("load_standards", {})
            
            assert len(result) == 1
            assert "Error: Test error" in result[0].text


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
    
    @pytest.mark.asyncio
    async def test_get_prompt_compliance_checklist(self):
        """Test compliance checklist prompt"""
        prompt = await get_prompt(
            "compliance-checklist",
            {"project_type": "mobile app", "profile": "high"}
        )
        
        assert prompt['description'] == "Compliance checklist for mobile app"
        assert len(prompt['messages']) == 1
        assert "mobile app" in prompt['messages'][0]['content']
        assert "high" in prompt['messages'][0]['content']
    
    @pytest.mark.asyncio
    async def test_get_prompt_unknown(self):
        """Test unknown prompt"""
        with pytest.raises(ValueError) as exc_info:
            await get_prompt("unknown-prompt", {})
        
        assert "Unknown prompt: unknown-prompt" in str(exc_info.value)


class TestServerInitialization:
    """Test server initialization and lifecycle"""
    
    @pytest.mark.asyncio
    async def test_initialize_server(self):
        """Test server initialization"""
        with patch('src.server.Path') as mock_path:
            mock_path.return_value.parent.parent = Path("/fake/path")
            
            await initialize_server()
            
            # Verify globals were set
            from src.server import standards_engine
            assert standards_engine is not None
    
    def test_main_function(self):
        """Test main entry point"""
        with patch('asyncio.run') as mock_run:
            with patch('builtins.print') as mock_print:
                main()
                
                # Verify initialize was called
                mock_run.assert_called_once()
                
                # Verify instructions were printed
                mock_print.assert_called_once()
                assert "MCP Standards Server initialized" in mock_print.call_args[0][0]
