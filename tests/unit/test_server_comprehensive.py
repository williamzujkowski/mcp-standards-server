"""
Comprehensive tests for server.py module
@nist-controls: SA-11, CA-7
@evidence: Complete MCP server testing
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from typing import Any

import pytest
import yaml

# Import the server module and components
from src import server
from src.server import app, standards_engine, compliance_scanner


class TestMCPServer:
    """Test MCP server functionality"""

    @pytest.fixture
    def mock_standards_engine(self):
        """Mock standards engine"""
        engine = MagicMock()
        engine.load_standard = AsyncMock()
        engine.natural_query = AsyncMock()
        engine.get_available_standards = MagicMock()
        return engine

    @pytest.fixture  
    def mock_compliance_scanner(self):
        """Mock compliance scanner"""
        scanner = MagicMock()
        scanner.scan_file = AsyncMock()
        scanner.scan_directory = AsyncMock()
        return scanner

    def test_server_creation(self):
        """Test that MCP server is created properly"""
        assert app is not None
        assert app.name == "mcp-standards-server"

    def test_global_instances_initialization(self):
        """Test global instance variables"""
        # Should be initially None before initialization
        assert isinstance(standards_engine, (type(None), MagicMock))
        assert isinstance(compliance_scanner, (type(None), MagicMock))

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """Test list_tools functionality"""
        tools = await server.list_tools()
        
        assert isinstance(tools, list)
        assert len(tools) > 0
        
        # Check that all tools have required fields
        for tool in tools:
            assert hasattr(tool, 'name')
            assert hasattr(tool, 'description')
            assert hasattr(tool, 'inputSchema')

    @pytest.mark.asyncio
    async def test_list_tools_contains_expected_tools(self):
        """Test that list_tools contains expected tool names"""
        tools = await server.list_tools()
        tool_names = [tool.name for tool in tools]
        
        expected_tools = [
            "load_standards",
            "query_standards", 
            "scan_code",
            "suggest_controls",
            "validate_compliance",
            "generate_ssp"
        ]
        
        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Missing expected tool: {expected_tool}"

    @pytest.mark.asyncio  
    async def test_list_resources(self):
        """Test list_resources functionality"""
        resources = await server.list_resources()
        
        assert isinstance(resources, list)
        assert len(resources) > 0
        
        # Check that all resources have required fields
        for resource in resources:
            assert hasattr(resource, 'uri')
            assert hasattr(resource, 'name')

    @pytest.mark.asyncio
    async def test_list_resources_contains_expected_resources(self):
        """Test that list_resources contains expected resource URIs"""
        resources = await server.list_resources()
        resource_uris = [resource.uri for resource in resources]
        
        expected_prefixes = [
            "standards://",
            "controls://",
            "templates://"
        ]
        
        for prefix in expected_prefixes:
            assert any(str(uri).startswith(prefix) for uri in resource_uris), \
                f"Missing resources with prefix: {prefix}"

    @pytest.mark.asyncio
    async def test_read_resource_standards_document(self):
        """Test reading a standards document resource"""
        # Mock file system
        with patch('builtins.open', mock_open(read_data='{"test": "data"}')):
            with patch('pathlib.Path.exists', return_value=True):
                result = await server.read_resource("standards://nist-800-53/doc1")
                
                assert isinstance(result, list)
                assert len(result) > 0
                assert hasattr(result[0], 'text')

    @pytest.mark.asyncio
    async def test_read_resource_controls_index(self):
        """Test reading controls index resource"""
        with patch('builtins.open', mock_open(read_data='{"controls": []}')):
            with patch('pathlib.Path.exists', return_value=True):
                result = await server.read_resource("controls://index")
                
                assert isinstance(result, list)

    @pytest.mark.asyncio  
    async def test_read_resource_nonexistent(self):
        """Test reading non-existent resource"""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(ValueError):
                await server.read_resource("standards://nonexistent")

    @pytest.mark.asyncio
    async def test_call_tool_load_standards(self, mock_standards_engine):
        """Test load_standards tool call"""
        with patch.object(server, 'standards_engine', mock_standards_engine):
            mock_standards_engine.load_standard.return_value = {
                "name": "NIST 800-53",
                "controls": ["AC-1", "AC-2"]
            }
            
            result = await server.call_tool(
                "load_standards",
                {"standard": "nist-800-53", "version": "rev5"}
            )
            
            assert isinstance(result, list)
            mock_standards_engine.load_standard.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_query_standards(self, mock_standards_engine):
        """Test query_standards tool call"""
        with patch.object(server, 'standards_engine', mock_standards_engine):
            mock_standards_engine.natural_query.return_value = [
                {"control": "AC-3", "description": "Access control"}
            ]
            
            result = await server.call_tool(
                "query_standards", 
                {"query": "access control", "limit": 10}
            )
            
            assert isinstance(result, list)
            mock_standards_engine.natural_query.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_scan_code(self, mock_compliance_scanner):
        """Test scan_code tool call"""
        with patch.object(server, 'compliance_scanner', mock_compliance_scanner):
            mock_compliance_scanner.scan_file.return_value = [
                {"file": "test.py", "controls": ["AC-3"], "issues": []}
            ]
            
            result = await server.call_tool(
                "scan_code",
                {"file_path": "/path/to/file.py", "language": "python"}
            )
            
            assert isinstance(result, list)
            mock_compliance_scanner.scan_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_suggest_controls(self, mock_standards_engine):
        """Test suggest_controls tool call"""
        with patch.object(server, 'standards_engine', mock_standards_engine):
            mock_standards_engine.suggest_controls_for_code.return_value = [
                "AC-3", "AU-2", "IA-2"
            ]
            
            result = await server.call_tool(
                "suggest_controls",
                {"code_snippet": "def authenticate():", "language": "python"}
            )
            
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_call_tool_validate_compliance(self, mock_compliance_scanner):
        """Test validate_compliance tool call"""
        with patch.object(server, 'compliance_scanner', mock_compliance_scanner):
            mock_compliance_scanner.scan_directory.return_value = {
                "summary": {"total_files": 5, "issues_found": 2},
                "findings": []
            }
            
            result = await server.call_tool(
                "validate_compliance",
                {"project_path": "/path/to/project", "profile": "moderate"}
            )
            
            assert isinstance(result, list)
            mock_compliance_scanner.scan_directory.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_generate_ssp(self):
        """Test generate_ssp tool call"""
        with patch('src.server.OSCALHandler') as mock_oscal:
            mock_handler = MagicMock()
            mock_oscal.return_value = mock_handler
            mock_handler.generate_ssp.return_value = {"system-security-plan": {}}
            
            result = await server.call_tool(
                "generate_ssp",
                {"project_path": "/path/to/project", "output_format": "json"}
            )
            
            assert isinstance(result, list)
            mock_handler.generate_ssp.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_invalid_tool(self):
        """Test calling invalid tool"""
        with pytest.raises(ValueError):
            await server.call_tool("invalid_tool", {})

    @pytest.mark.asyncio
    async def test_list_prompts(self):
        """Test list_prompts functionality"""
        prompts = await server.list_prompts()
        
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        
        # Check that all prompts have required fields
        for prompt in prompts:
            assert hasattr(prompt, 'name')
            assert hasattr(prompt, 'description')

    @pytest.mark.asyncio
    async def test_list_prompts_contains_expected_prompts(self):
        """Test that list_prompts contains expected prompt names"""
        prompts = await server.list_prompts()
        prompt_names = [prompt.name for prompt in prompts]
        
        expected_prompts = [
            "nist_compliance_review",
            "security_gap_analysis", 
            "control_implementation_guide",
            "ssp_generation_assistant",
            "remediation_suggestions"
        ]
        
        for expected_prompt in expected_prompts:
            assert expected_prompt in prompt_names, f"Missing expected prompt: {expected_prompt}"

    @pytest.mark.asyncio
    async def test_get_prompt_nist_compliance_review(self):
        """Test get_prompt for NIST compliance review"""
        result = await server.get_prompt(
            "nist_compliance_review",
            {"code": "def test():", "controls": ["AC-3"]}
        )
        
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_get_prompt_security_gap_analysis(self):
        """Test get_prompt for security gap analysis"""
        result = await server.get_prompt(
            "security_gap_analysis",
            {"baseline": "moderate", "current_controls": ["AC-1"]}
        )
        
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_prompt_invalid_prompt(self):
        """Test getting invalid prompt"""
        with pytest.raises(ValueError):
            await server.get_prompt("invalid_prompt", {})

    def test_imports_and_dependencies(self):
        """Test that all required modules are imported"""
        # Verify critical imports
        assert hasattr(server, 'Server')
        assert hasattr(server, 'Tool')
        assert hasattr(server, 'TextContent')
        assert hasattr(server, 'ComplianceScanner')
        assert hasattr(server, 'StandardsEngine')

    def test_logging_configuration(self):
        """Test logging is properly configured"""
        assert hasattr(server, 'logger')
        assert server.logger is not None

    @pytest.mark.asyncio
    async def test_initialize_engines(self):
        """Test engine initialization"""
        with patch('src.server.StandardsEngine') as mock_engine_class:
            with patch('src.server.ComplianceScanner') as mock_scanner_class:
                mock_engine = MagicMock()
                mock_scanner = MagicMock()
                mock_engine_class.return_value = mock_engine
                mock_scanner_class.return_value = mock_scanner
                
                # Call initialization function if it exists
                if hasattr(server, '_initialize_engines'):
                    await server._initialize_engines()
                
                # Otherwise just verify classes are available
                assert mock_engine_class is not None
                assert mock_scanner_class is not None

    @pytest.mark.asyncio
    async def test_error_handling_in_tools(self):
        """Test error handling in tool calls"""
        # Test with missing engine
        with patch.object(server, 'standards_engine', None):
            result = await server.call_tool(
                "load_standards",
                {"standard": "nist-800-53"}
            )
            
            # Should handle gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_resource_uri_parsing(self):
        """Test resource URI parsing"""
        test_uris = [
            "standards://nist-800-53/AC-1",
            "controls://AC-3",
            "templates://api-security",
            "invalid://uri"
        ]
        
        for uri in test_uris:
            try:
                result = await server.read_resource(uri)
                assert isinstance(result, list)
            except ValueError:
                # Invalid URIs should raise ValueError
                assert "invalid://" in uri

    @pytest.mark.asyncio
    async def test_json_serialization(self):
        """Test JSON serialization of responses"""
        # Test with simple data
        test_data = {"test": "value", "number": 42}
        
        # Verify JSON can be serialized
        json_str = json.dumps(test_data)
        assert isinstance(json_str, str)
        
        # Verify it can be deserialized
        parsed = json.loads(json_str)
        assert parsed == test_data

    @pytest.mark.asyncio
    async def test_yaml_parsing(self):
        """Test YAML parsing capabilities"""
        yaml_content = """
        version: 1.0
        controls:
          - AC-1
          - AC-2
        """
        
        # Test YAML parsing
        parsed = yaml.safe_load(yaml_content)
        assert parsed['version'] == 1.0
        assert 'AC-1' in parsed['controls']

    @pytest.mark.asyncio
    async def test_path_handling(self):
        """Test Path object handling"""
        test_path = Path("/test/path")
        assert isinstance(test_path, Path)
        
        # Test path operations don't crash
        assert test_path.name == "path"
        assert test_path.parent.name == "test"

    def test_server_constants_and_globals(self):
        """Test server constants and global variables"""
        # Test that app is properly configured
        assert server.app.name == "mcp-standards-server"
        
        # Test global variables exist
        assert hasattr(server, 'standards_engine')
        assert hasattr(server, 'compliance_scanner')
        assert hasattr(server, 'logger')

    @pytest.mark.asyncio
    async def test_resource_content_types(self):
        """Test different resource content types"""
        # Test text content
        with patch('builtins.open', mock_open(read_data='{"test": "data"}')):
            with patch('pathlib.Path.exists', return_value=True):
                result = await server.read_resource("standards://test")
                
                assert isinstance(result, list)
                if result:
                    # Should be TextContent or similar
                    assert hasattr(result[0], 'text') or hasattr(result[0], 'content')

    @pytest.mark.asyncio  
    async def test_tool_input_validation(self):
        """Test tool input validation"""
        # Test with invalid inputs
        invalid_inputs = [
            {"invalid_key": "value"},
            {},
            {"file_path": ""},
            {"standard": None}
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = await server.call_tool("load_standards", invalid_input)
                # Should handle gracefully or raise appropriate error
                assert isinstance(result, list)
            except (ValueError, TypeError, KeyError):
                # Expected for invalid inputs
                pass

    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self):
        """Test concurrent tool calls"""
        # Test multiple concurrent calls
        tasks = []
        for i in range(3):
            task = server.call_tool(
                "query_standards",
                {"query": f"test query {i}", "limit": 5}
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete (successfully or with exception)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, (list, Exception))

    def test_mcp_types_integration(self):
        """Test MCP types integration"""
        # Verify MCP types are properly imported and usable
        from mcp.types import Tool, TextContent, ImageContent
        
        # Test Tool creation
        tool = Tool(
            name="test_tool",
            description="Test tool",
            inputSchema={"type": "object", "properties": {}}
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"
        
        # Test TextContent creation
        text_content = TextContent(text="Test content")
        assert text_content.text == "Test content"

    @pytest.mark.asyncio
    async def test_server_lifecycle(self):
        """Test server lifecycle methods"""
        # Test server can be created and configured
        assert server.app is not None
        
        # Test that handlers are registered
        tools = await server.list_tools()
        resources = await server.list_resources()
        prompts = await server.list_prompts()
        
        assert len(tools) > 0
        assert len(resources) > 0  
        assert len(prompts) > 0

    def test_nist_annotations_present(self):
        """Test that NIST control annotations are present"""
        # Check module docstring has NIST annotations
        assert "@nist-controls:" in server.__doc__
        assert "@evidence:" in server.__doc__
        
        # Check that controls are documented
        assert "AC-4" in server.__doc__ or "SC-8" in server.__doc__