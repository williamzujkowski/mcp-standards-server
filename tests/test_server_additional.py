"""
Additional Tests for Main Server Module
@nist-controls: SA-11, CA-7
@evidence: Additional unit tests for server.py
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import TextContent
from pydantic import AnyUrl

from src.server import (
    call_tool,
    handle_analyze_code,
    handle_generate_template,
    handle_suggest_controls,
    handle_validate_compliance,
    main,
    read_resource,
)


class TestAdditionalServerFunctions:
    """Test additional server functions for better coverage"""
    
    @pytest.mark.asyncio
    async def test_handle_analyze_code_with_file_path(self):
        """Test analyze code handler with file path"""
        arguments = {
            "code": "def test(): pass",
            "language": "python",
            "file_path": "/test/file.py"
        }
        
        result = await handle_analyze_code(arguments)
        
        assert "Code Analysis Results" in result
        assert "python" in result
        assert "/test/file.py" in result
    
    @pytest.mark.asyncio
    async def test_handle_suggest_controls_with_encryption_keywords(self):
        """Test suggest controls with encryption keywords"""
        arguments = {
            "description": "Implementing data encryption with TLS and secure storage",
            "code_snippet": "encrypt_data(sensitive_info)"
        }
        
        result = await handle_suggest_controls(arguments)
        
        assert "NIST Control Suggestions" in result
        assert "SC-8" in result  # Transmission Confidentiality
        assert "SC-13" in result  # Cryptographic Protection
        assert "SC-28" in result  # Protection of Information at Rest
    
    @pytest.mark.asyncio
    async def test_handle_suggest_controls_with_logging_keywords(self):
        """Test suggest controls with logging keywords"""
        arguments = {
            "description": "Setting up audit logging and monitoring system"
        }
        
        result = await handle_suggest_controls(arguments)
        
        assert "AU-2" in result  # Audit Events
        assert "AU-3" in result  # Content of Audit Records
        assert "AU-12" in result  # Audit Generation
    
    @pytest.mark.asyncio
    async def test_handle_suggest_controls_no_matches(self):
        """Test suggest controls with no matching keywords"""
        arguments = {
            "description": "Generic development task"
        }
        
        result = await handle_suggest_controls(arguments)
        
        assert "No specific controls identified" in result
        assert "AC-3" in result  # Default suggestions
    
    @pytest.mark.asyncio
    async def test_handle_generate_template_unsupported_combination(self):
        """Test generate template with unsupported combination"""
        arguments = {
            "template_type": "logging-setup",
            "language": "go",
            "controls": ["AU-2", "AU-3"]
        }
        
        result = await handle_generate_template(arguments)
        
        assert "Template generation for this combination is not yet implemented" in result
        assert "AU-2, AU-3" in result
    
    @pytest.mark.asyncio
    async def test_handle_validate_compliance_default_profile(self):
        """Test validate compliance with default profile"""
        arguments = {
            "file_path": "/project/src"
        }
        
        result = await handle_validate_compliance(arguments)
        
        assert "Compliance Validation Report" in result
        assert "/project/src" in result
        assert "MODERATE" in result  # Default profile
    
    @pytest.mark.asyncio
    async def test_read_resource_templates(self):
        """Test reading templates resource"""
        with patch('src.server.str') as mock_str:
            mock_str.return_value = "standards://templates"
            
            # This should raise ValueError as it's not implemented
            with pytest.raises(ValueError) as exc_info:
                await read_resource(AnyUrl("standards://templates"))
            
            assert "Resource not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_call_tool_all_branches(self):
        """Test call_tool for all tool types"""
        # Test analyze_code
        result = await call_tool("analyze_code", {"code": "test", "language": "python"})
        assert len(result) == 1
        assert isinstance(result[0], TextContent)
        
        # Test suggest_controls
        result = await call_tool("suggest_controls", {"description": "test"})
        assert len(result) == 1
        
        # Test generate_template
        result = await call_tool("generate_template", {"template_type": "api-endpoint", "language": "python"})
        assert len(result) == 1
        
        # Test validate_compliance
        result = await call_tool("validate_compliance", {"file_path": "/test"})
        assert len(result) == 1
    
    def test_main_function_coverage(self):
        """Test main function"""
        with patch('asyncio.run') as mock_run:
            with patch('builtins.print') as mock_print:
                main()
                
                mock_run.assert_called_once()
                mock_print.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_read_resource_nist_controls(self):
        """Test reading NIST controls resource"""
        resource = await read_resource(AnyUrl("standards://nist-controls"))
        
        assert resource.uri == "standards://nist-controls"
        assert resource.mimeType == "application/json"
        assert "controls" in resource.text