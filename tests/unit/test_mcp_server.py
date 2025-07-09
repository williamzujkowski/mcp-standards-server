"""
Unit tests for the MCP Standards Server implementation.

Tests the core MCP server functionality including tool registration,
request handling, authentication, and error handling.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any

from src.mcp_server import MCPStandardsServer


class TestMCPStandardsServer:
    """Test cases for MCP Standards Server."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            "auth": {
                "enabled": False  # Disable auth for most tests
            },
            "privacy": {
                "enabled": False  # Disable privacy for most tests
            },
            "rate_limit_window": 60,
            "rate_limit_max_requests": 100
        }
    
    @pytest.fixture
    def mock_dependencies(self):
        """Mock all external dependencies."""
        with patch('src.mcp_server.RuleEngine') as mock_rule_engine, \
             patch('src.mcp_server.StandardsSynchronizer') as mock_synchronizer, \
             patch('src.mcp_server.CrossReferencer') as mock_cross_referencer, \
             patch('src.mcp_server.StandardsAnalytics') as mock_analytics, \
             patch('src.mcp_server.AuthManager') as mock_auth_manager, \
             patch('src.mcp_server.PrivacyFilter') as mock_privacy_filter, \
             patch('src.mcp_server.InputValidator') as mock_input_validator, \
             patch('src.mcp_server.RateLimiter') as mock_rate_limiter:
            
            yield {
                'rule_engine': mock_rule_engine,
                'synchronizer': mock_synchronizer,
                'cross_referencer': mock_cross_referencer,
                'analytics': mock_analytics,
                'auth_manager': mock_auth_manager,
                'privacy_filter': mock_privacy_filter,
                'input_validator': mock_input_validator,
                'rate_limiter': mock_rate_limiter
            }
    
    @pytest.fixture
    def server(self, config, mock_dependencies):
        """Create MCP server instance with mocked dependencies."""
        return MCPStandardsServer(config)
    
    def test_server_initialization(self, config, mock_dependencies):
        """Test server initialization with configuration."""
        server = MCPStandardsServer(config)
        
        assert server.config == config
        assert hasattr(server, 'rule_engine')
        assert hasattr(server, 'synchronizer')
        assert hasattr(server, 'cross_referencer')
        assert hasattr(server, 'analytics')
        assert hasattr(server, 'auth_manager')
        assert hasattr(server, 'privacy_filter')
        assert hasattr(server, 'input_validator')
        assert hasattr(server, 'rate_limiter')
        assert hasattr(server, 'metrics')
    
    def test_server_initialization_with_auth_enabled(self, mock_dependencies):
        """Test server initialization with authentication enabled."""
        config = {
            "auth": {
                "enabled": True,
                "secret_key": "test_secret",
                "algorithm": "HS256"
            }
        }
        
        server = MCPStandardsServer(config)
        
        # Verify auth manager was initialized with correct config
        mock_dependencies['auth_manager'].assert_called_once_with(config["auth"])
    
    async def test_get_applicable_standards_success(self, server):
        """Test successful get_applicable_standards operation."""
        # Mock rule engine response
        mock_standards = [
            {
                "id": "test-standard",
                "title": "Test Standard",
                "category": "testing",
                "description": "A test standard"
            }
        ]
        server.rule_engine.get_applicable_standards = AsyncMock(return_value=mock_standards)
        
        # Test arguments
        args = {
            "project_context": {
                "languages": ["python"],
                "frameworks": ["pytest"]
            },
            "requirements": ["testing"],
            "max_results": 10
        }
        
        result = await server._get_applicable_standards(**args)
        
        assert "standards" in result
        assert "metadata" in result
        assert len(result["standards"]) == 1
        assert result["standards"][0]["id"] == "test-standard"
        assert result["metadata"]["total_standards"] == 1
    
    async def test_get_applicable_standards_with_error(self, server):
        """Test get_applicable_standards with error handling."""
        # Mock rule engine to raise exception
        server.rule_engine.get_applicable_standards = AsyncMock(
            side_effect=Exception("Database error")
        )
        
        args = {
            "project_context": {"languages": ["python"]},
            "requirements": ["testing"]
        }
        
        result = await server._get_applicable_standards(**args)
        
        assert "error" in result
        assert "Database error" in result["error"]
    
    async def test_validate_against_standard_success(self, server):
        """Test successful validate_against_standard operation."""
        # Mock dependencies
        server.rule_engine.get_applicable_standards = AsyncMock(return_value=[
            {
                "id": "test-standard",
                "title": "Test Standard",
                "rules": {
                    "test_rule": {
                        "title": "Test Rule",
                        "severity": "error"
                    }
                }
            }
        ])
        
        # Mock analyzer
        with patch('src.analyzers.python_analyzer.PythonAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_code = Mock(return_value=Mock(
                issues=[],
                metrics={"lines_of_code": 10}
            ))
            mock_analyzer_class.return_value = mock_analyzer
            
            args = {
                "standard_id": "test-standard",
                "code_path": "/test/path.py",
                "code_content": "def test(): pass",
                "language": "python"
            }
            
            result = await server._validate_against_standard(**args)
            
            assert "validation_results" in result
            assert "compliance_score" in result
            assert "issues" in result
            assert "suggestions" in result
    
    async def test_search_standards_success(self, server):
        """Test successful search_standards operation."""
        # Mock semantic search engine
        mock_search_engine = Mock()
        mock_search_engine.search = AsyncMock(return_value=[
            {
                "id": "test-standard",
                "title": "Test Standard",
                "score": 0.9,
                "excerpt": "This is a test standard..."
            }
        ])
        
        with patch.object(server, '_get_semantic_search_engine', return_value=mock_search_engine):
            args = {
                "query": "test query",
                "filters": {"category": "testing"},
                "max_results": 5
            }
            
            result = await server._search_standards(**args)
            
            assert "results" in result
            assert "total_results" in result
            assert len(result["results"]) == 1
            assert result["results"][0]["id"] == "test-standard"
    
    async def test_suggest_improvements_success(self, server):
        """Test successful suggest_improvements operation."""
        # Mock dependencies
        server.rule_engine.get_applicable_standards = AsyncMock(return_value=[
            {
                "id": "test-standard",
                "title": "Test Standard",
                "rules": {}
            }
        ])
        
        with patch('src.analyzers.python_analyzer.PythonAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer.analyze_code = Mock(return_value=Mock(
                issues=[
                    Mock(
                        type="style",
                        severity="warning",
                        message="Missing docstring",
                        suggestion="Add a docstring"
                    )
                ]
            ))
            mock_analyzer_class.return_value = mock_analyzer
            
            args = {
                "code_content": "def test(): pass",
                "language": "python",
                "context": {"project_type": "library"}
            }
            
            result = await server._suggest_improvements(**args)
            
            assert "suggestions" in result
            assert "applicable_standards" in result
            assert len(result["suggestions"]) > 0
    
    async def test_get_compliance_mapping_success(self, server):
        """Test successful get_compliance_mapping operation."""
        server.rule_engine.get_applicable_standards = AsyncMock(return_value=[
            {
                "id": "test-standard",
                "title": "Test Standard",
                "compliance_mappings": {
                    "nist": ["AC-1", "AC-2"]
                }
            }
        ])
        
        args = {
            "standard_id": "test-standard",
            "framework": "nist"
        }
        
        result = await server._get_compliance_mapping(**args)
        
        assert "standard_id" in result
        assert "framework" in result
        assert "mappings" in result
        assert result["standard_id"] == "test-standard"
        assert result["framework"] == "nist"
    
    async def test_sync_standards_success(self, server):
        """Test successful sync_standards operation."""
        server.synchronizer.sync_standards = AsyncMock(return_value=Mock(
            status="success",
            synced_files=5,
            failed_files=0
        ))
        
        result = await server._sync_standards()
        
        assert result["status"] == "success"
        assert result["synced_files"] == 5
        assert result["failed_files"] == 0
    
    async def test_rate_limiting(self, server):
        """Test rate limiting functionality."""
        server.rate_limiter.check_rate_limit = Mock(return_value=False)
        
        # Test that rate limiting is checked
        assert server._check_rate_limit("test_user") is False
        server.rate_limiter.check_rate_limit.assert_called_once_with("test_user")
    
    async def test_input_validation(self, server):
        """Test input validation."""
        server.input_validator.validate_project_context = Mock(return_value=True)
        
        context = {"languages": ["python"]}
        assert server._validate_input(context, "project_context") is True
        server.input_validator.validate_project_context.assert_called_once_with(context)
    
    async def test_privacy_filtering(self, server):
        """Test privacy filtering."""
        server.privacy_filter.filter_response = Mock(return_value={"filtered": True})
        
        response = {"data": "sensitive"}
        filtered = server._filter_response(response)
        
        assert filtered == {"filtered": True}
        server.privacy_filter.filter_response.assert_called_once_with(response)
    
    async def test_metrics_recording(self, server):
        """Test metrics recording."""
        server.metrics.record_tool_execution = Mock()
        
        # Mock a tool execution
        with patch.object(server, 'rule_engine') as mock_engine:
            mock_engine.get_applicable_standards = AsyncMock(return_value=[])
            
            await server._get_applicable_standards(
                project_context={"languages": ["python"]},
                requirements=["testing"]
            )
            
            # Verify metrics were recorded
            server.metrics.record_tool_execution.assert_called()
    
    async def test_cross_reference_standards(self, server):
        """Test cross_reference_standards operation."""
        server.cross_referencer.find_related_standards = AsyncMock(return_value=[
            {
                "id": "related-standard",
                "title": "Related Standard",
                "relationship": "complementary",
                "confidence": 0.8
            }
        ])
        
        result = await server._cross_reference_standards(standard_id="test-standard")
        
        assert "related_standards" in result
        assert len(result["related_standards"]) == 1
        assert result["related_standards"][0]["id"] == "related-standard"
    
    async def test_get_analytics(self, server):
        """Test get_analytics operation."""
        server.analytics.get_usage_analytics = AsyncMock(return_value={
            "total_queries": 1000,
            "top_standards": ["python-best-practices"],
            "usage_trends": {"daily": [100, 120, 110]}
        })
        
        result = await server._get_analytics()
        
        assert result["total_queries"] == 1000
        assert "top_standards" in result
        assert "usage_trends" in result
    
    def test_get_semantic_search_engine(self, server):
        """Test semantic search engine initialization."""
        with patch('src.mcp_server.SemanticSearchEngine') as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            
            engine = server._get_semantic_search_engine()
            
            assert engine == mock_engine
            mock_engine_class.assert_called_once()
    
    async def test_error_handling_with_logging(self, server):
        """Test error handling and logging."""
        with patch('src.mcp_server.logger') as mock_logger:
            # Force an error
            server.rule_engine.get_applicable_standards = AsyncMock(
                side_effect=Exception("Test error")
            )
            
            result = await server._get_applicable_standards(
                project_context={"languages": ["python"]},
                requirements=["testing"]
            )
            
            # Verify error was logged
            mock_logger.error.assert_called()
            assert "error" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])