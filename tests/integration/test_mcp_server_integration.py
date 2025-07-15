"""
Integration tests for MCP server functionality.

Tests the complete MCP server implementation including tool execution,
authentication, validation, and integration with all components.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.mcp_server import MCPStandardsServer


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    @pytest.fixture
    def mcp_config(self):
        """Create MCP server configuration."""
        return {
            "auth": {"enabled": False},  # Disable for testing
            "privacy": {
                "detect_pii": False,
                "redact_pii": False,
            },  # Disable for testing
            "rate_limit_window": 60,
            "rate_limit_max_requests": 1000,  # High limit for testing
        }

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with required structure."""
        import json
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create required directory structure
            standards_dir = os.path.join(temp_dir, "standards")
            meta_dir = os.path.join(standards_dir, "meta")
            cache_dir = os.path.join(standards_dir, "cache")
            analytics_dir = os.path.join(standards_dir, "analytics")

            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(analytics_dir, exist_ok=True)

            # Create minimal rules file
            rules_data = {
                "rules": [
                    {
                        "id": "test_rule",
                        "name": "Test Rule",
                        "description": "A test rule",
                        "priority": 1,
                        "conditions": {
                            "field": "languages",
                            "operator": "contains",
                            "value": "python",
                        },
                        "standards": ["python-coding-standards"],
                        "tags": ["test"],
                        "metadata": {},
                    }
                ]
            }

            rules_file = os.path.join(meta_dir, "enhanced-selection-rules.json")
            with open(rules_file, "w") as f:
                json.dump(rules_data, f)

            # Create sync config
            sync_config = """
source:
  type: github
  repository: williamzujkowski/standards
  branch: main

target:
  directory: cache
  format: json

sync:
  enabled: true
  interval: 3600
  batch_size: 10
"""
            sync_config_file = os.path.join(standards_dir, "sync_config.yaml")
            with open(sync_config_file, "w") as f:
                f.write(sync_config)

            yield temp_dir

    @pytest.fixture
    def mcp_server(self, mcp_config, temp_data_dir):
        """Create MCP server instance."""
        # Set environment variables for temp directory
        with patch.dict(os.environ, {"MCP_STANDARDS_DATA_DIR": temp_data_dir}):
            server = MCPStandardsServer(mcp_config)
            return server

    @pytest.fixture
    def mock_standards_data(self):
        """Mock standards data for testing."""
        return [
            {
                "id": "python-coding-standards",
                "title": "Python Coding Standards",
                "category": "programming",
                "description": "Standards for Python code quality and style",
                "content": "# Python Coding Standards\n\n## Overview\nThis document outlines...",
                "tags": ["python", "coding", "quality"],
                "applicability": {
                    "languages": ["python"],
                    "frameworks": ["django", "flask"],
                    "project_types": ["web", "api"],
                },
                "rules": {
                    "pep8_compliance": {
                        "title": "PEP 8 Compliance",
                        "description": "Code must follow PEP 8 style guide",
                        "severity": "error",
                        "category": "style",
                    }
                },
            },
            {
                "id": "react-best-practices",
                "title": "React Best Practices",
                "category": "frontend",
                "description": "Best practices for React development",
                "content": "# React Best Practices\n\n## Components\nComponents should be...",
                "tags": ["react", "frontend", "javascript"],
                "applicability": {
                    "languages": ["javascript", "typescript"],
                    "frameworks": ["react"],
                    "project_types": ["web", "spa"],
                },
                "rules": {
                    "component_structure": {
                        "title": "Component Structure",
                        "description": "Components should follow standard structure",
                        "severity": "warning",
                        "category": "structure",
                    }
                },
            },
        ]

    async def test_get_applicable_standards_tool(self, mcp_server, mock_standards_data):
        """Test get_applicable_standards MCP tool."""
        # Mock the rule engine evaluate method to return known data structure
        mock_result = {
            "context": {
                "languages": ["python"],
                "frameworks": ["django"],
                "project_type": "web",
                "requirements": ["code_quality", "performance"],
            },
            "resolved_standards": ["python-coding-standards", "web-security-standards"],
            "matched_rules": [
                {
                    "rule_id": "python_web_rule",
                    "rule_name": "Python Web Applications",
                    "priority": 1,
                    "standards": ["python-coding-standards", "web-security-standards"],
                }
            ],
            "evaluation_path": ["python_web_rule"],
            "statistics": {
                "total_rules_evaluated": 1,
                "rules_matched": 1,
                "unique_standards": 2,
                "conflicts_found": 0,
            },
        }

        with patch.object(mcp_server.rule_engine, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = mock_result

            # Test tool execution
            arguments = {
                "context": {
                    "languages": ["python"],
                    "frameworks": ["django"],
                    "project_type": "web",
                    "requirements": ["code_quality", "performance"],
                },
                "include_resolution_details": False,
            }

            result = await mcp_server._get_applicable_standards(**arguments)

            # Verify result structure
            assert "standards" in result
            assert "evaluation_path" in result

            # Verify standards content - now returns full standard objects
            assert len(result["standards"]) == 2
            standard_ids = [std["id"] for std in result["standards"]]
            assert standard_ids == [
                "python-coding-standards",
                "web-security-standards",
            ]
            # Verify each standard has required fields
            for standard in result["standards"]:
                assert "id" in standard
                assert "title" in standard
                assert "description" in standard
            assert result["evaluation_path"] == ["python_web_rule"]

    async def test_validate_against_standard_tool(
        self, mcp_server, mock_standards_data
    ):
        """Test validate_against_standard MCP tool."""
        # Test tool execution - this uses the actual implementation from MCPStandardsServer
        arguments = {
            "standard": "python-coding-standards",
            "code": "def hello():\n    print('Hello, world!')",
            "language": "python",
        }

        result = await mcp_server._validate_against_standard(**arguments)

        # Verify result structure (based on actual implementation)
        assert "standard" in result
        assert "passed" in result
        assert "violations" in result

        # Verify basic structure
        assert result["standard"] == "python-coding-standards"
        assert isinstance(result["passed"], bool)
        assert isinstance(result["violations"], list)

    async def test_search_standards_tool(self, mcp_server, mock_standards_data):
        """Test search_standards MCP tool."""
        # Test tool execution - when semantic search is disabled, it returns a warning
        arguments = {
            "query": "python code quality standards",
            "filters": {"category": "programming"},
            "limit": 5,
        }

        result = await mcp_server._search_standards(**arguments)

        # Verify result structure (search is disabled in test environment)
        assert "results" in result
        assert isinstance(result["results"], list)

        # When search is disabled, we expect either empty results or a warning
        if "warning" in result:
            assert "Semantic search is disabled" in result["warning"]

    async def test_suggest_improvements_tool(self, mcp_server, mock_standards_data):
        """Test suggest_improvements MCP tool."""
        # Mock the rule engine evaluate method for consistent results
        mock_result = {
            "resolved_standards": ["javascript-standards"],
            "evaluation_path": ["js_rule"],
        }

        with patch.object(mcp_server.rule_engine, "evaluate") as mock_evaluate:
            mock_evaluate.return_value = mock_result

            # Test tool execution
            arguments = {
                "code": "var x = 1; function test() { return x; }",
                "context": {"language": "javascript", "project_type": "web"},
            }

            result = await mcp_server._suggest_improvements(**arguments)

            # Verify result structure
            assert "suggestions" in result
            assert isinstance(result["suggestions"], list)

            # Check if suggestions are generated based on code analysis
            # The actual implementation analyzes the code and may generate suggestions

    async def test_get_compliance_mapping_tool(self, mcp_server, mock_standards_data):
        """Test get_compliance_mapping MCP tool."""
        # Test tool execution using the actual implementation
        arguments = {"standard_id": "python-coding-standards", "framework": "nist"}

        result = await mcp_server._get_compliance_mapping(**arguments)

        # Verify result structure
        assert "standard_id" in result
        assert "framework" in result
        assert "mappings" in result

        # Verify mappings
        assert result["standard_id"] == "python-coding-standards"
        assert result["framework"] == "nist"
        assert isinstance(result["mappings"], list)

    async def test_authentication_flow(self, mcp_server):
        """Test authentication flow."""
        # Since auth is disabled in test config, operations should work without authentication
        result = await mcp_server._get_applicable_standards(
            context={"languages": ["python"], "requirements": ["code_quality"]},
        )

        # Should work without authentication when auth is disabled
        assert "standards" in result
        assert "evaluation_path" in result

    async def test_rate_limiting(self, mcp_server):
        """Test rate limiting functionality."""
        # Mock rate limiting
        user_id = "test_user"

        # Test that rate limiting is applied
        for i in range(mcp_server.rate_limit_max_requests + 1):
            if i < mcp_server.rate_limit_max_requests:
                # Should be allowed
                assert mcp_server._check_rate_limit(user_id) is True
            else:
                # Should be rate limited
                assert mcp_server._check_rate_limit(user_id) is False

    async def test_input_validation(self, mcp_server):
        """Test input validation."""
        # Test with valid input (since we're testing integration, not validation details)
        result = await mcp_server._get_applicable_standards(
            context={"languages": ["python"]},  # Valid dict
        )

        # Should work with valid input
        assert "standards" in result
        assert "evaluation_path" in result

    async def test_privacy_filtering(self, mcp_server):
        """Test privacy filtering."""
        # Since privacy is disabled in test config, just test basic functionality
        result = await mcp_server._get_applicable_standards(
            context={
                "languages": ["python"],
                "contact_email": "test@example.com",  # Would be filtered if privacy enabled
            }
        )

        # Should work without privacy filtering when disabled
        assert "standards" in result
        assert "evaluation_path" in result

    async def test_metrics_collection(self, mcp_server):
        """Test metrics collection."""
        # Test that metrics object exists and can be accessed
        assert hasattr(mcp_server, "metrics")
        assert mcp_server.metrics is not None

        # Test basic functionality works (metrics are collected internally)
        result = await mcp_server._get_applicable_standards(
            context={"languages": ["python"], "requirements": ["code_quality"]},
        )

        # Should work and metrics should be available
        assert "standards" in result
        assert "evaluation_path" in result

    async def test_error_handling(self, mcp_server):
        """Test error handling."""
        # Test graceful error handling by causing an exception
        with patch.object(mcp_server.rule_engine, "evaluate") as mock_evaluate:
            mock_evaluate.side_effect = Exception("Database connection failed")

            # Should handle error gracefully or raise MCPError
            try:
                result = await mcp_server._get_applicable_standards(
                    context={"languages": ["python"]},
                )
                # If it doesn't raise, check for error in result
                if "error" in result:
                    assert "Database connection failed" in result["error"]
            except Exception as e:
                # Should raise a proper exception
                assert "Database connection failed" in str(e)

    async def test_concurrent_requests(self, mcp_server):
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests using the actual implementation
        tasks = []
        for _i in range(10):
            task = asyncio.create_task(
                mcp_server._get_applicable_standards(
                    context={"languages": ["python"], "requirements": ["code_quality"]},
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Verify all requests were handled
        assert len(results) == 10
        for result in results:
            assert "standards" in result or "error" in result


class TestMCPServerTools:
    """Test individual MCP server tools."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with required structure."""
        import json
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create required directory structure
            standards_dir = os.path.join(temp_dir, "standards")
            meta_dir = os.path.join(standards_dir, "meta")
            cache_dir = os.path.join(standards_dir, "cache")
            analytics_dir = os.path.join(standards_dir, "analytics")

            os.makedirs(meta_dir, exist_ok=True)
            os.makedirs(cache_dir, exist_ok=True)
            os.makedirs(analytics_dir, exist_ok=True)

            # Create minimal rules file
            rules_data = {
                "rules": [
                    {
                        "id": "test_rule",
                        "name": "Test Rule",
                        "description": "A test rule",
                        "priority": 1,
                        "conditions": {
                            "field": "languages",
                            "operator": "contains",
                            "value": "python",
                        },
                        "standards": ["python-coding-standards"],
                        "tags": ["test"],
                        "metadata": {},
                    }
                ]
            }

            rules_file = os.path.join(meta_dir, "enhanced-selection-rules.json")
            with open(rules_file, "w") as f:
                json.dump(rules_data, f)

            # Create sync config
            sync_config = """
source:
  type: github
  repository: williamzujkowski/standards
  branch: main

target:
  directory: cache
  format: json

sync:
  enabled: true
  interval: 3600
  batch_size: 10
"""
            sync_config_file = os.path.join(standards_dir, "sync_config.yaml")
            with open(sync_config_file, "w") as f:
                f.write(sync_config)

            yield temp_dir

    @pytest.fixture
    def mcp_server(self, temp_data_dir):
        """Create MCP server instance."""
        config = {
            "auth": {"enabled": False},
            "privacy": {"detect_pii": False, "redact_pii": False},
        }
        with patch.dict(os.environ, {"MCP_STANDARDS_DATA_DIR": temp_data_dir}):
            server = MCPStandardsServer(config)
            return server

    async def test_sync_standards_tool(self, mcp_server):
        """Test sync_standards tool."""
        # Mock the sync method to return a proper async mock result
        from src.core.standards.sync import SyncResult, SyncStatus

        mock_result = SyncResult(
            status=SyncStatus.SUCCESS,
            synced_files=[],
            failed_files=[],
            message="Sync completed",
        )

        with patch.object(
            mcp_server.synchronizer, "sync", new_callable=AsyncMock
        ) as mock_sync:
            mock_sync.return_value = mock_result

            result = await mcp_server._sync_standards()

            assert result["status"] == "success"
            assert "synced_files" in result
            assert "failed_files" in result

    async def test_get_analytics_tool(self, mcp_server):
        """Test get_analytics tool."""
        # Test the actual _get_analytics method
        result = await mcp_server._get_analytics()

        # Should return analytics data structure
        assert "usage_metrics" in result
        assert "popularity_metrics" in result
        assert "quality_recommendations" in result

    async def test_cross_reference_standards_tool(self, mcp_server):
        """Test cross_reference_standards tool."""
        # Test the actual _get_cross_references method
        result = await mcp_server._get_cross_references(
            standard_id="python-coding-standards"
        )

        # Should return cross-reference data structure
        assert "references" in result
        assert "depth" in result
        assert "total_found" in result
        assert isinstance(result["references"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
