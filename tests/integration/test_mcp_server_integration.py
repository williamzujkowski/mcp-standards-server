"""
Integration tests for MCP server functionality.

Tests the complete MCP server implementation including tool execution,
authentication, validation, and integration with all components.
"""

import asyncio
from unittest.mock import Mock, patch

import pytest

from src.mcp_server import MCPStandardsServer


class TestMCPServerIntegration:
    """Integration tests for MCP server."""

    @pytest.fixture
    def mcp_config(self):
        """Create MCP server configuration."""
        return {
            "auth": {
                "enabled": True,
                "secret_key": "test_secret_key",
                "algorithm": "HS256",
            },
            "privacy": {
                "enabled": True,
                "pii_patterns": ["email", "phone", "ssn"],
                "anonymize_logs": True,
            },
            "rate_limit_window": 60,
            "rate_limit_max_requests": 100,
        }

    @pytest.fixture
    def mcp_server(self, mcp_config):
        """Create MCP server instance."""
        with (
            patch("src.mcp_server.RuleEngine"),
            patch("src.mcp_server.StandardsSynchronizer"),
            patch("src.mcp_server.CrossReferencer"),
            patch("src.mcp_server.StandardsAnalytics"),
        ):

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
        with patch.object(
            mcp_server.rule_engine, "get_applicable_standards"
        ) as mock_get_standards:
            mock_get_standards.return_value = mock_standards_data

            # Test tool execution
            arguments = {
                "project_context": {
                    "languages": ["python"],
                    "frameworks": ["django"],
                    "project_type": "web",
                },
                "requirements": ["code_quality", "performance"],
                "max_results": 10,
            }

            result = await mcp_server._get_applicable_standards(**arguments)

            # Verify result structure
            assert "standards" in result
            assert "metadata" in result
            assert len(result["standards"]) == 2

            # Verify standards content
            standard = result["standards"][0]
            assert standard["id"] == "python-coding-standards"
            assert standard["title"] == "Python Coding Standards"
            assert "applicability" in standard

            # Verify metadata
            assert result["metadata"]["total_standards"] == 2
            assert result["metadata"]["selection_criteria"]["languages"] == ["python"]

    async def test_validate_against_standard_tool(
        self, mcp_server, mock_standards_data
    ):
        """Test validate_against_standard MCP tool."""
        with (
            patch.object(
                mcp_server.rule_engine, "get_applicable_standards"
            ) as mock_get_standards,
            patch("src.analyzers.python_analyzer.PythonAnalyzer") as mock_analyzer,
        ):

            # Mock standards data
            mock_get_standards.return_value = [mock_standards_data[0]]

            # Mock analyzer result
            mock_analyzer_instance = Mock()
            mock_analyzer_instance.analyze_code.return_value = Mock(
                issues=[
                    Mock(
                        type="style",
                        severity="error",
                        message="Line too long",
                        line_number=10,
                        column=80,
                        suggestion="Break line at 79 characters",
                    )
                ],
                metrics={"lines_of_code": 100, "complexity": 5, "test_coverage": 85},
            )
            mock_analyzer.return_value = mock_analyzer_instance

            # Test tool execution
            arguments = {
                "standard_id": "python-coding-standards",
                "code_path": "/path/to/code.py",
                "code_content": "def hello():\n    print('Hello, world!')",
                "language": "python",
            }

            result = await mcp_server._validate_against_standard(**arguments)

            # Verify result structure
            assert "validation_results" in result
            assert "compliance_score" in result
            assert "issues" in result
            assert "suggestions" in result

            # Verify validation results
            validation = result["validation_results"][0]
            assert validation["rule_id"] == "pep8_compliance"
            assert validation["status"] in ["passed", "failed", "warning"]

            # Verify issues
            assert len(result["issues"]) == 1
            issue = result["issues"][0]
            assert issue["message"] == "Line too long"
            assert issue["line_number"] == 10

    async def test_search_standards_tool(self, mcp_server, mock_standards_data):
        """Test search_standards MCP tool."""
        with patch.object(mcp_server, "_get_semantic_search_engine") as mock_search:
            mock_search_engine = Mock()
            mock_search_engine.search.return_value = [
                {
                    "id": "python-coding-standards",
                    "title": "Python Coding Standards",
                    "score": 0.95,
                    "excerpt": "Standards for Python code quality and style...",
                    "metadata": {"category": "programming"},
                }
            ]
            mock_search.return_value = mock_search_engine

            # Test tool execution
            arguments = {
                "query": "python code quality standards",
                "filters": {"category": "programming"},
                "max_results": 5,
            }

            result = await mcp_server._search_standards(**arguments)

            # Verify result structure
            assert "results" in result
            assert "query" in result
            assert "total_results" in result

            # Verify search results
            assert len(result["results"]) == 1
            search_result = result["results"][0]
            assert search_result["id"] == "python-coding-standards"
            assert search_result["score"] == 0.95
            assert "excerpt" in search_result

    async def test_suggest_improvements_tool(self, mcp_server, mock_standards_data):
        """Test suggest_improvements MCP tool."""
        with (
            patch.object(
                mcp_server.rule_engine, "get_applicable_standards"
            ) as mock_get_standards,
            patch("src.analyzers.python_analyzer.PythonAnalyzer") as mock_analyzer,
        ):

            # Mock standards and analyzer
            mock_get_standards.return_value = [mock_standards_data[0]]
            mock_analyzer_instance = Mock()
            mock_analyzer_instance.analyze_code.return_value = Mock(
                issues=[
                    Mock(
                        type="style",
                        severity="warning",
                        message="Missing docstring",
                        line_number=1,
                        suggestion="Add docstring to function",
                    )
                ]
            )
            mock_analyzer.return_value = mock_analyzer_instance

            # Test tool execution
            arguments = {
                "code_content": "def hello():\n    print('Hello, world!')",
                "language": "python",
                "context": {"project_type": "web", "frameworks": ["django"]},
            }

            result = await mcp_server._suggest_improvements(**arguments)

            # Verify result structure
            assert "suggestions" in result
            assert "applicable_standards" in result
            assert "priority_ranking" in result

            # Verify suggestions
            assert len(result["suggestions"]) >= 1
            suggestion = result["suggestions"][0]
            assert "description" in suggestion
            assert "rationale" in suggestion
            assert "difficulty" in suggestion
            assert "impact" in suggestion

    async def test_get_compliance_mapping_tool(self, mcp_server, mock_standards_data):
        """Test get_compliance_mapping MCP tool."""
        with patch.object(
            mcp_server.rule_engine, "get_applicable_standards"
        ) as mock_get_standards:
            mock_get_standards.return_value = [mock_standards_data[0]]

            # Test tool execution
            arguments = {"standard_id": "python-coding-standards", "framework": "nist"}

            result = await mcp_server._get_compliance_mapping(**arguments)

            # Verify result structure
            assert "standard_id" in result
            assert "framework" in result
            assert "mappings" in result
            assert "coverage" in result

            # Verify mappings
            assert result["standard_id"] == "python-coding-standards"
            assert result["framework"] == "nist"
            assert isinstance(result["mappings"], list)

    async def test_authentication_flow(self, mcp_server):
        """Test authentication flow."""
        # Test without authentication (should fail if auth is enabled)
        with patch.object(mcp_server.auth_manager, "authenticate") as mock_auth:
            mock_auth.return_value = None  # No user authenticated

            # Test that protected operations require authentication
            with pytest.raises(RuntimeError):  # Should raise authentication error
                await mcp_server._get_applicable_standards(
                    project_context={"languages": ["python"]},
                    requirements=["code_quality"],
                )

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
        # Test with invalid input
        with pytest.raises(ValueError):  # Should raise validation error
            await mcp_server._get_applicable_standards(
                project_context="invalid_context",  # Should be dict
                requirements=["code_quality"],
            )

    async def test_privacy_filtering(self, mcp_server):
        """Test privacy filtering."""
        with patch.object(mcp_server.privacy_filter, "filter_response") as mock_filter:
            mock_filter.return_value = {
                "filtered": True,
                "redacted_fields": ["email", "phone"],
            }

            # Test that responses are filtered
            arguments = {
                "project_context": {
                    "languages": ["python"],
                    "contact_email": "test@example.com",  # Should be filtered
                },
                "requirements": ["code_quality"],
            }

            with patch.object(
                mcp_server.rule_engine, "get_applicable_standards"
            ) as mock_get_standards:
                mock_get_standards.return_value = []

                await mcp_server._get_applicable_standards(**arguments)

                # Verify privacy filtering was applied
                mock_filter.assert_called_once()

    async def test_metrics_collection(self, mcp_server):
        """Test metrics collection."""
        # Test that metrics are collected for tool execution
        with patch.object(mcp_server.metrics, "record_tool_execution") as mock_record:
            with patch.object(
                mcp_server.rule_engine, "get_applicable_standards"
            ) as mock_get_standards:
                mock_get_standards.return_value = []

                await mcp_server._get_applicable_standards(
                    project_context={"languages": ["python"]},
                    requirements=["code_quality"],
                )

                # Verify metrics were recorded
                mock_record.assert_called_once()

    async def test_error_handling(self, mcp_server):
        """Test error handling."""
        # Test graceful error handling
        with patch.object(
            mcp_server.rule_engine, "get_applicable_standards"
        ) as mock_get_standards:
            mock_get_standards.side_effect = Exception("Database connection failed")

            # Should handle error gracefully
            result = await mcp_server._get_applicable_standards(
                project_context={"languages": ["python"]}, requirements=["code_quality"]
            )

            # Should return error information
            assert "error" in result
            assert "Database connection failed" in result["error"]

    async def test_concurrent_requests(self, mcp_server):
        """Test handling of concurrent requests."""
        with patch.object(
            mcp_server.rule_engine, "get_applicable_standards"
        ) as mock_get_standards:
            mock_get_standards.return_value = []

            # Create multiple concurrent requests
            tasks = []
            for _i in range(10):
                task = asyncio.create_task(
                    mcp_server._get_applicable_standards(
                        project_context={"languages": ["python"]},
                        requirements=["code_quality"],
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
    def mcp_server(self):
        """Create MCP server instance."""
        with (
            patch("src.mcp_server.RuleEngine"),
            patch("src.mcp_server.StandardsSynchronizer"),
            patch("src.mcp_server.CrossReferencer"),
            patch("src.mcp_server.StandardsAnalytics"),
        ):

            server = MCPStandardsServer()
            return server

    async def test_sync_standards_tool(self, mcp_server):
        """Test sync_standards tool."""
        with patch.object(mcp_server.synchronizer, "sync_standards") as mock_sync:
            mock_sync.return_value = Mock(
                status="success", synced_files=5, failed_files=0, total_files=5
            )

            result = await mcp_server._sync_standards()

            assert result["status"] == "success"
            assert result["synced_files"] == 5
            assert result["failed_files"] == 0

    async def test_get_analytics_tool(self, mcp_server):
        """Test get_analytics tool."""
        with patch.object(
            mcp_server.analytics, "get_usage_analytics"
        ) as mock_analytics:
            mock_analytics.return_value = {
                "total_queries": 1000,
                "top_standards": ["python-coding-standards"],
                "usage_trends": {"daily": [100, 120, 110]},
            }

            result = await mcp_server._get_analytics()

            assert result["total_queries"] == 1000
            assert "top_standards" in result
            assert "usage_trends" in result

    async def test_cross_reference_standards_tool(self, mcp_server):
        """Test cross_reference_standards tool."""
        with patch.object(
            mcp_server.cross_referencer, "find_related_standards"
        ) as mock_xref:
            mock_xref.return_value = [
                {
                    "id": "related-standard",
                    "title": "Related Standard",
                    "relationship": "complementary",
                    "confidence": 0.8,
                }
            ]

            result = await mcp_server._cross_reference_standards(
                standard_id="python-coding-standards"
            )

            assert len(result["related_standards"]) == 1
            assert result["related_standards"][0]["id"] == "related-standard"
            assert result["related_standards"][0]["relationship"] == "complementary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
