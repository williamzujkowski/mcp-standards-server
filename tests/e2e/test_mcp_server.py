"""
End-to-end tests for MCP Standards Server functionality.

Tests cover:
- Server startup and shutdown
- All MCP tools functionality
- Standards synchronization workflow
- Rule engine integration
- Semantic search functionality
- Error handling scenarios
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from contextlib import asynccontextmanager

from src.core.standards.rule_engine import RuleEngine
from src.core.standards.sync import StandardsSynchronizer
from tests.e2e.test_data_setup import setup_test_data


class MCPTestClient:
    """Test client for interacting with MCP server."""
    
    def __init__(self, server_params: StdioServerParameters):
        self.server_params = server_params
        self.session: Optional[ClientSession] = None
        self._read = None
        self._write = None
        
    @asynccontextmanager
    async def connect(self):
        """Connect to the MCP server as an async context manager."""
        async with stdio_client(self.server_params) as (read, write):
            self._read = read
            self._write = write
            async with ClientSession(read, write) as session:
                self.session = session
                try:
                    # Initialize the session
                    await session.initialize()
                    yield self
                finally:
                    self.session = None
                    self._read = None
                    self._write = None
                
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool on the MCP server."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        result = await self.session.call_tool(tool_name, arguments)
        
        # Extract the content from the result
        if hasattr(result, 'content') and result.content:
            # Parse the JSON content from the first text content
            if result.content[0].text:
                return json.loads(result.content[0].text)
        
        return result


@pytest.fixture
async def mcp_server():
    """Fixture to start and stop MCP server for tests."""
    # Create temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Set up test data
        setup_test_data(temp_path)
        
        # Set up test environment
        env = os.environ.copy()
        env["MCP_STANDARDS_DATA_DIR"] = str(temp_path)
        env["MCP_CONFIG_PATH"] = str(Path(__file__).parent / "test_config.json")
        env["MCP_DISABLE_SEARCH"] = "true"  # Disable search to avoid heavy deps
        
        # Server parameters - run directly without coverage subprocess
        # Coverage will be handled by the parent process
        server_params = StdioServerParameters(
            command="python",
            args=["-m", "src"],
            env=env
        )
        
        # Start server process
        process = await asyncio.create_subprocess_exec(
            server_params.command,
            *server_params.args,
            env=server_params.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for server to start
        await asyncio.sleep(1)
        
        yield server_params
        
        # Stop server
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=5.0)
        except (ProcessLookupError, asyncio.TimeoutError):
            # Process may have already exited
            pass
        
        # Combine coverage data if we used coverage
        try:
            import coverage
            import subprocess
            # Try to combine coverage data
            subprocess.run(["coverage", "combine"], check=False)
        except (ImportError, FileNotFoundError):
            pass


@pytest.fixture
async def mcp_client(mcp_server):
    """Fixture to provide connected MCP client for tests."""
    client = MCPTestClient(mcp_server)
    async with client.connect() as connected_client:
        yield connected_client



class TestMCPServerStartupShutdown:
    """Test server startup and shutdown scenarios."""
    
    @pytest.mark.asyncio
    async def test_server_starts_successfully(self, mcp_server):
        """Test that MCP server starts without errors."""
        # Server fixture ensures server starts
        assert mcp_server is not None
        
    @pytest.mark.asyncio
    async def test_server_handles_graceful_shutdown(self, mcp_server):
        """Test graceful server shutdown."""
        # Create client connection
        client = MCPTestClient(mcp_server)
        async with client.connect() as connected_client:
            # Verify connection
            assert connected_client.session is not None
        
        # After context exit, connection should be closed
        assert client.session is None
        
    @pytest.mark.asyncio
    async def test_server_handles_multiple_connections(self, mcp_server):
        """Test server can handle multiple client connections."""
        clients = []
        
        # Create multiple clients
        for i in range(3):
            client = MCPTestClient(mcp_server)
            async with client.connect() as connected_client:
                clients.append(connected_client)
                
        # All clients should have connected successfully
        assert len(clients) == 3


class TestMCPTools:
    """Test all MCP tool implementations."""
    
    @pytest.mark.asyncio
    async def test_get_applicable_standards(self, mcp_client):
        """Test get_applicable_standards tool."""
        # Test with web application context
        result = await mcp_client.call_tool(
            "get_applicable_standards",
            {
                "context": {
                    "project_type": "web_application",
                    "framework": "react",
                    "language": "javascript",
                    "requirements": ["accessibility", "performance"]
                }
            }
        )
        
        assert "standards" in result
        assert isinstance(result["standards"], list)
        assert len(result["standards"]) > 0
        
        # Verify returned standards are relevant
        standards = result["standards"]
        assert any("react" in s.lower() for s in standards)
        assert any("javascript" in s.lower() for s in standards)
        
    @pytest.mark.asyncio
    async def test_validate_against_standard(self, mcp_client):
        """Test validate_against_standard tool."""
        # Sample code to validate
        code_content = """
        import React from 'react';
        
        const MyComponent = (props) => {
            return <div>{props.children}</div>;
        };
        
        export default MyComponent;
        """
        
        result = await mcp_client.call_tool(
            "validate_against_standard",
            {
                "code": code_content,
                "standard": "react-18-patterns",
                "language": "javascript"
            }
        )
        
        assert "violations" in result
        assert "passed" in result
        assert isinstance(result["violations"], list)
        
    @pytest.mark.asyncio
    async def test_search_standards(self, mcp_client):
        """Test search_standards tool with semantic search."""
        result = await mcp_client.call_tool(
            "search_standards",
            {
                "query": "How to implement accessibility in React components?",
                "limit": 5
            }
        )
        
        assert "results" in result
        assert isinstance(result["results"], list)
        assert len(result["results"]) <= 5
        
        # Verify search results contain relevant standards
        for item in result["results"]:
            assert "standard" in item
            assert "relevance_score" in item
            assert item["relevance_score"] >= 0.0
            
    @pytest.mark.asyncio
    async def test_get_standard_details(self, mcp_client):
        """Test get_standard_details tool."""
        result = await mcp_client.call_tool(
            "get_standard_details",
            {
                "standard_id": "react-18-patterns"
            }
        )
        
        assert "id" in result
        assert "name" in result
        assert "content" in result
        assert "metadata" in result
        assert result["id"] == "react-18-patterns"
        
    @pytest.mark.asyncio
    async def test_list_available_standards(self, mcp_client):
        """Test list_available_standards tool."""
        result = await mcp_client.call_tool(
            "list_available_standards",
            {
                "category": "frontend"
            }
        )
        
        assert "standards" in result
        assert isinstance(result["standards"], list)
        
        # Verify filtering by category works
        for standard in result["standards"]:
            assert "tags" in standard
            assert "frontend" in standard["tags"]
            
    @pytest.mark.asyncio
    async def test_suggest_improvements(self, mcp_client):
        """Test suggest_improvements tool."""
        code_content = """
        function fetchData() {
            fetch('/api/data')
                .then(res => res.json())
                .then(data => console.log(data));
        }
        """
        
        result = await mcp_client.call_tool(
            "suggest_improvements",
            {
                "code": code_content,
                "context": {
                    "language": "javascript",
                    "project_type": "web_application"
                }
            }
        )
        
        assert "suggestions" in result
        assert isinstance(result["suggestions"], list)
        assert len(result["suggestions"]) > 0
        
        # Verify suggestions have expected structure
        for suggestion in result["suggestions"]:
            assert "description" in suggestion
            assert "priority" in suggestion
            assert "standard_reference" in suggestion


class TestStandardsSynchronization:
    """Test standards synchronization workflow."""
    
    @pytest.mark.asyncio
    async def test_sync_standards_workflow(self, mcp_client):
        """Test complete standards synchronization workflow."""
        # Check sync status
        status_result = await mcp_client.call_tool(
            "get_sync_status",
            {}
        )
        
        assert "last_sync" in status_result
        assert "total_standards" in status_result
        assert "outdated_standards" in status_result
        
        # Trigger sync
        sync_result = await mcp_client.call_tool(
            "sync_standards",
            {
                "force": False
            }
        )
        
        assert "status" in sync_result
        assert "synced_files" in sync_result
        assert sync_result["status"] in ["success", "partial", "failed"]
        
        # Verify sync updated status
        new_status = await mcp_client.call_tool(
            "get_sync_status",
            {}
        )
        
        assert new_status["last_sync"] != status_result["last_sync"]
        
    @pytest.mark.asyncio
    async def test_sync_with_rate_limiting(self, mcp_client):
        """Test sync handles GitHub API rate limits gracefully."""
        with patch("src.core.standards.sync.StandardsSynchronizer.sync") as mock_sync:
            # Simulate rate limit error
            mock_sync.side_effect = Exception("API rate limit exceeded")
            
            result = await mcp_client.call_tool(
                "sync_standards",
                {
                    "force": True
                }
            )
            
            assert result["status"] == "failed"
            assert "rate limit" in result.get("error", "").lower()


class TestRuleEngineIntegration:
    """Test rule engine integration with MCP server."""
    
    @pytest.mark.asyncio
    async def test_rule_evaluation_through_mcp(self, mcp_client):
        """Test rule engine evaluation via MCP tools."""
        contexts = [
            {
                "project_type": "web_application",
                "framework": "react",
                "requirements": ["accessibility"]
            },
            {
                "project_type": "api",
                "language": "python",
                "framework": "fastapi"
            },
            {
                "project_type": "mobile_app",
                "framework": "react-native",
                "platform": ["ios", "android"]
            }
        ]
        
        for context in contexts:
            result = await mcp_client.call_tool(
                "get_applicable_standards",
                {"context": context}
            )
            
            assert "standards" in result
            assert len(result["standards"]) > 0
            assert "evaluation_path" in result
            
    @pytest.mark.asyncio
    async def test_rule_priority_resolution(self, mcp_client):
        """Test rule priority conflict resolution."""
        # Context that triggers multiple rules
        context = {
            "project_type": "web_application",
            "framework": "react",
            "language": "typescript",
            "requirements": ["accessibility", "performance", "security"],
            "team_size": "large"
        }
        
        result = await mcp_client.call_tool(
            "get_applicable_standards",
            {
                "context": context,
                "include_resolution_details": True
            }
        )
        
        assert "standards" in result
        assert "resolution_details" in result
        
        details = result["resolution_details"]
        assert "matched_rules" in details
        assert "conflicts_resolved" in details
        assert "final_priority_order" in details


class TestSemanticSearchFunctionality:
    """Test semantic search integration."""
    
    @pytest.mark.asyncio
    async def test_semantic_search_quality(self, mcp_client):
        """Test semantic search returns relevant results."""
        queries = [
            "How to optimize React component performance?",
            "Best practices for Python API error handling",
            "Implementing secure authentication in web apps",
            "Accessibility guidelines for mobile applications"
        ]
        
        for query in queries:
            result = await mcp_client.call_tool(
                "search_standards",
                {
                    "query": query,
                    "limit": 3,
                    "min_relevance": 0.7
                }
            )
            
            assert "results" in result
            assert len(result["results"]) <= 3
            
            # Verify minimum relevance threshold
            for item in result["results"]:
                assert item["relevance_score"] >= 0.7
                
    @pytest.mark.asyncio
    async def test_search_with_filters(self, mcp_client):
        """Test semantic search with category filters."""
        result = await mcp_client.call_tool(
            "search_standards",
            {
                "query": "testing best practices",
                "filters": {
                    "categories": ["testing", "quality"],
                    "languages": ["javascript", "python"]
                },
                "limit": 10
            }
        )
        
        assert "results" in result
        
        # Verify filters were applied
        for item in result["results"]:
            metadata = item.get("metadata", {})
            assert any(cat in metadata.get("categories", []) 
                      for cat in ["testing", "quality"])


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async def test_invalid_tool_name(self, mcp_client):
        """Test handling of invalid tool names."""
        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "non_existent_tool",
                {}
            )
        
        assert "unknown tool" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_invalid_parameters(self, mcp_client):
        """Test handling of invalid tool parameters."""
        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "get_applicable_standards",
                {
                    # Missing required 'context' parameter
                    "invalid_param": "value"
                }
            )
        
        assert "required parameter" in str(exc_info.value).lower()
        
    @pytest.mark.asyncio
    async def test_malformed_context(self, mcp_client):
        """Test handling of malformed context data."""
        result = await mcp_client.call_tool(
            "get_applicable_standards",
            {
                "context": {
                    # Invalid project type
                    "project_type": "invalid_type",
                    "framework": None
                }
            }
        )
        
        # Should return empty or default standards
        assert "standards" in result
        assert "warnings" in result
        
    @pytest.mark.asyncio
    async def test_server_timeout_handling(self, mcp_client):
        """Test handling of server timeout scenarios."""
        with patch("asyncio.wait_for") as mock_wait:
            mock_wait.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(asyncio.TimeoutError):
                await mcp_client.call_tool(
                    "sync_standards",
                    {"force": True}
                )


class TestConcurrentRequests:
    """Test concurrent request handling."""
    
    @pytest.mark.asyncio
    async def test_concurrent_tool_calls(self, mcp_client):
        """Test server handles concurrent tool calls correctly."""
        # Create multiple concurrent requests
        tasks = []
        
        for i in range(10):
            context = {
                "project_type": "web_application",
                "framework": "react",
                "request_id": f"test_{i}"
            }
            
            task = mcp_client.call_tool(
                "get_applicable_standards",
                {"context": context}
            )
            tasks.append(task)
        
        # Execute all requests concurrently
        results = await asyncio.gather(*tasks)
        
        # Verify all requests completed successfully
        assert len(results) == 10
        for result in results:
            assert "standards" in result
            
    @pytest.mark.asyncio
    async def test_concurrent_search_requests(self, mcp_client):
        """Test concurrent semantic search requests."""
        queries = [
            "React performance optimization",
            "Python testing strategies",
            "API security best practices",
            "Mobile app accessibility",
            "Database optimization techniques"
        ]
        
        tasks = [
            mcp_client.call_tool(
                "search_standards",
                {"query": query, "limit": 3}
            )
            for query in queries
        ]
        
        results = await asyncio.gather(*tasks)
        
        assert len(results) == len(queries)
        for result in results:
            assert "results" in result


class TestCachingBehavior:
    """Test caching functionality."""
    
    @pytest.mark.asyncio
    async def test_standards_caching(self, mcp_client):
        """Test that standards are cached after first access."""
        # First call - should load from source
        start_time = time.time()
        result1 = await mcp_client.call_tool(
            "get_standard_details",
            {"standard_id": "react-18-patterns"}
        )
        first_call_time = time.time() - start_time
        
        # Second call - should be cached
        start_time = time.time()
        result2 = await mcp_client.call_tool(
            "get_standard_details",
            {"standard_id": "react-18-patterns"}
        )
        second_call_time = time.time() - start_time
        
        # Cached call should be significantly faster
        assert second_call_time < first_call_time * 0.5
        assert result1 == result2
        
    @pytest.mark.asyncio
    async def test_cache_invalidation(self, mcp_client):
        """Test cache invalidation after sync."""
        # Get initial data
        result1 = await mcp_client.call_tool(
            "get_standard_details",
            {"standard_id": "python-testing"}
        )
        
        # Force sync
        await mcp_client.call_tool(
            "sync_standards",
            {"force": True}
        )
        
        # Get data again - should be refreshed
        result2 = await mcp_client.call_tool(
            "get_standard_details",
            {"standard_id": "python-testing"}
        )
        
        # Verify cache was invalidated
        assert "cache_refreshed" in result2.get("metadata", {})