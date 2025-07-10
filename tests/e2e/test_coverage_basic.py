"""Basic test to verify coverage is being collected."""

import pytest


class TestCoverageBasic:
    """Basic tests to ensure coverage collection works."""

    @pytest.mark.asyncio
    async def test_server_starts(self, mcp_server):
        """Test that server starts successfully."""
        assert mcp_server is not None

    @pytest.mark.asyncio
    async def test_client_connects(self, mcp_client):
        """Test that client can connect to server."""
        assert mcp_client is not None
        assert mcp_client.session is not None

    @pytest.mark.asyncio
    async def test_list_standards_tool(self, mcp_client):
        """Test basic tool call to ensure coverage tracks it."""
        result = await mcp_client.call_tool("list_available_standards", {})

        assert "standards" in result
        assert isinstance(result["standards"], list)
