"""Main entry point for running as module."""
import asyncio
from .mcp_server import main

if __name__ == "__main__":
    asyncio.run(main())