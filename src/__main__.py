"""Main entry point for running as module."""

import asyncio
import logging
import sys

# Configure logging before importing the server
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

try:
    from .mcp_server import main
except ImportError:
    # Fallback for direct execution
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mcp_server import main

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)
