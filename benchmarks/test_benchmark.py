#!/usr/bin/env python3
"""Simple test benchmark to verify setup."""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.mcp_server import MCPStandardsServer
from src.core.standards.engine import StandardsEngine
from src.core.standards.rule_engine import RuleEngine
from src.core.standards.semantic_search import SemanticSearch
from src.core.standards.token_optimizer import TokenOptimizer


async def test_mcp_server():
    """Test basic MCP server functionality."""
    print("Testing MCP Server setup...")
    
    try:
        # Initialize server with minimal config
        config = {
            "search": {"enabled": False},
            "token_model": "gpt-4",
            "default_token_budget": 8000
        }
        server = MCPStandardsServer(config)
        
        # Test basic method
        status = await server._get_sync_status()
        print(f"✓ MCP Server initialized successfully")
        print(f"  Status: {status}")
        
        return True
    except Exception as e:
        print(f"✗ MCP Server test failed: {e}")
        return False


async def test_standards_engine():
    """Test Standards Engine functionality."""
    print("\nTesting Standards Engine...")
    
    try:
        # Initialize with data directory
        data_dir = Path(project_root) / "data" / "standards"
        engine = StandardsEngine(data_dir)
        # Load any cached standards
        standards = list(engine.standards.values())
        print(f"✓ Standards Engine initialized")
        print(f"  Loaded {len(standards)} standards")
        
        return True
    except Exception as e:
        print(f"✗ Standards Engine test failed: {e}")
        return False


async def test_rule_engine():
    """Test Rule Engine functionality."""
    print("\nTesting Rule Engine...")
    
    try:
        rule_engine = RuleEngine()
        
        # Test with sample context
        context = {
            "language": "python",
            "framework": "fastapi",
            "project_type": "api"
        }
        
        matches = rule_engine.evaluate(context)
        print(f"✓ Rule Engine initialized")
        print(f"  Found {len(matches)} matching rules for context")
        
        return True
    except Exception as e:
        print(f"✗ Rule Engine test failed: {e}")
        return False


async def test_semantic_search():
    """Test Semantic Search functionality."""
    print("\nTesting Semantic Search...")
    
    try:
        # Initialize with empty config (will be disabled by default)
        search = SemanticSearch()
        print(f"✓ Semantic Search initialized")
        print(f"  Enabled: {search.enabled}")
        
        return True
    except Exception as e:
        print(f"✗ Semantic Search test failed: {e}")
        return False


async def test_token_optimizer():
    """Test Token Optimizer functionality."""
    print("\nTesting Token Optimizer...")
    
    try:
        optimizer = TokenOptimizer()
        
        # Test basic optimization
        test_content = {
            "title": "Test Standard",
            "description": "This is a test standard for benchmarking",
            "sections": ["Section 1", "Section 2", "Section 3"]
        }
        
        # Test get_optimized_content
        optimized = optimizer.get_optimized_content(
            test_content, 
            format_type="condensed",
            token_budget=100
        )
        print(f"✓ Token Optimizer initialized")
        print(f"  Optimized content type: {type(optimized)}")
        
        return True
    except Exception as e:
        print(f"✗ Token Optimizer test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("MCP Standards Server - Benchmark Setup Test")
    print("=" * 50)
    
    tests = [
        test_mcp_server,
        test_standards_engine,
        test_rule_engine,
        test_semantic_search,
        test_token_optimizer
    ]
    
    results = []
    for test in tests:
        result = await test()
        results.append(result)
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("\n✓ All components working! Ready to run benchmarks.")
        return 0
    else:
        print("\n✗ Some components failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))