#!/bin/bash
# Run only stable, fast tests for quick feedback

echo "Running stable test suite..."
echo "============================"

# Set test environment
export PYTHONDONTWRITEBYTECODE=1
export MCP_TEST_MODE=true
export MCP_DISABLE_TELEMETRY=true

# Run tests with optimized settings
python -m pytest \
    tests/unit/analyzers \
    tests/unit/core/standards/test_rule_engine.py \
    tests/unit/core/standards/test_token_optimizer.py \
    tests/unit/core/test_auth.py \
    tests/unit/core/test_security.py \
    tests/unit/core/test_validation.py \
    tests/unit/test_mcp_server.py \
    --no-cov \
    --tb=short \
    -v \
    --durations=10 \
    --timeout=10 \
    --timeout-method=thread \
    -x \
    "$@"

echo -e "\n✅ Stable test suite completed!"