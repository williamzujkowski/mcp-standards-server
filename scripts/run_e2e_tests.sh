#!/bin/bash
# Run E2E tests with various configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE_ENABLED=true
PERFORMANCE_TESTS=false
PARALLEL=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --no-coverage)
            COVERAGE_ENABLED=false
            shift
            ;;
        --performance)
            PERFORMANCE_TESTS=true
            shift
            ;;
        --parallel)
            PARALLEL=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --type <type>      Test type: all, e2e, unit, integration (default: all)"
            echo "  --no-coverage      Disable coverage reporting"
            echo "  --performance      Include performance tests"
            echo "  --parallel         Run tests in parallel"
            echo "  --verbose          Verbose output"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting E2E Test Suite${NC}"
echo "Configuration:"
echo "  Test Type: $TEST_TYPE"
echo "  Coverage: $COVERAGE_ENABLED"
echo "  Performance Tests: $PERFORMANCE_TESTS"
echo "  Parallel: $PARALLEL"
echo ""

# Check dependencies
echo -e "${YELLOW}Checking dependencies...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is required but not installed${NC}"
    exit 1
fi

# Check pytest
if ! python3 -m pytest --version &> /dev/null; then
    echo -e "${RED}pytest is required but not installed${NC}"
    echo "Run: pip install pytest pytest-asyncio pytest-cov"
    exit 1
fi

# Check for MCP
if ! python3 -c "import mcp" &> /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: MCP not installed. Some tests will be skipped.${NC}"
fi

# Check for Redis (if available)
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo "Redis server: Available"
    else
        echo -e "${YELLOW}Warning: Redis not running. Some tests may fail.${NC}"
    fi
fi

# Set environment variables
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export MCP_TEST_MODE="true"
export PYTEST_CURRENT_TEST=""

# Build pytest command
PYTEST_CMD="python3 -m pytest"

# Add test selection
case $TEST_TYPE in
    e2e)
        PYTEST_CMD="$PYTEST_CMD tests/e2e/"
        ;;
    unit)
        PYTEST_CMD="$PYTEST_CMD tests/unit/"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tests/integration/"
        ;;
    all)
        PYTEST_CMD="$PYTEST_CMD tests/"
        ;;
esac

# Add coverage options
if [ "$COVERAGE_ENABLED" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=src --cov-report=term-missing --cov-report=html --cov-report=xml"
fi

# Add performance test options
if [ "$PERFORMANCE_TESTS" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -m 'performance or benchmark'"
else
    PYTEST_CMD="$PYTEST_CMD -m 'not performance and not benchmark'"
fi

# Add parallel execution
if [ "$PARALLEL" = true ]; then
    # Check if pytest-xdist is installed
    if python3 -m pytest --version | grep -q "xdist"; then
        PYTEST_CMD="$PYTEST_CMD -n auto"
    else
        echo -e "${YELLOW}Warning: pytest-xdist not installed. Running tests sequentially.${NC}"
        echo "Install with: pip install pytest-xdist"
    fi
fi

# Add verbose flag
if [ "$VERBOSE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD -vv -s"
else
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add timeout
PYTEST_CMD="$PYTEST_CMD --timeout=300"

# Create test results directory
mkdir -p test-results

# Run tests
echo -e "${GREEN}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

# Execute tests and capture exit code
set +e
$PYTEST_CMD --junitxml=test-results/junit.xml
TEST_EXIT_CODE=$?
set -e

# Process results
echo ""
if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
else
    echo -e "${RED}Some tests failed!${NC}"
fi

# Generate coverage report
if [ "$COVERAGE_ENABLED" = true ] && [ -f coverage.xml ]; then
    echo ""
    echo -e "${YELLOW}Coverage Summary:${NC}"
    python3 -c "
import xml.etree.ElementTree as ET
tree = ET.parse('coverage.xml')
root = tree.getroot()
line_rate = float(root.get('line-rate', 0))
branch_rate = float(root.get('branch-rate', 0))
print(f'Line Coverage: {line_rate * 100:.2f}%')
print(f'Branch Coverage: {branch_rate * 100:.2f}%')
"
    echo "Detailed coverage report: htmlcov/index.html"
fi

# Check for performance regressions if performance tests were run
if [ "$PERFORMANCE_TESTS" = true ] && [ -f benchmark_results.json ]; then
    echo ""
    echo -e "${YELLOW}Checking for performance regressions...${NC}"
    
    if [ -f benchmark_baseline.json ]; then
        python3 scripts/detect_performance_regression.py \
            benchmark_results.json \
            benchmark_baseline.json \
            --threshold 10.0 \
            --output text
    else
        echo "No baseline found. Current results will be used as baseline."
        cp benchmark_results.json benchmark_baseline.json
    fi
fi

# Generate test report
echo ""
echo -e "${YELLOW}Generating test report...${NC}"

cat > test-results/summary.txt << EOF
Test Run Summary
================
Date: $(date)
Test Type: $TEST_TYPE
Total Duration: ${SECONDS}s

Results:
$(python3 -m pytest --collect-only -q 2>/dev/null | tail -1 || echo "Unable to count tests")

Exit Code: $TEST_EXIT_CODE
EOF

echo "Test summary saved to: test-results/summary.txt"

# Exit with test exit code
exit $TEST_EXIT_CODE