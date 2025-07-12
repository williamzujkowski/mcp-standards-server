# MCP Standards Server Evaluation

This directory contains the comprehensive evaluation framework for testing and validating the MCP Standards Server.

## Directory Structure

- **benchmarks/**: Performance benchmarking scripts and tools
- **e2e/**: End-to-end user workflow tests
- **scripts/**: Utility scripts for evaluation tasks
- **fixtures/**: Test data and fixtures
  - **standards/**: Test standard files
  - **code_samples/**: Sample code for validation testing
- **results/**: Evaluation execution results
  - **workflows/**: Workflow test results
  - **benchmarks/**: Performance benchmark results

## Running Evaluations

### Performance Benchmarks
```bash
python evaluation/benchmarks/mcp_performance_benchmark.py
```

### End-to-End Workflows
```bash
python evaluation/e2e/test_user_workflows.py
```

### Organization Script
```bash
python evaluation/scripts/organize_artifacts.py
```

## Key Files

- `MCP_EVALUATION_PLAN.md`: Comprehensive evaluation strategy
- `benchmarks/mcp_performance_benchmark.py`: Performance testing suite
- `e2e/test_user_workflows.py`: User workflow simulations
- `scripts/organize_artifacts.py`: Project organization utility
