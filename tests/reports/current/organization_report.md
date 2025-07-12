# Project Organization Report

**Generated:** 2025-07-11 20:57:13

## Summary

- **Files Moved:** 19
- **Files Archived:** 0
- **Duplicates Removed:** 0
- **Directories Created:** 16
- **Errors:** 0

## New Directory Structure

```
mcp-standards-server/
├── tests/
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── e2e/                     # End-to-end tests
│   ├── performance/             # Performance tests
│   ├── accuracy/                # Accuracy tests
│   ├── fixtures/                # Test fixtures
│   └── reports/                 # All test reports (organized)
│       ├── current/             # Latest test results
│       ├── historical/          # Previous test runs
│       ├── analysis/            # Analysis reports
│       ├── performance/         # Performance benchmarks
│       ├── compliance/          # Compliance reports
│       └── workflows/           # Workflow test results
├── evaluation/                  # Evaluation framework
│   ├── benchmarks/              # Benchmark scripts
│   ├── e2e/                     # E2E workflow tests
│   ├── scripts/                 # Utility scripts
│   ├── fixtures/                # Evaluation test data
│   │   ├── standards/           # Test standards
│   │   └── code_samples/        # Sample code for testing
│   └── results/                 # Evaluation results
│       ├── workflows/           # Workflow execution results
│       └── benchmarks/          # Benchmark results
├── archive/                     # Archived/old files
│   ├── test_files/              # Old test files
│   └── old_reports/             # Outdated reports
└── data/
    └── standards/
        └── search/              # Vector search data
            └── index.json       # Search index
```

## File Organization Details

### Reports Consolidated
- Historical test reports moved to `tests/reports/historical/`
- Performance data moved to `tests/reports/performance/`
- Compliance reports moved to `tests/reports/compliance/`
- Workflow reports moved to `tests/reports/workflows/`

### Duplicates Removed
- Removed 0 duplicate test files
- Kept most recent versions based on modification time
- Archived older versions to `archive/test_files/`

### Standards Search Data
- Indexed vector search files in `data/standards/search/`
- Created index.json for quick lookup


## Next Steps

1. **Review Archived Files**: Check `archive/` directory and delete if not needed
2. **Update Import Paths**: Update any imports that reference moved files
3. **Run Tests**: Ensure all tests still pass after reorganization
4. **Update Documentation**: Update README and docs to reflect new structure

## Cleanup Commands

To further clean up, you can run:

```bash
# Remove archived files older than 30 days
find archive/ -type f -mtime +30 -delete

# Remove empty directories
find . -type d -empty -delete

# Find and remove __pycache__ directories
find . -type d -name __pycache__ -exec rm -rf {} +

# Find large files that might need attention
find . -type f -size +10M -exec ls -lh {} \;
```
