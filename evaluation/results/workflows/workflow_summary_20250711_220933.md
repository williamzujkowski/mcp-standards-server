# MCP User Workflow Test Summary

**Generated:** 2025-07-11 22:09:33

## Overall Results

- **Total Workflows Tested:** 6
- **Successful Workflows:** 0 (0.0%)
- **Total Steps:** 38
- **Completed Steps:** 4 (10.5%)
- **Total Execution Time:** 1.51 seconds

## Workflow Results

| Workflow | Success | Steps Completed | Execution Time | Errors |
|----------|---------|-----------------|----------------|--------|
| new_project_setup | ❌ | 1/6 | 0.01s | 1 |
| security_audit | ❌ | 1/7 | 0.50s | 1 |
| performance_optimization | ❌ | 1/7 | 0.50s | 1 |
| team_onboarding | ❌ | 0/6 | 0.00s | 1 |
| compliance_verification | ❌ | 1/6 | 0.50s | 1 |
| continuous_improvement | ❌ | 0/6 | 0.00s | 1 |

## Detailed Error Summary

### new_project_setup
- Step get_applicable_standards: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/get_applicable_standards'

### security_audit
- Step search_security_standards: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/search_standards'

### performance_optimization
- Step search_performance_standards: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/search_standards'

### team_onboarding
- Step list_all_standards: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/list_available_standards'

### compliance_verification
- Step map_standards_to_controls: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/get_compliance_mapping'

### continuous_improvement
- Step get_current_standards: 405, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://localhost:8000/mcp/get_applicable_standards'


## Recommendations

1. **Failed Workflows**: Investigate and fix issues in failed workflows
2. **Incomplete Steps**: Review steps that consistently fail across workflows
3. **Performance**: Optimize workflows taking longer than expected
4. **Error Patterns**: Address common error patterns across workflows

## Next Steps

1. Fix identified issues in the MCP server implementation
2. Add more comprehensive error handling
3. Improve workflow resilience to handle partial failures
4. Enhance validation logic for better test coverage
