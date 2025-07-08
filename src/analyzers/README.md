# Language Analyzers

Comprehensive language analyzers for detecting security vulnerabilities, performance issues, and best practice violations in Go, Java, Rust, and TypeScript code.

## Features

### Security Analysis
- **SQL Injection Detection**: Identifies string concatenation and template literals in database queries
- **Command Injection**: Detects unsafe command execution patterns
- **XSS Prevention**: Finds dangerous DOM manipulation and React patterns
- **Cryptographic Issues**: Identifies weak algorithms and hardcoded secrets
- **Authentication/Authorization**: Detects missing security checks
- **OWASP Top 10**: Comprehensive coverage of OWASP security categories

### Performance Analysis
- **Memory Management**: Detects leaks, excessive allocations, and inefficient patterns
- **Concurrency Issues**: Identifies race conditions and thread safety problems
- **Algorithm Efficiency**: Finds O(n²) operations and inefficient data structures
- **I/O Optimization**: Detects blocking operations in async contexts
- **Bundle Size**: Identifies large imports and optimization opportunities

### Best Practices
- **Code Style**: Enforces language-specific naming conventions
- **Error Handling**: Detects empty catches and missing error checks
- **Type Safety**: Identifies unsafe type assertions and excessive `any` usage
- **Documentation**: Checks for missing docs on public APIs
- **Testing**: Recommends test coverage for complex modules

## Supported Languages

### Go Analyzer
- Goroutine race conditions
- Defer in loops
- Error handling patterns
- Context usage
- Interface complexity
- Memory safety with unsafe blocks

### Java Analyzer
- OWASP Top 10 compliance
- Spring Framework patterns
- Resource management
- Thread safety (SimpleDateFormat, etc.)
- N+1 query detection
- Design pattern analysis

### Rust Analyzer
- Unsafe code analysis
- Ownership and borrowing
- Memory safety guarantees
- Concurrency with Send/Sync
- Error handling (Result vs panic)
- Performance optimizations

### TypeScript Analyzer
- React performance patterns
- Type safety enforcement
- Modern JavaScript features
- Async/await optimization
- Bundle size analysis
- Security in browser context

## Usage

### With MCP Server

```python
from src.analyzers.mcp_integration import register_analyzer_tools

# Register with MCP server
register_analyzer_tools(mcp_server)

# Use via MCP tools
result = await mcp_server.call_tool("analyze_code", {
    "file_path": "/path/to/code.go",
    "checks": ["security", "performance"]
})
```

### Direct API Usage

```python
from src.analyzers.base import AnalyzerPlugin

# Get analyzer for a language
analyzer = AnalyzerPlugin.get_analyzer("go")

# Analyze a single file
result = analyzer.analyze_file(Path("example.go"))

# Analyze a directory
results = analyzer.analyze_directory(Path("src/"))

# Process results
for issue in result.issues:
    print(f"{issue.severity.value}: {issue.message}")
    print(f"  Location: {issue.file_path}:{issue.line_number}")
    print(f"  Recommendation: {issue.recommendation}")
```

### Command Line

```bash
# Analyze a single file
mcp-standards analyze-code --file path/to/file.ts

# Analyze a directory
mcp-standards analyze-code --dir ./src --language java

# List available analyzers
mcp-standards list-analyzers
```

## Architecture

### Plugin System

The analyzer uses a plugin architecture for easy language addition:

```python
from src.analyzers.base import BaseAnalyzer, AnalyzerPlugin

@AnalyzerPlugin.register("mylang")
class MyLanguageAnalyzer(BaseAnalyzer):
    @property
    def language(self) -> str:
        return "mylang"
    
    @property
    def file_extensions(self) -> List[str]:
        return [".ml", ".mli"]
    
    # Implement required methods...
```

### AST Parsing

Each analyzer parses source code into an AST for deep analysis:
- Simplified AST representation for cross-language patterns
- Pattern matching utilities for common vulnerabilities
- Security and performance pattern detectors

### Issue Reporting

Issues are categorized by:
- **Type**: Security, Performance, Best Practice, etc.
- **Severity**: Critical, High, Medium, Low, Info
- **CWE ID**: For security vulnerabilities
- **OWASP Category**: For web security issues

## Performance

Benchmark results on test files:

| Language   | Avg Time | Throughput    | Issues/Second |
|------------|----------|---------------|---------------|
| Go         | 0.0234s  | 2,136 lines/s | 45.3          |
| Java       | 0.0312s  | 1,603 lines/s | 38.1          |
| Rust       | 0.0289s  | 1,730 lines/s | 41.2          |
| TypeScript | 0.0267s  | 1,873 lines/s | 43.7          |

## Examples

See `examples/analyzer-test-samples/` for comprehensive examples demonstrating:
- Common vulnerabilities in each language
- Performance anti-patterns
- Best practice violations
- Correct implementations

## Testing

Run the test suite:

```bash
pytest tests/unit/analyzers/ -v
```

Run performance benchmarks:

```bash
python benchmarks/analyzer_performance.py
```

## Contributing

To add a new language analyzer:

1. Create a new analyzer class inheriting from `BaseAnalyzer`
2. Implement required methods for AST parsing and analysis
3. Register with `@AnalyzerPlugin.register("language")`
4. Add comprehensive tests
5. Update documentation

## Future Enhancements

- [ ] Tree-sitter integration for better AST parsing
- [ ] Machine learning for pattern detection
- [ ] IDE plugin support
- [ ] CI/CD integration
- [ ] Custom rule configuration
- [ ] Auto-fix suggestions
- [ ] Incremental analysis
- [ ] Cross-file analysis