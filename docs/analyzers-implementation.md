# Language Analyzers Implementation

This document describes the enhanced language analyzer implementations in the MCP Standards Server.

## Overview

We have implemented comprehensive security analyzers for four major programming languages:
- Python
- JavaScript/TypeScript
- Go
- Java

Each analyzer provides deep pattern detection for NIST 800-53 rev5 security controls.

## Key Features

### 1. Enhanced Pattern Detection
- **200+ security patterns** across all NIST control families
- Framework-specific pattern detection
- Configuration file analysis (requirements.txt, package.json, etc.)
- Confidence scoring for each detected control

### 2. Language-Specific Analysis

#### Python Analyzer
- Native AST parsing using Python's `ast` module
- Django, Flask, FastAPI framework detection
- Security decorator analysis (@login_required, @csrf_protect, etc.)
- Package dependency analysis from requirements.txt, setup.py, pyproject.toml

#### JavaScript/TypeScript Analyzer
- Pattern-based analysis for ES6+ and CommonJS
- Express.js, React, Vue, Angular framework support
- Security middleware detection (helmet, cors, csrf)
- NPM package analysis from package.json

#### Go Analyzer
- Import path analysis for security packages
- Gin, Echo framework-specific patterns
- Go module dependency analysis
- Gorilla toolkit security middleware detection

#### Java Analyzer
- Spring Security annotation analysis
- JAX-RS security patterns
- Maven (pom.xml) and Gradle dependency analysis
- Bean validation and OWASP library detection

### 3. NIST Control Mapping

Each analyzer maps code patterns to specific NIST controls:

| Pattern Type | Example Controls | Description |
|--------------|------------------|-------------|
| Authentication | IA-2, IA-5, IA-8 | Login, JWT, OAuth, MFA |
| Authorization | AC-3, AC-6 | RBAC, permissions, access control |
| Encryption | SC-13, SC-28 | Crypto libraries, TLS, data protection |
| Input Validation | SI-10 | Sanitization, escaping, validation |
| Logging | AU-2, AU-3, AU-9 | Audit trails, security events |
| Session Management | SC-23, AC-12 | Session handling, timeouts |

## Implementation Architecture

```
src/analyzers/
├── base.py                 # Abstract base analyzer
├── ast_utils.py           # AST parsing utilities
├── enhanced_patterns.py   # 200+ NIST control patterns
├── control_coverage_report.py  # Coverage analysis
├── python_analyzer.py     # Python analyzer
├── javascript_analyzer.py # JavaScript/TypeScript analyzer
├── go_analyzer.py        # Go analyzer
└── java_analyzer.py      # Java analyzer
```

## Usage Example

```python
from src.analyzers.python_analyzer import PythonAnalyzer

analyzer = PythonAnalyzer()
annotations = analyzer.analyze_file(Path("app.py"))

for annotation in annotations:
    print(f"Line {annotation.line_number}: {annotation.control_ids}")
    print(f"Evidence: {annotation.evidence}")
    print(f"Confidence: {annotation.confidence}")
```

## Pattern Examples

### Authentication Pattern (Python)
```python
@login_required  # Detects: IA-2, AC-3
def protected_view(request):
    pass
```

### Encryption Pattern (JavaScript)
```javascript
const bcrypt = require('bcrypt');  // Detects: IA-5, SC-13
const hash = await bcrypt.hash(password, 10);
```

### Authorization Pattern (Go)
```go
func AuthMiddleware() gin.HandlerFunc {  // Detects: IA-2, AC-3
    return func(c *gin.Context) {
        // Authorization logic
    }
}
```

### Security Annotation (Java)
```java
@PreAuthorize("hasRole('ADMIN')")  // Detects: AC-3, AC-6
public void adminMethod() {
    // Admin only
}
```

## Performance Considerations

- Pattern matching is optimized using compiled regex
- AST parsing (Python) provides accurate analysis
- File-level caching reduces redundant processing
- Project analysis skips common non-source directories

## Future Enhancements

1. **Full Tree-sitter Integration**: While current implementation works well, tree-sitter would provide:
   - Incremental parsing
   - Better performance for large files
   - Unified AST across all languages

2. **Additional Languages**: Support for Ruby, PHP, C++, Rust, C#

3. **Machine Learning**: Automatic pattern learning from annotated codebases

4. **IDE Integration**: Real-time analysis in VS Code, IntelliJ

## Testing

Comprehensive test coverage ensures analyzer accuracy:

```bash
pytest tests/unit/analyzers/test_analyzer_integration.py
```

All analyzers are tested with real-world code samples and verified against expected NIST control mappings.