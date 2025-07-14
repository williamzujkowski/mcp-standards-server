# Writing Validators Guide

This guide explains how to write custom validators for MCP Standards Server.

## Validator Overview

Validators are Python classes that analyze code and detect violations of standards. They can use pattern matching, AST analysis, or custom logic.

## Validator Architecture

```
BaseAnalyzer (Abstract)
    ├── PythonAnalyzer
    ├── JavaScriptAnalyzer
    ├── GoAnalyzer
    └── CustomValidator (Your validator)
```

## Creating a Basic Validator

### 1. Inherit from BaseAnalyzer

```python
# src/validators/my_validator.py
from typing import List, Dict, Any
from src.analyzers.base import BaseAnalyzer, AnalysisResult

class MyValidator(BaseAnalyzer):
    """Custom validator for specific standards."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()
        self.config = config or {}
    
    def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
        """Analyze code and return violations."""
        violations = []
        issues = []
        
        # Your analysis logic here
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if self._check_violation(line):
                violations.append({
                    'rule': 'my-rule',
                    'message': 'Violation detected',
                    'line': i,
                    'column': 0,
                    'severity': 'warning'
                })
        
        return AnalysisResult(
            violations=violations,
            security_issues=issues,
            metrics={'lines_analyzed': len(lines)}
        )
    
    def _check_violation(self, line: str) -> bool:
        """Check if line violates rules."""
        # Implement your logic
        return 'TODO' in line
```

### 2. Pattern-Based Validator

```python
import re
from src.analyzers.base import BaseAnalyzer

class PatternValidator(BaseAnalyzer):
    """Validator using regex patterns."""
    
    def __init__(self):
        super().__init__()
        self.patterns = {
            'hardcoded-secret': re.compile(r'(api_key|password)\s*=\s*["\'][\w]+["\']'),
            'console-log': re.compile(r'console\.(log|debug|info)'),
            'sql-injection': re.compile(r'f["\']\s*SELECT.*{.*}')
        }
    
    def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
        violations = []
        
        for line_num, line in enumerate(code.split('\n'), 1):
            for rule_id, pattern in self.patterns.items():
                if pattern.search(line):
                    violations.append({
                        'rule': rule_id,
                        'message': f'Pattern "{rule_id}" detected',
                        'line': line_num,
                        'column': pattern.search(line).start(),
                        'severity': 'error' if 'secret' in rule_id else 'warning'
                    })
        
        return AnalysisResult(violations=violations)
```

### 3. AST-Based Validator

```python
import ast
from src.analyzers.base import BaseAnalyzer

class ASTValidator(BaseAnalyzer):
    """Validator using Abstract Syntax Tree analysis."""
    
    def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
        violations = []
        
        try:
            tree = ast.parse(code)
            visitor = ViolationVisitor()
            visitor.visit(tree)
            violations = visitor.violations
        except SyntaxError as e:
            # Handle syntax errors gracefully
            violations.append({
                'rule': 'syntax-error',
                'message': str(e),
                'line': e.lineno,
                'severity': 'error'
            })
        
        return AnalysisResult(violations=violations)

class ViolationVisitor(ast.NodeVisitor):
    """AST visitor to find violations."""
    
    def __init__(self):
        self.violations = []
    
    def visit_FunctionDef(self, node):
        # Check function complexity
        if self._calculate_complexity(node) > 10:
            self.violations.append({
                'rule': 'high-complexity',
                'message': f'Function {node.name} has high complexity',
                'line': node.lineno,
                'severity': 'warning'
            })
        
        # Check docstring
        if not ast.get_docstring(node):
            self.violations.append({
                'rule': 'missing-docstring',
                'message': f'Function {node.name} missing docstring',
                'line': node.lineno,
                'severity': 'warning'
            })
        
        self.generic_visit(node)
    
    def _calculate_complexity(self, node):
        # Simplified complexity calculation
        return len([n for n in ast.walk(node) if isinstance(n, ast.If)])
```

## Advanced Validator Features

### 1. Multi-Language Support

```python
class MultiLanguageValidator(BaseAnalyzer):
    """Validator supporting multiple languages."""
    
    def __init__(self):
        super().__init__()
        self.language_handlers = {
            'python': self._analyze_python,
            'javascript': self._analyze_javascript,
            'go': self._analyze_go
        }
    
    def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
        language = self._detect_language(file_path)
        handler = self.language_handlers.get(language, self._analyze_generic)
        return handler(code)
    
    def _detect_language(self, file_path: str) -> str:
        if not file_path:
            return 'unknown'
        
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.go': 'go'
        }
        
        import os
        ext = os.path.splitext(file_path)[1]
        return ext_map.get(ext, 'unknown')
```

### 2. Context-Aware Validation

```python
class ContextAwareValidator(BaseAnalyzer):
    """Validator that uses project context."""
    
    def analyze(self, code: str, file_path: str = None, context: dict = None) -> AnalysisResult:
        violations = []
        context = context or {}
        
        # Different rules for different project types
        if context.get('project_type') == 'library':
            violations.extend(self._check_library_rules(code))
        elif context.get('project_type') == 'application':
            violations.extend(self._check_application_rules(code))
        
        # Framework-specific rules
        if 'django' in context.get('frameworks', []):
            violations.extend(self._check_django_rules(code))
        
        return AnalysisResult(violations=violations)
```

### 3. Performance Optimization

```python
import functools
from concurrent.futures import ThreadPoolExecutor

class OptimizedValidator(BaseAnalyzer):
    """Performance-optimized validator."""
    
    def __init__(self):
        super().__init__()
        self._cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    @functools.lru_cache(maxsize=1000)
    def _cached_analysis(self, code_hash: str) -> List[dict]:
        """Cache analysis results."""
        # Expensive analysis here
        return self._do_analysis(code_hash)
    
    def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
        # Use hash for caching
        import hashlib
        code_hash = hashlib.md5(code.encode()).hexdigest()
        
        # Check cache first
        if code_hash in self._cache:
            return AnalysisResult(violations=self._cache[code_hash])
        
        # Parallel analysis for large files
        if len(code) > 10000:
            chunks = self._split_code(code)
            futures = [self.executor.submit(self._analyze_chunk, chunk) for chunk in chunks]
            violations = []
            for future in futures:
                violations.extend(future.result())
        else:
            violations = self._do_analysis(code)
        
        self._cache[code_hash] = violations
        return AnalysisResult(violations=violations)
```

## Testing Your Validator

### Unit Tests

```python
# tests/validators/test_my_validator.py
import pytest
from src.validators.my_validator import MyValidator

class TestMyValidator:
    def setup_method(self):
        self.validator = MyValidator()
    
    def test_detects_violation(self):
        code = """
        # TODO: Fix this later
        def bad_function():
            pass
        """
        
        result = self.validator.analyze(code)
        assert len(result.violations) == 1
        assert result.violations[0]['rule'] == 'my-rule'
        assert result.violations[0]['line'] == 2
    
    def test_clean_code_passes(self):
        code = """
        def good_function():
            '''Well documented function.'''
            return 42
        """
        
        result = self.validator.analyze(code)
        assert len(result.violations) == 0
    
    @pytest.mark.parametrize("code,expected_count", [
        ("TODO: fix", 1),
        ("# TODO: fix\n# TODO: another", 2),
        ("No todos here", 0)
    ])
    def test_multiple_cases(self, code, expected_count):
        result = self.validator.analyze(code)
        assert len(result.violations) == expected_count
```

### Integration Tests

```python
def test_validator_integration():
    """Test validator with standards engine."""
    from src.core.standards import StandardsEngine
    
    engine = StandardsEngine()
    engine.register_validator('my-validator', MyValidator)
    
    # Test with actual standard
    result = engine.validate_file(
        'test_file.py',
        validators=['my-validator']
    )
    
    assert result is not None
```

## Best Practices

### 1. Error Handling

```python
def analyze(self, code: str, file_path: str = None) -> AnalysisResult:
    try:
        # Your analysis
        pass
    except Exception as e:
        # Log error but don't crash
        import logging
        logging.error(f"Validator error: {e}")
        
        # Return partial results if possible
        return AnalysisResult(
            violations=[],
            errors=[str(e)]
        )
```

### 2. Performance Guidelines

- Cache expensive computations
- Use generators for large files
- Implement early exit conditions
- Profile your validator

### 3. Clear Messages

```python
violations.append({
    'rule': 'function-too-long',
    'message': f'Function "{func_name}" is {lines} lines long (max: 50)',
    'line': start_line,
    'severity': 'warning',
    'suggestion': 'Consider breaking this function into smaller functions'
})
```

## Registering Your Validator

### 1. Add to Registry

```python
# src/validators/__init__.py
from .my_validator import MyValidator

VALIDATORS = {
    'my-validator': MyValidator,
    # ... other validators
}
```

### 2. Configure in Standard

```yaml
# standards/my-standard.yaml
validators:
  - type: 'my-validator'
    config:
      strict_mode: true
      ignore_patterns: ['test_*']
```

## Debugging Tips

1. **Enable Debug Logging**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Add Breakpoints**
   ```python
   import pdb; pdb.set_trace()
   ```

3. **Test Incrementally**
   ```python
   # Test one rule at a time
   validator = MyValidator()
   validator.patterns = {'test-rule': re.compile(r'test')}
   ```

## Related Documentation

- [Standards Format](../api/standards-format.md)
- [Testing Guidelines](./testing.md)
- [Performance Optimization](../reference/performance.md)

Happy validator writing! 🚀