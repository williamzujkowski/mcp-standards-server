"""
AST utilities for language analyzers (simplified implementation)
@nist-controls: SA-11, SA-15
@evidence: AST parsing infrastructure for security analysis
"""
import ast
import re
from typing import Any


def get_python_functions(code: str) -> list[dict[str, Any]]:
    """Extract function definitions from Python code using AST"""
    functions = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append({
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'decorators': [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else [],
                    'is_async': isinstance(node, ast.AsyncFunctionDef)
                })

    except SyntaxError:
        # Fallback to regex
        pattern = r'^(?:async\s+)?def\s+(\w+)\s*\('
        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(pattern, line.strip())
            if match:
                functions.append({
                    'name': match.group(1),
                    'start_line': i,
                    'end_line': i,
                    'decorators': [],
                    'is_async': line.strip().startswith('async')
                })

    return functions


def get_python_classes(code: str) -> list[dict[str, Any]]:
    """Extract class definitions from Python code using AST"""
    classes = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                classes.append({
                    'name': node.name,
                    'start_line': node.lineno,
                    'end_line': node.end_lineno or node.lineno,
                    'decorators': [ast.unparse(d) for d in node.decorator_list] if hasattr(ast, 'unparse') else [],
                    'bases': [ast.unparse(b) for b in node.bases] if hasattr(ast, 'unparse') else []
                })

    except SyntaxError:
        # Fallback to regex
        pattern = r'^class\s+(\w+)(?:\((.*?)\))?:'
        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(pattern, line.strip())
            if match:
                classes.append({
                    'name': match.group(1),
                    'start_line': i,
                    'end_line': i,
                    'decorators': [],
                    'bases': [b.strip() for b in match.group(2).split(',')] if match.group(2) else []
                })

    return classes


def get_python_imports(code: str) -> list[dict[str, Any]]:
    """Extract import statements from Python code using AST"""
    imports = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                        'type': 'import'
                    })
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno,
                        'type': 'from'
                    })

    except SyntaxError:
        # Fallback to regex
        import_pattern = r'^import\s+([\w.]+)(?:\s+as\s+(\w+))?'
        from_pattern = r'^from\s+([\w.]+)\s+import\s+(.+)'

        for i, line in enumerate(code.splitlines(), 1):
            line = line.strip()

            match = re.match(import_pattern, line)
            if match:
                imports.append({
                    'module': match.group(1),
                    'alias': match.group(2),
                    'line': i,
                    'type': 'import'
                })
                continue

            match = re.match(from_pattern, line)
            if match:
                module = match.group(1)
                items = match.group(2)
                # Simple parsing, doesn't handle all cases
                for item in items.split(','):
                    item = item.strip()
                    if ' as ' in item:
                        name, alias = item.split(' as ')
                        imports.append({
                            'module': module,
                            'name': name.strip(),
                            'alias': alias.strip(),
                            'line': i,
                            'type': 'from'
                        })
                    else:
                        imports.append({
                            'module': module,
                            'name': item,
                            'alias': None,
                            'line': i,
                            'type': 'from'
                        })

    return imports


def get_python_decorators(code: str) -> list[dict[str, Any]]:
    """Extract decorators from Python code"""
    decorators = []

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.ClassDef):
                for decorator in node.decorator_list:
                    decorator_info = {
                        'line': decorator.lineno,
                        'target_line': node.lineno,
                        'target_name': node.name,
                        'target_type': 'function' if isinstance(node, ast.FunctionDef) else 'class'
                    }

                    # Extract decorator name
                    if isinstance(decorator, ast.Name):
                        decorator_info['name'] = decorator.id
                    elif isinstance(decorator, ast.Attribute):
                        decorator_info['name'] = decorator.attr
                    elif isinstance(decorator, ast.Call):
                        if isinstance(decorator.func, ast.Name):
                            decorator_info['name'] = decorator.func.id
                        elif isinstance(decorator.func, ast.Attribute):
                            decorator_info['name'] = decorator.func.attr

                    decorators.append(decorator_info)

    except SyntaxError:
        # Fallback to regex
        decorator_pattern = r'^@(\w+)(?:\((.*?)\))?'
        lines = code.splitlines()

        for i, line in enumerate(lines):
            line = line.strip()
            match = re.match(decorator_pattern, line)
            if match:
                # Look for the decorated function/class
                for j in range(i + 1, min(i + 10, len(lines))):
                    target_line = lines[j].strip()
                    if target_line and not target_line.startswith('@'):
                        func_match = re.match(r'(?:async\s+)?def\s+(\w+)', target_line)
                        class_match = re.match(r'class\s+(\w+)', target_line)

                        if func_match or class_match:
                            decorators.append({
                                'line': i + 1,
                                'name': match.group(1),
                                'target_line': j + 1,
                                'target_name': func_match.group(1) if func_match else class_match.group(1),
                                'target_type': 'function' if func_match else 'class'
                            })
                            break

    return decorators


def get_javascript_functions(code: str) -> list[dict[str, Any]]:
    """Extract function definitions from JavaScript code using regex"""
    functions = []

    # Function patterns
    patterns = [
        # Function declaration
        (r'^(?:async\s+)?function\s+(\w+)\s*\(', 'declaration'),
        # Function expression
        (r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function', 'expression'),
        # Arrow function
        (r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(', 'arrow'),
        # Method definition
        (r'^\s*(?:async\s+)?(\w+)\s*\(.*?\)\s*{', 'method'),
        # Class method
        (r'^\s*(?:static\s+)?(?:async\s+)?(\w+)\s*\(', 'class_method'),
    ]

    lines = code.splitlines()
    for i, line in enumerate(lines):
        for pattern, func_type in patterns:
            match = re.search(pattern, line)
            if match:
                functions.append({
                    'name': match.group(1),
                    'start_line': i + 1,
                    'type': func_type,
                    'is_async': 'async' in line
                })
                break

    return functions


def get_javascript_classes(code: str) -> list[dict[str, Any]]:
    """Extract class definitions from JavaScript code"""
    classes = []

    pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
    for i, line in enumerate(code.splitlines(), 1):
        match = re.search(pattern, line)
        if match:
            classes.append({
                'name': match.group(1),
                'start_line': i,
                'extends': match.group(2)
            })

    return classes


def get_javascript_imports(code: str) -> list[dict[str, Any]]:
    """Extract import statements from JavaScript code"""
    imports = []

    patterns = [
        # ES6 imports
        (r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)', 'default'),
        (r'import\s*\{([^}]+)\}\s*from\s+[\'"]([^\'"]+)', 'named'),
        (r'import\s*\*\s*as\s+(\w+)\s+from\s+[\'"]([^\'"]+)', 'namespace'),
        # CommonJS
        (r'(?:const|let|var)\s+(\w+)\s*=\s*require\s*\([\'"]([^\'"]+)', 'commonjs'),
        (r'(?:const|let|var)\s+\{([^}]+)\}\s*=\s*require\s*\([\'"]([^\'"]+)', 'commonjs_destructure'),
    ]

    for i, line in enumerate(code.splitlines(), 1):
        for pattern, import_type in patterns:
            match = re.search(pattern, line)
            if match:
                imports.append({
                    'line': i,
                    'type': import_type,
                    'names': match.group(1),
                    'module': match.group(2) if match.lastindex > 1 else match.group(1)
                })

    return imports
