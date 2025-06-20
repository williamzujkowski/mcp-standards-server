"""
Enhanced Python code analyzer with tree-sitter
@nist-controls: SA-11, SA-15
@evidence: Advanced Python AST analysis for security controls
"""
import re
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, CodeAnnotation
from .ast_utils import (
    get_python_functions,
    get_python_imports,
    get_python_classes,
    get_python_decorators
)


class PythonAnalyzer(BaseAnalyzer):
    """
    Enhanced Python analyzer using tree-sitter for deeper AST analysis
    @nist-controls: SA-11, CA-7
    @evidence: Tree-sitter based Python security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.py']
        self.language = 'python'
        
        # Security-relevant imports mapping
        self.security_imports = {
            # Authentication/Authorization
            'django.contrib.auth': ['IA-2', 'AC-3'],
            'flask_login': ['IA-2', 'AC-3'],
            'flask_jwt_extended': ['IA-2', 'SC-8'],
            'passlib': ['IA-5', 'SC-13'],
            'argon2': ['IA-5', 'SC-13'],
            'bcrypt': ['IA-5', 'SC-13'],
            'jwt': ['IA-2', 'SC-8'],
            'oauthlib': ['IA-2', 'IA-8'],
            'authlib': ['IA-2', 'IA-8'],
            
            # Cryptography
            'cryptography': ['SC-13', 'SC-28'],
            'pycryptodome': ['SC-13', 'SC-28'],
            'hashlib': ['SC-13', 'SI-7'],
            'hmac': ['SC-13', 'SI-7'],
            'secrets': ['SC-13'],
            'ssl': ['SC-8', 'SC-13'],
            
            # Input Validation
            'bleach': ['SI-10'],
            'html': ['SI-10'],
            'validators': ['SI-10'],
            'marshmallow': ['SI-10'],
            'pydantic': ['SI-10'],
            'cerberus': ['SI-10'],
            
            # Logging/Auditing
            'logging': ['AU-2', 'AU-3'],
            'structlog': ['AU-2', 'AU-3'],
            'loguru': ['AU-2', 'AU-3'],
            'sentry_sdk': ['AU-2', 'AU-14'],
            
            # Session Management
            'flask.sessions': ['SC-23', 'AC-12'],
            'django.contrib.sessions': ['SC-23', 'AC-12'],
            'beaker': ['SC-23', 'AC-12'],
            
            # Security Middleware
            'django.middleware.security': ['SC-8', 'SC-18'],
            'flask_talisman': ['SC-8', 'SC-18'],
            'secure': ['SC-8', 'SC-18'],
        }
        
        # Security function patterns
        self.security_functions = {
            # Authentication
            'authenticate': ['IA-2', 'AC-7'],
            'login': ['IA-2', 'AC-7'],
            'logout': ['AC-12'],
            'verify_password': ['IA-2', 'IA-5'],
            'check_password': ['IA-2', 'IA-5'],
            'validate_token': ['IA-2', 'SC-8'],
            'verify_token': ['IA-2', 'SC-8'],
            
            # Authorization
            'authorize': ['AC-3', 'AC-6'],
            'check_permission': ['AC-3', 'AC-6'],
            'has_permission': ['AC-3', 'AC-6'],
            'require_role': ['AC-3', 'AC-6'],
            'is_admin': ['AC-6'],
            
            # Encryption
            'encrypt': ['SC-13', 'SC-28'],
            'decrypt': ['SC-13', 'SC-28'],
            'hash_password': ['IA-5', 'SC-13'],
            'generate_key': ['SC-13'],
            'sign': ['SC-13', 'SI-7'],
            'verify_signature': ['SC-13', 'SI-7'],
            
            # Input Validation
            'validate': ['SI-10'],
            'sanitize': ['SI-10'],
            'escape': ['SI-10'],
            'clean': ['SI-10'],
            'filter_input': ['SI-10'],
            
            # Logging
            'audit_log': ['AU-2', 'AU-3'],
            'log_event': ['AU-2', 'AU-3'],
            'log_security': ['AU-2', 'AU-9'],
            'log_access': ['AU-2', 'AC-3'],
        }
        
        # Security decorators
        self.security_decorators = {
            # Django
            'login_required': ['IA-2', 'AC-3'],
            'permission_required': ['AC-3', 'AC-6'],
            'user_passes_test': ['AC-3', 'AC-6'],
            'csrf_protect': ['SI-10', 'SC-8'],
            'require_http_methods': ['AC-4'],
            'cache_control': ['SC-28'],
            
            # Flask
            'jwt_required': ['IA-2', 'SC-8'],
            'roles_required': ['AC-3', 'AC-6'],
            'auth_required': ['IA-2', 'AC-3'],
            'limiter.limit': ['SC-5'],
            
            # FastAPI
            'Depends': ['AC-3'],
            'HTTPBearer': ['IA-2', 'SC-8'],
            'OAuth2PasswordBearer': ['IA-2', 'IA-8'],
            
            # General
            'authenticated': ['IA-2'],
            'authorized': ['AC-3'],
            'validate_input': ['SI-10'],
            'rate_limit': ['SC-5'],
            'cache': ['SC-28'],
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Python file using tree-sitter for NIST controls"""
        if file_path.suffix not in self.file_extensions:
            return []

        try:
            with open(file_path, encoding='utf-8') as f:
                code = f.read()
        except Exception:
            return []

        annotations = []

        # Extract explicit annotations
        annotations.extend(self.extract_annotations(code, str(file_path)))

        # AST-based analysis
        annotations.extend(self._analyze_with_ast(code, str(file_path)))

        # Pattern-based analysis (fallback and additional patterns)
        annotations.extend(self._analyze_implicit_patterns(code, str(file_path)))
        
        # Enhanced pattern detection
        annotations.extend(self.analyze_with_enhanced_patterns(code, str(file_path)))

        # Deduplicate annotations
        seen = set()
        unique_annotations = []
        for ann in annotations:
            key = (ann.file_path, ann.line_number, tuple(ann.control_ids))
            if key not in seen:
                seen.add(key)
                unique_annotations.append(ann)

        return unique_annotations

    def _analyze_with_ast(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code using AST parsing"""
        annotations = []
        
        # Analyze imports
        imports = get_python_imports(code)
        for import_info in imports:
            module = import_info.get('module', '')
            if module:
                # Check against security imports
                for sec_module, controls in self.security_imports.items():
                    if sec_module in module or module.startswith(sec_module):
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=import_info['line'],
                            control_ids=controls,
                            evidence=f"Security module import: {module}",
                            component="imports",
                            confidence=0.9
                        ))
                        break
        
        # Analyze function definitions
        functions = get_python_functions(code)
        for func in functions:
            func_name = func.get('name', '').lower()
            if func_name:
                # Check against security functions
                for pattern, controls in self.security_functions.items():
                    if pattern in func_name:
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=func['start_line'],
                            control_ids=controls,
                            evidence=f"Security function: {func['name']}",
                            component="function",
                            confidence=0.85
                        ))
                        break
        
        # Analyze decorators
        decorators = get_python_decorators(code)
        for decorator in decorators:
            decorator_name = decorator.get('name', '').lower()
            for dec_name, controls in self.security_decorators.items():
                if dec_name.lower() in decorator_name:
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=decorator['line'],
                        control_ids=controls,
                        evidence=f"Security decorator: @{decorator['name']}",
                        component="decorator",
                        confidence=0.9
                    ))
                    break
        
        # Analyze class definitions for security patterns
        classes = get_python_classes(code)
        for cls in classes:
            class_name = cls.get('name', '').lower()
            if any(sec in class_name for sec in ['auth', 'permission', 'security', 'crypto', 'validator']):
                # Determine controls based on class name
                controls = []
                if 'auth' in class_name:
                    controls = ['IA-2', 'AC-3']
                elif 'permission' in class_name or 'role' in class_name:
                    controls = ['AC-3', 'AC-6']
                elif 'crypto' in class_name or 'encrypt' in class_name:
                    controls = ['SC-13', 'SC-28']
                elif 'validator' in class_name:
                    controls = ['SI-10']
                
                if controls:
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=cls['start_line'],
                        control_ids=controls,
                        evidence=f"Security class: {cls['name']}",
                        component="class",
                        confidence=0.8
                    ))
        
        # Check for exception handling patterns
        try_except_pattern = r'except\s+\w*(?:Authentication|Permission|Unauthorized|Forbidden)'
        for i, line in enumerate(code.splitlines(), 1):
            if re.search(try_except_pattern, line, re.IGNORECASE):
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=i,
                    control_ids=['IA-2', 'AC-7'],
                    evidence="Authentication/authorization error handling",
                    component="error-handling",
                    confidence=0.7
                ))
        
        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns using regex"""
        annotations = []

        # Framework-specific security configurations
        django_patterns = [
            (r'ALLOWED_HOSTS\s*=', ['SC-8', 'SC-18'], "Django allowed hosts configuration"),
            (r'SECURE_SSL_REDIRECT\s*=\s*True', ['SC-8', 'SC-13'], "Django SSL redirect enabled"),
            (r'SESSION_COOKIE_SECURE\s*=\s*True', ['SC-8', 'SC-23'], "Secure session cookies"),
            (r'CSRF_COOKIE_SECURE\s*=\s*True', ['SC-8', 'SI-10'], "Secure CSRF cookies"),
            (r'X_FRAME_OPTIONS\s*=', ['SC-18'], "Clickjacking protection"),
            (r'SECURE_HSTS_SECONDS\s*=', ['SC-8'], "HSTS security header"),
        ]

        flask_patterns = [
            (r'app\.config\[.SECRET_KEY.\]', ['SC-13', 'SC-28'], "Flask secret key configuration"),
            (r'SESSION_COOKIE_SECURE\s*=\s*True', ['SC-8', 'SC-23'], "Secure session cookies"),
            (r'SESSION_COOKIE_HTTPONLY\s*=\s*True', ['SC-8', 'SC-23'], "HTTPOnly session cookies"),
            (r'WTF_CSRF_ENABLED\s*=\s*True', ['SI-10'], "CSRF protection enabled"),
        ]

        # SQL injection prevention patterns
        sql_patterns = [
            (r'execute\s*\(\s*["\'].*%s', ['SI-10'], "Parameterized SQL query"),
            (r'execute\s*\(\s*["\'].*\?', ['SI-10'], "Parameterized SQL query"),
            (r'prepare\s*\(|prepared\s+statement', ['SI-10'], "Prepared statement"),
            (r'escape_string|quote|escapeshellarg', ['SI-10'], "Input escaping"),
        ]

        # Rate limiting patterns
        rate_limit_patterns = [
            (r'@limiter\.limit|RateLimiter|throttle', ['SC-5'], "Rate limiting implementation"),
            (r'requests_per_minute|rate_limit|quota', ['SC-5'], "Rate limiting configuration"),
        ]

        # All pattern groups
        all_patterns = [
            (django_patterns, "Django security"),
            (flask_patterns, "Flask security"),
            (sql_patterns, "SQL injection prevention"),
            (rate_limit_patterns, "Rate limiting"),
        ]

        for pattern_group, group_name in all_patterns:
            for pattern, controls, evidence in pattern_group:
                if re.search(pattern, code, re.IGNORECASE):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=evidence,
                        component=group_name,
                        confidence=0.85
                    ))

        return annotations

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for Python code using AST analysis"""
        suggestions = set()
        
        # Use AST to analyze the code
        imports = get_python_imports(code)
        functions = get_python_functions(code)
        classes = get_python_classes(code)
        
        # Check imports
        for import_info in imports:
            module = import_info.get('module', '').lower()
            
            # Web frameworks
            if any(fw in module for fw in ['django', 'flask', 'fastapi', 'pyramid']):
                suggestions.update(['AC-3', 'AC-4', 'SC-8', 'SI-10', 'AU-2'])
            
            # Authentication libraries
            if any(auth in module for auth in ['auth', 'login', 'jwt', 'oauth']):
                suggestions.update(['IA-2', 'IA-5', 'IA-8', 'AC-3'])
            
            # Cryptography
            if any(crypto in module for crypto in ['crypto', 'hashlib', 'hmac', 'ssl']):
                suggestions.update(['SC-13', 'SC-28', 'SC-8'])
            
            # Database
            if any(db in module for db in ['sqlalchemy', 'psycopg', 'pymongo', 'redis']):
                suggestions.update(['SC-28', 'SI-10', 'AU-2'])
            
            # Cloud SDKs
            if any(cloud in module for cloud in ['boto3', 'azure', 'google.cloud']):
                suggestions.update(['AC-2', 'AU-2', 'SC-28', 'SC-8'])
        
        # Check functions
        for func in functions:
            func_name = func.get('name', '').lower()
            
            if any(auth in func_name for auth in ['auth', 'login', 'verify']):
                suggestions.update(['IA-2', 'AC-3', 'AC-7'])
            
            if any(crypto in func_name for crypto in ['encrypt', 'decrypt', 'hash']):
                suggestions.update(['SC-13', 'SC-28'])
            
            if any(val in func_name for val in ['validate', 'sanitize', 'clean']):
                suggestions.update(['SI-10'])
            
            if any(log in func_name for log in ['log', 'audit', 'track']):
                suggestions.update(['AU-2', 'AU-3'])
        
        # Check classes
        for cls in classes:
            class_name = cls.get('name', '').lower()
            
            if any(sec in class_name for sec in ['auth', 'user', 'permission']):
                suggestions.update(['IA-2', 'AC-3', 'AC-6'])
            
            if any(crypto in class_name for crypto in ['crypto', 'cipher', 'key']):
                suggestions.update(['SC-13', 'SC-28'])
        
        # Also run pattern-based suggestions
        patterns = self.find_security_patterns(code, "temp.py")
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)
        
        return sorted(suggestions)

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire Python project with tree-sitter"""
        results = {}

        # Common directories to skip
        skip_dirs = {
            'venv', '__pycache__', '.env', '.venv', 'env',
            'build', 'dist', '.git', '.pytest_cache', '.tox',
            'node_modules', 'migrations', '.mypy_cache', 'htmlcov'
        }

        for file_path in project_path.rglob('*.py'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip test files unless they contain security tests
            if 'test' in file_path.name or file_path.name.startswith('test_'):
                with open(file_path, encoding='utf-8') as f:
                    content = f.read().lower()
                    if not any(term in content for term in ['security', 'auth', 'permission', 'crypto']):
                        continue

            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations

        # Analyze project configuration files
        config_files = [
            'requirements.txt', 'setup.py', 'pyproject.toml',
            'Pipfile', 'poetry.lock', 'setup.cfg'
        ]
        
        for config_file in config_files:
            config_path = project_path / config_file
            if config_path.exists():
                config_annotations = self._analyze_config_file(config_path)
                if config_annotations:
                    results[str(config_path)] = config_annotations

        return results

    def _analyze_config_file(self, config_path: Path) -> list[CodeAnnotation]:
        """Analyze Python project configuration files"""
        annotations = []
        
        security_packages = {
            # Authentication/Authorization
            'django-allauth': ['IA-2', 'IA-8'],
            'django-guardian': ['AC-3', 'AC-6'],
            'flask-login': ['IA-2', 'AC-3'],
            'flask-security': ['IA-2', 'AC-3'],
            'flask-jwt-extended': ['IA-2', 'SC-8'],
            'python-jose': ['IA-2', 'SC-8'],
            'pyjwt': ['IA-2', 'SC-8'],
            'oauthlib': ['IA-2', 'IA-8'],
            'authlib': ['IA-2', 'IA-8'],
            
            # Cryptography
            'cryptography': ['SC-13', 'SC-28'],
            'pycryptodome': ['SC-13', 'SC-28'],
            'bcrypt': ['IA-5', 'SC-13'],
            'passlib': ['IA-5', 'SC-13'],
            'argon2-cffi': ['IA-5', 'SC-13'],
            
            # Input Validation
            'bleach': ['SI-10'],
            'python-html-sanitizer': ['SI-10'],
            'validators': ['SI-10'],
            'marshmallow': ['SI-10'],
            'pydantic': ['SI-10'],
            'cerberus': ['SI-10'],
            
            # Security Tools
            'bandit': ['SA-11', 'SA-15'],
            'safety': ['SA-11', 'SI-2'],
            'python-dotenv': ['CM-7', 'SC-28'],
            
            # Logging/Monitoring
            'python-json-logger': ['AU-2', 'AU-3'],
            'structlog': ['AU-2', 'AU-3'],
            'loguru': ['AU-2', 'AU-3'],
            'sentry-sdk': ['AU-2', 'AU-14'],
        }
        
        try:
            with open(config_path, encoding='utf-8') as f:
                content = f.read()
                
            for i, line in enumerate(content.splitlines(), 1):
                line_lower = line.lower().strip()
                if line_lower and not line_lower.startswith('#'):
                    for pkg, controls in security_packages.items():
                        if pkg in line_lower:
                            annotations.append(CodeAnnotation(
                                file_path=str(config_path),
                                line_number=i,
                                control_ids=controls,
                                evidence=f"Security dependency: {pkg}",
                                component="dependencies",
                                confidence=0.85
                            ))
                            break
        except Exception:
            pass
            
        return annotations