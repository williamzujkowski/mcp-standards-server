"""
Python code analyzer
@nist-controls: SA-11, SA-15
@evidence: Python static analysis for security controls
"""
import ast
import re
from pathlib import Path

from .base import BaseAnalyzer, CodeAnnotation


class PythonAnalyzer(BaseAnalyzer):
    """
    Analyzes Python code for NIST control implementations
    @nist-controls: SA-11, CA-7
    @evidence: Python-specific security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.py']
        self.framework_patterns = {
            'django': ['django', 'from django', 'DJANGO_SETTINGS'],
            'flask': ['flask', 'from flask', 'Flask(__name__)'],
            'fastapi': ['fastapi', 'from fastapi', 'FastAPI()'],
            'pyramid': ['pyramid', 'from pyramid']
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Python file for NIST controls"""
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

        # Find implicit patterns
        implicit_annotations = self._analyze_implicit_patterns(code, str(file_path))
        annotations.extend(implicit_annotations)

        # AST-based analysis
        try:
            tree = ast.parse(code)
            ast_annotations = self._analyze_ast(tree, code, str(file_path))
            annotations.extend(ast_annotations)
        except SyntaxError:
            # If we can't parse, continue with pattern matching
            pass

        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []

        # Authentication patterns
        auth_patterns = [
            (r'@login_required|@require_auth|@authenticated', ["IA-2", "AC-3"], "Authentication decorator"),
            (r'authenticate\(|login\(|verify_password', ["IA-2", "AC-7"], "Authentication function"),
            (r'JWT|jwt|JsonWebToken', ["IA-2", "SC-8"], "JWT token handling"),
            (r'OAuth|oauth2', ["IA-2", "IA-8"], "OAuth authentication"),
            (r'Session\(|session\[', ["SC-23", "AC-12"], "Session management")
        ]

        for pattern, controls, evidence in auth_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="authentication",
                    confidence=0.85
                ))

        # Encryption patterns
        crypto_patterns = [
            (r'from cryptography|import cryptography', ["SC-13", "SC-28"], "Cryptography library"),
            (r'hashlib\.|bcrypt\.|argon2', ["IA-5", "SC-13"], "Password hashing"),
            (r'Fernet\(|AES\.new|RSA\.', ["SC-13", "SC-28"], "Encryption implementation"),
            (r'ssl\.|TLS|https://', ["SC-8", "SC-13"], "SSL/TLS usage"),
            (r'secrets\.|os\.urandom', ["SC-13"], "Secure random generation")
        ]

        for pattern, controls, evidence in crypto_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="encryption",
                    confidence=0.9
                ))

        # Input validation patterns
        validation_patterns = [
            (r'validate|sanitize|clean|escape', ["SI-10", "SI-15"], "Input validation"),
            (r'schema\.|Schema\(|marshmallow', ["SI-10"], "Schema validation"),
            (r'bleach\.|html\.escape', ["SI-10"], "Output sanitization"),
            (r'parameterized|execute.*%s|\\?', ["SI-10"], "SQL injection prevention"),
            (r're\.match|re\.compile', ["SI-10"], "Regular expression validation")
        ]

        for pattern, controls, evidence in validation_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="input-validation",
                    confidence=0.85
                ))

        # Logging patterns
        logging_patterns = [
            (r'logging\.|logger\.|log\.', ["AU-2", "AU-3"], "Logging implementation"),
            (r'audit_log|security_log|event_log', ["AU-2", "AU-9"], "Audit logging"),
            (r'structlog|loguru', ["AU-2", "AU-3"], "Structured logging"),
            (r'syslog\.|SysLogHandler', ["AU-3", "AU-9"], "System logging")
        ]

        for pattern, controls, evidence in logging_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="logging",
                    confidence=0.8
                ))

        # Access control patterns
        authz_patterns = [
            (r'@permission_required|@require_role|has_permission', ["AC-3", "AC-6"], "Permission checking"),
            (r'check_permission|authorize|can_access', ["AC-3", "AC-6"], "Authorization function"),
            (r'rbac|role_based|ACL', ["AC-2", "AC-3"], "Role-based access control"),
            (r'@admin_required|is_admin|is_superuser', ["AC-6"], "Privileged access control")
        ]

        for pattern, controls, evidence in authz_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="authorization",
                    confidence=0.85
                ))

        # Django-specific patterns
        django_patterns = [
            (r'@csrf_protect|{% csrf_token', ["SI-10", "SC-8"], "CSRF protection"),
            (r'SECURE_SSL_REDIRECT|SecurityMiddleware', ["SC-8", "SC-13"], "Django security middleware"),
            (r'UserPassesTestMixin|PermissionRequiredMixin', ["AC-3", "AC-6"], "Django authorization"),
            (r'ContentTypeOptionsMiddleware|XFrameOptionsMiddleware', ["SC-18"], "Security headers")
        ]

        for pattern, controls, evidence in django_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="django-security",
                    confidence=0.9
                ))

        # Flask-specific patterns
        flask_patterns = [
            (r'@app\.before_request|@login_manager', ["IA-2", "AC-3"], "Flask authentication"),
            (r'flask_login|Flask-Security', ["IA-2", "AC-3"], "Flask security extension"),
            (r'flask_cors|CORS\(app', ["AC-4", "SC-8"], "CORS configuration"),
            (r'flask_limiter|RateLimiter', ["SC-5"], "Rate limiting")
        ]

        for pattern, controls, evidence in flask_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="flask-security",
                    confidence=0.85
                ))

        return annotations

    def _analyze_ast(self, tree: ast.AST, code: str, file_path: str) -> list[CodeAnnotation]:  # noqa: ARG002
        """Analyze Python AST for security patterns"""
        annotations = []

        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, analyzer):
                self.analyzer = analyzer
                self.annotations = []

            def visit_FunctionDef(self, node):
                # Check for authentication decorators
                for decorator in node.decorator_list:
                    decorator_name = self._get_decorator_name(decorator)
                    if decorator_name and any(auth in decorator_name.lower()
                                            for auth in ['auth', 'login', 'permission', 'role']):
                        self.annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=node.lineno,
                            control_ids=["IA-2", "AC-3"],
                            evidence=f"Security decorator: {decorator_name}",
                            component="authentication",
                            confidence=0.9
                        ))

                # Check function names
                if any(sec in node.name.lower()
                      for sec in ['authenticate', 'authorize', 'login', 'verify', 'validate']):
                    controls = []
                    if 'auth' in node.name.lower() or 'login' in node.name.lower():
                        controls = ["IA-2", "AC-7"]
                    elif 'validate' in node.name.lower():
                        controls = ["SI-10"]

                    if controls:
                        self.annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=node.lineno,
                            control_ids=controls,
                            evidence=f"Security function: {node.name}",
                            component="security-function",
                            confidence=0.8
                        ))

                self.generic_visit(node)

            def visit_Import(self, node):
                for alias in node.names:
                    self._check_security_import(alias.name, node.lineno)
                self.generic_visit(node)

            def visit_ImportFrom(self, node):
                if node.module:
                    self._check_security_import(node.module, node.lineno)
                self.generic_visit(node)

            def _get_decorator_name(self, decorator):
                if isinstance(decorator, ast.Name):
                    return decorator.id
                elif isinstance(decorator, ast.Attribute):
                    return decorator.attr
                elif isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Name):
                        return decorator.func.id
                    elif isinstance(decorator.func, ast.Attribute):
                        return decorator.func.attr
                return None

            def _check_security_import(self, module_name, lineno):
                security_modules = {
                    'cryptography': ["SC-13", "SC-28"],
                    'hashlib': ["SC-13", "SI-7"],
                    'secrets': ["SC-13"],
                    'ssl': ["SC-8", "SC-13"],
                    'hmac': ["SC-13", "SI-7"],
                    'jwt': ["IA-2", "SC-8"],
                    'bcrypt': ["IA-5", "SC-13"],
                    'passlib': ["IA-5", "SC-13"],
                    'django.contrib.auth': ["IA-2", "AC-3"],
                    'flask_login': ["IA-2", "AC-3"],
                    'oauthlib': ["IA-2", "IA-8"]
                }

                for sec_module, controls in security_modules.items():
                    if sec_module in module_name:
                        self.annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=lineno,
                            control_ids=controls,
                            evidence=f"Security module import: {module_name}",
                            component="imports",
                            confidence=0.8
                        ))
                        break

        visitor = SecurityVisitor(self)
        visitor.visit(tree)
        annotations.extend(visitor.annotations)

        return annotations

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire Python project"""
        results = {}

        # Common directories to skip
        skip_dirs = {
            'venv', '__pycache__', '.env', '.venv', 'env',
            'build', 'dist', '.git', '.pytest_cache', '.tox',
            'node_modules', 'migrations'
        }

        for file_path in project_path.rglob('*.py'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip test files unless they contain security tests
            if 'test' in file_path.name or file_path.name.startswith('test_'):
                with open(file_path, encoding='utf-8') as f:
                    content = f.read().lower()
                    if 'security' not in content and 'auth' not in content:
                        continue

            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations

        # Analyze requirements.txt or setup.py for security insights
        requirements_path = project_path / 'requirements.txt'
        setup_path = project_path / 'setup.py'
        pyproject_path = project_path / 'pyproject.toml'

        if requirements_path.exists():
            req_annotations = self._analyze_requirements(requirements_path)
            if req_annotations:
                results[str(requirements_path)] = req_annotations

        if setup_path.exists():
            setup_annotations = self.analyze_file(setup_path)
            if setup_annotations:
                results[str(setup_path)] = setup_annotations

        if pyproject_path.exists():
            pyproject_annotations = self._analyze_pyproject(pyproject_path)
            if pyproject_annotations:
                results[str(pyproject_path)] = pyproject_annotations

        return results

    def _analyze_requirements(self, requirements_path: Path) -> list[CodeAnnotation]:
        """Analyze requirements.txt for security-relevant packages"""
        annotations = []

        security_packages = {
            'cryptography': ["SC-13", "SC-28"],
            'pycryptodome': ["SC-13", "SC-28"],
            'bcrypt': ["IA-5", "SC-13"],
            'passlib': ["IA-5", "SC-13"],
            'pyjwt': ["IA-2", "SC-8"],
            'python-jose': ["IA-2", "SC-8"],
            'django': ["AC-3", "SC-8", "SI-10"],
            'flask-security': ["IA-2", "AC-3"],
            'flask-login': ["IA-2", "AC-3"],
            'oauthlib': ["IA-2", "IA-8"],
            'python-dotenv': ["CM-7", "SC-28"],
            'bleach': ["SI-10"],
            'python-secrets': ["SC-13"]
        }

        try:
            with open(requirements_path) as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        for pkg, controls in security_packages.items():
                            if pkg in line.lower():
                                annotations.append(CodeAnnotation(
                                    file_path=str(requirements_path),
                                    line_number=i,
                                    control_ids=controls,
                                    evidence=f"Security dependency: {pkg}",
                                    component="dependencies",
                                    confidence=0.8
                                ))
                                break
        except Exception:
            pass

        return annotations

    def _analyze_pyproject(self, pyproject_path: Path) -> list[CodeAnnotation]:
        """Analyze pyproject.toml for security-relevant dependencies"""
        # Similar analysis to requirements.txt
        return self._analyze_requirements(pyproject_path)

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for Python code"""
        suggestions = set()
        patterns = self.find_security_patterns(code, "temp.py")

        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)

        # Framework-specific suggestions
        code_lower = code.lower()

        # Django controls
        if 'django' in code_lower:
            suggestions.update(["AC-3", "AC-4", "SC-8", "SI-10", "AU-2"])

        # Flask controls
        if 'flask' in code_lower:
            suggestions.update(["AC-3", "SC-8", "SI-10", "AU-2"])

        # FastAPI controls
        if 'fastapi' in code_lower:
            suggestions.update(["AC-3", "SC-8", "SI-10", "AC-4"])

        # Database controls
        if any(db in code_lower for db in ['sqlalchemy', 'psycopg', 'pymongo', 'redis']):
            suggestions.update(["SC-28", "SI-10", "AU-2"])

        # Cloud SDK controls
        if any(cloud in code_lower for cloud in ['boto3', 'azure', 'google-cloud']):
            suggestions.update(["AC-2", "AU-2", "SC-28", "SC-8"])

        # ML/AI controls
        if any(ml in code_lower for ml in ['tensorflow', 'pytorch', 'sklearn']):
            suggestions.update(["SI-10", "AC-4", "SC-28"])

        return sorted(suggestions)
