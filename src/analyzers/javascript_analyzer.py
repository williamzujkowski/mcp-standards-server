"""
Enhanced JavaScript/TypeScript code analyzer
@nist-controls: SA-11, SA-15
@evidence: Advanced JavaScript/TypeScript analysis for security controls
"""
import re
from pathlib import Path

from .ast_utils import (
    get_javascript_functions,
    get_javascript_imports,
)
from .base import BaseAnalyzer, CodeAnnotation


class JavaScriptAnalyzer(BaseAnalyzer):
    """
    Enhanced JavaScript/TypeScript analyzer with improved pattern detection
    @nist-controls: SA-11, CA-7
    @evidence: JavaScript/TypeScript security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']
        self.language = 'javascript'

        # Security-relevant imports/packages
        self.security_packages = {
            # Authentication/Authorization
            'passport': ['IA-2', 'IA-8'],
            'passport-jwt': ['IA-2', 'SC-8'],
            'jsonwebtoken': ['IA-2', 'SC-8'],
            'express-jwt': ['IA-2', 'SC-8'],
            '@auth0/': ['IA-2', 'IA-8'],
            'firebase-auth': ['IA-2', 'IA-8'],
            'express-session': ['SC-23', 'AC-12'],
            'cookie-session': ['SC-23', 'AC-12'],

            # Cryptography
            'crypto': ['SC-13', 'SC-28'],
            'bcrypt': ['IA-5', 'SC-13'],
            'bcryptjs': ['IA-5', 'SC-13'],
            'argon2': ['IA-5', 'SC-13'],
            'node-forge': ['SC-13', 'SC-28'],
            'crypto-js': ['SC-13', 'SC-28'],

            # Input Validation
            'validator': ['SI-10'],
            'express-validator': ['SI-10'],
            'joi': ['SI-10'],
            'yup': ['SI-10'],
            'ajv': ['SI-10'],
            'dompurify': ['SI-10'],
            'sanitize-html': ['SI-10'],

            # Security Middleware
            'helmet': ['SC-8', 'SC-18'],
            'cors': ['AC-4', 'SC-8'],
            'express-rate-limit': ['SC-5'],
            'express-brute': ['SC-5'],
            'csurf': ['SI-10', 'SC-8'],

            # Logging
            'winston': ['AU-2', 'AU-3'],
            'morgan': ['AU-2', 'AU-3'],
            'bunyan': ['AU-2', 'AU-3'],
            'pino': ['AU-2', 'AU-3'],
        }

        # Security function patterns
        self.security_functions = {
            # Authentication
            'authenticate': ['IA-2', 'AC-7'],
            'login': ['IA-2', 'AC-7'],
            'logout': ['AC-12'],
            'signin': ['IA-2', 'AC-7'],
            'signout': ['AC-12'],
            'verifytoken': ['IA-2', 'SC-8'],
            'checkauth': ['IA-2', 'AC-3'],

            # Authorization
            'authorize': ['AC-3', 'AC-6'],
            'checkpermission': ['AC-3', 'AC-6'],
            'hasrole': ['AC-3', 'AC-6'],
            'canaccess': ['AC-3', 'AC-6'],
            'isadmin': ['AC-6'],

            # Encryption
            'encrypt': ['SC-13', 'SC-28'],
            'decrypt': ['SC-13', 'SC-28'],
            'hash': ['SC-13', 'IA-5'],
            'sign': ['SC-13', 'SI-7'],
            'verify': ['SC-13', 'SI-7'],

            # Validation
            'validate': ['SI-10'],
            'sanitize': ['SI-10'],
            'escape': ['SI-10'],
            'clean': ['SI-10'],
            'purify': ['SI-10'],

            # Logging
            'audit': ['AU-2', 'AU-3'],
            'logaccess': ['AU-2', 'AC-3'],
            'logevent': ['AU-2', 'AU-3'],
        }

        # Middleware patterns
        self.middleware_patterns = {
            'authenticate': ['IA-2', 'AC-3'],
            'requireauth': ['IA-2', 'AC-3'],
            'isAuthenticated': ['IA-2', 'AC-3'],
            'ensureAuthenticated': ['IA-2', 'AC-3'],
            'checkAuth': ['IA-2', 'AC-3'],
            'verifyToken': ['IA-2', 'SC-8'],
            'authorize': ['AC-3', 'AC-6'],
            'checkPermission': ['AC-3', 'AC-6'],
            'hasRole': ['AC-3', 'AC-6'],
            'rateLimit': ['SC-5'],
            'csrf': ['SI-10', 'SC-8'],
            'helmet': ['SC-8', 'SC-18'],
            'cors': ['AC-4', 'SC-8'],
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze JavaScript/TypeScript file for NIST controls"""
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

        # Analyze imports and requires
        annotations.extend(self._analyze_imports(code, str(file_path)))

        # Analyze functions and methods
        annotations.extend(self._analyze_functions(code, str(file_path)))

        # Find implicit patterns
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

    def _analyze_imports(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze imports and requires for security packages"""
        annotations = []

        # Get imports using AST utils
        imports = get_javascript_imports(code)

        for import_info in imports:
            module = import_info.get('module', '')
            if module:
                # Check against security packages
                for pkg, controls in self.security_packages.items():
                    if pkg in module or module.startswith(pkg):
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=import_info['line'],
                            control_ids=controls,
                            evidence=f"Security package: {module}",
                            component="imports",
                            confidence=0.9
                        ))
                        break

        return annotations

    def _analyze_functions(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze functions for security patterns"""
        annotations = []

        # Get functions using AST utils
        functions = get_javascript_functions(code)

        for func in functions:
            func_name = func.get('name', '').lower()
            if func_name:
                # Check against security functions
                for pattern, controls in self.security_functions.items():
                    if pattern in func_name.replace('_', '').replace('-', ''):
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=func['start_line'],
                            control_ids=controls,
                            evidence=f"Security function: {func['name']}",
                            component="function",
                            confidence=0.85
                        ))
                        break

        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []

        # Authentication middleware patterns
        auth_patterns = [
            (r'app\.use\s*\(\s*[\'"`]?/?auth', ["IA-2", "AC-3"], "Authentication middleware"),
            (r'router\.use\s*\(\s*authenticate', ["IA-2", "AC-3"], "Router authentication"),
            (r'passport\.(authenticate|use)', ["IA-2", "IA-8"], "Passport.js authentication"),
            (r'jwt\.(sign|verify|decode)', ["IA-2", "SC-8"], "JWT token handling"),
            (r'req\.(user|isAuthenticated)', ["IA-2", "AC-3"], "Request authentication check"),
            (r'Bearer\s+[\'"`]?\$?\{?token', ["IA-2", "SC-8"], "Bearer token usage"),
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

        # Authorization patterns
        authz_patterns = [
            (r'(checkPermission|hasRole|can|isAuthorized)', ["AC-3", "AC-6"], "Authorization check"),
            (r'req\.user\.(role|permissions|scope)', ["AC-3", "AC-6"], "Role-based access control"),
            (r'@(Authorized|RequireRole|Permission)', ["AC-3", "AC-6"], "Authorization decorator"),
            (r'if\s*\(.*(?:role|permission|admin).*\)', ["AC-3", "AC-6"], "Permission check"),
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
                    confidence=0.8
                ))

        # Encryption patterns
        crypto_patterns = [
            (r'crypto\.(createCipher|createHash|pbkdf2|randomBytes)', ["SC-13", "SC-28"], "Node.js crypto"),
            (r'bcrypt\.(hash|compare|genSalt)', ["IA-5", "SC-13"], "Bcrypt password hashing"),
            (r'CryptoJS\.(AES|SHA256|HmacSHA)', ["SC-13", "SC-28"], "CryptoJS encryption"),
            (r'https\.(createServer|request)', ["SC-8", "SC-13"], "HTTPS/TLS usage"),
            (r'tls\.(createServer|connect)', ["SC-8", "SC-13"], "TLS connection"),
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
            (r'(validate|sanitize|escape|clean)\s*\(', ["SI-10"], "Input validation function"),
            (r'validator\.(isEmail|isURL|escape|trim)', ["SI-10"], "Validator.js usage"),
            (r'DOMPurify\.sanitize', ["SI-10"], "DOM sanitization"),
            (r'\.replace\s*\(\s*[/\\<>]', ["SI-10"], "Manual sanitization"),
            (r'express-validator', ["SI-10"], "Express validator middleware"),
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

        # Security headers and middleware
        security_patterns = [
            (r'helmet\s*\(\s*\)', ["SC-8", "SC-18"], "Helmet security headers"),
            (r'app\.use\s*\(\s*cors', ["AC-4", "SC-8"], "CORS configuration"),
            (r'csurf\s*\(\s*\)', ["SI-10", "SC-8"], "CSRF protection"),
            (r'rateLimit\s*\(\s*{', ["SC-5"], "Rate limiting middleware"),
            (r'X-Frame-Options|Content-Security-Policy', ["SC-18"], "Security headers"),
        ]

        for pattern, controls, evidence in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="security-middleware",
                    confidence=0.9
                ))

        # Logging patterns
        logging_patterns = [
            (r'(winston|morgan|bunyan|pino)\.', ["AU-2", "AU-3"], "Logging framework"),
            (r'console\.(log|info|warn|error).*(?:auth|security|access)', ["AU-2"], "Security logging"),
            (r'audit.*log|log.*audit', ["AU-2", "AU-9"], "Audit logging"),
            (r'req\.log|logger\.', ["AU-2", "AU-3"], "Request logging"),
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

        # Framework-specific patterns
        self._analyze_framework_patterns(code, file_path, annotations)

        return annotations

    def _analyze_framework_patterns(self, code: str, file_path: str, annotations: list[CodeAnnotation]):
        """Analyze framework-specific security patterns"""

        # Express.js patterns
        if 'express' in code.lower():
            express_patterns = [
                (r'app\.set\s*\(\s*[\'"`]trust proxy', ["SC-8"], "Express trust proxy setting"),
                (r'res\.cookie\s*\(.*secure\s*:\s*true', ["SC-8", "SC-23"], "Secure cookie flag"),
                (r'res\.cookie\s*\(.*httpOnly\s*:\s*true', ["SC-8", "SC-23"], "HTTPOnly cookie flag"),
                (r'app\.disable\s*\(\s*[\'"`]x-powered-by', ["SC-18"], "Hide Express version"),
            ]

            for pattern, controls, evidence in express_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Express.js: {evidence}",
                        component="framework-security",
                        confidence=0.85
                    ))

        # React patterns
        if 'react' in code.lower() or 'jsx' in file_path:
            react_patterns = [
                (r'dangerouslySetInnerHTML', ["SI-10"], "React XSS risk - needs review"),
                (r'(sanitize|escape|purify).*dangerouslySetInnerHTML', ["SI-10"], "React sanitized HTML"),
                (r'createContext.*(?:Auth|User|Permission)', ["IA-2", "AC-3"], "React auth context"),
            ]

            for pattern, controls, evidence in react_patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"React: {evidence}",
                        component="framework-security",
                        confidence=0.7 if 'dangerously' in pattern else 0.85
                    ))

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for JavaScript/TypeScript code"""
        suggestions = set()

        # Analyze imports
        imports = get_javascript_imports(code)
        for import_info in imports:
            module = import_info.get('module', '').lower()

            # Web frameworks
            if any(fw in module for fw in ['express', 'fastify', 'koa', 'hapi']):
                suggestions.update(['AC-3', 'AC-4', 'SC-8', 'SI-10', 'AU-2'])

            # Authentication
            if any(auth in module for auth in ['passport', 'jwt', 'auth', 'oauth']):
                suggestions.update(['IA-2', 'IA-5', 'IA-8', 'AC-3'])

            # Frontend frameworks
            if any(fw in module for fw in ['react', 'vue', 'angular']):
                suggestions.update(['SI-10', 'SC-8', 'AC-3'])

            # Database
            if any(db in module for db in ['mongoose', 'sequelize', 'typeorm', 'prisma']):
                suggestions.update(['SC-28', 'SI-10', 'AU-2'])

        # Analyze functions
        functions = get_javascript_functions(code)
        for func in functions:
            func_name = func.get('name', '').lower()

            if any(auth in func_name for auth in ['auth', 'login', 'verify']):
                suggestions.update(['IA-2', 'AC-3', 'AC-7'])

            if any(crypto in func_name for crypto in ['encrypt', 'decrypt', 'hash']):
                suggestions.update(['SC-13', 'SC-28'])

            if any(val in func_name for val in ['validate', 'sanitize', 'escape']):
                suggestions.update(['SI-10'])

        # Pattern-based suggestions
        patterns = self.find_security_patterns(code, "temp.js")
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)

        # TypeScript specific
        if any(ext in code for ext in ['.ts', '.tsx', 'interface', 'type']):
            suggestions.update(['SI-10', 'SA-11'])  # Type safety helps with validation

        return sorted(suggestions)

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire JavaScript/TypeScript project"""
        results = {}

        # Common directories to skip
        skip_dirs = {
            'node_modules', 'bower_components', '.git', 'dist', 'build',
            'coverage', '.nyc_output', '.next', '.nuxt', 'out'
        }

        for file_path in project_path.rglob('*'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Check if it's a JS/TS file
            if file_path.suffix in self.file_extensions:
                # Skip test files unless they contain security tests
                if any(test in file_path.name for test in ['test', 'spec', '.test.', '.spec.']):
                    with open(file_path, encoding='utf-8') as f:
                        content = f.read().lower()
                        if not any(term in content for term in ['security', 'auth', 'permission']):
                            continue

                annotations = self.analyze_file(file_path)
                if annotations:
                    results[str(file_path)] = annotations

        # Analyze package.json for security dependencies
        package_json = project_path / 'package.json'
        if package_json.exists():
            pkg_annotations = self._analyze_package_json(package_json)
            if pkg_annotations:
                results[str(package_json)] = pkg_annotations

        return results

    def _analyze_package_json(self, package_path: Path) -> list[CodeAnnotation]:
        """Analyze package.json for security-relevant dependencies"""
        annotations = []

        try:
            import json
            with open(package_path, encoding='utf-8') as f:
                package_data = json.load(f)

            # Combine all dependencies
            all_deps = {}
            for dep_type in ['dependencies', 'devDependencies', 'peerDependencies']:
                if dep_type in package_data:
                    all_deps.update(package_data[dep_type])

            # Check each dependency
            line_num = 1
            for dep_name in all_deps:
                for pkg, controls in self.security_packages.items():
                    if pkg in dep_name:
                        annotations.append(CodeAnnotation(
                            file_path=str(package_path),
                            line_number=line_num,
                            control_ids=controls,
                            evidence=f"Security dependency: {dep_name}",
                            component="dependencies",
                            confidence=0.85
                        ))
                        break
                line_num += 1

        except Exception:
            pass

        return annotations
