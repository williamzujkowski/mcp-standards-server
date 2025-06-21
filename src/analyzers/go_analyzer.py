"""
Enhanced Go code analyzer
@nist-controls: SA-11, SA-15
@evidence: Advanced Go analysis for security controls
"""
import re
from pathlib import Path

from .base import BaseAnalyzer, CodeAnnotation


class GoAnalyzer(BaseAnalyzer):
    """
    Enhanced Go analyzer with improved pattern detection
    @nist-controls: SA-11, CA-7
    @evidence: Go-specific security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.go']
        self.language = 'go'

        # Security-relevant imports
        self.security_imports = {
            # Cryptography
            'crypto/aes': ['SC-13', 'SC-28'],
            'crypto/cipher': ['SC-13', 'SC-28'],
            'crypto/des': ['SC-13', 'SC-28'],
            'crypto/dsa': ['SC-13'],
            'crypto/ecdsa': ['SC-13'],
            'crypto/ed25519': ['SC-13'],
            'crypto/elliptic': ['SC-13'],
            'crypto/hmac': ['SC-13', 'SI-7'],
            'crypto/md5': ['SC-13'],  # Note: Weak algorithm
            'crypto/rand': ['SC-13'],
            'crypto/rc4': ['SC-13'],  # Note: Deprecated
            'crypto/rsa': ['SC-13', 'SC-28'],
            'crypto/sha1': ['SC-13', 'SI-7'],  # Note: Weak algorithm
            'crypto/sha256': ['SC-13', 'SI-7'],
            'crypto/sha512': ['SC-13', 'SI-7'],
            'crypto/tls': ['SC-8', 'SC-13'],
            'crypto/x509': ['SC-8', 'SC-13'],
            'golang.org/x/crypto/bcrypt': ['IA-5', 'SC-13'],
            'golang.org/x/crypto/argon2': ['IA-5', 'SC-13'],
            'golang.org/x/crypto/scrypt': ['IA-5', 'SC-13'],
            'golang.org/x/crypto/pbkdf2': ['IA-5', 'SC-13'],

            # Authentication/Authorization
            'github.com/dgrijalva/jwt-go': ['IA-2', 'SC-8'],
            'github.com/golang-jwt/jwt': ['IA-2', 'SC-8'],
            'github.com/gorilla/sessions': ['SC-23', 'AC-12'],
            'github.com/casbin/casbin': ['AC-3', 'AC-6'],
            'golang.org/x/oauth2': ['IA-2', 'IA-8'],

            # Input Validation
            'github.com/go-playground/validator': ['SI-10'],
            'github.com/asaskevich/govalidator': ['SI-10'],
            'html/template': ['SI-10'],
            'text/template': ['SI-10'],
            'encoding/json': ['SI-10'],

            # Security Tools
            'github.com/gorilla/csrf': ['SI-10', 'SC-8'],
            'github.com/gorilla/secure': ['SC-8', 'SC-18'],
            'github.com/unrolled/secure': ['SC-8', 'SC-18'],
            'golang.org/x/time/rate': ['SC-5'],

            # Logging
            'log': ['AU-2', 'AU-3'],
            'log/syslog': ['AU-2', 'AU-9'],
            'github.com/sirupsen/logrus': ['AU-2', 'AU-3'],
            'go.uber.org/zap': ['AU-2', 'AU-3'],
            'github.com/rs/zerolog': ['AU-2', 'AU-3'],
            
            # Framework packages
            'github.com/gin-gonic/gin': ['AC-3', 'AC-4', 'SC-8', 'SI-10', 'AU-2'],
            'github.com/gin-contrib/secure': ['SC-8', 'SC-18'],
            'github.com/gin-contrib/cors': ['AC-4', 'SC-8'],
            'github.com/ulule/limiter/v3': ['SC-5'],
            
            # Additional crypto
            'golang.org/x/crypto': ['SC-13', 'SC-28'],
        }

        # Security function patterns
        self.security_functions = {
            # Authentication
            'Authenticate': ['IA-2', 'AC-7'],
            'Login': ['IA-2', 'AC-7'],
            'Logout': ['AC-12'],
            'VerifyPassword': ['IA-2', 'IA-5'],
            'CheckPassword': ['IA-2', 'IA-5'],
            'ValidateToken': ['IA-2', 'SC-8'],
            'VerifyToken': ['IA-2', 'SC-8'],
            'GenerateToken': ['IA-2', 'SC-8'],

            # Authorization
            'Authorize': ['AC-3', 'AC-6'],
            'CheckPermission': ['AC-3', 'AC-6'],
            'HasPermission': ['AC-3', 'AC-6'],
            'RequireRole': ['AC-3', 'AC-6'],
            'IsAdmin': ['AC-6'],
            'CanAccess': ['AC-3', 'AC-6'],

            # Encryption
            'Encrypt': ['SC-13', 'SC-28'],
            'Decrypt': ['SC-13', 'SC-28'],
            'Hash': ['SC-13', 'IA-5'],
            'Sign': ['SC-13', 'SI-7'],
            'Verify': ['SC-13', 'SI-7'],
            'GenerateKey': ['SC-13'],

            # Validation
            'Validate': ['SI-10'],
            'Sanitize': ['SI-10'],
            'Escape': ['SI-10'],
            'Clean': ['SI-10'],
            'ValidateInput': ['SI-10'],

            # Logging
            'AuditLog': ['AU-2', 'AU-3'],
            'LogEvent': ['AU-2', 'AU-3'],
            'LogSecurity': ['AU-2', 'AU-9'],
            'LogAccess': ['AU-2', 'AC-3'],
        }

        # Middleware/Handler patterns
        self.middleware_patterns = {
            'AuthMiddleware': ['IA-2', 'AC-3'],
            'RequireAuth': ['IA-2', 'AC-3'],
            'AuthenticationMiddleware': ['IA-2', 'AC-3'],
            'AuthorizationMiddleware': ['AC-3', 'AC-6'],
            'PermissionMiddleware': ['AC-3', 'AC-6'],
            'CSRFMiddleware': ['SI-10', 'SC-8'],
            'SecurityHeaders': ['SC-8', 'SC-18'],
            'RateLimiter': ['SC-5'],
            'CORSMiddleware': ['AC-4', 'SC-8'],
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Go file for NIST controls"""
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

        # Analyze imports
        annotations.extend(self._analyze_imports(code, str(file_path)))

        # Analyze functions
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
        """Analyze import statements for security packages"""
        annotations: list[CodeAnnotation] = []

        # Extract imports - Go uses import blocks or single imports
        import_pattern = r'import\s+(?:\(\s*((?:[^)]+))\s*\)|"([^"]+)")'

        for match in re.finditer(import_pattern, code, re.MULTILINE | re.DOTALL):
            if match.group(1):  # Import block
                imports_block = match.group(1)
                import_lines = imports_block.split('\n')
                for i, line in enumerate(import_lines):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        # Extract package from quotes
                        pkg_match = re.search(r'"([^"]+)"', line)
                        if pkg_match:
                            pkg = pkg_match.group(1)
                            self._check_import(pkg, match.start() + i, file_path, annotations)
            else:  # Single import
                pkg = match.group(2)
                self._check_import(pkg, match.start(), file_path, annotations)

        return annotations

    def _check_import(self, pkg: str, position: int, file_path: str, annotations: list[CodeAnnotation]):
        """Check if import is security-relevant"""
        for sec_pkg, controls in self.security_imports.items():
            if pkg == sec_pkg or pkg.startswith(sec_pkg + '/'):
                # Calculate line number
                line_num = self._calculate_line_number(position, file_path)

                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=f"Security import: {pkg}",
                    component="imports",
                    confidence=0.9
                ))
                break

    def _analyze_functions(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze function definitions for security patterns"""
        annotations = []

        # Go function pattern: func (receiver) FunctionName(params) returnType
        func_pattern = r'^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\('

        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(func_pattern, line.strip())
            if match:
                func_name = match.group(1)

                # Check against security functions
                for pattern, controls in self.security_functions.items():
                    if pattern.lower() in func_name.lower():
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=i,
                            control_ids=controls,
                            evidence=f"Security function: {func_name}",
                            component="function",
                            confidence=0.85
                        ))
                        break

                # Check for middleware/handler patterns
                for pattern, controls in self.middleware_patterns.items():
                    if pattern.lower() in func_name.lower():
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=i,
                            control_ids=controls,
                            evidence=f"Security middleware: {func_name}",
                            component="middleware",
                            confidence=0.85
                        ))
                        break

        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []

        # Authentication patterns
        auth_patterns = [
            (r'jwt\.Parse|jwt\.Valid|jwt\.Sign', ["IA-2", "SC-8"], "JWT token handling"),
            (r'bcrypt\.GenerateFromPassword|bcrypt\.CompareHashAndPassword', ["IA-5", "SC-13"], "Bcrypt password handling"),
            (r'session\.Get|session\.Save', ["SC-23", "AC-12"], "Session management"),
            (r'Bearer\s+token|Authorization:\s*Bearer', ["IA-2", "SC-8"], "Bearer token authentication"),
            (r'oauth2\.Config|oauth2\.Token', ["IA-2", "IA-8"], "OAuth2 implementation"),
            (r'verifyToken|ValidateToken', ["IA-2"], "Token verification"),
            (r'grpc\.UnaryInterceptor|grpc\.StreamInterceptor', ["IA-2", "AC-3"], "gRPC interceptors"),
            (r'codes\.Unauthenticated|status\.Error.*Unauthenticated', ["IA-2"], "gRPC authentication"),
            (r'RequireRole|codes\.PermissionDenied', ["AC-3", "AC-6"], "gRPC authorization"),
        ]

        for pattern, controls, evidence in auth_patterns:
            if re.search(pattern, code):
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
            (r'casbin\.NewEnforcer|enforcer\.Enforce', ["AC-3", "AC-6"], "Casbin authorization"),
            (r'if\s+.*\.Role\s*==|if\s+.*\.IsAdmin', ["AC-3", "AC-6"], "Role-based access control"),
            (r'permission\.Check|HasPermission', ["AC-3", "AC-6"], "Permission checking"),
            (r'ctx\.Value\("user"\)|context\.Value\("user"\)', ["AC-3"], "Context-based authorization"),
        ]

        for pattern, controls, evidence in authz_patterns:
            if re.search(pattern, code):
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
            (r'aes\.NewCipher|cipher\.NewGCM', ["SC-13", "SC-28"], "AES encryption"),
            (r'rsa\.GenerateKey|rsa\.EncryptPKCS1v15', ["SC-13", "SC-28"], "RSA encryption"),
            (r'tls\.Config|tls\.LoadX509KeyPair', ["SC-8", "SC-13"], "TLS configuration"),
            (r'x509\.CreateCertificate', ["SC-8", "SC-13"], "Certificate generation"),
            (r'rand\.Read|crypto/rand', ["SC-13"], "Secure random generation"),
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
            (r'validator\.Validate|validate\.Struct', ["SI-10"], "Struct validation"),
            (r'html\.EscapeString|template\.HTMLEscapeString', ["SI-10"], "HTML escaping"),
            (r'regexp\.MustCompile|regexp\.Match', ["SI-10"], "Regular expression validation"),
            (r'strings\.Replace.*[<>"]|html/template', ["SI-10"], "Output sanitization"),
            (r'json\.Valid|json\.Unmarshal', ["SI-10"], "JSON validation"),
            (r'(?:sql|SQL).*(?:injection|Injection)|sqlPatterns', ["SI-10"], "SQL injection prevention"),
            (r'(?:Sanitize|sanitize|Clean|clean).*(?:Input|String|Text)', ["SI-10"], "Input sanitization"),
        ]

        for pattern, controls, evidence in validation_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="input-validation",
                    confidence=0.85
                ))

        # Security middleware patterns
        security_patterns = [
            (r'secure\.New|secure\.Options', ["SC-8", "SC-18"], "Security headers middleware"),
            (r'csrf\.Protect|nosurf\.New', ["SI-10", "SC-8"], "CSRF protection"),
            (r'cors\.New|cors\.Default', ["AC-4", "SC-8"], "CORS configuration"),
            (r'ratelimit\.New|limiter\.NewLimiter|limiter\.Rate|limiter/v3', ["SC-5"], "Rate limiting"),
            (r'helmet\.Default|secure\.Headers', ["SC-8", "SC-18"], "Security headers"),
        ]

        for pattern, controls, evidence in security_patterns:
            if re.search(pattern, code):
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
            (r'log\.Printf.*(?:auth|security|access)', ["AU-2"], "Security logging"),
            (r'logger\.Info.*(?:login|logout|auth)', ["AU-2", "AU-3"], "Authentication logging"),
            (r'zap\.L\(\)|zerolog\.', ["AU-2", "AU-3"], "Structured logging"),
            (r'syslog\.New|syslog\.Write', ["AU-2", "AU-9"], "System logging"),
            (r'AuditLog|LogSecurityEvent|SecurityEvent', ["AU-2", "AU-3"], "Security event logging"),
            (r'timestamp.*ISO8601|EncodeTime.*ISO8601', ["AU-3"], "Timestamp formatting"),
            (r'OutputPaths.*\[|config\.OutputPaths', ["AU-4"], "Audit storage configuration"),
            (r'Sampling\s*=\s*nil', ["AU-2"], "Comprehensive event logging"),
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

        # Database security patterns
        db_patterns = [
            (r'NamedExec|NamedQuery|Named\(', ["SI-10"], "Parameterized queries with named parameters"),
            (r'\$\d+|:\w+', ["SI-10"], "SQL parameter placeholders"),
            (r'Prepare\(|PrepareContext\(|Prepared|Preparex', ["SI-10"], "Prepared statements"),
            (r'BeginTx|tx\.Rollback|tx\.Commit', ["AC-3"], "Transaction management"),
            (r'audit_log|audit.*log', ["AU-2", "AU-3"], "Database audit logging"),
        ]

        for pattern, controls, evidence in db_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="database-security",
                    confidence=0.85
                ))

        # Framework-specific patterns
        self._analyze_framework_patterns(code, file_path, annotations)

        return annotations

    def _analyze_framework_patterns(self, code: str, file_path: str, annotations: list[CodeAnnotation]):
        """Analyze Go framework-specific patterns"""

        # Gin framework
        if 'gin-gonic/gin' in code:
            gin_patterns = [
                (r'gin\.BasicAuth|gin\.BasicAuthForRealm', ["IA-2", "AC-3"], "Gin basic authentication"),
                (r'c\.ShouldBind|c\.Bind', ["SI-10"], "Gin input binding/validation"),
                (r'gin\.Recovery|gin\.Logger', ["AU-2", "AU-14"], "Gin middleware"),
                (r'c\.SecureJSON|c\.SetCookie.*Secure', ["SC-8", "SC-23"], "Gin secure response"),
            ]

            for pattern, controls, evidence in gin_patterns:
                if re.search(pattern, code):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Gin framework: {evidence}",
                        component="framework-security",
                        confidence=0.85
                    ))

        # Echo framework
        if 'labstack/echo' in code:
            echo_patterns = [
                (r'echo\.JWT|middleware\.JWT', ["IA-2", "SC-8"], "Echo JWT middleware"),
                (r'middleware\.BasicAuth', ["IA-2", "AC-3"], "Echo basic auth"),
                (r'middleware\.CSRF', ["SI-10", "SC-8"], "Echo CSRF protection"),
                (r'middleware\.Secure', ["SC-8", "SC-18"], "Echo security headers"),
            ]

            for pattern, controls, evidence in echo_patterns:
                if re.search(pattern, code):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Echo framework: {evidence}",
                        component="framework-security",
                        confidence=0.85
                    ))

        # Fiber framework
        if 'gofiber/fiber' in code:
            fiber_patterns = [
                (r'fiber/v2/middleware/csrf', ["SI-10", "SC-8"], "Fiber CSRF protection"),
                (r'fiber/v2/middleware/helmet', ["SC-8", "SC-18"], "Fiber security headers"),
                (r'fiber/v2/middleware/limiter', ["SC-5"], "Fiber rate limiting"),
                (r'fiber/v2/middleware/cors', ["AC-4", "SC-8"], "Fiber CORS configuration"),
                (r'fiber/v2/middleware/logger', ["AU-2", "AU-3"], "Fiber logging middleware"),
                (r'limiter\.New\(|Max:\s*\d+|Expiration:\s*time\.', ["SC-5"], "Rate limiting configuration"),
            ]

            for pattern, controls, evidence in fiber_patterns:
                if re.search(pattern, code):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Fiber framework: {evidence}",
                        component="framework-security",
                        confidence=0.85
                    ))

    def _calculate_line_number(self, position: int, file_path: str) -> int:
        """Calculate line number from string position"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                return content[:position].count('\n') + 1
        except Exception:
            return 1

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for Go code"""
        suggestions = set()

        # Check imports
        import_pattern = r'import\s+(?:\(\s*((?:[^)]+))\s*\)|"([^"]+)")'
        for match in re.finditer(import_pattern, code, re.MULTILINE | re.DOTALL):
            imports_text = match.group(0).lower()

            # Web frameworks
            if any(fw in imports_text for fw in ['gin', 'echo', 'fiber', 'gorilla']):
                suggestions.update(['AC-3', 'AC-4', 'SC-8', 'SI-10', 'AU-2'])

            # Authentication/crypto
            if any(auth in imports_text for auth in ['jwt', 'oauth', 'crypto', 'bcrypt']):
                suggestions.update(['IA-2', 'IA-5', 'SC-13', 'SC-28'])

            # Database
            if any(db in imports_text for db in ['sql', 'gorm', 'mongo', 'redis']):
                suggestions.update(['SC-28', 'SI-10', 'AU-2'])

        # Check function names
        func_pattern = r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\('
        for match in re.finditer(func_pattern, code):
            func_name = match.group(1).lower()

            if any(auth in func_name for auth in ['auth', 'login', 'verify']):
                suggestions.update(['IA-2', 'AC-3', 'AC-7'])

            if any(crypto in func_name for crypto in ['encrypt', 'decrypt', 'hash']):
                suggestions.update(['SC-13', 'SC-28'])

        # Pattern-based suggestions
        patterns = self.find_security_patterns(code, "temp.go")
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)

        return sorted(suggestions)

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire Go project"""
        results = {}

        # Common directories to skip
        skip_dirs = {
            'vendor', '.git', 'testdata', '.idea'
        }

        for file_path in project_path.rglob('*.go'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip test files unless they contain security tests
            if file_path.name.endswith('_test.go'):
                with open(file_path, encoding='utf-8') as f:
                    content = f.read().lower()
                    if not any(term in content for term in ['security', 'auth', 'crypto']):
                        continue

            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations

        # Analyze go.mod for security dependencies
        go_mod = project_path / 'go.mod'
        if go_mod.exists():
            mod_annotations = self._analyze_go_mod(go_mod)
            if mod_annotations:
                results[str(go_mod)] = mod_annotations

        return results

    def _analyze_go_mod(self, mod_path: Path) -> list[CodeAnnotation]:
        """Analyze go.mod for security-relevant dependencies"""
        annotations = []

        try:
            with open(mod_path, encoding='utf-8') as f:
                content = f.read()

            # Extract require blocks
            require_pattern = r'require\s*\((.*?)\)'
            require_match = re.search(require_pattern, content, re.DOTALL)

            if require_match:
                requires = require_match.group(1)
                for i, line in enumerate(requires.splitlines(), 1):
                    line = line.strip()
                    if line and not line.startswith('//'):
                        for pkg, controls in self.security_imports.items():
                            if pkg in line:
                                annotations.append(CodeAnnotation(
                                    file_path=str(mod_path),
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
