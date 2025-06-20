"""
Go code analyzer
@nist-controls: SA-11, SA-15
@evidence: Go static analysis for security controls
"""
import re
from pathlib import Path

from .base import BaseAnalyzer, CodeAnnotation


class GoAnalyzer(BaseAnalyzer):
    """
    Analyzes Go code for NIST control implementations
    @nist-controls: SA-11, CA-7
    @evidence: Go-specific security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.go']
        self.import_patterns = {
            'crypto': ["SC-13", "SC-28"],
            'net/http': ["SC-8", "AC-4"],
            'database/sql': ["SI-10", "SC-28"],
            'log': ["AU-2", "AU-3"],
            'context': ["AC-4", "SC-5"],
            'sync': ["SC-5", "SI-16"],
            'encoding/json': ["SI-10", "SC-8"]
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

        # Find implicit patterns
        implicit_annotations = self._analyze_implicit_patterns(code, str(file_path))
        annotations.extend(implicit_annotations)

        # Analyze imports
        import_annotations = self._analyze_imports(code, str(file_path))
        annotations.extend(import_annotations)

        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []

        # Authentication patterns
        auth_patterns = [
            (r'func.*(?:Auth|Login|Authenticate|ValidateToken)', ["IA-2", "AC-3"], "Authentication function"),
            (r'jwt\.(?:Parse|Sign|Valid)', ["IA-2", "SC-8"], "JWT token handling"),
            (r'bcrypt\.(?:GenerateFromPassword|CompareHashAndPassword)', ["IA-5", "SC-13"], "Password hashing"),
            (r'oauth2\.Config', ["IA-2", "IA-8"], "OAuth2 implementation"),
            (r'BasicAuth|BearerAuth', ["IA-2"], "HTTP authentication")
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

        # Encryption patterns
        crypto_patterns = [
            (r'aes\.New(?:Cipher|GCM)', ["SC-13", "SC-28"], "AES encryption"),
            (r'rsa\.(?:Encrypt|Decrypt|Sign|Verify)', ["SC-13", "SI-7"], "RSA cryptography"),
            (r'tls\.Config|tls\.Listen', ["SC-8", "SC-13"], "TLS configuration"),
            (r'crypto/rand', ["SC-13"], "Cryptographic random number generation"),
            (r'sha256\.(?:Sum|New)', ["SI-7", "SC-13"], "SHA-256 hashing")
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

        # Access control patterns
        authz_patterns = [
            (r'(?:CheckPermission|HasRole|Authorize|CanAccess)', ["AC-3", "AC-6"], "Authorization check"),
            (r'middleware\.(?:Auth|RequireRole|Permission)', ["AC-3", "AC-4"], "Authorization middleware"),
            (r'if\s+.*\.Role\s*==|switch\s+.*\.Role', ["AC-3", "AC-6"], "Role-based access control")
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

        # Input validation patterns
        validation_patterns = [
            (r'(?:Validate|Sanitize|Clean|Escape)', ["SI-10", "SI-15"], "Input validation"),
            (r'regexp\.(?:Match|Compile|MustCompile)', ["SI-10"], "Regular expression validation"),
            (r'strconv\.(?:Atoi|ParseInt|ParseFloat)', ["SI-10"], "Type conversion with validation"),
            (r'sql\.DB.*Prepare|\\?', ["SI-10"], "Prepared statements for SQL injection prevention"),
            (r'html\.EscapeString', ["SI-10"], "HTML escaping")
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

        # Logging patterns
        logging_patterns = [
            (r'log\.(?:Printf|Println|Fatal|Error|Warn|Info)', ["AU-2", "AU-3"], "Logging implementation"),
            (r'logrus\.(?:Info|Warn|Error|Debug)', ["AU-2", "AU-3"], "Structured logging with logrus"),
            (r'zap\.(?:Info|Error|Debug|Sugar)', ["AU-2", "AU-3"], "Structured logging with zap"),
            (r'slog\.(?:Info|Error|Debug)', ["AU-2", "AU-3"], "Structured logging with slog"),
            (r'AuditLog|SecurityLog', ["AU-2", "AU-9"], "Security audit logging")
        ]

        for pattern, controls, evidence in logging_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="logging",
                    confidence=0.8
                ))

        # Error handling patterns
        error_patterns = [
            (r'if\s+err\s*!=\s*nil', ["SI-11"], "Error handling"),
            (r'defer.*(?:Close|Unlock|Done)', ["SI-11", "SC-5"], "Resource cleanup with defer"),
            (r'panic\(|recover\(', ["SI-11", "SC-24"], "Panic and recovery handling"),
            (r'errors\.(?:New|Wrap|Wrapf)', ["SI-11"], "Error wrapping and context")
        ]

        for pattern, controls, evidence in error_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="error-handling",
                    confidence=0.75
                ))

        # Concurrency and rate limiting
        concurrency_patterns = [
            (r'sync\.(?:Mutex|RWMutex|WaitGroup)', ["SC-5", "SI-16"], "Thread-safe operations"),
            (r'rate\.(?:Limiter|NewLimiter)', ["SC-5"], "Rate limiting"),
            (r'context\.(?:WithTimeout|WithDeadline)', ["SC-5", "SC-24"], "Request timeout handling"),
            (r'chan\s+|<-|->|select\s*{', ["SC-5", "SI-16"], "Channel-based concurrency")
        ]

        for pattern, controls, evidence in concurrency_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="concurrency",
                    confidence=0.8
                ))

        return annotations

    def _analyze_imports(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze import statements for security-relevant packages"""
        annotations = []

        # Extract import block
        import_match = re.search(r'import\s*\((.*?)\)', code, re.DOTALL)
        if not import_match:
            # Try single line imports
            import_match = re.search(r'import\s+"([^"]+)"', code)

        if import_match:
            imports_text = import_match.group(1) if import_match.group(1) else import_match.group(0)

            for pkg, controls in self.import_patterns.items():
                if pkg in imports_text:
                    line_num = self._find_pattern_line(code, f'"{pkg}"')
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Import of security-relevant package: {pkg}",
                        component="imports",
                        confidence=0.7
                    ))

        # Check for security-specific imports
        security_imports = {
            'golang.org/x/crypto': ["SC-13", "SC-28"],
            'github.com/dgrijalva/jwt-go': ["IA-2", "SC-8"],
            'github.com/gorilla/sessions': ["SC-23", "AC-12"],
            'github.com/gorilla/csrf': ["SI-10", "SC-8"],
            'golang.org/x/oauth2': ["IA-2", "IA-8"],
            'github.com/casbin/casbin': ["AC-3", "AC-6"],
            'github.com/go-playground/validator': ["SI-10"]
        }

        for pkg, controls in security_imports.items():
            if pkg in code:
                line_num = self._find_pattern_line(code, pkg)
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=f"Security library import: {pkg}",
                    component="imports",
                    confidence=0.85
                ))

        return annotations

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire Go project"""
        results = {}

        # Common directories to skip
        skip_dirs = {'.git', 'vendor', '.idea', '.vscode', 'testdata'}

        for file_path in project_path.rglob('*.go'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip test files unless they contain security tests
            if file_path.name.endswith('_test.go'):
                with open(file_path, encoding='utf-8') as f:
                    if 'security' not in f.read().lower():
                        continue

            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations

        # Analyze go.mod for security insights
        go_mod_path = project_path / 'go.mod'
        if go_mod_path.exists():
            mod_annotations = self._analyze_go_mod(go_mod_path)
            if mod_annotations:
                results[str(go_mod_path)] = mod_annotations

        return results

    def _analyze_go_mod(self, go_mod_path: Path) -> list[CodeAnnotation]:
        """Analyze go.mod for security-relevant dependencies"""
        annotations = []

        try:
            with open(go_mod_path) as f:
                content = f.read()

            security_deps = {
                'golang.org/x/crypto': ["SC-13", "SC-28"],
                'github.com/golang-jwt/jwt': ["IA-2", "SC-8"],
                'github.com/gorilla/sessions': ["SC-23", "AC-12"],
                'github.com/gorilla/csrf': ["SI-10", "SC-8"],
                'golang.org/x/oauth2': ["IA-2", "IA-8"],
                'github.com/casbin/casbin': ["AC-3", "AC-6"],
                'github.com/sirupsen/logrus': ["AU-2", "AU-3"],
                'go.uber.org/zap': ["AU-2", "AU-3"]
            }

            for dep, controls in security_deps.items():
                if dep in content:
                    line_num = self._find_pattern_line(content, dep)
                    annotations.append(CodeAnnotation(
                        file_path=str(go_mod_path),
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Security dependency: {dep}",
                        component="dependencies",
                        confidence=0.8
                    ))

        except Exception:
            pass

        return annotations

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for Go code"""
        suggestions = set()
        patterns = self.find_security_patterns(code, "temp.go")

        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)

        # Go-specific suggestions
        code_lower = code.lower()

        # Web framework controls
        if any(framework in code_lower for framework in ['gin', 'echo', 'fiber', 'gorilla/mux']):
            suggestions.update(["AC-3", "AC-4", "SC-8", "SI-10"])

        # Database controls
        if any(db in code_lower for db in ['database/sql', 'gorm', 'mongo', 'redis']):
            suggestions.update(["SC-28", "SI-10", "AU-2"])

        # Cloud SDK controls
        if any(cloud in code_lower for cloud in ['aws-sdk-go', 'azure-sdk-for-go', 'cloud.google.com']):
            suggestions.update(["AC-2", "AU-2", "SC-28", "SC-8"])

        # gRPC controls
        if 'grpc' in code_lower:
            suggestions.update(["SC-8", "SC-13", "AC-4", "IA-2"])

        # Kubernetes controls
        if any(k8s in code_lower for k8s in ['k8s.io', 'kubernetes']):
            suggestions.update(["AC-3", "AC-6", "CM-7", "SC-28"])

        return sorted(suggestions)
