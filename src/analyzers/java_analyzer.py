"""
Enhanced Java code analyzer
@nist-controls: SA-11, SA-15
@evidence: Advanced Java analysis for security controls
"""
import re
from pathlib import Path

from .base import BaseAnalyzer, CodeAnnotation


class JavaAnalyzer(BaseAnalyzer):
    """
    Enhanced Java analyzer with improved pattern detection
    @nist-controls: SA-11, CA-7
    @evidence: Java-specific security analysis
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.java']
        self.language = 'java'

        # Security-relevant imports
        self.security_imports = {
            # Cryptography
            'javax.crypto': ['SC-13', 'SC-28'],
            'java.security': ['SC-13', 'SC-28'],
            'java.security.MessageDigest': ['SC-13', 'SI-7'],
            'java.security.SecureRandom': ['SC-13'],
            'javax.crypto.Cipher': ['SC-13', 'SC-28'],
            'javax.crypto.KeyGenerator': ['SC-13'],
            'javax.net.ssl': ['SC-8', 'SC-13'],
            'org.bouncycastle': ['SC-13', 'SC-28'],
            'org.mindrot.jbcrypt': ['IA-5', 'SC-13'],
            'org.springframework.security.crypto': ['IA-5', 'SC-13'],
            'com.password4j': ['IA-5', 'SC-13'],

            # Authentication/Authorization
            'org.springframework.security': ['IA-2', 'AC-3'],
            'org.springframework.security.core': ['IA-2', 'AC-3'],
            'org.springframework.security.authentication': ['IA-2', 'AC-7'],
            'org.springframework.security.access': ['AC-3', 'AC-6'],
            'org.springframework.security.oauth2': ['IA-2', 'IA-8'],
            'org.springframework.security.jwt': ['IA-2', 'SC-8'],
            'io.jsonwebtoken': ['IA-2', 'SC-8'],
            'com.auth0.jwt': ['IA-2', 'SC-8'],
            'javax.servlet.http.HttpSession': ['SC-23', 'AC-12'],
            'org.apache.shiro': ['IA-2', 'AC-3'],
            'org.keycloak': ['IA-2', 'IA-8'],

            # Input Validation
            'javax.validation': ['SI-10'],
            'org.hibernate.validator': ['SI-10'],
            'org.owasp.encoder': ['SI-10'],
            'org.owasp.html': ['SI-10'],
            'com.google.common.html': ['SI-10'],
            'org.apache.commons.text': ['SI-10'],
            'org.apache.commons.validator': ['SI-10'],

            # Security Tools
            'org.springframework.security.web.csrf': ['SI-10', 'SC-8'],
            'org.springframework.security.config': ['SC-8', 'SC-18'],
            'org.owasp.esapi': ['SI-10', 'SC-8'],
            'com.google.common.util.concurrent.RateLimiter': ['SC-5'],

            # Logging
            'org.slf4j': ['AU-2', 'AU-3'],
            'org.apache.logging.log4j': ['AU-2', 'AU-3'],
            'java.util.logging': ['AU-2', 'AU-3'],
            'ch.qos.logback': ['AU-2', 'AU-3'],
            'org.springframework.security.core.context.SecurityContextHolder': ['AU-2', 'AC-3'],
        }

        # Security annotations
        self.security_annotations = {
            '@PreAuthorize': ['AC-3', 'AC-6'],
            '@PostAuthorize': ['AC-3', 'AC-6'],
            '@Secured': ['AC-3', 'AC-6'],
            '@RolesAllowed': ['AC-3', 'AC-6'],
            '@PermitAll': ['AC-3'],
            '@DenyAll': ['AC-3'],
            '@WithMockUser': ['IA-2'],  # Test annotation
            '@Valid': ['SI-10'],
            '@Validated': ['SI-10'],
            '@NotNull': ['SI-10'],
            '@NotEmpty': ['SI-10'],
            '@Pattern': ['SI-10'],
            '@CrossOrigin': ['AC-4', 'SC-8'],
            '@EnableWebSecurity': ['SC-8', 'AC-3'],
            '@EnableGlobalMethodSecurity': ['AC-3', 'AC-6'],
            '@EnableOAuth2Sso': ['IA-2', 'IA-8'],
        }

        # Security method patterns
        self.security_methods = {
            # Authentication
            'authenticate': ['IA-2', 'AC-7'],
            'login': ['IA-2', 'AC-7'],
            'logout': ['AC-12'],
            'verifyPassword': ['IA-2', 'IA-5'],
            'checkPassword': ['IA-2', 'IA-5'],
            'validateToken': ['IA-2', 'SC-8'],
            'verifyToken': ['IA-2', 'SC-8'],
            'generateToken': ['IA-2', 'SC-8'],

            # Authorization
            'authorize': ['AC-3', 'AC-6'],
            'checkPermission': ['AC-3', 'AC-6'],
            'hasPermission': ['AC-3', 'AC-6'],
            'hasRole': ['AC-3', 'AC-6'],
            'hasAuthority': ['AC-3', 'AC-6'],
            'isAdmin': ['AC-6'],
            'canAccess': ['AC-3', 'AC-6'],

            # Encryption
            'encrypt': ['SC-13', 'SC-28'],
            'decrypt': ['SC-13', 'SC-28'],
            'hash': ['SC-13', 'IA-5'],
            'sign': ['SC-13', 'SI-7'],
            'verify': ['SC-13', 'SI-7'],
            'generateKey': ['SC-13'],

            # Validation
            'validate': ['SI-10'],
            'sanitize': ['SI-10'],
            'escape': ['SI-10'],
            'clean': ['SI-10'],
            'validateInput': ['SI-10'],
            'encode': ['SI-10'],

            # Logging
            'auditLog': ['AU-2', 'AU-3'],
            'logEvent': ['AU-2', 'AU-3'],
            'logSecurity': ['AU-2', 'AU-9'],
            'logAccess': ['AU-2', 'AC-3'],
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Java file for NIST controls"""
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

        # Analyze security annotations
        annotations.extend(self._analyze_security_annotations(code, str(file_path)))

        # Analyze methods
        annotations.extend(self._analyze_methods(code, str(file_path)))

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
        annotations = []

        # Java import pattern
        import_pattern = r'^import\s+(?:static\s+)?([a-zA-Z0-9_.]+(?:\.[A-Z][a-zA-Z0-9_]*)?);'

        for i, line in enumerate(code.splitlines(), 1):
            match = re.match(import_pattern, line.strip())
            if match:
                import_pkg = match.group(1)

                # Check against security imports
                for sec_pkg, controls in self.security_imports.items():
                    if import_pkg.startswith(sec_pkg):
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=i,
                            control_ids=controls,
                            evidence=f"Security import: {import_pkg}",
                            component="imports",
                            confidence=0.9
                        ))
                        break

        return annotations

    def _analyze_security_annotations(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze Java security annotations"""
        annotations = []

        for annotation, controls in self.security_annotations.items():
            pattern = rf'{re.escape(annotation)}(?:\([^)]*\))?'

            for match in re.finditer(pattern, code):
                line_num = code[:match.start()].count('\n') + 1

                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=f"Security annotation: {annotation}",
                    component="annotation",
                    confidence=0.95
                ))

        return annotations

    def _analyze_methods(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze method definitions for security patterns"""
        annotations = []

        # Java method pattern with modifiers
        method_pattern = r'(?:public|private|protected|static|final|synchronized|native|abstract|\s)+[\w<>\[\]]+\s+(\w+)\s*\('

        for match in re.finditer(method_pattern, code):
            method_name = match.group(1)
            line_num = code[:match.start()].count('\n') + 1

            # Check against security methods
            for pattern, controls in self.security_methods.items():
                if pattern.lower() in method_name.lower():
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Security method: {method_name}",
                        component="method",
                        confidence=0.85
                    ))
                    break

        return annotations

    def _analyze_implicit_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []

        # Spring Security patterns
        spring_patterns = [
            (r'SecurityContextHolder\.getContext', ["IA-2", "AC-3"], "Spring Security context"),
            (r'@EnableWebSecurity', ["SC-8", "AC-3"], "Spring Security configuration"),
            (r'HttpSecurity.*authorizeRequests', ["AC-3", "AC-6"], "Spring Security authorization"),
            (r'\.hasRole\(|\.hasAuthority\(', ["AC-3", "AC-6"], "Spring role-based access"),
            (r'\.authenticated\(\)|\.permitAll\(\)', ["AC-3"], "Spring authentication requirement"),
            (r'CsrfTokenRepository|csrf\(\)', ["SI-10", "SC-8"], "Spring CSRF protection"),
            (r'SessionCreationPolicy\.|sessionManagement\(\)', ["SC-23", "AC-12"], "Spring session management"),
        ]

        for pattern, controls, evidence in spring_patterns:
            if re.search(pattern, code):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="spring-security",
                    confidence=0.9
                ))

        # JWT patterns
        jwt_patterns = [
            (r'JWT\.create\(|JWT\.decode\(', ["IA-2", "SC-8"], "JWT token handling"),
            (r'Jwts\.builder\(|Jwts\.parser\(', ["IA-2", "SC-8"], "JJWT library usage"),
            (r'Algorithm\.HMAC|Algorithm\.RSA', ["SC-13", "SC-8"], "JWT algorithm specification"),
            (r'Bearer\s+["\']?\$?\{?token', ["IA-2", "SC-8"], "Bearer token authentication"),
        ]

        for pattern, controls, evidence in jwt_patterns:
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
            (r'Cipher\.getInstance\(', ["SC-13", "SC-28"], "Java cipher usage"),
            (r'MessageDigest\.getInstance\(', ["SC-13", "SI-7"], "Message digest algorithm"),
            (r'KeyGenerator\.getInstance\(', ["SC-13"], "Key generation"),
            (r'SecureRandom\(\)|SecureRandom\.', ["SC-13"], "Secure random generation"),
            (r'BCrypt\.hashpw|BCrypt\.checkpw', ["IA-5", "SC-13"], "BCrypt password hashing"),
            (r'SSLContext\.getInstance\(', ["SC-8", "SC-13"], "SSL/TLS configuration"),
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
            (r'@Valid|@Validated', ["SI-10"], "Bean validation"),
            (r'ESAPI\.encoder\(\)|Encode\.', ["SI-10"], "OWASP ESAPI encoding"),
            (r'HtmlUtils\.htmlEscape|StringEscapeUtils\.', ["SI-10"], "HTML escaping"),
            (r'PreparedStatement\.|setString\(|setInt\(', ["SI-10"], "SQL injection prevention"),
            (r'Pattern\.compile\(|matcher\(', ["SI-10"], "Regular expression validation"),
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
            (r'logger\.info.*(?:login|logout|auth)', ["AU-2", "AU-3"], "Authentication logging"),
            (r'AuditEvent|audit\.log', ["AU-2", "AU-9"], "Audit logging"),
            (r'SecurityEvent|security\.log', ["AU-2", "AU-9"], "Security event logging"),
            (r'MDC\.put|ThreadContext\.put', ["AU-3"], "Contextual logging"),
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
        """Analyze Java framework-specific patterns"""

        # Spring Boot patterns
        if '@SpringBootApplication' in code or 'springframework.boot' in code:
            boot_patterns = [
                (r'@EnableGlobalMethodSecurity', ["AC-3", "AC-6"], "Spring method security"),
                (r'@EnableResourceServer', ["AC-3", "SC-8"], "OAuth2 resource server"),
                (r'@EnableAuthorizationServer', ["IA-2", "AC-3"], "OAuth2 authorization server"),
                (r'SecurityFilterChain', ["SC-8", "AC-3"], "Spring Security filter chain"),
                (r'CorsConfigurationSource', ["AC-4", "SC-8"], "CORS configuration"),
            ]

            for pattern, controls, evidence in boot_patterns:
                if re.search(pattern, code):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Spring Boot: {evidence}",
                        component="framework-security",
                        confidence=0.9
                    ))

        # JAX-RS patterns
        if '@Path' in code or 'javax.ws.rs' in code:
            jaxrs_patterns = [
                (r'@RolesAllowed|@PermitAll|@DenyAll', ["AC-3", "AC-6"], "JAX-RS authorization"),
                (r'SecurityContext\.', ["IA-2", "AC-3"], "JAX-RS security context"),
                (r'@Context\s+SecurityContext', ["IA-2", "AC-3"], "Security context injection"),
            ]

            for pattern, controls, evidence in jaxrs_patterns:
                if re.search(pattern, code):
                    line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"JAX-RS: {evidence}",
                        component="framework-security",
                        confidence=0.85
                    ))

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for Java code"""
        suggestions = set()

        # Check imports
        import_pattern = r'^import\s+(?:static\s+)?([a-zA-Z0-9_.]+)'
        for match in re.finditer(import_pattern, code, re.MULTILINE):
            import_text = match.group(1).lower()

            # Web frameworks
            if any(fw in import_text for fw in ['spring', 'javax.servlet', 'jakarta.servlet']):
                suggestions.update(['AC-3', 'AC-4', 'SC-8', 'SI-10', 'AU-2'])

            # Security libraries
            if any(sec in import_text for sec in ['security', 'crypto', 'jwt', 'oauth']):
                suggestions.update(['IA-2', 'IA-5', 'SC-13', 'SC-28', 'AC-3'])

            # Database
            if any(db in import_text for db in ['sql', 'jdbc', 'jpa', 'hibernate']):
                suggestions.update(['SC-28', 'SI-10', 'AU-2'])

        # Check for security annotations
        for annotation in self.security_annotations:
            if annotation in code:
                suggestions.update(self.security_annotations[annotation])

        # Pattern-based suggestions
        patterns = self.find_security_patterns(code, "temp.java")
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)

        return sorted(suggestions)

    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire Java project"""
        results = {}

        # Common directories to skip
        skip_dirs = {
            'target', 'build', '.gradle', '.idea', '.mvn', 'out', 'bin'
        }

        for file_path in project_path.rglob('*.java'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            # Skip test files unless they contain security tests
            if 'test' in file_path.parts or file_path.name.endswith('Test.java'):
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                    if not any(term in content for term in ['Security', 'Auth', 'Crypto']):
                        continue

            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations

        # Analyze pom.xml for security dependencies (Maven)
        pom_xml = project_path / 'pom.xml'
        if pom_xml.exists():
            pom_annotations = self._analyze_pom_xml(pom_xml)
            if pom_annotations:
                results[str(pom_xml)] = pom_annotations

        # Analyze build.gradle for security dependencies (Gradle)
        build_gradle = project_path / 'build.gradle'
        if build_gradle.exists():
            gradle_annotations = self._analyze_build_gradle(build_gradle)
            if gradle_annotations:
                results[str(build_gradle)] = gradle_annotations

        return results

    def _analyze_pom_xml(self, pom_path: Path) -> list[CodeAnnotation]:
        """Analyze pom.xml for security-relevant dependencies"""
        annotations = []

        security_artifacts = {
            'spring-security': ['IA-2', 'AC-3'],
            'spring-boot-starter-security': ['IA-2', 'AC-3'],
            'jjwt': ['IA-2', 'SC-8'],
            'java-jwt': ['IA-2', 'SC-8'],
            'bcrypt': ['IA-5', 'SC-13'],
            'jasypt': ['SC-13', 'SC-28'],
            'owasp-esapi': ['SI-10', 'SC-8'],
            'owasp-java-html-sanitizer': ['SI-10'],
            'hibernate-validator': ['SI-10'],
            'logback': ['AU-2', 'AU-3'],
            'log4j': ['AU-2', 'AU-3'],
        }

        try:
            with open(pom_path, encoding='utf-8') as f:
                content = f.read()

            for i, line in enumerate(content.splitlines(), 1):
                for artifact, controls in security_artifacts.items():
                    if f'<artifactId>{artifact}' in line:
                        annotations.append(CodeAnnotation(
                            file_path=str(pom_path),
                            line_number=i,
                            control_ids=controls,
                            evidence=f"Security dependency: {artifact}",
                            component="dependencies",
                            confidence=0.85
                        ))
                        break
        except Exception:
            pass

        return annotations

    def _analyze_build_gradle(self, gradle_path: Path) -> list[CodeAnnotation]:
        """Analyze build.gradle for security-relevant dependencies"""
        annotations = []

        security_deps = {
            'spring-security': ['IA-2', 'AC-3'],
            'spring-boot-starter-security': ['IA-2', 'AC-3'],
            'jjwt': ['IA-2', 'SC-8'],
            'java-jwt': ['IA-2', 'SC-8'],
            'bcrypt': ['IA-5', 'SC-13'],
            'jasypt': ['SC-13', 'SC-28'],
            'esapi': ['SI-10', 'SC-8'],
            'owasp': ['SI-10'],
            'hibernate-validator': ['SI-10'],
        }

        try:
            with open(gradle_path, encoding='utf-8') as f:
                content = f.read()

            for i, line in enumerate(content.splitlines(), 1):
                for dep, controls in security_deps.items():
                    if dep in line and ('implementation' in line or 'compile' in line):
                        annotations.append(CodeAnnotation(
                            file_path=str(gradle_path),
                            line_number=i,
                            control_ids=controls,
                            evidence=f"Security dependency: {dep}",
                            component="dependencies",
                            confidence=0.85
                        ))
                        break
        except Exception:
            pass

        return annotations
