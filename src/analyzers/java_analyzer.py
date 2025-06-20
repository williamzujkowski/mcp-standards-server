"""
Java code analyzer
@nist-controls: SA-11, SA-15
@evidence: Java static analysis for security controls
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAnalyzer, CodeAnnotation, SecurityPattern


class JavaAnalyzer(BaseAnalyzer):
    """
    Analyzes Java code for NIST control implementations
    @nist-controls: SA-11, CA-7
    @evidence: Java-specific security analysis
    """
    
    def __init__(self):
        super().__init__()
        self.file_extensions = ['.java']
        self.framework_patterns = {
            'spring': ['@SpringBootApplication', '@RestController', '@Service', 'org.springframework'],
            'jakarta': ['jakarta.', 'javax.servlet', '@WebServlet'],
            'micronaut': ['io.micronaut', '@Controller'],
            'quarkus': ['io.quarkus', '@QuarkusTest']
        }
        
    def analyze_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze Java file for NIST controls"""
        if file_path.suffix not in self.file_extensions:
            return []
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
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
    
    def _analyze_implicit_patterns(self, code: str, file_path: str) -> List[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []
        
        # Spring Security patterns
        spring_security_patterns = [
            (r'@Secured|@PreAuthorize|@PostAuthorize', ["AC-3", "AC-6"], "Spring Security authorization"),
            (r'@EnableWebSecurity|SecurityConfig', ["AC-3", "SC-8"], "Spring Security configuration"),
            (r'AuthenticationManager|UserDetailsService', ["IA-2", "IA-8"], "Spring authentication"),
            (r'@EnableOAuth2|OAuth2', ["IA-2", "IA-8"], "OAuth2 authentication"),
            (r'CsrfTokenRepository|csrf\(\)', ["SI-10", "SC-8"], "CSRF protection")
        ]
        
        for pattern, controls, evidence in spring_security_patterns:
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
        
        # General authentication patterns
        auth_patterns = [
            (r'(?:authenticate|login|signin)\s*\(', ["IA-2", "AC-7"], "Authentication method"),
            (r'JWT|JsonWebToken', ["IA-2", "SC-8"], "JWT token handling"),
            (r'BCrypt|PasswordEncoder', ["IA-5", "SC-13"], "Password hashing"),
            (r'Session(?:Factory|Manager)', ["SC-23", "AC-12"], "Session management"),
            (r'Principal|Subject|Authentication', ["IA-2", "AC-3"], "Security principal")
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
            (r'Cipher\.getInstance|KeyGenerator', ["SC-13", "SC-28"], "Java cryptography"),
            (r'MessageDigest\.getInstance', ["SC-13", "SI-7"], "Message digest/hashing"),
            (r'KeyStore|TrustStore', ["SC-13", "IA-5"], "Key management"),
            (r'SSLContext|TLSv', ["SC-8", "SC-13"], "SSL/TLS configuration"),
            (r'@Encrypt|encrypted', ["SC-28"], "Data encryption annotation")
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
            (r'@Valid|@Validated|@NotNull|@Size', ["SI-10", "SI-15"], "Bean validation"),
            (r'PreparedStatement|setParameter', ["SI-10"], "SQL injection prevention"),
            (r'OWASP|AntiSamy|ESAPI', ["SI-10", "SI-15"], "OWASP security library"),
            (r'Pattern\.compile|Matcher', ["SI-10"], "Regular expression validation"),
            (r'StringEscapeUtils|HtmlUtils', ["SI-10"], "Output encoding")
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
            (r'Logger\.getLogger|LoggerFactory\.getLogger', ["AU-2", "AU-3"], "Logging framework"),
            (r'@Slf4j|log\.(info|warn|error|debug)', ["AU-2", "AU-3"], "SLF4J logging"),
            (r'AuditEvent|SecurityEvent', ["AU-2", "AU-9"], "Security audit logging"),
            (r'MDC\.put|ThreadContext', ["AU-3"], "Logging context"),
            (r'log4j|logback', ["AU-2", "AU-3"], "Logging configuration")
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
            (r'@RolesAllowed|@PermitAll|@DenyAll', ["AC-3", "AC-6"], "JAX-RS authorization"),
            (r'hasRole|hasAuthority|hasPermission', ["AC-3", "AC-6"], "Spring authorization"),
            (r'AccessDecisionManager|AccessDecisionVoter', ["AC-3"], "Access control framework"),
            (r'@Secured\(|SecurityContext', ["AC-3", "AC-4"], "Security annotations")
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
                    confidence=0.85
                ))
        
        # Exception and error handling
        error_patterns = [
            (r'try\s*{.*?catch', ["SI-11"], "Exception handling"),
            (r'@ExceptionHandler|@ControllerAdvice', ["SI-11", "AU-5"], "Global exception handling"),
            (r'finally\s*{', ["SI-11", "SC-5"], "Resource cleanup"),
            (r'SecurityException|AccessDeniedException', ["AC-3", "SI-11"], "Security exception handling")
        ]
        
        for pattern, controls, evidence in error_patterns:
            if re.search(pattern, code, re.DOTALL):
                line_num = self._find_pattern_line(code, 'try' if 'try' in pattern else pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="error-handling",
                    confidence=0.75
                ))
        
        return annotations
    
    def _analyze_imports(self, code: str, file_path: str) -> List[CodeAnnotation]:
        """Analyze import statements for security-relevant packages"""
        annotations = []
        
        # Security-relevant imports
        security_imports = {
            'java.security': ["SC-13", "IA-5"],
            'javax.crypto': ["SC-13", "SC-28"],
            'org.springframework.security': ["AC-3", "IA-2", "SC-8"],
            'org.apache.shiro': ["AC-3", "IA-2"],
            'com.auth0': ["IA-2", "IA-8"],
            'io.jsonwebtoken': ["IA-2", "SC-8"],
            'org.owasp': ["SI-10", "SI-15"],
            'org.slf4j': ["AU-2", "AU-3"],
            'javax.validation': ["SI-10"],
            'org.hibernate.validator': ["SI-10"],
            'java.sql.PreparedStatement': ["SI-10"]
        }
        
        for pkg, controls in security_imports.items():
            if f"import {pkg}" in code or f"import static {pkg}" in code:
                line_num = self._find_pattern_line(code, pkg)
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=f"Security-relevant import: {pkg}",
                    component="imports",
                    confidence=0.7
                ))
        
        return annotations
    
    def analyze_project(self, project_path: Path) -> Dict[str, List[CodeAnnotation]]:
        """Analyze entire Java project"""
        results = {}
        
        # Common directories to skip
        skip_dirs = {'.git', 'target', 'build', '.idea', '.gradle', 'out'}
        
        for file_path in project_path.rglob('*.java'):
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
                
            # Skip test files unless they contain security tests
            if 'test' in file_path.parts or file_path.name.endswith('Test.java'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if 'security' not in content and 'auth' not in content:
                        continue
                        
            annotations = self.analyze_file(file_path)
            if annotations:
                results[str(file_path)] = annotations
                
        # Analyze pom.xml or build.gradle for security insights
        pom_path = project_path / 'pom.xml'
        gradle_path = project_path / 'build.gradle'
        
        if pom_path.exists():
            pom_annotations = self._analyze_pom_xml(pom_path)
            if pom_annotations:
                results[str(pom_path)] = pom_annotations
                
        if gradle_path.exists():
            gradle_annotations = self._analyze_build_gradle(gradle_path)
            if gradle_annotations:
                results[str(gradle_path)] = gradle_annotations
                
        return results
    
    def _analyze_pom_xml(self, pom_path: Path) -> List[CodeAnnotation]:
        """Analyze pom.xml for security-relevant dependencies"""
        annotations = []
        
        try:
            with open(pom_path, 'r') as f:
                content = f.read()
                
            security_deps = {
                'spring-security': ["AC-3", "IA-2", "SC-8"],
                'spring-boot-starter-security': ["AC-3", "IA-2", "SC-8"],
                'shiro': ["AC-3", "IA-2"],
                'jwt': ["IA-2", "SC-8"],
                'bcrypt': ["IA-5", "SC-13"],
                'owasp': ["SI-10", "SI-15"],
                'hibernate-validator': ["SI-10"],
                'logback': ["AU-2", "AU-3"],
                'log4j': ["AU-2", "AU-3"]
            }
            
            for dep, controls in security_deps.items():
                if dep in content:
                    line_num = self._find_pattern_line(content, dep)
                    annotations.append(CodeAnnotation(
                        file_path=str(pom_path),
                        line_number=line_num,
                        control_ids=controls,
                        evidence=f"Security dependency: {dep}",
                        component="dependencies",
                        confidence=0.8
                    ))
                    
        except Exception:
            pass
            
        return annotations
    
    def _analyze_build_gradle(self, gradle_path: Path) -> List[CodeAnnotation]:
        """Analyze build.gradle for security-relevant dependencies"""
        # Similar to pom.xml analysis
        return self._analyze_pom_xml(gradle_path)
    
    def suggest_controls(self, code: str) -> List[str]:
        """Suggest NIST controls for Java code"""
        suggestions = set()
        patterns = self.find_security_patterns(code, "temp.java")
        
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)
            
        # Framework-specific suggestions
        code_lower = code.lower()
        
        # Spring framework controls
        if 'spring' in code_lower:
            suggestions.update(["AC-3", "AC-4", "SC-8", "SI-10", "IA-2"])
            
        # JEE/Jakarta controls
        if any(jee in code_lower for jee in ['jakarta', 'javax', 'ejb', 'servlet']):
            suggestions.update(["AC-3", "SC-8", "SI-10", "AU-2"])
            
        # Database controls
        if any(db in code_lower for db in ['jdbc', 'jpa', 'hibernate', 'sql']):
            suggestions.update(["SC-28", "SI-10", "AU-2", "AC-3"])
            
        # Microservices controls
        if any(ms in code_lower for ms in ['microservice', 'rest', 'grpc', 'kafka']):
            suggestions.update(["SC-8", "AC-4", "AU-2", "SI-10"])
            
        # Cloud controls
        if any(cloud in code_lower for cloud in ['aws', 'azure', 'gcp', 'cloud']):
            suggestions.update(["AC-2", "AU-2", "SC-28", "SC-8"])
            
        return sorted(list(suggestions))