"""
JavaScript/TypeScript code analyzer
@nist-controls: SA-11, SA-15
@evidence: JavaScript/TypeScript static analysis for security controls
"""
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import BaseAnalyzer, CodeAnnotation, SecurityPattern


class JavaScriptAnalyzer(BaseAnalyzer):
    """
    Analyzes JavaScript/TypeScript code for NIST control implementations
    @nist-controls: SA-11, CA-7
    @evidence: JavaScript/TypeScript-specific security analysis
    """
    
    def __init__(self):
        super().__init__()
        self.file_extensions = ['.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs']
        self.framework_patterns = {
            'express': ['express', 'app.get', 'app.post', 'router'],
            'fastify': ['fastify', 'fastify()', 'reply.send'],
            'nextjs': ['next/', 'getServerSideProps', 'getStaticProps'],
            'react': ['React', 'useState', 'useEffect', 'Component'],
            'vue': ['Vue', 'createApp', 'defineComponent'],
            'angular': ['@angular', 'NgModule', '@Component']
        }
        
    def analyze_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze JavaScript/TypeScript file for NIST controls"""
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
        
        return annotations
    
    def _analyze_implicit_patterns(self, code: str, file_path: str) -> List[CodeAnnotation]:
        """Analyze code for implicit security patterns"""
        annotations = []
        
        # Authentication middleware patterns
        auth_patterns = [
            (r'(authenticate|requireAuth|isAuthenticated|checkAuth)\s*\(', ["IA-2", "AC-3"], "Authentication middleware"),
            (r'passport\.(authenticate|use)', ["IA-2", "IA-8"], "Passport.js authentication"),
            (r'jwt\.(sign|verify)', ["IA-2", "SC-8"], "JWT token handling"),
            (r'express-session|cookie-session', ["SC-23", "AC-12"], "Session management"),
            (r'bearer\s+token|Authorization:\s*Bearer', ["IA-2", "SC-8"], "Bearer token authentication")
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
            (r'(authorize|checkPermission|hasRole|can\()', ["AC-3", "AC-6"], "Authorization checks"),
            (r'@(Authorized|RequireRole|Permission)', ["AC-3", "AC-6"], "Authorization decorators"),
            (r'rbac|role-based|permissions', ["AC-2", "AC-3"], "Role-based access control")
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
            (r'crypto\.(createCipher|createHash|pbkdf2)', ["SC-13", "SC-28"], "Node.js crypto usage"),
            (r'bcrypt\.(hash|compare)', ["IA-5", "SC-13"], "Password hashing with bcrypt"),
            (r'https\.createServer|tls\.createServer', ["SC-8", "SC-13"], "TLS/HTTPS implementation"),
            (r'helmet\(\)', ["SC-8", "SI-10"], "Helmet.js security headers")
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
            (r'(joi|yup|zod)\.', ["SI-10", "SI-15"], "Schema validation library"),
            (r'express-validator|validator\.', ["SI-10"], "Express input validation"),
            (r'sanitize|escape|clean', ["SI-10", "SI-15"], "Input sanitization"),
            (r'xss\(|DOMPurify', ["SI-10"], "XSS prevention"),
            (r'parameterized.*query|prepared.*statement', ["SI-10"], "SQL injection prevention")
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
            (r'(winston|pino|bunyan|log4js)', ["AU-2", "AU-3"], "Structured logging library"),
            (r'console\.(log|error|warn)', ["AU-2"], "Basic logging"),
            (r'morgan\(|express.*logger', ["AU-3", "AU-12"], "HTTP request logging"),
            (r'audit.*log|security.*log', ["AU-2", "AU-9"], "Security audit logging")
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
        
        # CORS and security headers
        security_patterns = [
            (r'cors\(|Access-Control-Allow', ["AC-4", "SC-8"], "CORS configuration"),
            (r'csp\s*=|Content-Security-Policy', ["SI-10", "SC-8"], "Content Security Policy"),
            (r'X-Frame-Options|frameguard', ["SC-18"], "Clickjacking protection"),
            (r'rate.*limit|express-rate-limit', ["SC-5"], "Rate limiting")
        ]
        
        for pattern, controls, evidence in security_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                line_num = self._find_pattern_line(code, pattern.split('\\')[0])
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_num,
                    control_ids=controls,
                    evidence=evidence,
                    component="security-headers",
                    confidence=0.85
                ))
        
        return annotations
    
    def analyze_project(self, project_path: Path) -> Dict[str, List[CodeAnnotation]]:
        """Analyze entire JavaScript/TypeScript project"""
        results = {}
        
        # Common directories to skip
        skip_dirs = {
            'node_modules', '.git', 'dist', 'build', 'coverage',
            '.next', '.nuxt', 'out', '.cache', 'vendor'
        }
        
        for file_path in project_path.rglob('*'):
            # Skip directories
            if file_path.is_dir():
                continue
                
            # Skip if in excluded directory
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue
                
            # Only process JS/TS files
            if file_path.suffix in self.file_extensions:
                annotations = self.analyze_file(file_path)
                if annotations:
                    results[str(file_path)] = annotations
                    
        return results
    
    def suggest_controls(self, code: str) -> List[str]:
        """Suggest NIST controls for JavaScript/TypeScript code"""
        suggestions = set()
        patterns = self.find_security_patterns(code, "temp.js")
        
        for pattern in patterns:
            suggestions.update(pattern.suggested_controls)
            
        # Framework-specific suggestions
        code_lower = code.lower()
        
        # Web framework controls
        if any(framework in code_lower for framework in ['express', 'fastify', 'koa', 'hapi']):
            suggestions.update(["AC-3", "AC-4", "SC-8", "SI-10"])
            
        # Frontend framework controls
        if any(framework in code_lower for framework in ['react', 'vue', 'angular']):
            suggestions.update(["SI-10", "SC-18", "AC-4"])
            
        # Database controls
        if any(db in code_lower for db in ['mongodb', 'postgres', 'mysql', 'redis']):
            suggestions.update(["SC-28", "SI-10", "AU-2"])
            
        # Cloud service controls
        if any(cloud in code_lower for cloud in ['aws-sdk', 'azure', 'google-cloud']):
            suggestions.update(["AC-2", "AU-2", "SC-28", "SC-8"])
            
        # Authentication libraries
        if any(auth in code_lower for auth in ['passport', 'auth0', 'okta', 'firebase-auth']):
            suggestions.update(["IA-2", "IA-8", "AC-7"])
            
        return sorted(list(suggestions))
    
    def _detect_framework(self, code: str) -> Optional[str]:
        """Detect which framework is being used"""
        for framework, patterns in self.framework_patterns.items():
            if any(pattern in code for pattern in patterns):
                return framework
        return None
    
    def _analyze_package_json(self, project_path: Path) -> Dict[str, Any]:
        """Analyze package.json for security-relevant dependencies"""
        package_json_path = project_path / 'package.json'
        security_info = {
            'dependencies': [],
            'security_tools': [],
            'suggested_controls': []
        }
        
        if not package_json_path.exists():
            return security_info
            
        try:
            import json
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
                
            deps = list(package_data.get('dependencies', {}).keys())
            deps.extend(package_data.get('devDependencies', {}).keys())
            
            # Security-relevant dependencies
            security_deps = {
                'helmet': ["SC-8", "SI-10"],
                'cors': ["AC-4"],
                'express-rate-limit': ["SC-5"],
                'express-validator': ["SI-10"],
                'jsonwebtoken': ["IA-2", "SC-8"],
                'bcrypt': ["IA-5", "SC-13"],
                'crypto-js': ["SC-13"],
                'express-session': ["SC-23", "AC-12"],
                'passport': ["IA-2", "IA-8"],
                'winston': ["AU-2", "AU-3"],
                'dotenv': ["CM-7", "SC-28"]
            }
            
            for dep in deps:
                if dep in security_deps:
                    security_info['security_tools'].append(dep)
                    security_info['suggested_controls'].extend(security_deps[dep])
                    
            security_info['dependencies'] = deps
            security_info['suggested_controls'] = list(set(security_info['suggested_controls']))
            
        except Exception:
            pass
            
        return security_info