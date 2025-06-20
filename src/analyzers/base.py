"""
Base code analyzer with language-specific implementations
@nist-controls: SA-11, SA-15, CA-7
@evidence: Continuous code analysis and security testing
"""
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .enhanced_patterns import EnhancedNISTPatterns


@dataclass
class SecurityPattern:
    """Represents a security pattern in code"""
    pattern_type: str
    location: str
    line_number: int
    confidence: float
    details: dict[str, Any]
    suggested_controls: list[str]


@dataclass
class CodeAnnotation:
    """Represents a NIST annotation in code"""
    file_path: str
    line_number: int
    control_ids: list[str]
    evidence: str | None
    component: str | None
    confidence: float


class BaseAnalyzer(ABC):
    """
    Abstract base class for language analyzers
    @nist-controls: PM-5, SA-11
    @evidence: Systematic security analysis across languages
    """

    def __init__(self):
        self.security_patterns: list[SecurityPattern] = []
        self.enhanced_patterns = EnhancedNISTPatterns()

    @abstractmethod
    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze a single file for NIST controls"""
        pass

    @abstractmethod
    def analyze_project(self, project_path: Path) -> dict[str, list[CodeAnnotation]]:
        """Analyze entire project"""
        pass

    @abstractmethod
    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls for given code"""
        pass

    def find_security_patterns(self, code: str, file_path: str) -> list[SecurityPattern]:
        """Find common security patterns in code"""
        patterns = []

        # Authentication patterns
        if self._has_authentication(code):
            patterns.append(SecurityPattern(
                pattern_type="authentication",
                location=file_path,
                line_number=self._find_pattern_line(code, "auth"),
                confidence=0.8,
                details={"type": "basic"},
                suggested_controls=["IA-2", "IA-5", "AC-7"]
            ))

        # Encryption patterns
        if self._has_encryption(code):
            patterns.append(SecurityPattern(
                pattern_type="encryption",
                location=file_path,
                line_number=self._find_pattern_line(code, "encrypt"),
                confidence=0.9,
                details={"algorithms": self._find_crypto_algorithms(code)},
                suggested_controls=["SC-8", "SC-13", "SC-28"]
            ))

        # Access control patterns
        if self._has_access_control(code):
            patterns.append(SecurityPattern(
                pattern_type="access_control",
                location=file_path,
                line_number=self._find_pattern_line(code, "permission"),
                confidence=0.85,
                details={"type": "rbac"},
                suggested_controls=["AC-2", "AC-3", "AC-6"]
            ))

        # Logging patterns
        if self._has_logging(code):
            patterns.append(SecurityPattern(
                pattern_type="logging",
                location=file_path,
                line_number=self._find_pattern_line(code, "log"),
                confidence=0.9,
                details={"type": "security"},
                suggested_controls=["AU-2", "AU-3", "AU-12"]
            ))

        # Input validation patterns
        if self._has_input_validation(code):
            patterns.append(SecurityPattern(
                pattern_type="input_validation",
                location=file_path,
                line_number=self._find_pattern_line(code, "validat"),
                confidence=0.85,
                details={"type": "input_sanitization"},
                suggested_controls=["SI-10", "SI-15"]
            ))

        return patterns

    def extract_annotations(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Extract @nist-controls annotations from code"""
        annotations = []

        # Regex for @nist-controls annotations
        pattern = r'@nist-controls:\s*([A-Z]{2}-\d+(?:\(\d+\))?(?:\s*,\s*[A-Z]{2}-\d+(?:\(\d+\))?)*)'
        evidence_pattern = r'@evidence:\s*(.+?)(?=\n|$)'
        component_pattern = r'@oscal-component:\s*(.+?)(?=\n|$)'

        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Find control annotations
            control_match = re.search(pattern, line)
            if control_match:
                controls = [c.strip() for c in control_match.group(1).split(',')]

                # Look for evidence in nearby lines
                evidence = None
                component = None

                for j in range(max(0, i-2), min(len(lines), i+3)):
                    evidence_match = re.search(evidence_pattern, lines[j])
                    if evidence_match:
                        evidence = evidence_match.group(1).strip()

                    component_match = re.search(component_pattern, lines[j])
                    if component_match:
                        component = component_match.group(1).strip()

                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=i + 1,
                    control_ids=controls,
                    evidence=evidence,
                    component=component,
                    confidence=1.0  # Explicit annotations have full confidence
                ))

        return annotations

    def analyze_with_enhanced_patterns(self, code: str, file_path: str) -> list[CodeAnnotation]:
        """Analyze code using enhanced NIST pattern detection"""
        annotations = []
        
        # Get all matching patterns
        pattern_matches = self.enhanced_patterns.get_patterns_for_code(code)
        
        for pattern, match in pattern_matches:
            # Find the line number
            lines_before = code[:match.start()].count('\n')
            line_number = lines_before + 1
            
            # Generate evidence
            evidence = pattern.evidence_template.format(match=match.group(0))
            
            annotations.append(CodeAnnotation(
                file_path=file_path,
                line_number=line_number,
                control_ids=pattern.control_ids,
                evidence=evidence,
                component=pattern.pattern_type,
                confidence=pattern.confidence
            ))
        
        return annotations

    def _has_authentication(self, code: str) -> bool:
        """Check if code has authentication patterns"""
        auth_keywords = [
            "authenticate", "login", "signin", "auth",
            "credential", "password", "token", "jwt",
            "oauth", "saml", "ldap", "mfa", "2fa"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in auth_keywords)

    def _has_encryption(self, code: str) -> bool:
        """Check if code has encryption patterns"""
        crypto_keywords = [
            "encrypt", "decrypt", "cipher", "aes", "rsa",
            "sha", "hash", "crypto", "tls", "ssl", "https",
            "certificate", "key", "kms"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in crypto_keywords)

    def _has_access_control(self, code: str) -> bool:
        """Check if code has access control patterns"""
        ac_keywords = [
            "permission", "authorize", "role", "rbac", "abac",
            "policy", "grant", "deny", "access", "privilege",
            "can", "cannot", "allow", "forbidden"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in ac_keywords)

    def _has_logging(self, code: str) -> bool:
        """Check if code has logging patterns"""
        log_keywords = [
            "log", "audit", "trace", "monitor", "event",
            "track", "record", "journal", "syslog"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in log_keywords)

    def _has_input_validation(self, code: str) -> bool:
        """Check if code has input validation patterns"""
        validation_keywords = [
            "validate", "sanitize", "escape", "filter",
            "clean", "strip", "check", "verify", "schema"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in validation_keywords)

    def _find_pattern_line(self, code: str, pattern: str) -> int:
        """Find line number where pattern first appears"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                return i + 1
        return 1

    def _find_crypto_algorithms(self, code: str) -> list[str]:
        """Find cryptographic algorithms mentioned in code"""
        algorithms = []
        crypto_patterns = {
            "AES": r"aes[-_]?\d{3}",
            "RSA": r"rsa[-_]?\d{4}",
            "SHA": r"sha[-_]?\d{3}",
            "ECDSA": r"ecdsa|ec[-_]?dsa",
            "HMAC": r"hmac",
            "PBKDF2": r"pbkdf2",
            "Bcrypt": r"bcrypt",
            "Argon2": r"argon2"
        }

        code_lower = code.lower()
        for name, pattern in crypto_patterns.items():
            if re.search(pattern, code_lower):
                algorithms.append(name)

        return algorithms
