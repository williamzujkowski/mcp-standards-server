"""
Dockerfile analyzer for container security
@nist-controls: CM-2, CM-6, AC-6, SC-28, SI-2, IA-5, AU-12
@evidence: Comprehensive Dockerfile security analysis
@oscal-component: container-analyzer
"""

import logging
import re
from pathlib import Path
from typing import Any

from .base import BaseAnalyzer, CodeAnnotation

logger = logging.getLogger(__name__)


class DockerfileAnalyzer(BaseAnalyzer):
    """
    Analyzes Dockerfiles for security issues and best practices
    """

    def __init__(self):
        super().__init__()
        self.file_patterns = ['Dockerfile', 'Dockerfile.*', '*.dockerfile']
        self.instruction_patterns = self._initialize_instruction_patterns()
        self.base_image_patterns = self._initialize_base_image_patterns()
        self.secure_base_images = self._initialize_secure_base_images()

    def _initialize_instruction_patterns(self) -> list[dict[str, Any]]:
        """Initialize Dockerfile instruction security patterns"""
        return [
            {
                "pattern": r'^FROM\s+[\w/\-:]+:latest',
                "controls": ["CM-2"],
                "evidence": "Using 'latest' tag - unpinned base image version",
                "confidence": 0.90,
                "severity": "medium"
            },
            {
                "pattern": r'^USER\s+root\s*$',
                "controls": ["AC-6"],
                "evidence": "Explicitly running as root user",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "pattern": r'^RUN\s+.*sudo\s+',
                "controls": ["AC-6"],
                "evidence": "Using sudo indicates running with elevated privileges",
                "confidence": 0.85,
                "severity": "medium"
            },
            {
                "pattern": r'^ENV\s+\w*(?:PASSWORD|SECRET|KEY|TOKEN|API_KEY|PRIVATE_KEY)\w*\s*=',
                "controls": ["IA-5"],
                "evidence": "Hardcoded secrets in ENV instruction",
                "confidence": 0.95,
                "severity": "critical"
            },
            {
                "pattern": r'^ARG\s+\w*(?:PASSWORD|SECRET|KEY|TOKEN|API_KEY|PRIVATE_KEY)\w*\s*=',
                "controls": ["IA-5"],
                "evidence": "Hardcoded secrets in ARG instruction",
                "confidence": 0.95,
                "severity": "critical"
            },
            {
                "pattern": r'^EXPOSE\s+22\b',
                "controls": ["IA-2", "SC-7"],
                "evidence": "Exposing SSH port 22",
                "confidence": 0.90,
                "severity": "high"
            },
            {
                "pattern": r'^ADD\s+https?://',
                "controls": ["SI-2", "CM-6"],
                "evidence": "Using ADD with URL - prefer COPY with verified files",
                "confidence": 0.80,
                "severity": "medium"
            },
            {
                "pattern": r'^RUN\s+.*curl.*\|\s*sh',
                "controls": ["SI-2", "CM-6"],
                "evidence": "Piping curl to shell - security risk",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "pattern": r'^RUN\s+.*wget.*\|\s*sh',
                "controls": ["SI-2", "CM-6"],
                "evidence": "Piping wget to shell - security risk",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "pattern": r'^RUN\s+apt-get\s+install.*-y\s+ssh',
                "controls": ["CM-6", "IA-2"],
                "evidence": "Installing SSH server in container",
                "confidence": 0.85,
                "severity": "medium"
            }
        ]

    def _initialize_base_image_patterns(self) -> list[dict[str, Any]]:
        """Initialize base image security patterns"""
        return [
            {
                "pattern": r'^FROM\s+(?:ubuntu|debian|centos):(?:\d+\.?\d*|latest)',
                "controls": ["AC-6", "SI-2"],
                "evidence": "Using full OS base image - consider minimal/distroless",
                "confidence": 0.70,
                "severity": "low"
            },
            {
                "pattern": r'^FROM\s+[\w/\-:]+:?(?:alpha|beta|rc|dev|snapshot)',
                "controls": ["CM-2", "SI-2"],
                "evidence": "Using pre-release base image version",
                "confidence": 0.85,
                "severity": "medium"
            }
        ]

    def _initialize_secure_base_images(self) -> set[str]:
        """Initialize list of known secure base images"""
        return {
            "gcr.io/distroless",
            "alpine",
            "scratch",
            "busybox",
            "cgr.dev/chainguard"
        }

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze a Dockerfile for security issues"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            annotations = []
            lines = content.split('\n')

            # Track context
            has_user_instruction = False
            has_healthcheck = False
            base_image = None
            exposed_ports = []

            for i, line in enumerate(lines, 1):
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Check for USER instruction
                if line.startswith('USER ') and not line.startswith('USER root'):
                    has_user_instruction = True

                # Check for HEALTHCHECK
                if line.startswith('HEALTHCHECK'):
                    has_healthcheck = True

                # Extract base image
                if line.startswith('FROM '):
                    base_image = self._extract_base_image(line)
                    annotations.extend(self._analyze_base_image(base_image, i, file_path))

                # Track exposed ports
                if line.startswith('EXPOSE '):
                    ports = re.findall(r'\d+', line)
                    exposed_ports.extend(ports)

                # Check instruction patterns
                for pattern_def in self.instruction_patterns:
                    if re.match(pattern_def["pattern"], line, re.IGNORECASE):
                        annotations.append(CodeAnnotation(
                            file_path=str(file_path),
                            line_number=i,
                            control_ids=pattern_def["controls"],
                            evidence=pattern_def["evidence"],
                            confidence=pattern_def["confidence"],
                            component="dockerfile-instruction"
                        ))

                # Check for apt/yum without cleanup
                annotations.extend(self._check_package_manager(line, i, file_path))

                # Check for COPY/ADD with wrong ownership
                annotations.extend(self._check_file_operations(line, i, file_path))

            # Add context-based checks
            if not has_user_instruction:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["AC-6"],
                    evidence="No USER instruction - container runs as root by default",
                    confidence=0.90,
                    component="dockerfile-context"
                ))

            if not has_healthcheck:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["AU-12"],
                    evidence="No HEALTHCHECK instruction - container health monitoring missing",
                    confidence=0.80,
                    component="dockerfile-context"
                ))

            # Check for security best practices
            annotations.extend(self._check_best_practices(content, file_path))

            return annotations

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return []

    def _extract_base_image(self, from_line: str) -> str:
        """Extract base image from FROM instruction"""
        match = re.match(r'^FROM\s+(?:--platform=\S+\s+)?(\S+)', from_line)
        return match.group(1) if match else ""

    def _analyze_base_image(self, base_image: str, line_number: int, file_path: Path) -> list[CodeAnnotation]:
        """Analyze base image for security issues"""
        annotations = []

        if not base_image:
            return annotations

        # Check for missing tag
        if ':' not in base_image and '@' not in base_image:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=line_number,
                control_ids=["CM-2"],
                evidence=f"Base image '{base_image}' without tag - defaults to latest",
                confidence=0.85,
                component="dockerfile-base-image"
            ))

        # Check for outdated base images
        outdated_images = {
            "node:8": "Node.js 8 is end-of-life",
            "node:10": "Node.js 10 is end-of-life",
            "python:2": "Python 2 is end-of-life",
            "ubuntu:16.04": "Ubuntu 16.04 is end-of-life",
            "debian:8": "Debian 8 is end-of-life"
        }

        for old_image, reason in outdated_images.items():
            if base_image.startswith(old_image):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=["SI-2"],
                    evidence=f"Outdated base image: {reason}",
                    confidence=0.95,
                    component="dockerfile-base-image"
                ))

        # Check base image patterns
        for pattern_def in self.base_image_patterns:
            if re.match(pattern_def["pattern"], f"FROM {base_image}", re.IGNORECASE):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=pattern_def["controls"],
                    evidence=pattern_def["evidence"],
                    confidence=pattern_def["confidence"],
                    component="dockerfile-base-image"
                ))

        return annotations

    def _check_package_manager(self, line: str, line_number: int, file_path: Path) -> list[CodeAnnotation]:
        """Check package manager usage for security issues"""
        annotations = []

        # Check for missing cleanup after package installation
        if re.match(r'^RUN\s+(?:apt-get|yum|apk)\s+install', line, re.IGNORECASE):
            if not any(cleanup in line for cleanup in ['&& rm -rf', '&& apt-get clean', '&& yum clean']):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=["CM-6"],
                    evidence="Package manager cache not cleaned - increases image size and may contain sensitive data",
                    confidence=0.80,
                    component="dockerfile-packages"
                ))

        # Check for missing package version pinning
        if re.match(r'^RUN\s+.*(?:apt-get|yum|apk)\s+install\s+(?!.*=)', line, re.IGNORECASE):
            if not re.search(r'[=@:][\d\.]', line):  # No version specification
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=["CM-2"],
                    evidence="Package installation without version pinning",
                    confidence=0.70,
                    component="dockerfile-packages"
                ))

        return annotations

    def _check_file_operations(self, line: str, line_number: int, file_path: Path) -> list[CodeAnnotation]:
        """Check COPY/ADD operations for security issues"""
        annotations = []

        # Check for COPY/ADD without --chown
        if re.match(r'^(?:COPY|ADD)\s+(?!--chown)', line):
            # Only flag if we're in a multi-stage build or have USER instruction
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=line_number,
                control_ids=["AC-6"],
                evidence="COPY/ADD without --chown may result in incorrect file ownership",
                confidence=0.60,
                component="dockerfile-files"
            ))

        # Check for copying sensitive files
        sensitive_patterns = [
            r'\.env',
            r'\.git',
            r'private_key',
            r'id_rsa',
            r'\.ssh',
            r'credentials',
            r'\.aws'
        ]

        for pattern in sensitive_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=["IA-5", "SC-28"],
                    evidence=f"Potentially copying sensitive files matching pattern: {pattern}",
                    confidence=0.85,
                    component="dockerfile-files"
                ))

        return annotations

    def _check_best_practices(self, content: str, file_path: Path) -> list[CodeAnnotation]:
        """Check for Docker security best practices"""
        annotations = []

        # Check for multi-stage builds (good practice)
        if content.count('FROM ') == 1:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=1,
                control_ids=["CM-6"],
                evidence="Consider using multi-stage builds to reduce final image size and attack surface",
                confidence=0.50,
                component="dockerfile-practices"
            ))

        # Check for LABEL maintainer
        if 'LABEL maintainer' not in content and 'MAINTAINER' not in content:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=1,
                control_ids=["CM-2"],
                evidence="Missing maintainer information",
                confidence=0.40,
                component="dockerfile-metadata"
            ))

        # Check for security scanning labels
        security_labels = ['security.scan', 'version', 'build-date']
        for label in security_labels:
            if f'LABEL {label}' not in content:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["SI-2", "CM-2"],
                    evidence=f"Consider adding LABEL {label} for better security tracking",
                    confidence=0.30,
                    component="dockerfile-metadata"
                ))

        # Check for WORKDIR
        if 'WORKDIR' not in content:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=1,
                control_ids=["CM-6"],
                evidence="No WORKDIR set - using root directory by default",
                confidence=0.60,
                component="dockerfile-practices"
            ))

        return annotations

    def suggest_controls(self, code: str) -> set[str]:
        """Suggest NIST controls based on Dockerfile content"""
        controls = set()

        # Always relevant for containers
        controls.update(["CM-2", "CM-6", "AC-6"])

        # Conditional controls
        if re.search(r'(?:COPY|ADD).*(?:ssl|tls|cert|key)', code, re.IGNORECASE):
            controls.update(["SC-8", "SC-12", "SC-13"])

        if re.search(r'USER|chown|chmod', code, re.IGNORECASE):
            controls.update(["AC-3", "AC-6"])

        if re.search(r'SECRET|PASSWORD|KEY|TOKEN', code, re.IGNORECASE):
            controls.update(["IA-5", "SC-28"])

        if 'HEALTHCHECK' in code:
            controls.update(["AU-12", "SI-4"])

        if re.search(r'apt-get\s+update|yum\s+update|apk\s+update', code):
            controls.update(["SI-2"])

        return controls

    def _analyze_config_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Docker-related configuration files"""
        annotations = []

        # Check docker-compose.yml files
        if file_path.name == 'docker-compose.yml' or file_path.name.endswith('compose.yml'):
            annotations.extend(self._analyze_compose_file(file_path))

        # Check .dockerignore
        elif file_path.name == '.dockerignore':
            annotations.extend(self._analyze_dockerignore(file_path))

        return annotations

    def _analyze_compose_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze docker-compose.yml for security issues"""
        # This would be implemented in a separate compose_analyzer.py
        # Placeholder for now
        return []

    def _analyze_dockerignore(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze .dockerignore for missing entries"""
        annotations = []

        try:
            with open(file_path) as f:
                content = f.read()

            # Important files that should be in .dockerignore
            should_ignore = [
                '.git',
                '.env',
                '*.key',
                '*.pem',
                '.aws',
                '.ssh',
                'node_modules',
                '__pycache__',
                '*.log'
            ]

            for pattern in should_ignore:
                if pattern not in content:
                    annotations.append(CodeAnnotation(
                        file_path=str(file_path),
                        line_number=1,
                        control_ids=["CM-6", "IA-5"],
                        evidence=f"Consider adding '{pattern}' to .dockerignore",
                        confidence=0.60,
                        component="dockerignore"
                    ))

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")

        return annotations

    def analyze_project(self, project_path: Path) -> dict[str, Any]:
        """Analyze entire Docker project"""
        from src.compliance.scanner import ComplianceScanner

        # Use the compliance scanner for project-wide analysis
        scanner = ComplianceScanner()
        results = scanner.scan_directory(project_path)

        # Convert to expected format
        docker_files = []
        total_controls = set()

        for file_path, annotations in results.items():
            if any(pattern in str(file_path) for pattern in self.file_patterns):
                docker_files.append({
                    'file': str(file_path),
                    'annotations': [
                        {
                            'line': ann.line_number,
                            'controls': ann.control_ids,
                            'evidence': ann.evidence,
                            'confidence': ann.confidence
                        }
                        for ann in annotations
                    ]
                })
                for ann in annotations:
                    total_controls.update(ann.control_ids)

        return {
            'summary': {
                'files_analyzed': len(docker_files),
                'total_controls': len(total_controls),
                'docker_images': self._count_images(project_path)
            },
            'files': docker_files,
            'controls': sorted(total_controls)
        }

    def _count_images(self, project_path: Path) -> dict[str, int]:
        """Count Docker images in project"""
        image_counts = {}

        for file_path in project_path.rglob('*'):
            if any(pattern in file_path.name for pattern in self.file_patterns):
                try:
                    with open(file_path) as f:
                        content = f.read()

                    # Count base images
                    from_pattern = r'^FROM\s+(?:--platform=\S+\s+)?(\S+)'
                    for match in re.finditer(from_pattern, content, re.MULTILINE):
                        image = match.group(1)
                        base_name = image.split(':')[0]
                        image_counts[base_name] = image_counts.get(base_name, 0) + 1
                except Exception:
                    continue

        return image_counts
