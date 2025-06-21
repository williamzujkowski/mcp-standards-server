"""
Kubernetes manifest analyzer for container orchestration security
@nist-controls: AC-3, AC-4, AC-6, SC-5, SC-7, CM-2, CM-6, AU-12, SI-4
@evidence: Comprehensive Kubernetes security analysis
@oscal-component: k8s-analyzer
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from .base import BaseAnalyzer, CodeAnnotation

logger = logging.getLogger(__name__)


class KubernetesAnalyzer(BaseAnalyzer):
    """
    Analyzes Kubernetes manifests for security issues
    """

    def __init__(self):
        super().__init__()
        self.file_extensions = ['.yaml', '.yml']
        self.k8s_kinds = {
            'Pod', 'Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob',
            'Service', 'Ingress', 'NetworkPolicy', 'PodSecurityPolicy',
            'Role', 'ClusterRole', 'RoleBinding', 'ClusterRoleBinding',
            'ServiceAccount', 'Secret', 'ConfigMap'
        }
        self.security_contexts = self._initialize_security_contexts()
        self.rbac_patterns = self._initialize_rbac_patterns()
        self.container_patterns = self._initialize_container_patterns()

    def _initialize_security_contexts(self) -> list[dict[str, Any]]:
        """Initialize security context patterns"""
        return [
            {
                "check": "privileged_container",
                "controls": ["AC-6"],
                "evidence": "Container running in privileged mode",
                "confidence": 0.95,
                "severity": "critical"
            },
            {
                "check": "run_as_root",
                "controls": ["AC-6"],
                "evidence": "Container running as root user (UID 0)",
                "confidence": 0.90,
                "severity": "high"
            },
            {
                "check": "allow_privilege_escalation",
                "controls": ["AC-6"],
                "evidence": "Container allows privilege escalation",
                "confidence": 0.90,
                "severity": "high"
            },
            {
                "check": "host_network",
                "controls": ["SC-7", "AC-4"],
                "evidence": "Pod using host network namespace",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "check": "host_pid",
                "controls": ["AC-6", "SC-7"],
                "evidence": "Pod using host PID namespace",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "check": "host_ipc",
                "controls": ["AC-6", "SC-7"],
                "evidence": "Pod using host IPC namespace",
                "confidence": 0.95,
                "severity": "high"
            },
            {
                "check": "no_security_context",
                "controls": ["CM-6"],
                "evidence": "Pod/Container missing security context",
                "confidence": 0.80,
                "severity": "medium"
            },
            {
                "check": "writable_root_filesystem",
                "controls": ["AC-6", "CM-6"],
                "evidence": "Container filesystem is writable",
                "confidence": 0.75,
                "severity": "medium"
            },
            {
                "check": "no_resource_limits",
                "controls": ["SC-5"],
                "evidence": "Container missing resource limits",
                "confidence": 0.85,
                "severity": "medium"
            }
        ]

    def _initialize_rbac_patterns(self) -> list[dict[str, Any]]:
        """Initialize RBAC patterns"""
        return [
            {
                "pattern": r'rules:\s*-\s*apiGroups:\s*\[\s*["\']?\*["\']?\s*\].*resources:\s*\[\s*["\']?\*["\']?\s*\].*verbs:\s*\[\s*["\']?\*["\']?\s*\]',
                "controls": ["AC-6", "AC-3"],
                "evidence": "Overly permissive RBAC rule (*:*:*)",
                "confidence": 0.95,
                "severity": "critical"
            },
            {
                "pattern": r'verbs:\s*\[.*["\']?\*["\']?.*\]',
                "controls": ["AC-6"],
                "evidence": "RBAC rule with wildcard verbs",
                "confidence": 0.85,
                "severity": "high"
            },
            {
                "pattern": r'resources:\s*\[.*["\']?secrets["\']?.*\].*verbs:\s*\[.*["\']?get["\']?.*\]',
                "controls": ["AC-3", "IA-5"],
                "evidence": "RBAC rule allowing secret access",
                "confidence": 0.80,
                "severity": "medium"
            },
            {
                "pattern": r'clusterRole:\s*cluster-admin',
                "controls": ["AC-6"],
                "evidence": "Binding to cluster-admin role",
                "confidence": 0.95,
                "severity": "critical"
            }
        ]

    def _initialize_container_patterns(self) -> list[dict[str, Any]]:
        """Initialize container-specific patterns"""
        return [
            {
                "pattern": r'image:\s*[^:]+:latest',
                "controls": ["CM-2"],
                "evidence": "Using 'latest' tag for container image",
                "confidence": 0.90,
                "severity": "medium"
            },
            {
                "pattern": r'imagePullPolicy:\s*Always',
                "controls": ["CM-2", "SI-2"],
                "evidence": "Always pulling images (good for updates)",
                "confidence": 0.60,
                "severity": "info"
            },
            {
                "pattern": r'env:\s*-\s*name:\s*(?:PASSWORD|SECRET|KEY|TOKEN)',
                "controls": ["IA-5"],
                "evidence": "Potential secret in environment variable",
                "confidence": 0.85,
                "severity": "high"
            }
        ]

    def analyze_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze a Kubernetes manifest file"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Skip non-Kubernetes YAML files
            if not self._is_kubernetes_manifest(content):
                return []

            annotations = []

            # Parse YAML documents (handle multi-doc files)
            documents = list(yaml.safe_load_all(content))

            for doc_index, doc in enumerate(documents):
                if not doc or not isinstance(doc, dict):
                    continue

                kind = doc.get('kind', '')
                if kind not in self.k8s_kinds:
                    continue

                # Analyze based on resource type
                if kind in ['Pod', 'Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob']:
                    annotations.extend(self._analyze_workload(doc, file_path, doc_index))
                elif kind in ['Role', 'ClusterRole']:
                    annotations.extend(self._analyze_rbac(doc, file_path, doc_index))
                elif kind in ['RoleBinding', 'ClusterRoleBinding']:
                    annotations.extend(self._analyze_rbac_binding(doc, file_path, doc_index))
                elif kind == 'NetworkPolicy':
                    annotations.extend(self._analyze_network_policy(doc, file_path, doc_index))
                elif kind == 'Ingress':
                    annotations.extend(self._analyze_ingress(doc, file_path, doc_index))
                elif kind == 'Service':
                    annotations.extend(self._analyze_service(doc, file_path, doc_index))
                elif kind == 'Secret':
                    annotations.extend(self._analyze_secret(doc, file_path, doc_index))

            # Also check for patterns in raw content
            annotations.extend(self._analyze_raw_patterns(content, file_path))

            return annotations

        except Exception as e:
            logger.error(f"Error analyzing {file_path}: {e}")
            return []

    def _is_kubernetes_manifest(self, content: str) -> bool:
        """Check if file is a Kubernetes manifest"""
        # Look for Kubernetes API version and kind
        return bool(re.search(r'apiVersion:\s*\S+', content) and
                   re.search(r'kind:\s*(' + '|'.join(self.k8s_kinds) + ')', content))

    def _analyze_workload(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze workload resources (Pod, Deployment, etc.)"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        # Get pod spec
        spec = manifest.get('spec', {})
        if manifest['kind'] == 'Pod':
            pod_spec = spec
        elif manifest['kind'] in ['CronJob', 'Job']:
            # CronJob and Job have additional nesting
            pod_spec = spec.get('jobTemplate', spec).get('spec', {}).get('template', {}).get('spec', {})
        else:
            # For Deployment, StatefulSet, etc.
            pod_spec = spec.get('template', {}).get('spec', {})

        # Check pod-level security
        annotations.extend(self._check_pod_security(pod_spec, file_path, base_line))

        # Check containers
        containers = pod_spec.get('containers', [])
        for i, container in enumerate(containers):
            annotations.extend(self._check_container_security(container, file_path, base_line, i))

        # Check init containers
        init_containers = pod_spec.get('initContainers', [])
        for i, container in enumerate(init_containers):
            annotations.extend(self._check_container_security(container, file_path, base_line, i, is_init=True))

        # Check volumes
        annotations.extend(self._check_volumes(pod_spec.get('volumes', []), file_path, base_line))

        # Check service account
        if not pod_spec.get('serviceAccountName') and not pod_spec.get('serviceAccount'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["AC-3", "AC-6"],
                evidence="Pod using default service account",
                confidence=0.70,
                component="k8s-service-account"
            ))

        return annotations

    def _check_pod_security(self, pod_spec: dict[str, Any], file_path: Path, base_line: int) -> list[CodeAnnotation]:
        """Check pod-level security settings"""
        annotations = []

        # Check host namespaces
        if pod_spec.get('hostNetwork'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-7", "AC-4"],
                evidence="Pod using host network namespace",
                confidence=0.95,
                component="k8s-pod-security"
            ))

        if pod_spec.get('hostPID'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["AC-6", "SC-7"],
                evidence="Pod using host PID namespace",
                confidence=0.95,
                component="k8s-pod-security"
            ))

        if pod_spec.get('hostIPC'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["AC-6", "SC-7"],
                evidence="Pod using host IPC namespace",
                confidence=0.95,
                component="k8s-pod-security"
            ))

        # Check pod security context
        pod_security_context = pod_spec.get('securityContext', {})
        if not pod_security_context:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["CM-6"],
                evidence="Pod missing security context",
                confidence=0.75,
                component="k8s-pod-security"
            ))
        else:
            # Check if running as root
            if pod_security_context.get('runAsUser') == 0:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6"],
                    evidence="Pod configured to run as root (UID 0)",
                    confidence=0.90,
                    component="k8s-pod-security"
                ))

            # Good practice: runAsNonRoot
            if pod_security_context.get('runAsNonRoot'):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6"],
                    evidence="Pod enforces non-root execution (good)",
                    confidence=0.90,
                    component="k8s-pod-security"
                ))

        return annotations

    def _check_container_security(self, container: dict[str, Any], file_path: Path,
                                  base_line: int, index: int, is_init: bool = False) -> list[CodeAnnotation]:
        """Check container-level security settings"""
        annotations = []
        container_type = "init container" if is_init else "container"
        container_name = container.get('name', f'{container_type} {index}')

        # Check image tag
        image = container.get('image', '')
        if image.endswith(':latest') or (':' not in image and '@' not in image):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["CM-2"],
                evidence=f"Container '{container_name}' using unpinned image version",
                confidence=0.85,
                component="k8s-container"
            ))

        # Check security context
        security_context = container.get('securityContext', {})
        if not security_context:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["CM-6"],
                evidence=f"Container '{container_name}' missing security context",
                confidence=0.70,
                component="k8s-container"
            ))
        else:
            # Check privileged
            if security_context.get('privileged'):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6"],
                    evidence=f"Container '{container_name}' running in privileged mode",
                    confidence=0.95,
                    component="k8s-container"
                ))

            # Check allowPrivilegeEscalation
            if security_context.get('allowPrivilegeEscalation') is not False:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6"],
                    evidence=f"Container '{container_name}' allows privilege escalation",
                    confidence=0.85,
                    component="k8s-container"
                ))

            # Check readOnlyRootFilesystem
            if not security_context.get('readOnlyRootFilesystem'):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6", "CM-6"],
                    evidence=f"Container '{container_name}' has writable root filesystem",
                    confidence=0.60,
                    component="k8s-container"
                ))

            # Check capabilities
            capabilities = security_context.get('capabilities', {})
            if capabilities.get('add'):
                dangerous_caps = ['SYS_ADMIN', 'NET_ADMIN', 'SYS_PTRACE', 'SYS_MODULE']
                added_caps = capabilities['add']
                for cap in dangerous_caps:
                    if cap in added_caps:
                        annotations.append(CodeAnnotation(
                            file_path=str(file_path),
                            line_number=base_line,
                            control_ids=["AC-6"],
                            evidence=f"Container '{container_name}' adds dangerous capability: {cap}",
                            confidence=0.90,
                            component="k8s-container"
                        ))

        # Check resource limits
        resources = container.get('resources', {})
        if not resources.get('limits'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-5"],
                evidence=f"Container '{container_name}' missing resource limits",
                confidence=0.80,
                component="k8s-container"
            ))

        # Check for secrets in env
        env_vars = container.get('env', [])
        for env in env_vars:
            name = env.get('name', '')
            if any(secret in name.upper() for secret in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'CREDENTIAL']):
                if env.get('value'):  # Direct value (not from secret)
                    annotations.append(CodeAnnotation(
                        file_path=str(file_path),
                        line_number=base_line,
                        control_ids=["IA-5"],
                        evidence=f"Container '{container_name}' has potential secret in env var: {name}",
                        confidence=0.85,
                        component="k8s-container"
                    ))

        # Check health checks
        if not container.get('livenessProbe') and not container.get('readinessProbe'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["AU-12", "SI-4"],
                evidence=f"Container '{container_name}' missing health checks",
                confidence=0.70,
                component="k8s-container"
            ))

        return annotations

    def _check_volumes(self, volumes: list[dict[str, Any]], file_path: Path, base_line: int) -> list[CodeAnnotation]:
        """Check volume security"""
        annotations = []

        for volume in volumes:
            # Check hostPath volumes
            if 'hostPath' in volume:
                path = volume['hostPath'].get('path', '')
                dangerous_paths = ['/', '/etc', '/var', '/usr', '/root', '/home']
                if path in dangerous_paths or any(path.startswith(dp + '/') for dp in dangerous_paths[1:]):
                    annotations.append(CodeAnnotation(
                        file_path=str(file_path),
                        line_number=base_line,
                        control_ids=["AC-6", "SC-7"],
                        evidence=f"Volume mounting sensitive host path: {path}",
                        confidence=0.90,
                        component="k8s-volume"
                    ))

        return annotations

    def _analyze_rbac(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze RBAC Role/ClusterRole"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        rules = manifest.get('rules', [])
        for rule in rules:
            api_groups = rule.get('apiGroups', [])
            resources = rule.get('resources', [])
            verbs = rule.get('verbs', [])

            # Check for overly permissive rules
            if '*' in api_groups and '*' in resources and '*' in verbs:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6", "AC-3"],
                    evidence="RBAC rule grants cluster-admin equivalent permissions (*:*:*)",
                    confidence=0.95,
                    component="k8s-rbac"
                ))
            elif '*' in verbs:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-6"],
                    evidence="RBAC rule uses wildcard verbs",
                    confidence=0.85,
                    component="k8s-rbac"
                ))

            # Check for sensitive resource access
            sensitive_resources = ['secrets', 'configmaps', 'serviceaccounts', 'pods/exec']
            for resource in resources:
                if resource in sensitive_resources:
                    annotations.append(CodeAnnotation(
                        file_path=str(file_path),
                        line_number=base_line,
                        control_ids=["AC-3", "IA-5"],
                        evidence=f"RBAC rule grants access to sensitive resource: {resource}",
                        confidence=0.75,
                        component="k8s-rbac"
                    ))

        return annotations

    def _analyze_rbac_binding(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze RBAC RoleBinding/ClusterRoleBinding"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        role_ref = manifest.get('roleRef', {})
        if role_ref.get('name') == 'cluster-admin':
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["AC-6"],
                evidence="Binding to cluster-admin role",
                confidence=0.95,
                component="k8s-rbac-binding"
            ))

        # Check subjects
        subjects = manifest.get('subjects', [])
        for subject in subjects:
            if subject.get('kind') == 'User' and subject.get('name') == 'system:anonymous':
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["AC-3", "IA-2"],
                    evidence="RBAC binding includes anonymous user",
                    confidence=0.90,
                    component="k8s-rbac-binding"
                ))

        return annotations

    def _analyze_network_policy(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze NetworkPolicy"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        spec = manifest.get('spec', {})

        # Good practice: having network policies
        annotations.append(CodeAnnotation(
            file_path=str(file_path),
            line_number=base_line,
            control_ids=["SC-7", "AC-4"],
            evidence="Network policy defined for network segmentation (good)",
            confidence=0.80,
            component="k8s-network-policy"
        ))

        # Check for overly permissive policies
        ingress = spec.get('ingress', [])
        egress = spec.get('egress', [])

        if not ingress and 'ingress' in spec.get('policyTypes', []):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-7"],
                evidence="Network policy allows all ingress traffic",
                confidence=0.85,
                component="k8s-network-policy"
            ))

        if not egress and 'egress' in spec.get('policyTypes', []):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-7"],
                evidence="Network policy allows all egress traffic",
                confidence=0.85,
                component="k8s-network-policy"
            ))

        return annotations

    def _analyze_ingress(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze Ingress resources"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        spec = manifest.get('spec', {})

        # Check TLS configuration
        if not spec.get('tls'):
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-8", "SC-13"],
                evidence="Ingress without TLS configuration",
                confidence=0.85,
                component="k8s-ingress"
            ))

        # Check annotations for security headers
        annotations_dict = manifest.get('metadata', {}).get('annotations', {})
        security_headers = [
            'nginx.ingress.kubernetes.io/force-ssl-redirect',
            'nginx.ingress.kubernetes.io/ssl-protocols',
            'nginx.ingress.kubernetes.io/ssl-ciphers'
        ]

        for header in security_headers:
            if header not in annotations_dict:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["SC-8", "SC-13"],
                    evidence=f"Ingress missing security annotation: {header}",
                    confidence=0.60,
                    component="k8s-ingress"
                ))

        return annotations

    def _analyze_service(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze Service resources"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        spec = manifest.get('spec', {})

        # Check for NodePort services (potential security risk)
        if spec.get('type') == 'NodePort':
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["SC-7", "AC-4"],
                evidence="Service exposed via NodePort - consider LoadBalancer or Ingress",
                confidence=0.70,
                component="k8s-service"
            ))

        # Check for LoadBalancer without annotations
        if spec.get('type') == 'LoadBalancer':
            annotations_dict = manifest.get('metadata', {}).get('annotations', {})
            if not any('service.beta.kubernetes.io' in key for key in annotations_dict):
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=base_line,
                    control_ids=["SC-7"],
                    evidence="LoadBalancer service without security annotations",
                    confidence=0.60,
                    component="k8s-service"
                ))

        return annotations

    def _analyze_secret(self, manifest: dict[str, Any], file_path: Path, doc_index: int) -> list[CodeAnnotation]:
        """Analyze Secret resources"""
        annotations = []
        base_line = self._get_line_number(file_path, doc_index)

        # Check if secret type is specified
        secret_type = manifest.get('type', 'Opaque')
        if secret_type == 'Opaque':
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["IA-5"],
                evidence="Using generic Opaque secret type - consider specific types",
                confidence=0.50,
                component="k8s-secret"
            ))

        # Check for unencrypted data (should use stringData for clarity)
        if 'data' in manifest:
            annotations.append(CodeAnnotation(
                file_path=str(file_path),
                line_number=base_line,
                control_ids=["IA-5", "SC-28"],
                evidence="Secret data should be stored encrypted at rest",
                confidence=0.70,
                component="k8s-secret"
            ))

        return annotations

    def _analyze_raw_patterns(self, content: str, file_path: Path) -> list[CodeAnnotation]:
        """Analyze raw content for patterns"""
        annotations = []
        content.split('\n')

        # Check RBAC patterns
        for pattern_def in self.rbac_patterns:
            for match in re.finditer(pattern_def["pattern"], content, re.MULTILINE | re.DOTALL):
                line_number = content[:match.start()].count('\n') + 1
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=pattern_def["controls"],
                    evidence=pattern_def["evidence"],
                    confidence=pattern_def["confidence"],
                    component="k8s-pattern"
                ))

        # Check container patterns
        for pattern_def in self.container_patterns:
            for match in re.finditer(pattern_def["pattern"], content, re.MULTILINE):
                line_number = content[:match.start()].count('\n') + 1
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=line_number,
                    control_ids=pattern_def["controls"],
                    evidence=pattern_def["evidence"],
                    confidence=pattern_def["confidence"],
                    component="k8s-pattern"
                ))

        return annotations

    def _get_line_number(self, file_path: Path, doc_index: int) -> int:
        """Get approximate line number for a YAML document"""
        # For multi-document YAML files, this is approximate
        # In practice, you'd parse the file to get exact positions
        return 1 + (doc_index * 10)  # Rough estimate

    def suggest_controls(self, code: str) -> list[str]:
        """Suggest NIST controls based on Kubernetes resources"""
        controls = set()

        # Resource type to controls mapping
        resource_controls = {
            "networkpolicy": ["SC-7", "AC-4"],
            "rbac": ["AC-2", "AC-3", "AC-6"],
            "podsecuritypolicy": ["AC-6", "CM-6"],
            "securitycontext": ["AC-6", "CM-6"],
            "secret": ["IA-5", "SC-28"],
            "tls": ["SC-8", "SC-13"],
            "ingress": ["SC-8", "SC-13", "SC-7"],
            "audit": ["AU-2", "AU-3", "AU-12"],
            "resourcequota": ["SC-5"],
            "limitrange": ["SC-5"],
            "monitoring": ["AU-12", "SI-4"]
        }

        for resource_type, type_controls in resource_controls.items():
            if resource_type in code.lower():
                controls.update(type_controls)

        return list(controls)

    def _analyze_config_file(self, file_path: Path) -> list[CodeAnnotation]:
        """Analyze Kubernetes configuration files"""
        annotations = []

        # Handle kubeconfig files
        if file_path.name in ['config', 'kubeconfig'] or file_path.suffix == '.kubeconfig':
            with open(file_path) as f:
                content = f.read()

            # Check for embedded certificates/keys
            if 'client-certificate-data:' in content or 'client-key-data:' in content:
                annotations.append(CodeAnnotation(
                    file_path=str(file_path),
                    line_number=1,
                    control_ids=["IA-5", "SC-28"],
                    evidence="Kubeconfig contains embedded credentials - should use external files",
                    confidence=0.80,
                    component="k8s-kubeconfig"
                ))

        return annotations

    async def analyze_project(self, project_path: Path) -> dict[str, Any]:
        """Analyze entire Kubernetes project"""
        from src.compliance.scanner import ComplianceScanner

        # Use the compliance scanner for project-wide analysis
        scanner = ComplianceScanner()
        results = await scanner.scan_directory(project_path)

        # Convert to expected format
        k8s_files = []
        total_controls = set()

        for result in results:
            file_path = result.get("file_path", "")
            if any(str(file_path).endswith(ext) for ext in self.file_extensions):
                # Extract annotations from result
                annotations = []
                for control in result.get("controls_found", []):
                    annotations.append({
                        'line': 1,  # Scanner doesn't provide line numbers
                        'controls': [control],
                        'evidence': f"Control {control} detected",
                        'confidence': 0.8
                    })

                k8s_files.append({
                    'file': str(file_path),
                    'annotations': annotations
                })
                total_controls.update(result.get("controls_found", []))

        return {
            'summary': {
                'files_analyzed': len(k8s_files),
                'total_controls': len(total_controls),
                'k8s_resources': self._count_resources(project_path)
            },
            'files': k8s_files,
            'controls': sorted(total_controls)
        }

    def _count_resources(self, project_path: Path) -> dict[str, int]:
        """Count Kubernetes resources in project"""
        resource_counts: dict[str, int] = {}

        for file_path in project_path.rglob('*.yaml'):
            try:
                with open(file_path) as f:
                    content = f.read()

                if self._is_kubernetes_manifest(content):
                    documents = list(yaml.safe_load_all(content))
                    for doc in documents:
                        if doc and isinstance(doc, dict):
                            kind = doc.get('kind', 'Unknown')
                            resource_counts[kind] = resource_counts.get(kind, 0) + 1
            except Exception:
                continue

        return resource_counts
