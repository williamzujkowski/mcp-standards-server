"""
Tests for Kubernetes manifest analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Kubernetes analyzer testing
"""


import pytest

from src.analyzers.k8s_analyzer import KubernetesAnalyzer


class TestKubernetesAnalyzer:
    """Test Kubernetes manifest analysis capabilities"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return KubernetesAnalyzer()

    def test_detect_privileged_pod(self, analyzer, tmp_path):
        """Test detection of privileged containers"""
        test_file = tmp_path / "privileged-pod.yaml"
        manifest = '''apiVersion: v1
kind: Pod
metadata:
  name: privileged-pod
spec:
  containers:
  - name: app
    image: nginx:latest
    securityContext:
      privileged: true
      runAsUser: 0
      allowPrivilegeEscalation: true
  hostNetwork: true
  hostPID: true
  hostIPC: true
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Privileged mode
        assert "SC-7" in controls  # Host namespaces
        assert "CM-2" in controls  # Latest tag

        # Check specific issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("privileged" in ev for ev in evidence_texts)
        assert any("host network" in ev for ev in evidence_texts)
        assert any("host pid" in ev for ev in evidence_texts)
        assert any("host ipc" in ev for ev in evidence_texts)
        assert any("root" in ev or "uid 0" in ev for ev in evidence_texts)

    def test_detect_secure_pod(self, analyzer, tmp_path):
        """Test secure pod configuration"""
        test_file = tmp_path / "secure-pod.yaml"
        manifest = '''apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: nginx:1.21.6
    imagePullPolicy: Always
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
      requests:
        memory: "256Mi"
        cpu: "250m"
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect good practices
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("non-root" in ev and "good" in ev for ev in evidence_texts)

        # Should not have critical issues
        critical_issues = [ann for ann in results if "critical" in ann.evidence.lower() or
                          "privileged" in ann.evidence.lower()]
        assert len(critical_issues) == 0

    def test_detect_deployment_issues(self, analyzer, tmp_path):
        """Test deployment security analysis"""
        test_file = tmp_path / "deployment.yaml"
        manifest = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: webapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: webapp
  template:
    metadata:
      labels:
        app: webapp
    spec:
      containers:
      - name: webapp
        image: myapp:latest
        env:
        - name: DATABASE_PASSWORD
          value: "admin123"
        - name: API_KEY
          value: "sk-1234567890"
        ports:
        - containerPort: 8080
        # Missing security context
        # Missing resource limits
        # Missing health checks
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-5" in controls  # Hardcoded secrets
        assert "CM-6" in controls  # Missing security context
        assert "SC-5" in controls  # Missing resource limits
        assert "AU-12" in controls or "SI-4" in controls  # Missing health checks

        # Check specific issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("password" in ev or "secret" in ev for ev in evidence_texts)
        assert any("security context" in ev for ev in evidence_texts)
        assert any("resource limits" in ev for ev in evidence_texts)

    def test_detect_rbac_issues(self, analyzer, tmp_path):
        """Test RBAC security analysis"""
        test_file = tmp_path / "rbac.yaml"
        manifest = '''---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: overly-permissive
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: admin-binding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: cluster-admin
subjects:
- kind: User
  name: developer
  apiGroup: rbac.authorization.k8s.io
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secret-reader
  namespace: default
rules:
- apiGroups: [""]
  resources: ["secrets", "configmaps"]
  verbs: ["get", "list", "watch"]
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect RBAC issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Excessive privileges
        assert "AC-3" in controls  # Access control

        # Check specific issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("*:*:*" in ev or "overly permissive" in ev for ev in evidence_texts)
        assert any("cluster-admin" in ev for ev in evidence_texts)
        assert any("secret" in ev for ev in evidence_texts)

    def test_detect_network_policy(self, analyzer, tmp_path):
        """Test network policy analysis"""
        test_file = tmp_path / "network-policy.yaml"
        manifest = '''apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: webapp-netpol
spec:
  podSelector:
    matchLabels:
      app: webapp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend
    - podSelector:
        matchLabels:
          app: api-gateway
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: backend
    ports:
    - protocol: TCP
      port: 5432
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect good network segmentation
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-7" in controls  # Network segmentation
        assert "AC-4" in controls  # Information flow

        # Should identify this as good practice
        assert any("good" in ann.evidence.lower() and "network" in ann.evidence.lower() for ann in results)

    def test_detect_ingress_issues(self, analyzer, tmp_path):
        """Test ingress security analysis"""
        test_file = tmp_path / "ingress.yaml"
        manifest = '''apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: webapp-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: webapp.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: webapp
            port:
              number: 80
  # Missing TLS configuration
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect missing TLS
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-8" in controls  # Transmission security
        assert "SC-13" in controls  # Cryptographic protection

        # Check specific issues
        assert any("tls" in ann.evidence.lower() for ann in results)

    def test_detect_service_issues(self, analyzer, tmp_path):
        """Test service security analysis"""
        test_file = tmp_path / "service.yaml"
        manifest = '''apiVersion: v1
kind: Service
metadata:
  name: webapp-nodeport
spec:
  type: NodePort
  selector:
    app: webapp
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
---
apiVersion: v1
kind: Service
metadata:
  name: webapp-lb
spec:
  type: LoadBalancer
  selector:
    app: webapp
  ports:
  - port: 80
    targetPort: 8080
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect service exposure issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-7" in controls  # Boundary protection
        assert "AC-4" in controls  # Information flow

        # Check specific issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("nodeport" in ev for ev in evidence_texts)
        assert any("loadbalancer" in ev for ev in evidence_texts)

    def test_detect_statefulset_security(self, analyzer, tmp_path):
        """Test StatefulSet security analysis"""
        test_file = tmp_path / "statefulset.yaml"
        manifest = '''apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: database
spec:
  serviceName: database
  replicas: 3
  selector:
    matchLabels:
      app: database
  template:
    metadata:
      labels:
        app: database
    spec:
      serviceAccountName: database-sa
      securityContext:
        runAsNonRoot: true
        runAsUser: 999
        fsGroup: 999
      containers:
      - name: postgres
        image: postgres:14.5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false  # PostgreSQL needs writable filesystem
          capabilities:
            drop:
            - ALL
            add:
            - CHOWN
            - SETUID
            - SETGID
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        - name: POSTGRES_USER
          value: appuser
        volumeMounts:
        - name: data
          mountPath: /var/lib/postgresql/data
        resources:
          limits:
            memory: "1Gi"
            cpu: "1000m"
          requests:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - appuser
          periodSeconds: 30
  volumeClaimTemplates:
  - metadata:
      name: data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 10Gi
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should recognize secure configuration with some necessary exceptions
        evidence_texts = [ann.evidence.lower() for ann in results]

        # Should see good practices
        assert any("non-root" in ev and "good" in ev for ev in evidence_texts)

        # Writable filesystem is noted but understood for database
        writable_issues = [ann for ann in results if "writable" in ann.evidence.lower()]
        assert len(writable_issues) > 0  # Should note it
        assert all(ann.confidence <= 0.60 for ann in writable_issues)  # But low confidence

    def test_detect_cronjob_security(self, analyzer, tmp_path):
        """Test CronJob security analysis"""
        test_file = tmp_path / "cronjob.yaml"
        manifest = '''apiVersion: batch/v1
kind: CronJob
metadata:
  name: backup-job
spec:
  schedule: "0 2 * * *"
  jobTemplate:
    spec:
      template:
        spec:
          restartPolicy: OnFailure
          containers:
          - name: backup
            image: backup-tool
            command: ["/bin/sh"]
            args: ["-c", "backup.sh"]
            volumeMounts:
            - name: host-root
              mountPath: /host
          volumes:
          - name: host-root
            hostPath:
              path: /
              type: Directory
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect dangerous volume mount
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Excessive access
        assert "SC-7" in controls  # Boundary violation

        # Check specific issues
        assert any("host path" in ann.evidence.lower() and "/" in ann.evidence for ann in results)

    def test_detect_secret_configuration(self, analyzer, tmp_path):
        """Test Secret resource analysis"""
        test_file = tmp_path / "secret.yaml"
        manifest = '''apiVersion: v1
kind: Secret
metadata:
  name: app-secret
type: Opaque
data:
  username: YWRtaW4=
  password: MWYyZDFlMmU2N2Rm
---
apiVersion: v1
kind: Secret
metadata:
  name: tls-secret
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTi...
  tls.key: LS0tLS1CRUdJTi...
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect secret configuration
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-5" in controls  # Authenticator management

        # Check specific issues
        assert any("opaque" in ann.evidence.lower() for ann in results)
        assert any("encrypted at rest" in ann.evidence.lower() for ann in results)

    def test_detect_daemonset_security(self, analyzer, tmp_path):
        """Test DaemonSet security analysis"""
        test_file = tmp_path / "daemonset.yaml"
        manifest = '''apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: node-exporter
spec:
  selector:
    matchLabels:
      app: node-exporter
  template:
    metadata:
      labels:
        app: node-exporter
    spec:
      hostNetwork: true
      hostPID: true
      containers:
      - name: node-exporter
        image: prom/node-exporter:v1.3.1
        securityContext:
          privileged: true
        ports:
        - containerPort: 9100
          hostPort: 9100
        volumeMounts:
        - name: proc
          mountPath: /host/proc
          readOnly: true
        - name: sys
          mountPath: /host/sys
          readOnly: true
      volumes:
      - name: proc
        hostPath:
          path: /proc
      - name: sys
        hostPath:
          path: /sys
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple host-level access issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Privileged access
        assert "SC-7" in controls  # Host namespaces

        # Should detect all major issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("privileged" in ev for ev in evidence_texts)
        assert any("host network" in ev for ev in evidence_texts)
        assert any("host pid" in ev for ev in evidence_texts)

    def test_multi_document_yaml(self, analyzer, tmp_path):
        """Test multi-document YAML file analysis"""
        test_file = tmp_path / "multi-resource.yaml"
        manifest = '''---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: app-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: app-rolebinding
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: app-role
subjects:
- kind: ServiceAccount
  name: app-sa
  namespace: default
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 1
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      serviceAccountName: app-sa
      containers:
      - name: app
        image: myapp:v1.0.0
        securityContext:
          runAsNonRoot: true
          runAsUser: 1000
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should analyze all documents
        assert len(results) > 0

        # Should recognize good RBAC setup
        [ann.evidence.lower() for ann in results]

        # Should not flag the limited RBAC as a problem
        rbac_issues = [ann for ann in results if "rbac" in ann.evidence.lower() and
                       ann.confidence > 0.80]
        assert len(rbac_issues) == 0  # Limited permissions are good

    def test_non_k8s_yaml_ignored(self, analyzer, tmp_path):
        """Test that non-Kubernetes YAML files are ignored"""
        test_file = tmp_path / "config.yaml"
        manifest = '''database:
  host: localhost
  port: 5432
  username: admin
  password: secret123

server:
  port: 8080
  tls:
    enabled: true
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should not analyze non-K8s files
        assert len(results) == 0

    def test_suggest_controls(self, analyzer):
        """Test control suggestions for Kubernetes manifests"""
        code = '''
        apiVersion: networking.k8s.io/v1
        kind: NetworkPolicy
        spec:
          podSelector: {}
        ---
        apiVersion: rbac.authorization.k8s.io/v1
        kind: Role
        rules:
        - apiGroups: [""]
          resources: ["secrets"]
          verbs: ["get"]
        ---
        apiVersion: v1
        kind: Pod
        spec:
          securityContext:
            runAsNonRoot: true
        ---
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        spec:
          tls:
          - secretName: tls-secret
        '''

        controls = analyzer.suggest_controls(code)

        # Should suggest appropriate controls for detected resources
        assert 'SC-7' in controls  # Network policy
        assert 'AC-3' in controls  # RBAC
        assert 'AC-6' in controls  # Security context, RBAC
        assert 'IA-5' in controls  # Secrets
        assert 'SC-8' in controls  # TLS/Ingress

    @pytest.mark.asyncio
    async def test_analyze_project(self, analyzer, tmp_path):
        """Test project-wide analysis"""
        # Create Kubernetes project structure
        deployment_file = tmp_path / "deployment.yaml"
        deployment_file.write_text("""
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: web-app
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: web-app
          template:
            metadata:
              labels:
                app: web-app
            spec:
              containers:
              - name: web
                image: nginx:1.21
                ports:
                - containerPort: 80
        """)

        service_file = tmp_path / "service.yaml"
        service_file.write_text("""
        apiVersion: v1
        kind: Service
        metadata:
          name: web-service
        spec:
          selector:
            app: web-app
          ports:
          - port: 80
            targetPort: 80
          type: ClusterIP
        """)

        # Non-K8s file (should be ignored)
        config_file = tmp_path / "app-config.yaml"
        config_file.write_text("""
        database:
          host: localhost
          port: 5432
        """)

        # Run project analysis
        results = await analyzer.analyze_project(tmp_path)

        # Should analyze Kubernetes project
        assert 'summary' in results
        assert 'files' in results
        assert 'controls' in results

        # Should have resource counts
        assert 'k8s_resources' in results['summary']
        resource_counts = results['summary']['k8s_resources']
        assert isinstance(resource_counts, dict)

    def test_is_kubernetes_manifest(self, analyzer):
        """Test Kubernetes manifest detection"""
        # Valid K8s manifest
        k8s_content = '''
        apiVersion: v1
        kind: Pod
        metadata:
          name: test
        '''
        assert analyzer._is_kubernetes_manifest(k8s_content) is True

        # Invalid - missing apiVersion
        invalid_content = '''
        kind: Pod
        metadata:
          name: test
        '''
        assert analyzer._is_kubernetes_manifest(invalid_content) is False

        # Invalid - unknown kind
        unknown_kind = '''
        apiVersion: v1
        kind: UnknownResource
        metadata:
          name: test
        '''
        assert analyzer._is_kubernetes_manifest(unknown_kind) is False

        # Valid - different API version
        valid_apps = '''
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: test
        '''
        assert analyzer._is_kubernetes_manifest(valid_apps) is True

    def test_count_resources_function(self, analyzer, tmp_path):
        """Test resource counting functionality"""
        # Create multiple K8s files
        multi_resource = tmp_path / "multi.yaml"
        multi_resource.write_text('''
        ---
        apiVersion: v1
        kind: Pod
        metadata:
          name: pod1
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: svc1
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: deploy1
        ''')

        single_resource = tmp_path / "single.yaml"
        single_resource.write_text('''
        apiVersion: v1
        kind: ConfigMap
        metadata:
          name: config1
        ''')

        # Non-K8s file
        non_k8s = tmp_path / "config.yaml"
        non_k8s.write_text('database: localhost')

        # Count resources
        resource_counts = analyzer._count_resources(tmp_path)

        # Should return a dictionary with resource counts
        assert isinstance(resource_counts, dict)
        assert len(resource_counts) > 0
        
        # Should count at least some resources
        total_resources = sum(resource_counts.values())
        assert total_resources >= 1

    def test_get_line_number_function(self, analyzer, tmp_path):
        """Test line number estimation for YAML documents"""
        fake_path = tmp_path / "test.yaml"

        # Test line number calculation
        assert analyzer._get_line_number(fake_path, 0) == 1
        assert analyzer._get_line_number(fake_path, 1) == 11
        assert analyzer._get_line_number(fake_path, 2) == 21

    def test_dangerous_capabilities(self, analyzer, tmp_path):
        """Test detection of dangerous Linux capabilities"""
        test_file = tmp_path / "dangerous-caps.yaml"
        manifest = '''apiVersion: v1
kind: Pod
metadata:
  name: dangerous-pod
spec:
  containers:
  - name: app
    image: alpine:3.14
    securityContext:
      capabilities:
        add:
        - SYS_ADMIN
        - NET_ADMIN
        - SYS_PTRACE
        - SYS_MODULE
        - SETUID
        drop:
        - ALL
'''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect dangerous capabilities
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        # Should detect capability-related controls
        assert len(controls) > 0
        assert any(control.startswith("AC-") for control in controls) or "SC-7" in controls

        # Should identify dangerous capabilities (if any found)
        evidence_texts = [ann.evidence.lower() for ann in results]
        if evidence_texts:
            dangerous_caps = ['sys_admin', 'net_admin', 'sys_ptrace', 'sys_module', 'capabilities']
            assert any(any(cap in ev for cap in dangerous_caps) for ev in evidence_texts)

    def test_hostpath_volume_analysis(self, analyzer, tmp_path):
        """Test hostPath volume security analysis"""
        test_file = tmp_path / "hostpath-volumes.yaml"
        manifest = '''apiVersion: v1
        kind: Pod
        metadata:
          name: hostpath-pod
        spec:
          containers:
          - name: app
            image: alpine:3.14
            volumeMounts:
            - name: host-root
              mountPath: /host
            - name: host-etc
              mountPath: /host-etc
            - name: host-var
              mountPath: /host-var
            - name: safe-tmp
              mountPath: /tmp-mount
          volumes:
          - name: host-root
            hostPath:
              path: /
              type: Directory
          - name: host-etc
            hostPath:
              path: /etc
              type: Directory
          - name: host-var
            hostPath:
              path: /var/log
              type: Directory
          - name: safe-tmp
            hostPath:
              path: /tmp/safe
              type: Directory
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect dangerous hostPath mounts
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls
        assert "SC-7" in controls

        # Should identify specific dangerous paths
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("/" in ev and "sensitive" in ev for ev in evidence_texts)
        assert any("/etc" in ev for ev in evidence_texts)
        assert any("/var" in ev for ev in evidence_texts)

    def test_anonymous_rbac_binding(self, analyzer, tmp_path):
        """Test detection of anonymous user in RBAC bindings"""
        test_file = tmp_path / "anonymous-rbac.yaml"
        manifest = '''apiVersion: rbac.authorization.k8s.io/v1
        kind: ClusterRoleBinding
        metadata:
          name: anonymous-binding
        roleRef:
          apiGroup: rbac.authorization.k8s.io
          kind: ClusterRole
          name: view
        subjects:
        - kind: User
          name: system:anonymous
          apiGroup: rbac.authorization.k8s.io
        - kind: ServiceAccount
          name: default
          namespace: default
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect anonymous user binding
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-3" in controls
        assert "IA-2" in controls

        # Should identify anonymous user
        assert any("anonymous" in ann.evidence.lower() for ann in results)

    def test_network_policy_permissive_rules(self, analyzer, tmp_path):
        """Test detection of overly permissive network policies"""
        test_file = tmp_path / "permissive-netpol.yaml"
        manifest = '''apiVersion: networking.k8s.io/v1
        kind: NetworkPolicy
        metadata:
          name: allow-all-ingress
        spec:
          podSelector: {}
          policyTypes:
          - Ingress
          # Empty ingress rules = allow all
        ---
        apiVersion: networking.k8s.io/v1
        kind: NetworkPolicy
        metadata:
          name: allow-all-egress
        spec:
          podSelector: {}
          policyTypes:
          - Egress
          # Empty egress rules = allow all
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect permissive policies
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "SC-7" in controls

        # Should identify both ingress and egress issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("ingress" in ev and "all" in ev for ev in evidence_texts)
        assert any("egress" in ev and "all" in ev for ev in evidence_texts)

    def test_secure_ingress_configuration(self, analyzer, tmp_path):
        """Test secure ingress configuration analysis"""
        test_file = tmp_path / "secure-ingress.yaml"
        manifest = '''apiVersion: networking.k8s.io/v1
        kind: Ingress
        metadata:
          name: secure-ingress
          annotations:
            nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
            nginx.ingress.kubernetes.io/ssl-protocols: "TLSv1.2 TLSv1.3"
            nginx.ingress.kubernetes.io/ssl-ciphers: "ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512"
            nginx.ingress.kubernetes.io/backend-protocol: "HTTPS"
        spec:
          tls:
          - hosts:
            - secure.example.com
            secretName: tls-secret
          rules:
          - host: secure.example.com
            http:
              paths:
              - path: /
                pathType: Prefix
                backend:
                  service:
                    name: web-service
                    port:
                      number: 443
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should not have TLS-related issues due to proper configuration
        tls_issues = [ann for ann in results if "tls" in ann.evidence.lower() and
                     ann.confidence > 0.8]
        assert len(tls_issues) == 0

        # Should not have missing security annotation issues
        security_annotation_issues = [ann for ann in results if "annotation" in ann.evidence.lower()]
        assert len(security_annotation_issues) == 0

    def test_job_security_analysis(self, analyzer, tmp_path):
        """Test Job resource security analysis"""
        test_file = tmp_path / "job.yaml"
        manifest = '''apiVersion: batch/v1
        kind: Job
        metadata:
          name: data-migration
        spec:
          template:
            spec:
              restartPolicy: Never
              securityContext:
                runAsUser: 0
                runAsGroup: 0
              containers:
              - name: migration
                image: migrate-tool:latest
                securityContext:
                  privileged: true
                  allowPrivilegeEscalation: true
                command: ["migrate.sh"]
                env:
                - name: DB_PASSWORD
                  value: "admin123"
                volumeMounts:
                - name: host-data
                  mountPath: /data
              volumes:
              - name: host-data
                hostPath:
                  path: /var/lib/mysql
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Root user, privileged mode
        assert "IA-5" in controls  # Hardcoded password
        assert "CM-2" in controls  # Latest tag
        assert "SC-7" in controls  # Host path volume

        # Should identify specific issues
        evidence_texts = [ann.evidence.lower() for ann in results]
        assert any("root" in ev or "uid 0" in ev for ev in evidence_texts)
        assert any("privileged" in ev for ev in evidence_texts)
        assert any("password" in ev or "secret" in ev for ev in evidence_texts)
        assert any("latest" in ev for ev in evidence_texts)

    def test_init_container_analysis(self, analyzer, tmp_path):
        """Test init container security analysis"""
        test_file = tmp_path / "init-containers.yaml"
        manifest = '''apiVersion: v1
        kind: Pod
        metadata:
          name: app-with-init
        spec:
          initContainers:
          - name: init-db
            image: busybox:latest
            securityContext:
              runAsUser: 0
              privileged: true
            command: ['sh', '-c', 'echo Setting up database']
          - name: init-files
            image: alpine:3.14
            command: ['sh', '-c', 'cp /config/* /shared/']
            volumeMounts:
            - name: config
              mountPath: /config
            - name: shared
              mountPath: /shared
          containers:
          - name: app
            image: myapp:v1.0.0
            securityContext:
              runAsNonRoot: true
              runAsUser: 1000
            volumeMounts:
            - name: shared
              mountPath: /app/config
          volumes:
          - name: config
            configMap:
              name: app-config
          - name: shared
            emptyDir: {}
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect init container security issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls  # Root user, privileged mode
        assert "CM-2" in controls  # Latest tag

        # Should identify init container issues specifically
        evidence_texts = [ann.evidence.lower() for ann in results]
        init_issues = [ev for ev in evidence_texts if "init container" in ev]
        assert len(init_issues) >= 2  # Root user and privileged mode

        # Should also recognize the secure main container
        assert any("non-root" in ev and "good" in ev for ev in evidence_texts)

    def test_analyze_config_file_kubeconfig(self, analyzer, tmp_path):
        """Test kubeconfig file analysis"""
        kubeconfig_file = tmp_path / "config"
        kubeconfig_content = '''
        apiVersion: v1
        kind: Config
        clusters:
        - cluster:
            certificate-authority-data: LS0tLS1CRUdJTi...
            server: https://kubernetes.example.com
          name: production
        contexts:
        - context:
            cluster: production
            user: admin
          name: production
        current-context: production
        users:
        - name: admin
          user:
            client-certificate-data: LS0tLS1CRUdJTi...
            client-key-data: LS0tLS1CRUdJTi...
        '''
        kubeconfig_file.write_text(kubeconfig_content)

        results = analyzer._analyze_config_file(kubeconfig_file)

        # Should detect embedded credentials
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-5" in controls
        assert "SC-28" in controls

        # Should identify embedded credentials issue
        assert any("embedded credentials" in ann.evidence.lower() for ann in results)

    def test_complex_multi_resource_analysis(self, analyzer, tmp_path):
        """Test complex multi-resource file with various security issues"""
        test_file = tmp_path / "complex-manifest.yaml"
        manifest = '''---
        apiVersion: v1
        kind: Namespace
        metadata:
          name: production
        ---
        apiVersion: v1
        kind: ServiceAccount
        metadata:
          name: app-sa
          namespace: production
        ---
        apiVersion: rbac.authorization.k8s.io/v1
        kind: ClusterRole
        metadata:
          name: dangerous-role
        rules:
        - apiGroups: ["*"]
          resources: ["*"]
          verbs: ["*"]
        ---
        apiVersion: rbac.authorization.k8s.io/v1
        kind: ClusterRoleBinding
        metadata:
          name: dangerous-binding
        roleRef:
          apiGroup: rbac.authorization.k8s.io
          kind: ClusterRole
          name: cluster-admin
        subjects:
        - kind: ServiceAccount
          name: app-sa
          namespace: production
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: insecure-app
          namespace: production
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: insecure-app
          template:
            metadata:
              labels:
                app: insecure-app
            spec:
              serviceAccountName: app-sa
              hostNetwork: true
              hostPID: true
              containers:
              - name: app
                image: myapp:latest
                securityContext:
                  privileged: true
                  runAsUser: 0
                  allowPrivilegeEscalation: true
                  capabilities:
                    add:
                    - SYS_ADMIN
                    - NET_ADMIN
                env:
                - name: DB_PASSWORD
                  value: "supersecret123"
                - name: API_KEY
                  value: "sk-1234567890abcdef"
                volumeMounts:
                - name: host-root
                  mountPath: /host
              volumes:
              - name: host-root
                hostPath:
                  path: /
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: insecure-service
          namespace: production
        spec:
          type: NodePort
          selector:
            app: insecure-app
          ports:
          - port: 80
            targetPort: 8080
            nodePort: 30080
        ---
        apiVersion: networking.k8s.io/v1
        kind: Ingress
        metadata:
          name: insecure-ingress
          namespace: production
        spec:
          rules:
          - host: app.example.com
            http:
              paths:
              - path: /
                pathType: Prefix
                backend:
                  service:
                    name: insecure-service
                    port:
                      number: 80
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect numerous security issues
        assert len(results) >= 10

        # Should detect various control families
        all_controls = set()
        for ann in results:
            all_controls.update(ann.control_ids)

        # Access control violations
        assert "AC-6" in all_controls  # Privileged mode, root user, excessive RBAC
        assert "AC-3" in all_controls  # RBAC violations

        # Network security violations
        assert "SC-7" in all_controls  # Host network, NodePort, hostPath

        # Cryptographic/transmission security
        assert "SC-8" in all_controls or "SC-13" in all_controls  # Missing TLS

        # Authentication/secrets management
        assert "IA-5" in all_controls  # Hardcoded secrets

        # Configuration management
        assert "CM-2" in all_controls  # Latest tag
        assert "CM-6" in all_controls  # Missing security contexts

    def test_error_handling(self, analyzer, tmp_path):
        """Test error handling for malformed YAML files"""
        test_file = tmp_path / "malformed.yaml"
        test_file.write_text("invalid: yaml: content: [unclosed")

        # Should not crash on malformed YAML
        results = analyzer.analyze_file(test_file)
        assert isinstance(results, list)

    def test_file_not_found(self, analyzer, tmp_path):
        """Test handling of non-existent files"""
        fake_file = tmp_path / "does_not_exist.yaml"
        results = analyzer.analyze_file(fake_file)
        assert results == []

    def test_empty_yaml_file(self, analyzer, tmp_path):
        """Test handling of empty YAML files"""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        results = analyzer.analyze_file(empty_file)
        assert results == []

    def test_yaml_with_only_comments(self, analyzer, tmp_path):
        """Test handling of YAML files with only comments"""
        comments_file = tmp_path / "comments.yaml"
        comments_file.write_text("""
        # This is a comment
        # Another comment

        # apiVersion: v1 (commented out)
        # kind: Pod (commented out)
        """)
        results = analyzer.analyze_file(comments_file)
        assert results == []

    def test_file_extensions_matching(self, analyzer):
        """Test that analyzer recognizes correct file extensions"""
        extensions = analyzer.file_extensions

        # Should match YAML file extensions
        assert '.yaml' in extensions
        assert '.yml' in extensions

    def test_k8s_kinds_recognition(self, analyzer):
        """Test recognition of Kubernetes resource kinds"""
        kinds = analyzer.k8s_kinds

        # Should include major Kubernetes resource types
        workload_kinds = {'Pod', 'Deployment', 'StatefulSet', 'DaemonSet', 'Job', 'CronJob'}
        assert workload_kinds.issubset(kinds)

        network_kinds = {'Service', 'Ingress', 'NetworkPolicy'}
        assert network_kinds.issubset(kinds)

        rbac_kinds = {'Role', 'ClusterRole', 'RoleBinding', 'ClusterRoleBinding'}
        assert rbac_kinds.issubset(kinds)

        storage_kinds = {'Secret', 'ConfigMap'}
        assert storage_kinds.issubset(kinds)

    def test_raw_pattern_analysis(self, analyzer, tmp_path):
        """Test raw pattern analysis functionality"""
        test_file = tmp_path / "pattern-test.yaml"
        manifest = '''apiVersion: rbac.authorization.k8s.io/v1
        kind: Role
        metadata:
          name: test-role
        rules:
        - apiGroups: ["*"]
          resources: ["*"]
          verbs: ["*"]
        ---
        apiVersion: v1
        kind: Pod
        metadata:
          name: test-pod
        spec:
          containers:
          - name: app
            image: nginx:latest
            env:
            - name: SECRET_KEY
              value: "hardcoded-secret"
        '''
        test_file.write_text(manifest)

        results = analyzer.analyze_file(test_file)

        # Should detect patterns in addition to structured analysis
        # The raw pattern analysis should catch RBAC and container patterns
        pattern_results = [ann for ann in results if ann.component == "k8s-pattern"]
        assert len(pattern_results) >= 1

        # Should detect RBAC wildcard pattern
        rbac_patterns = [ann for ann in pattern_results if
                        any(control in ["AC-6", "AC-3"] for control in ann.control_ids)]
        assert len(rbac_patterns) >= 1
