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
