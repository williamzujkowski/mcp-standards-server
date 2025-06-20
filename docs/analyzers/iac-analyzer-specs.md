# Infrastructure as Code Analyzer Specifications

This document provides detailed implementation specifications for IaC and container analyzers.

## Implementation Status

### ✅ Completed Analyzers
- **Terraform Analyzer**: Full HCL parsing with multi-provider support (AWS, Azure, GCP)
- **Dockerfile Analyzer**: Comprehensive security best practices and anti-pattern detection
- **Kubernetes Analyzer**: Complete manifest validation with RBAC and security context analysis

### 🚧 Future Analyzers
- CloudFormation Analyzer
- Helm Chart Analyzer
- Ansible Analyzer
- Docker Compose Analyzer

## Terraform Analyzer Detailed Specification (✅ IMPLEMENTED)

### File Patterns
- `*.tf` - Terraform configuration files
- `*.tfvars` - Variable definition files
- `terraform.tfstate` - State files (sensitive!)
- `*.tf.json` - JSON format Terraform files

### Security Patterns to Detect

#### AWS Provider Patterns
```hcl
# INSECURE: Open security group
resource "aws_security_group" "insecure" {
  ingress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]  # SI-4, SC-7: Unrestricted access
  }
}

# SECURE: Restricted security group
resource "aws_security_group" "secure" {
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]  # SC-7: Network segmentation
  }
}

# INSECURE: Unencrypted S3 bucket
resource "aws_s3_bucket" "insecure" {
  bucket = "my-bucket"
  # Missing encryption configuration - SC-28
}

# SECURE: Encrypted S3 bucket
resource "aws_s3_bucket" "secure" {
  bucket = "my-secure-bucket"
}

resource "aws_s3_bucket_server_side_encryption_configuration" "secure" {
  bucket = aws_s3_bucket.secure.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"  # SC-28: Protection at rest
    }
  }
}

# INSECURE: Overly permissive IAM
resource "aws_iam_policy" "insecure" {
  policy = jsonencode({
    Statement = [{
      Effect   = "Allow"
      Action   = "*"
      Resource = "*"  # AC-6: Excessive privileges
    }]
  })
}
```

#### Azure Provider Patterns
```hcl
# INSECURE: Public storage account
resource "azurerm_storage_account" "insecure" {
  allow_blob_public_access = true  # AC-3: Public access
}

# SECURE: Private storage with encryption
resource "azurerm_storage_account" "secure" {
  allow_blob_public_access = false
  enable_https_traffic_only = true  # SC-8: Transmission protection
  
  blob_properties {
    delete_retention_policy {
      days = 30  # CP-9: Backup retention
    }
  }
}

# Key Vault with access policies
resource "azurerm_key_vault" "secure" {
  sku_name = "standard"
  
  network_acls {
    default_action = "Deny"  # AC-4: Information flow
    ip_rules       = ["10.0.0.0/8"]
  }
}
```

#### GCP Provider Patterns
```hcl
# INSECURE: Public GCS bucket
resource "google_storage_bucket" "insecure" {
  name          = "public-bucket"
  location      = "US"
  force_destroy = true
  
  # Missing IAM binding restrictions
}

# SECURE: Private GCS bucket with encryption
resource "google_storage_bucket" "secure" {
  name     = "secure-bucket"
  location = "US"
  
  encryption {
    default_kms_key_name = google_kms_crypto_key.key.id  # SC-28
  }
  
  lifecycle_rule {
    action {
      type = "Delete"
    }
    condition {
      age = 30  # Data retention
    }
  }
  
  versioning {
    enabled = true  # CP-9: Backup
  }
}
```

### NIST Control Mappings

| Pattern | NIST Controls | Evidence |
|---------|--------------|----------|
| Open security groups | SC-7, SI-4 | Unrestricted network access |
| Unencrypted storage | SC-28 | Missing encryption at rest |
| Public access | AC-3, AC-4 | Unrestricted information flow |
| No backup policy | CP-9, CP-10 | Missing recovery capability |
| Weak IAM policies | AC-6, AC-3 | Excessive privileges |
| Missing monitoring | AU-2, AU-12 | No audit capability |
| No MFA requirement | IA-2(1) | Weak authentication |
| Hardcoded secrets | IA-5 | Credential exposure |

## Dockerfile Analyzer Detailed Specification (✅ IMPLEMENTED)

### Security Anti-Patterns

```dockerfile
# INSECURE: Running as root
FROM ubuntu:latest
RUN apt-get update && apt-get install -y curl
# No USER instruction - runs as root (AC-6)

# INSECURE: Hardcoded secrets
ENV API_KEY="sk-1234567890abcdef"  # IA-5: Exposed credential
ENV DATABASE_PASSWORD="admin123"    # IA-5: Weak password

# INSECURE: Using latest tag
FROM node:latest  # CM-2: Unpinned version

# INSECURE: Exposed sensitive port
EXPOSE 22  # IA-2: SSH exposed

# INSECURE: No health check
# Missing HEALTHCHECK instruction (AU-12)
```

### Secure Patterns

```dockerfile
# SECURE: Multi-stage build with non-root user
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

FROM node:18-alpine
RUN apk add --no-cache dumb-init
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
USER nodejs  # AC-6: Least privilege

WORKDIR /app
COPY --from=builder --chown=nodejs:nodejs /app/node_modules ./node_modules
COPY --chown=nodejs:nodejs . .

# SECURE: Health check defined
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js  # AU-12: Monitoring

# SECURE: Specific port
EXPOSE 3000

# SECURE: Security options
LABEL security.scan="true"
LABEL security.updates="auto"

ENTRYPOINT ["dumb-init", "--"]
CMD ["node", "server.js"]
```

### Container Security Checks

| Check | NIST Control | Severity |
|-------|--------------|----------|
| Root user | AC-6 | High |
| Latest tag | CM-2 | Medium |
| Exposed secrets | IA-5 | Critical |
| No health check | AU-12 | Medium |
| Vulnerable base | SI-2 | High |
| Missing updates | SI-2 | High |
| No resource limits | SC-5 | Medium |
| Privileged mode | AC-6 | Critical |

## Kubernetes Analyzer Detailed Specification (✅ IMPLEMENTED)

### Pod Security Patterns

```yaml
# INSECURE: Privileged pod with host access
apiVersion: v1
kind: Pod
metadata:
  name: insecure-pod
spec:
  hostNetwork: true      # SC-7: Host network access
  hostPID: true         # AC-6: Host PID namespace
  containers:
  - name: app
    image: myapp:latest  # CM-2: Unpinned image
    securityContext:
      privileged: true   # AC-6: Privileged container
      runAsUser: 0      # AC-6: Running as root
    resources: {}       # SC-5: No resource limits

# SECURE: Hardened pod specification
apiVersion: v1
kind: Pod
metadata:
  name: secure-pod
  labels:
    app: myapp
    security: hardened
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: app
    image: myapp:v1.2.3  # CM-2: Pinned version
    imagePullPolicy: Always
    securityContext:
      allowPrivilegeEscalation: false
      readOnlyRootFilesystem: true
      capabilities:
        drop:
        - ALL  # AC-6: Least privilege
    resources:
      limits:
        memory: "512Mi"
        cpu: "500m"
      requests:
        memory: "256Mi"
        cpu: "250m"  # SC-5: Resource limits
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      periodSeconds: 10  # AU-12: Health monitoring
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
```

### RBAC Security Patterns

```yaml
# INSECURE: Overly permissive RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: insecure-role
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]  # AC-6: Excessive permissions

# SECURE: Least privilege RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: secure-role
  namespace: myapp
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]  # AC-6: Minimal permissions
- apiGroups: [""]
  resources: ["configmaps"]
  resourceNames: ["myapp-config"]
  verbs: ["get"]  # AC-3: Specific resource access
```

### Network Policy Patterns

```yaml
# SECURE: Network segmentation
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: myapp-netpol
spec:
  podSelector:
    matchLabels:
      app: myapp
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: frontend  # SC-7: Network segmentation
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
      port: 5432  # SC-7: Controlled egress
```

## Implementation Architecture

### Base IaC Analyzer Class

```python
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
from ..base import BaseAnalyzer, CodeAnnotation

class BaseIaCAnalyzer(BaseAnalyzer):
    """Base class for Infrastructure as Code analyzers"""
    
    def __init__(self):
        super().__init__()
        self.iac_patterns = self._load_iac_patterns()
        self.compliance_rules = self._load_compliance_rules()
    
    @abstractmethod
    def parse_configuration(self, content: str) -> Dict[str, Any]:
        """Parse IaC configuration into analyzable structure"""
        pass
    
    @abstractmethod
    def analyze_security_groups(self, config: Dict[str, Any]) -> List[CodeAnnotation]:
        """Analyze network security configurations"""
        pass
    
    @abstractmethod
    def analyze_encryption(self, config: Dict[str, Any]) -> List[CodeAnnotation]:
        """Analyze encryption settings"""
        pass
    
    @abstractmethod
    def analyze_iam(self, config: Dict[str, Any]) -> List[CodeAnnotation]:
        """Analyze identity and access management"""
        pass
    
    def analyze_compliance(self, config: Dict[str, Any]) -> List[CodeAnnotation]:
        """Check compliance with frameworks (CIS, PCI-DSS, HIPAA)"""
        annotations = []
        
        for rule in self.compliance_rules:
            if self._check_rule(config, rule):
                annotations.append(self._create_compliance_annotation(rule))
        
        return annotations
    
    def suggest_remediation(self, issue: CodeAnnotation) -> Dict[str, Any]:
        """Suggest remediation for detected issues"""
        return {
            "issue": issue.evidence,
            "severity": issue.confidence,
            "remediation": self._get_remediation(issue.control_ids),
            "example": self._get_secure_example(issue.control_ids)
        }
```

### Integration with Existing System

```python
# In src/cli/main.py
@app.command()
def scan_iac(
    path: Path = typer.Argument(..., help="Path to IaC files"),
    format: str = typer.Option("terraform", help="IaC format (terraform, cloudformation, k8s)"),
    output: str = typer.Option("json", help="Output format"),
    compliance: str = typer.Option(None, help="Compliance framework (cis, pci, hipaa)")
):
    """Scan Infrastructure as Code for security issues"""
    
    # Select appropriate analyzer
    analyzer = get_iac_analyzer(format)
    
    # Scan files
    results = analyzer.scan_directory(path)
    
    # Check compliance if specified
    if compliance:
        compliance_results = analyzer.check_compliance(results, compliance)
        results.extend(compliance_results)
    
    # Output results
    output_results(results, output)
```

## Testing Requirements

### Unit Test Structure

```python
# tests/unit/analyzers/test_terraform_analyzer.py
class TestTerraformAnalyzer:
    def test_detect_open_security_group(self):
        """Test detection of unrestricted security groups"""
        
    def test_detect_unencrypted_storage(self):
        """Test detection of unencrypted storage resources"""
        
    def test_detect_privileged_iam(self):
        """Test detection of overly permissive IAM policies"""
        
    def test_detect_hardcoded_secrets(self):
        """Test detection of hardcoded credentials"""
```

### Integration Test Scenarios

1. **Multi-provider Terraform**
2. **Complex Kubernetes deployments**
3. **Multi-stage Docker builds**
4. **Nested CloudFormation stacks**
5. **Helm charts with dependencies**

## Performance Considerations

1. **Lazy parsing** - Parse only when needed
2. **Parallel analysis** - Analyze multiple files concurrently
3. **Caching** - Cache parsed configurations
4. **Incremental analysis** - Analyze only changed files
5. **Resource limits** - Set memory/CPU limits for large files