"""
Tests for Dockerfile analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Dockerfile analyzer testing
"""

import pytest
from pathlib import Path

from src.analyzers.dockerfile_analyzer import DockerfileAnalyzer
from src.analyzers.base import CodeAnnotation


class TestDockerfileAnalyzer:
    """Test Dockerfile analysis capabilities"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return DockerfileAnalyzer()
    
    def test_detect_running_as_root(self, analyzer, tmp_path):
        """Test detection of containers running as root"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

COPY app.py /app/
WORKDIR /app

# No USER instruction - runs as root
CMD ["python", "app.py"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect root user issue
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AC-6" in controls  # Least privilege
        
        # Should identify missing USER instruction
        assert any("root" in ann.evidence.lower() for ann in results)
        assert any("USER instruction" in ann.evidence for ann in results)
    
    def test_detect_latest_tag(self, analyzer, tmp_path):
        """Test detection of latest tag usage"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM node:latest

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .

USER node
EXPOSE 3000
CMD ["npm", "start"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect latest tag
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CM-2" in controls  # Configuration management
        
        # Should identify latest tag issue
        assert any("latest" in ann.evidence.lower() for ann in results)
        assert any("unpinned" in ann.evidence.lower() for ann in results)
    
    def test_detect_hardcoded_secrets(self, analyzer, tmp_path):
        """Test detection of hardcoded secrets in Dockerfile"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM python:3.9-slim

ENV DATABASE_PASSWORD=admin123
ENV API_KEY=sk-1234567890abcdef
ENV SECRET_TOKEN=mysecrettoken

ARG AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
ARG AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

USER 1000
CMD ["python", "app.py"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect hardcoded secrets
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-5" in controls  # Authenticator management
        
        # Should find multiple secrets
        secret_findings = [ann for ann in results if "IA-5" in ann.control_ids]
        assert len(secret_findings) >= 3  # ENV and ARG secrets
    
    def test_detect_exposed_ssh(self, analyzer, tmp_path):
        """Test detection of exposed SSH port"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    openssh-server \
    && mkdir /var/run/sshd

# Configure SSH
RUN echo 'root:password' | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22
EXPOSE 80

CMD ["/usr/sbin/sshd", "-D"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect SSH exposure
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-2" in controls  # Identification and authentication
        assert "SC-7" in controls  # Boundary protection
        
        # Should identify SSH port exposure
        assert any("ssh" in ann.evidence.lower() and "22" in ann.evidence for ann in results)
    
    def test_detect_curl_pipe_bash(self, analyzer, tmp_path):
        """Test detection of curl | sh pattern"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM alpine:3.14

RUN apk add --no-cache curl

# Dangerous pattern
RUN curl -L https://get.docker.com | sh
RUN wget -O - https://example.com/install.sh | sh

# Also dangerous
RUN curl https://raw.githubusercontent.com/example/repo/master/install.sh | bash

WORKDIR /app
USER nobody
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect pipe to shell
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SI-2" in controls  # Flaw remediation
        assert "CM-6" in controls  # Configuration settings
        
        # Should find multiple instances
        pipe_findings = [ann for ann in results if "pipe" in ann.evidence.lower() or "shell" in ann.evidence.lower()]
        assert len(pipe_findings) >= 2
    
    def test_detect_missing_healthcheck(self, analyzer, tmp_path):
        """Test detection of missing HEALTHCHECK"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM node:16-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .

USER node
EXPOSE 3000

# No HEALTHCHECK instruction
CMD ["node", "server.js"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect missing healthcheck
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AU-12" in controls  # Audit generation
        
        # Should identify missing healthcheck
        assert any("healthcheck" in ann.evidence.lower() for ann in results)
    
    def test_secure_multistage_build(self, analyzer, tmp_path):
        """Test analysis of secure multi-stage build"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''# Build stage
FROM golang:1.17-alpine AS builder
RUN apk add --no-cache git ca-certificates
WORKDIR /build
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .

# Final stage
FROM scratch
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=builder /build/app /app
USER 1000
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["/app", "health"]
ENTRYPOINT ["/app"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should recognize good practices but still note scratch base
        evidence_texts = [ann.evidence.lower() for ann in results]
        
        # Should see multi-stage build suggestion is already implemented
        multistage_issues = [ann for ann in results if "multi-stage" in ann.evidence.lower() and ann.confidence > 0.40]
        assert len(multistage_issues) == 0  # Should not suggest multi-stage since it's already used
    
    def test_detect_package_manager_cleanup(self, analyzer, tmp_path):
        """Test detection of package manager cleanup issues"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

# Bad - no cleanup
RUN apt-get update && apt-get install -y python3 python3-pip

# Good - with cleanup
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Bad - separate RUN commands
RUN yum install -y nodejs
RUN yum clean all

USER 1001
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect cleanup issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CM-6" in controls  # Configuration settings
        
        # Should find cleanup issues
        assert any("cache" in ann.evidence.lower() or "cleanup" in ann.evidence.lower() for ann in results)
    
    def test_detect_add_vs_copy(self, analyzer, tmp_path):
        """Test detection of ADD usage with URLs"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM alpine:3.14

# Bad - using ADD with URL
ADD https://github.com/example/file.tar.gz /tmp/
ADD http://example.com/script.sh /usr/local/bin/

# Good - using COPY for local files
COPY app.tar.gz /tmp/
COPY script.sh /usr/local/bin/

RUN chmod +x /usr/local/bin/script.sh

USER nobody
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect ADD with URL
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SI-2" in controls  # Flaw remediation
        assert "CM-6" in controls  # Configuration settings
        
        # Should find ADD issues
        assert any("ADD" in ann.evidence and "URL" in ann.evidence for ann in results)
    
    def test_detect_base_image_issues(self, analyzer, tmp_path):
        """Test detection of base image security issues"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:latest

# Using full OS image instead of minimal
RUN apt-get update && apt-get install -y \
    python3 \
    && rm -rf /var/lib/apt/lists/*

FROM node:8

# Using outdated Node.js version
WORKDIR /app
COPY . .
RUN npm install

USER node
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect base image issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "SI-2" in controls  # Flaw remediation
        assert "CM-2" in controls  # Configuration management
        
        # Should find outdated and full OS issues
        assert any("minimal" in ann.evidence.lower() or "distroless" in ann.evidence.lower() for ann in results)
        assert any("end-of-life" in ann.evidence.lower() or "outdated" in ann.evidence.lower() for ann in results)
    
    def test_detect_copy_ownership(self, analyzer, tmp_path):
        """Test detection of COPY without --chown"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM node:16-alpine

RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001

WORKDIR /app

# Bad - no chown
COPY package*.json ./
COPY . .

# Good - with chown
COPY --chown=nodejs:nodejs app.js ./

RUN npm ci --only=production

USER nodejs
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect ownership issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AC-6" in controls  # Least privilege
        
        # Should find COPY without chown
        assert any("chown" in ann.evidence.lower() and "ownership" in ann.evidence.lower() for ann in results)
    
    def test_detect_sensitive_files(self, analyzer, tmp_path):
        """Test detection of sensitive file copying"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM python:3.9

WORKDIR /app

# Dangerous - copying sensitive files
COPY .env /app/
COPY .git /app/.git
COPY id_rsa /root/.ssh/
COPY credentials.json /app/

# Copying AWS credentials
COPY .aws /root/.aws

RUN pip install -r requirements.txt

USER 1000
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect sensitive files
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "IA-5" in controls  # Authenticator management
        assert "SC-28" in controls  # Protection at rest
        
        # Should find multiple sensitive files
        sensitive_findings = [ann for ann in results if "sensitive" in ann.evidence.lower()]
        assert len(sensitive_findings) >= 3
    
    def test_detect_workdir_and_labels(self, analyzer, tmp_path):
        """Test detection of missing WORKDIR and labels"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM alpine:3.14

# Using root directory
COPY app.py /
RUN chmod +x /app.py

# No security labels
# No maintainer information

USER nobody
CMD ["python", "/app.py"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect missing best practices
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "CM-6" in controls  # Configuration settings
        assert "CM-2" in controls  # Baseline configuration
        
        # Should find missing elements
        assert any("workdir" in ann.evidence.lower() for ann in results)
        assert any("maintainer" in ann.evidence.lower() or "label" in ann.evidence.lower() for ann in results)
    
    def test_privileged_mode_detection(self, analyzer, tmp_path):
        """Test detection of privileged mode indicators"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM alpine:3.14

USER root

# Installing packages that might need privileged mode
RUN apk add --no-cache \
    docker \
    iptables \
    sudo

# Running as root explicitly
USER root

EXPOSE 80 443

CMD ["dockerd"]
'''
        test_file.write_text(dockerfile)
        
        results = analyzer.analyze_file(test_file)
        
        # Should detect root user
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)
        
        assert "AC-6" in controls  # Least privilege
        
        # Should find explicit root user
        assert any("USER root" in ann.evidence or "root user" in ann.evidence.lower() for ann in results)