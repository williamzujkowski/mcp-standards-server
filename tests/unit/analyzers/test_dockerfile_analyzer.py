"""
Tests for Dockerfile analyzer
@nist-controls: SA-11, CA-7
@evidence: Comprehensive Dockerfile analyzer testing
"""


import pytest

from src.analyzers.dockerfile_analyzer import DockerfileAnalyzer


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
        [ann.evidence.lower() for ann in results]

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

    def test_suggest_controls(self, analyzer):
        """Test control suggestions for Dockerfile code"""
        code = '''
        FROM alpine:3.14
        USER nobody
        COPY --chown=nobody ssl-cert.pem /app/
        ENV SECRET_KEY=mysecret
        HEALTHCHECK --interval=30s CMD curl -f http://localhost:8080/health
        RUN apk update && apk add curl
        '''
        
        controls = analyzer.suggest_controls(code)
        
        # Should suggest appropriate controls for detected patterns
        assert 'CM-2' in controls  # Configuration management
        assert 'AC-6' in controls  # Least privilege (USER)
        assert 'SC-8' in controls  # Transmission security (SSL cert)
        assert 'IA-5' in controls  # Authenticator management (SECRET)
        assert 'AU-12' in controls  # Audit generation (HEALTHCHECK)
        assert 'SI-2' in controls  # Flaw remediation (package update)

    @pytest.mark.asyncio
    async def test_analyze_project(self, analyzer, tmp_path):
        """Test project-wide analysis"""
        # Create Docker project structure
        dockerfile1 = tmp_path / "Dockerfile"
        dockerfile1.write_text("""
        FROM python:3.9-slim
        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt
        COPY . .
        USER 1000
        HEALTHCHECK --interval=30s CMD curl -f http://localhost:8000/health
        CMD ["python", "app.py"]
        """)
        
        dockerfile2 = tmp_path / "services" / "Dockerfile.web"
        dockerfile2.parent.mkdir()
        dockerfile2.write_text("""
        FROM nginx:alpine
        COPY nginx.conf /etc/nginx/nginx.conf
        RUN adduser -D -s /bin/sh nginx || true
        USER nginx
        EXPOSE 80
        """)
        
        # Non-Docker file (should be ignored)
        readme_file = tmp_path / "README.md"
        readme_file.write_text("# Docker Project")
        
        # Run project analysis
        results = await analyzer.analyze_project(tmp_path)
        
        # Should analyze Docker project
        assert 'summary' in results
        assert 'files' in results
        assert 'controls' in results
        
        # Should have image counts
        assert 'docker_images' in results['summary']
        image_counts = results['summary']['docker_images']
        assert isinstance(image_counts, dict)

    def test_extract_base_image(self, analyzer):
        """Test base image extraction functionality"""
        # Test various FROM instruction formats
        test_cases = [
            ("FROM ubuntu:20.04", "ubuntu:20.04"),
            ("FROM --platform=linux/amd64 node:16", "node:16"),
            ("FROM alpine", "alpine"),
            ("FROM gcr.io/distroless/java:11", "gcr.io/distroless/java:11"),
            ("FROM scratch", "scratch"),
        ]
        
        for from_line, expected in test_cases:
            result = analyzer._extract_base_image(from_line)
            assert result == expected

    def test_count_images_function(self, analyzer, tmp_path):
        """Test image counting functionality"""
        # Create multiple Dockerfiles
        dockerfile1 = tmp_path / "Dockerfile"
        dockerfile1.write_text('''
        FROM python:3.9
        FROM alpine:3.14
        ''')
        
        dockerfile2 = tmp_path / "backend" / "Dockerfile"
        dockerfile2.parent.mkdir()
        dockerfile2.write_text('''
        FROM node:16
        FROM nginx:alpine
        ''')
        
        dockerfile3 = tmp_path / "Dockerfile.prod"
        dockerfile3.write_text('''
        FROM python:3.9
        FROM scratch
        ''')
        
        # Count images
        image_counts = analyzer._count_images(tmp_path)
        
        # Should count different base images
        assert "python" in image_counts
        assert image_counts["python"] == 2  # Used in 2 files
        assert "alpine" in image_counts
        assert image_counts["alpine"] == 1
        assert "node" in image_counts
        assert image_counts["node"] == 1
        assert "nginx" in image_counts
        assert image_counts["nginx"] == 1
        assert "scratch" in image_counts
        assert image_counts["scratch"] == 1

    def test_dockerfile_without_tag(self, analyzer, tmp_path):
        """Test detection of base images without tags"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu
        FROM python
        FROM alpine:3.14

        WORKDIR /app
        COPY . .
        USER 1000
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect missing tags
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "CM-2" in controls

        # Should find missing tag issues
        tag_issues = [ann for ann in results if "without tag" in ann.evidence.lower()]
        assert len(tag_issues) >= 2  # ubuntu and python without tags

    def test_pre_release_base_images(self, analyzer, tmp_path):
        """Test detection of pre-release base images"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM node:18-alpha
        FROM python:3.11-rc
        FROM ubuntu:22.04-beta
        FROM redis:7.0-dev
        FROM postgres:15-snapshot

        WORKDIR /app
        USER 1000
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect pre-release versions
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "CM-2" in controls or "SI-2" in controls

        # Should find pre-release issues
        prerelease_issues = [ann for ann in results if "pre-release" in ann.evidence.lower()]
        assert len(prerelease_issues) >= 3

    def test_package_version_pinning(self, analyzer, tmp_path):
        """Test detection of package version pinning issues"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

        # Bad - no version pinning
        RUN apt-get update && apt-get install -y python3 curl

        # Good - with version pinning
        RUN apt-get update && apt-get install -y \
            python3=3.8.* \
            curl=7.68.* \
            && rm -rf /var/lib/apt/lists/*

        # Bad - no version pinning with yum
        RUN yum install -y nodejs npm

        # Good - with version pinning
        RUN apk add --no-cache \
            nodejs=16.* \
            npm=8.*

        USER 1001
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect version pinning issues
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "CM-2" in controls

        # Should find version pinning issues
        version_issues = [ann for ann in results if "version pinning" in ann.evidence.lower()]
        assert len(version_issues) >= 2

    def test_dockerignore_analysis(self, analyzer, tmp_path):
        """Test .dockerignore file analysis"""
        dockerignore_file = tmp_path / ".dockerignore"
        content = '''# Basic ignore patterns
        .git
        *.log
        node_modules
        __pycache__
        
        # Missing important patterns like .env, *.key, etc.
        '''
        dockerignore_file.write_text(content)

        results = analyzer._analyze_dockerignore(dockerignore_file)

        # Should detect missing patterns
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "CM-6" in controls or "IA-5" in controls

        # Should suggest adding missing patterns
        missing_patterns = [ann for ann in results if "consider adding" in ann.evidence.lower()]
        assert len(missing_patterns) >= 3  # .env, *.key, .aws, .ssh, etc.

    def test_sudo_usage_detection(self, analyzer, tmp_path):
        """Test detection of sudo usage in RUN commands"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

        # Using sudo indicates elevated privileges
        RUN sudo apt-get update
        RUN sudo apt-get install -y python3
        RUN sudo chmod 755 /app
        
        # Non-sudo commands (good)
        RUN apt-get update && apt-get install -y curl
        
        USER 1000
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect sudo usage
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "AC-6" in controls

        # Should find sudo usage
        sudo_issues = [ann for ann in results if "sudo" in ann.evidence.lower()]
        assert len(sudo_issues) >= 3

    def test_ssh_installation_detection(self, analyzer, tmp_path):
        """Test detection of SSH server installation"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:20.04

        # Installing SSH server
        RUN apt-get update && apt-get install -y ssh openssh-server
        RUN yum install -y openssh-server
        
        # Other installations (should not trigger)
        RUN apt-get install -y curl git
        
        EXPOSE 22
        USER root
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect SSH installation and port exposure
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "CM-6" in controls
        assert "IA-2" in controls
        assert "SC-7" in controls

        # Should find SSH-related issues
        ssh_issues = [ann for ann in results if "ssh" in ann.evidence.lower()]
        assert len(ssh_issues) >= 2

    def test_complex_dockerfile_analysis(self, analyzer, tmp_path):
        """Test analysis of complex Dockerfile with multiple issues"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM ubuntu:latest

        # Multiple security issues
        ENV DATABASE_PASSWORD=admin123
        ENV API_KEY=sk-1234567890abcdef
        
        # No cleanup, no version pinning
        RUN apt-get update && apt-get install -y \
            python3 \
            curl \
            openssh-server
            
        # Dangerous patterns
        RUN curl -L https://get.docker.com | sh
        ADD https://example.com/file.tar.gz /tmp/
        
        # Copying sensitive files
        COPY .env /app/
        COPY id_rsa /root/.ssh/
        
        # No chown
        COPY app.py /app/
        
        # Exposing SSH
        EXPOSE 22 80
        
        # No USER instruction - runs as root
        # No HEALTHCHECK
        # No WORKDIR
        
        CMD ["python3", "/app/app.py"]
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect multiple security issues
        assert len(results) >= 8

        # Should detect various control families
        all_controls = set()
        for ann in results:
            all_controls.update(ann.control_ids)

        # Configuration management
        assert "CM-2" in all_controls  # Latest tag, no version pinning
        assert "CM-6" in all_controls  # Package cleanup, best practices
        
        # Access control
        assert "AC-6" in all_controls  # Root user, file ownership
        
        # Authentication/Authorization
        assert "IA-2" in all_controls  # SSH exposure
        assert "IA-5" in all_controls  # Hardcoded secrets
        
        # System integrity
        assert "SI-2" in all_controls  # Curl pipe to shell, ADD with URL
        
        # Data protection
        assert "SC-28" in all_controls  # Sensitive files
        
        # Boundary protection
        assert "SC-7" in all_controls  # SSH port exposure
        
        # Audit generation
        assert "AU-12" in all_controls  # Missing healthcheck

    def test_secure_best_practices_dockerfile(self, analyzer, tmp_path):
        """Test analysis of secure Dockerfile following best practices"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''# Multi-stage build for security
        FROM golang:1.19-alpine AS builder
        
        LABEL maintainer="security-team@example.com"
        LABEL version="1.0.0"
        LABEL security.scan="enabled"
        LABEL build-date="2023-01-01"
        
        WORKDIR /build
        COPY go.mod go.sum ./
        RUN go mod download
        COPY . .
        RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .
        
        # Final minimal image
        FROM gcr.io/distroless/static:nonroot
        
        WORKDIR /app
        COPY --from=builder --chown=nonroot:nonroot /build/app .
        
        USER nonroot
        EXPOSE 8080
        
        HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
          CMD ["/app", "health"]
          
        ENTRYPOINT ["/app"]
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should have minimal issues due to good practices
        critical_issues = [ann for ann in results if getattr(ann, 'severity', None) == 'critical']
        assert len(critical_issues) == 0

        # May have some low-confidence suggestions but should be mostly clean
        high_confidence_issues = [ann for ann in results if ann.confidence > 0.8]
        assert len(high_confidence_issues) <= 2

    def test_error_handling(self, analyzer, tmp_path):
        """Test error handling for malformed files"""
        test_file = tmp_path / "Dockerfile"
        test_file.write_text("This is not a valid Dockerfile {{{ unclosed")
        
        # Should not crash on malformed files
        results = analyzer.analyze_file(test_file)
        assert isinstance(results, list)

    def test_file_not_found(self, analyzer, tmp_path):
        """Test handling of non-existent files"""
        fake_file = tmp_path / "does_not_exist"
        results = analyzer.analyze_file(fake_file)
        assert results == []

    def test_empty_dockerfile(self, analyzer, tmp_path):
        """Test handling of empty Dockerfiles"""
        empty_file = tmp_path / "Dockerfile"
        empty_file.write_text("")
        results = analyzer.analyze_file(empty_file)
        assert results == []

    def test_dockerfile_with_only_comments(self, analyzer, tmp_path):
        """Test handling of Dockerfiles with only comments"""
        comments_file = tmp_path / "Dockerfile"
        comments_file.write_text("""
        # This is a comment
        # Another comment
        
        # FROM ubuntu:20.04 (commented out)
        """)
        results = analyzer.analyze_file(comments_file)
        
        # Should detect missing essential instructions
        assert len(results) >= 2  # Missing USER, HEALTHCHECK, etc.

    def test_file_patterns_matching(self, analyzer):
        """Test that analyzer recognizes different Dockerfile patterns"""
        patterns = analyzer.file_patterns
        
        # Should match various Dockerfile naming conventions
        assert 'Dockerfile' in patterns
        assert 'Dockerfile.*' in patterns
        assert '*.dockerfile' in patterns

    def test_secure_base_images_recognition(self, analyzer):
        """Test recognition of secure base images"""
        secure_images = analyzer.secure_base_images
        
        # Should include known secure base images
        assert 'gcr.io/distroless' in secure_images
        assert 'alpine' in secure_images
        assert 'scratch' in secure_images
        assert 'busybox' in secure_images
        assert 'cgr.dev/chainguard' in secure_images

    def test_analyze_config_file_method(self, analyzer, tmp_path):
        """Test the _analyze_config_file method"""
        # Test docker-compose.yml detection
        compose_file = tmp_path / "docker-compose.yml"
        compose_file.write_text("version: '3.8'")
        
        results = analyzer._analyze_config_file(compose_file)
        # Currently returns empty list as compose analysis is placeholder
        assert isinstance(results, list)
        
        # Test .dockerignore detection
        dockerignore_file = tmp_path / ".dockerignore"
        dockerignore_file.write_text(".git\n*.log")
        
        results = analyzer._analyze_config_file(dockerignore_file)
        assert isinstance(results, list)
        assert len(results) >= 1  # Should suggest additional patterns

    def test_contextual_analysis_with_user_instruction(self, analyzer, tmp_path):
        """Test that context tracking works correctly with USER instruction"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM alpine:3.14
        
        # Has USER instruction
        USER nobody
        
        WORKDIR /app
        COPY app.py .
        
        # Has HEALTHCHECK
        HEALTHCHECK --interval=30s CMD ping -c 1 localhost
        
        CMD ["python", "app.py"]
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should NOT complain about missing USER or HEALTHCHECK
        root_issues = [ann for ann in results if "runs as root" in ann.evidence.lower()]
        assert len(root_issues) == 0
        
        healthcheck_issues = [ann for ann in results if "healthcheck" in ann.evidence.lower()]
        assert len(healthcheck_issues) == 0

    def test_platform_specific_from_instruction(self, analyzer, tmp_path):
        """Test handling of platform-specific FROM instructions"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM --platform=linux/amd64 ubuntu:20.04
        
        RUN apt-get update && apt-get install -y python3
        
        FROM --platform=linux/arm64 alpine:3.14
        
        RUN apk add --no-cache nodejs
        
        USER 1000
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should properly extract base images despite platform flag
        # Base image analysis should still work
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        # Should still detect configuration management issues if any
        assert isinstance(results, list)

    def test_multiple_exposed_ports(self, analyzer, tmp_path):
        """Test handling of multiple EXPOSE instructions"""
        test_file = tmp_path / "Dockerfile"
        dockerfile = '''FROM nginx:alpine
        
        # Multiple ports exposed
        EXPOSE 80
        EXPOSE 443  
        EXPOSE 8080
        EXPOSE 22   # This should be flagged
        EXPOSE 3000
        
        USER nginx
        '''
        test_file.write_text(dockerfile)

        results = analyzer.analyze_file(test_file)

        # Should detect SSH port exposure
        controls = set()
        for ann in results:
            controls.update(ann.control_ids)

        assert "IA-2" in controls or "SC-7" in controls

        # Should specifically flag port 22
        ssh_port_issues = [ann for ann in results if "22" in ann.evidence]
        assert len(ssh_port_issues) >= 1
