# CI/CD Integration Guide

Integrate MCP Standards Server into your CI/CD pipeline for automated code quality enforcement.

## GitHub Actions

### Basic Workflow

Create `.github/workflows/standards-check.yml`:

```yaml
name: Standards Check

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install MCP Standards Server
      run: |
        pip install mcp-standards-server
    
    - name: Sync Standards
      run: |
        mcp-standards sync
    
    - name: Validate Code
      run: |
        mcp-standards validate --format sarif --output results.sarif
    
    - name: Upload SARIF to GitHub
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: results.sarif
```

### Advanced Workflow with Caching

```yaml
name: Standards Check (Advanced)

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  MCP_CACHE_DIR: ~/.mcp-standards/cache

jobs:
  validate:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
      with:
        fetch-depth: 0  # Full history for better analysis
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        cache: 'pip'
    
    - name: Cache MCP Standards
      uses: actions/cache@v3
      with:
        path: ${{ env.MCP_CACHE_DIR }}
        key: mcp-standards-${{ runner.os }}-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          mcp-standards-${{ runner.os }}-
    
    - name: Install dependencies
      run: |
        pip install mcp-standards-server
        pip install -r requirements.txt
    
    - name: Configure MCP Standards
      run: |
        mcp-standards config init
        mcp-standards config set validation.severity error
        mcp-standards config set validation.auto_fix false
    
    - name: Sync Standards
      run: |
        mcp-standards sync --force
    
    - name: Validate Changed Files Only
      if: github.event_name == 'pull_request'
      run: |
        # Get list of changed files
        git diff --name-only ${{ github.event.pull_request.base.sha }}..HEAD > changed_files.txt
        
        # Validate only changed files
        cat changed_files.txt | xargs -I {} mcp-standards validate {} --format sarif --output results.sarif
    
    - name: Validate All Files
      if: github.event_name == 'push'
      run: |
        mcp-standards validate . --format sarif --output results.sarif
    
    - name: Upload SARIF Results
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: results.sarif
    
    - name: Comment on PR
      if: github.event_name == 'pull_request' && failure()
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const results = JSON.parse(fs.readFileSync('results.sarif', 'utf8'));
          
          let comment = '## 🚨 Standards Validation Failed\n\n';
          
          results.runs[0].results.forEach(result => {
            const rule = result.ruleId;
            const message = result.message.text;
            const location = result.locations[0].physicalLocation;
            const file = location.artifactLocation.uri;
            const line = location.region.startLine;
            
            comment += `- **${rule}** in \`${file}:${line}\`: ${message}\n`;
          });
          
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: comment
          });
```

### Matrix Strategy for Multiple Languages

```yaml
name: Multi-Language Standards Check

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        language: [python, javascript, typescript, go]
        include:
          - language: python
            path: "**/*.py"
            standards: "python-pep8,python-security"
          - language: javascript
            path: "**/*.js"
            standards: "javascript-es6,security-eslint"
          - language: typescript
            path: "**/*.ts"
            standards: "typescript-strict,react-patterns"
          - language: go
            path: "**/*.go"
            standards: "go-effective,go-security"
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install MCP Standards
      run: pip install mcp-standards-server
    
    - name: Validate ${{ matrix.language }} code
      run: |
        mcp-standards validate \
          --language ${{ matrix.language }} \
          --standard ${{ matrix.standards }} \
          --format sarif \
          --output ${{ matrix.language }}-results.sarif \
          .
    
    - name: Upload SARIF for ${{ matrix.language }}
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: ${{ matrix.language }}-results.sarif
        category: ${{ matrix.language }}
```

## GitLab CI

### Basic Pipeline

Create `.gitlab-ci.yml`:

```yaml
stages:
  - validate
  - report

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  MCP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/mcp"

cache:
  paths:
    - .cache/pip
    - .cache/mcp

standards_check:
  stage: validate
  image: python:3.11
  
  before_script:
    - pip install mcp-standards-server
    - mcp-standards config init
    - mcp-standards sync
  
  script:
    - mcp-standards validate --format junit --output junit-report.xml
  
  artifacts:
    reports:
      junit: junit-report.xml
    paths:
      - junit-report.xml
    expire_in: 1 week
  
  rules:
    - if: $CI_PIPELINE_SOURCE == "merge_request_event"
    - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
```

### Advanced Pipeline with Security Scanning

```yaml
stages:
  - prepare
  - validate
  - security
  - report

variables:
  SECURE_LOG_LEVEL: info

include:
  - template: Security/SAST.gitlab-ci.yml
  - template: Security/Secret-Detection.gitlab-ci.yml

prepare_standards:
  stage: prepare
  image: python:3.11
  script:
    - pip install mcp-standards-server
    - mcp-standards sync
    - mcp-standards cache warm
  artifacts:
    paths:
      - .mcp-standards/
    expire_in: 1 hour

validate_python:
  stage: validate
  image: python:3.11
  dependencies:
    - prepare_standards
  script:
    - pip install mcp-standards-server
    - mcp-standards validate --language python --format sarif --output python-results.sarif
  artifacts:
    reports:
      sast: python-results.sarif
    expire_in: 1 week

validate_javascript:
  stage: validate
  image: node:18
  dependencies:
    - prepare_standards
  before_script:
    - apt-get update && apt-get install -y python3 python3-pip
    - pip3 install mcp-standards-server
  script:
    - mcp-standards validate --language javascript --format sarif --output js-results.sarif
  artifacts:
    reports:
      sast: js-results.sarif
    expire_in: 1 week

security_scan:
  stage: security
  dependencies:
    - prepare_standards
  script:
    - mcp-standards validate --standard security-comprehensive --format sarif --output security-results.sarif
  artifacts:
    reports:
      sast: security-results.sarif
```

## Jenkins

### Declarative Pipeline

Create `Jenkinsfile`:

```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.11'
        MCP_CACHE_DIR = '${WORKSPACE}/.mcp-cache'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    python${PYTHON_VERSION} -m venv venv
                    . venv/bin/activate
                    pip install mcp-standards-server
                '''
            }
        }
        
        stage('Sync Standards') {
            steps {
                sh '''
                    . venv/bin/activate
                    mcp-standards sync
                '''
            }
        }
        
        stage('Validate Code') {
            parallel {
                stage('Python Validation') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            mcp-standards validate \
                                --language python \
                                --format junit \
                                --output python-results.xml
                        '''
                    }
                    post {
                        always {
                            junit 'python-results.xml'
                        }
                    }
                }
                
                stage('Security Validation') {
                    steps {
                        sh '''
                            . venv/bin/activate
                            mcp-standards validate \
                                --standard security-comprehensive \
                                --format sarif \
                                --output security-results.sarif
                        '''
                    }
                    post {
                        always {
                            archiveArtifacts artifacts: 'security-results.sarif'
                        }
                    }
                }
            }
        }
    }
    
    post {
        always {
            cleanWs()
        }
        failure {
            emailext (
                subject: "Standards Validation Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: "Build failed. Check console output at ${env.BUILD_URL}",
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Scripted Pipeline with Docker

```groovy
node {
    def image
    
    stage('Build Docker Image') {
        checkout scm
        image = docker.build("mcp-standards:${env.BUILD_ID}")
    }
    
    stage('Validate in Container') {
        image.inside {
            sh '''
                mcp-standards sync
                mcp-standards validate --format junit --output results.xml
            '''
        }
    }
    
    stage('Publish Results') {
        junit 'results.xml'
        
        if (currentBuild.result == 'FAILURE') {
            slackSend(
                channel: '#development',
                color: 'danger',
                message: "Standards validation failed for ${env.JOB_NAME} ${env.BUILD_NUMBER}"
            )
        }
    }
}
```

## Azure DevOps

### Pipeline YAML

Create `azure-pipelines.yml`:

```yaml
trigger:
- main
- develop

pool:
  vmImage: 'ubuntu-latest'

variables:
  pythonVersion: '3.11'
  mcpCacheDir: '$(Pipeline.Workspace)/.mcp-cache'

stages:
- stage: Validate
  displayName: 'Standards Validation'
  jobs:
  - job: ValidateCode
    displayName: 'Validate Code Standards'
    
    steps:
    - task: UsePythonVersion@0
      inputs:
        versionSpec: $(pythonVersion)
      displayName: 'Use Python $(pythonVersion)'
    
    - task: Cache@2
      inputs:
        key: 'mcp-standards | "$(Agent.OS)" | requirements.txt'
        restoreKeys: |
          mcp-standards | "$(Agent.OS)"
        path: $(mcpCacheDir)
      displayName: 'Cache MCP Standards'
    
    - script: |
        pip install mcp-standards-server
        mcp-standards config init
        mcp-standards config set standards.cache_directory $(mcpCacheDir)
      displayName: 'Install MCP Standards'
    
    - script: |
        mcp-standards sync
      displayName: 'Sync Standards'
    
    - script: |
        mcp-standards validate --format junit --output $(System.DefaultWorkingDirectory)/test-results.xml
      displayName: 'Validate Code'
    
    - task: PublishTestResults@2
      condition: always()
      inputs:
        testResultsFormat: 'JUnit'
        testResultsFiles: '**/test-results.xml'
        searchFolder: '$(System.DefaultWorkingDirectory)'
      displayName: 'Publish Test Results'
    
    - script: |
        mcp-standards validate --format sarif --output $(System.DefaultWorkingDirectory)/sarif-results.sarif
      displayName: 'Generate SARIF Results'
    
    - task: PublishBuildArtifacts@1
      inputs:
        pathToPublish: '$(System.DefaultWorkingDirectory)/sarif-results.sarif'
        artifactName: 'sarif-results'
      displayName: 'Publish SARIF Results'
```

## CircleCI

### Configuration

Create `.circleci/config.yml`:

```yaml
version: 2.1

orbs:
  python: circleci/python@2.0.0

jobs:
  validate-standards:
    docker:
      - image: cimg/python:3.11
    
    steps:
      - checkout
      
      - restore_cache:
          keys:
            - mcp-standards-v1-{{ checksum "requirements.txt" }}
            - mcp-standards-v1-
      
      - run:
          name: Install MCP Standards
          command: |
            pip install mcp-standards-server
            mcp-standards config init
      
      - run:
          name: Sync Standards
          command: mcp-standards sync
      
      - save_cache:
          key: mcp-standards-v1-{{ checksum "requirements.txt" }}
          paths:
            - ~/.mcp-standards/cache
      
      - run:
          name: Validate Code
          command: |
            mcp-standards validate \
              --format junit \
              --output test-results.xml
      
      - store_test_results:
          path: test-results.xml
      
      - store_artifacts:
          path: test-results.xml
          destination: test-results

workflows:
  version: 2
  validate:
    jobs:
      - validate-standards
```

## Docker Integration

### Dockerfile for CI

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MCP Standards Server
RUN pip install mcp-standards-server

# Create cache directory
RUN mkdir -p /app/.mcp-standards/cache
WORKDIR /app

# Pre-sync standards for faster CI
RUN mcp-standards config init
RUN mcp-standards sync

# Copy validation script
COPY validate.sh /usr/local/bin/validate
RUN chmod +x /usr/local/bin/validate

ENTRYPOINT ["validate"]
```

### Validation Script

Create `validate.sh`:

```bash
#!/bin/bash
set -e

# Configuration
FORMAT=${MCP_FORMAT:-junit}
OUTPUT=${MCP_OUTPUT:-results.xml}
LANGUAGE=${MCP_LANGUAGE:-auto}
SEVERITY=${MCP_SEVERITY:-warning}

# Copy source code to container
cp -r /src/* /app/

# Update standards if needed
if [ "$MCP_UPDATE_STANDARDS" = "true" ]; then
    mcp-standards sync --force
fi

# Run validation
echo "Running MCP Standards validation..."
mcp-standards validate \
    --language "$LANGUAGE" \
    --format "$FORMAT" \
    --output "$OUTPUT" \
    --severity "$SEVERITY" \
    /app

echo "Validation complete. Results saved to $OUTPUT"
```

## Pre-commit Hooks

### Setup

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: mcp-standards-validate
        name: MCP Standards Validation
        entry: mcp-standards validate
        language: system
        files: \.(py|js|ts|go|rs)$
        pass_filenames: true
        args: [--severity, error, --fix]
```

### Install

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Test hooks
pre-commit run --all-files
```

## Integration Best Practices

### Performance Optimization

1. **Cache standards data** between builds
2. **Validate only changed files** in PR builds
3. **Use parallel validation** for multiple languages
4. **Pre-warm cache** in Docker images

### Security Considerations

1. **Limit network access** in CI environments
2. **Use secrets** for private repositories
3. **Scan for sensitive data** before validation
4. **Isolate validation** in containers

### Error Handling

1. **Continue on non-critical errors**
2. **Provide clear error messages**
3. **Log validation details**
4. **Notify teams** of critical failures

### Reporting

1. **Use SARIF format** for security tools integration
2. **Generate JUnit reports** for test dashboards
3. **Archive artifacts** for investigation
4. **Trend analysis** over time

---

For more CI/CD examples and troubleshooting, see our [GitHub repository](https://github.com/williamzujkowski/mcp-standards-server) and [community discussions](https://github.com/williamzujkowski/mcp-standards-server/discussions).
