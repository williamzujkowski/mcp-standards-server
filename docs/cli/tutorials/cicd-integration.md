# CI/CD Integration Guide

This guide covers how to integrate MCP Standards Server into your continuous integration and deployment pipelines.

## Table of Contents

1. [GitHub Actions](#github-actions)
2. [GitLab CI/CD](#gitlab-cicd)
3. [Jenkins](#jenkins)
4. [CircleCI](#circleci)
5. [Azure DevOps](#azure-devops)
6. [Bitbucket Pipelines](#bitbucket-pipelines)
7. [Docker-based CI](#docker-based-ci)
8. [Performance Optimization](#performance-optimization)
9. [Reporting and Notifications](#reporting-and-notifications)

## GitHub Actions

### Basic Workflow

```yaml
# .github/workflows/mcp-standards.yml
name: MCP Standards Validation

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
      
      - name: Cache MCP Standards
        uses: actions/cache@v3
        with:
          path: |
            ~/.cache/mcp-standards
            ~/.cache/pip
          key: ${{ runner.os }}-mcp-${{ hashFiles('.mcp-standards.yaml') }}-${{ hashFiles('**/requirements*.txt') }}
          restore-keys: |
            ${{ runner.os }}-mcp-${{ hashFiles('.mcp-standards.yaml') }}-
            ${{ runner.os }}-mcp-
      
      - name: Install MCP Standards Server
        run: |
          python -m pip install --upgrade pip
          pip install mcp-standards-server
          mcp-standards --version
      
      - name: Sync Standards
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=$GITHUB_TOKEN
          mcp-standards sync
      
      - name: Validate Code
        run: |
          mcp-standards validate . \
            --format sarif \
            --output results.sarif \
            --fail-on error
      
      - name: Upload SARIF results
        uses: github/codeql-action/upload-sarif@v2
        if: always()
        with:
          sarif_file: results.sarif
      
      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const { execSync } = require('child_process');
            
            // Generate markdown report
            execSync('mcp-standards report --format markdown > report.md');
            const report = fs.readFileSync('report.md', 'utf8');
            
            // Find or create comment
            const { data: comments } = await github.rest.issues.listComments({
              owner: context.repo.owner,
              repo: context.repo.repo,
              issue_number: context.issue.number,
            });
            
            const botComment = comments.find(comment => 
              comment.body.includes('MCP Standards Validation Report'));
            
            const body = `## 🔍 MCP Standards Validation Report\n\n${report}`;
            
            if (botComment) {
              await github.rest.issues.updateComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                comment_id: botComment.id,
                body
              });
            } else {
              await github.rest.issues.createComment({
                owner: context.repo.owner,
                repo: context.repo.repo,
                issue_number: context.issue.number,
                body
              });
            }
```

### Matrix Strategy for Multiple Environments

```yaml
name: Multi-Environment Validation

on: [push, pull_request]

jobs:
  validate:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
        node-version: ['16.x', '18.x', '20.x']
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version 1.0.0
      
      - name: Set up Node.js
        uses: actions/setup-node@v3
        with:
          node-version: ${{ matrix.node-version 1.0.0
      
      - name: Install dependencies
        run: |
          pip install mcp-standards-server
          npm ci
      
      - name: Run validation
        run: |
          mcp-standards validate . --fail-on warning
          
      - name: Upload results
        uses: actions/upload-artifact@v3
        if: failure()
        with:
          name: validation-results-${{ matrix.os }}-py${{ matrix.python-version 1.0.0
          path: |
            validation-report.*
            **/*.log
```

### Reusable Workflow

```yaml
# .github/workflows/mcp-standards-reusable.yml
name: MCP Standards Check

on:
  workflow_call:
    inputs:
      severity:
        required: false
        type: string
        default: 'error'
      config-file:
        required: false
        type: string
        default: '.mcp-standards.yaml'
    secrets:
      github-token:
        required: true

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: MCP Standards Validation
        uses: williamzujkowski/mcp-standards-action@v1
        with:
          severity: ${{ inputs.severity }}
          config-file: ${{ inputs.config-file }}
          github-token: ${{ secrets.github-token }}
```

## GitLab CI/CD

### Basic Pipeline

```yaml
# .gitlab-ci.yml
stages:
  - prepare
  - validate
  - report

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"
  MCP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/mcp-standards"

cache:
  key: "$CI_JOB_NAME-$CI_COMMIT_REF_SLUG"
  paths:
    - .cache/pip
    - .cache/mcp-standards

before_script:
  - python -m pip install --upgrade pip
  - pip install mcp-standards-server
  - export MCP_STANDARDS_CACHE_DIRECTORY=$MCP_CACHE_DIR

sync-standards:
  stage: prepare
  script:
    - mcp-standards sync
  artifacts:
    paths:
      - .cache/mcp-standards
    expire_in: 1 week

validate-code:
  stage: validate
  dependencies:
    - sync-standards
  script:
    - mcp-standards validate . --format junit --output standards-report.xml
  artifacts:
    reports:
      junit: standards-report.xml
    paths:
      - standards-report.xml
      - validation-details.json
    when: always
    expire_in: 1 month

generate-report:
  stage: report
  dependencies:
    - validate-code
  script:
    - mcp-standards report --input validation-details.json --format html --output standards-report.html
  artifacts:
    paths:
      - standards-report.html
    expose_as: 'Standards Compliance Report'
    expire_in: 1 month
  only:
    - merge_requests
    - main

# Merge request pipeline
validate-mr:
  stage: validate
  script:
    - |
      # Get changed files
      CHANGED_FILES=$(git diff --name-only origin/$CI_MERGE_REQUEST_TARGET_BRANCH_NAME...HEAD)
      
      # Validate only changed files
      echo "$CHANGED_FILES" | xargs mcp-standards validate --format gitlab --output mr-report.json
      
      # Post comment to MR
      curl --request POST \
        --header "PRIVATE-TOKEN: $CI_JOB_TOKEN" \
        --data-urlencode "body@mr-report.json" \
        "$CI_API_V4_URL/projects/$CI_PROJECT_ID/merge_requests/$CI_MERGE_REQUEST_IID/notes"
  only:
    - merge_requests
```

### GitLab Templates

```yaml
# .gitlab/ci/mcp-standards.yml
.mcp-standards:
  image: python:3.11-slim
  before_script:
    - apt-get update && apt-get install -y git curl
    - pip install mcp-standards-server
    - mcp-standards sync
  cache:
    key: mcp-standards-$CI_COMMIT_REF_SLUG
    paths:
      - .cache/mcp-standards
      - .cache/pip

include:
  - local: '.gitlab/ci/mcp-standards.yml'

validate:
  extends: .mcp-standards
  script:
    - mcp-standards validate src/ --fail-on error
```

## Jenkins

### Declarative Pipeline

```groovy
// Jenkinsfile
pipeline {
    agent {
        docker {
            image 'python:3.11-slim'
            args '-v $HOME/.cache:/root/.cache'
        }
    }
    
    environment {
        MCP_STANDARDS_REPOSITORY_AUTH_TOKEN = credentials('github-token')
    }
    
    stages {
        stage('Setup') {
            steps {
                sh '''
                    pip install mcp-standards-server
                    mcp-standards --version
                '''
            }
        }
        
        stage('Sync Standards') {
            steps {
                sh 'mcp-standards sync'
            }
        }
        
        stage('Validate') {
            steps {
                script {
                    def validation = sh(
                        script: 'mcp-standards validate . --format json --output validation.json',
                        returnStatus: true
                    )
                    
                    if (validation != 0) {
                        unstable('Standards validation found issues')
                    }
                }
            }
            post {
                always {
                    recordIssues(
                        enabledForFailure: true,
                        tool: groovyScript(
                            parserId: 'mcp-standards',
                            pattern: 'validation.json'
                        )
                    )
                }
            }
        }
        
        stage('Generate Report') {
            steps {
                sh 'mcp-standards report --format html --output standards-report.html'
                publishHTML([
                    allowMissing: false,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: '.',
                    reportFiles: 'standards-report.html',
                    reportName: 'Standards Compliance Report'
                ])
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'validation.json,standards-report.html', 
                             allowEmptyArchive: true
        }
        failure {
            emailext (
                subject: "Standards Validation Failed: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
                body: '''${SCRIPT, template="mcp-standards-email.template"}''',
                to: "${env.CHANGE_AUTHOR_EMAIL}"
            )
        }
    }
}
```

### Shared Library

```groovy
// vars/mcpStandardsValidation.groovy
def call(Map config = [:]) {
    def severity = config.severity ?: 'error'
    def path = config.path ?: '.'
    def format = config.format ?: 'json'
    
    pipeline {
        agent any
        stages {
            stage('MCP Standards Validation') {
                steps {
                    script {
                        docker.image('python:3.11-slim').inside {
                            sh """
                                pip install mcp-standards-server
                                mcp-standards sync
                                mcp-standards validate ${path} \
                                    --severity ${severity} \
                                    --format ${format} \
                                    --output validation-results.${format}
                            """
                        }
                    }
                }
            }
        }
    }
}
```

## CircleCI

### Configuration

```yaml
# .circleci/config.yml
version: 2.1

orbs:
  python: circleci/python@2.1.1

executors:
  mcp-standards:
    docker:
      - image: cimg/python:3.11
    resource_class: medium

commands:
  install-mcp:
    steps:
      - python/install-packages:
          pkg-manager: pip
          packages: mcp-standards-server
      
  sync-standards:
    steps:
      - restore_cache:
          keys:
            - v1-mcp-standards-{{ checksum ".mcp-standards.yaml" }}
            - v1-mcp-standards-
      - run:
          name: Sync Standards
          command: mcp-standards sync
      - save_cache:
          key: v1-mcp-standards-{{ checksum ".mcp-standards.yaml" }}
          paths:
            - ~/.cache/mcp-standards

jobs:
  validate:
    executor: mcp-standards
    steps:
      - checkout
      - install-mcp
      - sync-standards
      - run:
          name: Validate Code
          command: |
            mcp-standards validate . \
              --format junit \
              --output test-results/mcp-standards.xml
      - store_test_results:
          path: test-results
      - store_artifacts:
          path: test-results
      
  report:
    executor: mcp-standards
    steps:
      - checkout
      - install-mcp
      - sync-standards
      - run:
          name: Generate Compliance Report
          command: |
            mcp-standards validate . --format json --output validation.json
            mcp-standards report --input validation.json --format html --output report.html
      - store_artifacts:
          path: report.html
          destination: compliance-report

workflows:
  version: 2
  standards-check:
    jobs:
      - validate
      - report:
          requires:
            - validate
          filters:
            branches:
              only:
                - main
                - develop
```

### Orb Definition

```yaml
# mcp-standards-orb.yml
version: 2.1

description: MCP Standards validation orb

executors:
  default:
    docker:
      - image: cimg/python:3.11

commands:
  validate:
    parameters:
      path:
        type: string
        default: "."
      severity:
        type: enum
        enum: ["error", "warning", "info"]
        default: "error"
    steps:
      - run:
          name: Install MCP Standards
          command: pip install mcp-standards-server
      - run:
          name: Sync Standards
          command: mcp-standards sync
      - run:
          name: Validate
          command: |
            mcp-standards validate << parameters.path >> \
              --fail-on << parameters.severity >> \
              --format junit \
              --output $CIRCLE_TEST_REPORTS/mcp-standards.xml

jobs:
  validate:
    executor: default
    parameters:
      path:
        type: string
        default: "."
    steps:
      - checkout
      - validate:
          path: << parameters.path >>
```

## Azure DevOps

### Azure Pipeline

```yaml
# azure-pipelines.yml
trigger:
  branches:
    include:
      - main
      - develop
  paths:
    exclude:
      - docs/*
      - README.md

pool:
  vmImage: 'ubuntu-latest'

variables:
  pythonVersion: '3.11'
  mcpCacheDir: $(Pipeline.Workspace)/.mcp-standards

stages:
  - stage: Validate
    displayName: 'Standards Validation'
    jobs:
      - job: ValidateCode
        displayName: 'Validate Code Standards'
        steps:
          - task: UsePythonVersion@0
            inputs:
              versionSpec: '$(pythonVersion)'
            displayName: 'Use Python $(pythonVersion)'
          
          - task: Cache@2
            inputs:
              key: 'mcp | "$(Agent.OS)" | .mcp-standards.yaml'
              restoreKeys: |
                mcp | "$(Agent.OS)"
              path: $(mcpCacheDir)
            displayName: 'Cache MCP Standards'
          
          - script: |
              python -m pip install --upgrade pip
              pip install mcp-standards-server
              echo "##vso[task.setvariable variable=MCP_STANDARDS_CACHE_DIRECTORY]$(mcpCacheDir)"
            displayName: 'Install MCP Standards'
          
          - script: |
              mcp-standards sync
            displayName: 'Sync Standards'
            env:
              MCP_STANDARDS_REPOSITORY_AUTH_TOKEN: $(System.AccessToken)
          
          - script: |
              mcp-standards validate . \
                --format azurepipelines \
                --output $(Agent.TempDirectory)/validation-results.json
            displayName: 'Validate Code'
            continueOnError: true
          
          - task: PublishTestResults@2
            inputs:
              testResultsFormat: 'JUnit'
              testResultsFiles: '$(Agent.TempDirectory)/**/validation-*.xml'
              testRunTitle: 'MCP Standards Validation'
            condition: always()
          
          - task: PublishCodeQualityResults@1
            inputs:
              summaryFileLocation: '$(Agent.TempDirectory)/validation-results.json'
              baselineFile: '$(Build.SourcesDirectory)/.mcp-baseline.json'
            condition: always()
  
  - stage: Report
    displayName: 'Generate Reports'
    dependsOn: Validate
    condition: and(succeeded(), eq(variables['Build.SourceBranch'], 'refs/heads/main'))
    jobs:
      - job: GenerateReport
        displayName: 'Generate Compliance Report'
        steps:
          - script: |
              mcp-standards report \
                --type compliance \
                --format html \
                --output $(Build.ArtifactStagingDirectory)/compliance-report.html
            displayName: 'Generate Report'
          
          - task: PublishBuildArtifacts@1
            inputs:
              pathToPublish: '$(Build.ArtifactStagingDirectory)'
              artifactName: 'compliance-reports'
```

### Task Group

```yaml
# mcp-standards-task-group.yml
steps:
  - task: UsePythonVersion@0
    inputs:
      versionSpec: '3.11'
    displayName: 'Setup Python'
  
  - task: PythonScript@0
    inputs:
      scriptSource: 'inline'
      script: |
        import subprocess
        import json
        import sys
        
        # Install MCP Standards
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mcp-standards-server'])
        
        # Sync standards
        subprocess.check_call(['mcp-standards', 'sync'])
        
        # Run validation
        result = subprocess.run(
            ['mcp-standards', 'validate', '.', '--format', 'json'],
            capture_output=True,
            text=True
        )
        
        # Parse results
        if result.returncode != 0:
            issues = json.loads(result.stdout)
            for issue in issues['issues']:
                print(f"##vso[task.logissue type={issue['severity']};sourcepath={issue['file']};"
                      f"linenumber={issue['line']};columnnumber={issue['column']}]{issue['message']}")
            
            if any(i['severity'] == 'error' for i in issues['issues']):
                sys.exit(1)
    displayName: 'MCP Standards Validation'
```

## Bitbucket Pipelines

### Configuration

```yaml
# bitbucket-pipelines.yml
image: python:3.11-slim

definitions:
  caches:
    mcp-standards: ~/.cache/mcp-standards
    pip: ~/.cache/pip
  
  steps:
    - step: &install-mcp
        name: Install MCP Standards
        caches:
          - pip
        script:
          - pip install mcp-standards-server
          - mcp-standards --version
    
    - step: &sync-standards
        name: Sync Standards
        caches:
          - mcp-standards
        script:
          - export MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=$GITHUB_TOKEN
          - mcp-standards sync
    
    - step: &validate-code
        name: Validate Code
        script:
          - mcp-standards validate . --format junit --output test-results/mcp-standards.xml
        after-script:
          - pipe: atlassian/checkstyle-report:0.3.0
            variables:
              REPORT_FILE: test-results/mcp-standards.xml

pipelines:
  default:
    - step: *install-mcp
    - step: *sync-standards
    - step: *validate-code
  
  pull-requests:
    '**':
      - step: *install-mcp
      - step: *sync-standards
      - step:
          name: Validate PR
          script:
            # Get changed files
            - git diff --name-only origin/$BITBUCKET_PR_DESTINATION_BRANCH...HEAD > changed_files.txt
            
            # Validate only changed files
            - cat changed_files.txt | xargs mcp-standards validate --format bitbucket
            
            # Post comment to PR
            - |
              if [ -f validation-report.md ]; then
                curl -X POST \
                  -H "Authorization: Bearer $BITBUCKET_TOKEN" \
                  -H "Content-Type: application/json" \
                  -d "{\"content\": {\"raw\": \"$(cat validation-report.md | jq -Rs .)\"}}" \
                  "https://api.bitbucket.org/2.0/repositories/$BITBUCKET_WORKSPACE/$BITBUCKET_REPO_SLUG/pullrequests/$BITBUCKET_PR_ID/comments"
              fi
  
  branches:
    main:
      - step: *install-mcp
      - step: *sync-standards
      - step: *validate-code
      - step:
          name: Generate Compliance Report
          script:
            - mcp-standards report --type compliance --format html --output compliance-report.html
          artifacts:
            - compliance-report.html
```

## Docker-based CI

### Dockerfile for CI

```dockerfile
# Dockerfile.ci
FROM python:3.11-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install MCP Standards
RUN pip install --no-cache-dir mcp-standards-server

# Create non-root user
RUN useradd -m -u 1000 mcp && \
    mkdir -p /home/mcp/.cache/mcp-standards && \
    chown -R mcp:mcp /home/mcp

USER mcp
WORKDIR /workspace

# Pre-cache standards (optional)
ARG GITHUB_TOKEN
ENV MCP_STANDARDS_REPOSITORY_AUTH_TOKEN=$GITHUB_TOKEN
RUN mcp-standards sync || true

# Validation stage
FROM base as validator

COPY --chown=mcp:mcp . /workspace

RUN mcp-standards validate . --format json --output /tmp/validation.json || true

# Report stage
FROM base as reporter

COPY --from=validator /tmp/validation.json /tmp/validation.json

RUN mcp-standards report \
    --input /tmp/validation.json \
    --format html \
    --output /workspace/report.html

# Final stage
FROM nginx:alpine as final

COPY --from=reporter /workspace/report.html /usr/share/nginx/html/index.html
```

### Docker Compose for CI

```yaml
# docker-compose.ci.yml
version: '3.8'

services:
  validator:
    build:
      context: .
      dockerfile: Dockerfile.ci
      target: validator
      args:
        GITHUB_TOKEN: ${GITHUB_TOKEN}
    volumes:
      - ./validation-results:/tmp/results
    command: |
      sh -c "
        mcp-standards validate /workspace \
          --format json \
          --output /tmp/results/validation.json &&
        mcp-standards report \
          --input /tmp/results/validation.json \
          --format junit \
          --output /tmp/results/junit.xml
      "
    
  report-server:
    build:
      context: .
      dockerfile: Dockerfile.ci
      target: final
    ports:
      - "8080:80"
    depends_on:
      - validator
```

## Performance Optimization

### Caching Strategies

```yaml
# Optimized GitHub Actions with multiple cache layers
- name: Cache MCP Standards
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/mcp-standards
      ~/.cache/pip
      ~/.local/share/mcp-standards
    key: |
      mcp-${{ runner.os }}-${{ hashFiles('.mcp-standards.yaml') }}-${{ hashFiles('**/*.py') }}-${{ hashFiles('**/*.js') }}
    restore-keys: |
      mcp-${{ runner.os }}-${{ hashFiles('.mcp-standards.yaml') }}-${{ hashFiles('**/*.py') }}-
      mcp-${{ runner.os }}-${{ hashFiles('.mcp-standards.yaml') }}-
      mcp-${{ runner.os }}-
```

### Parallel Validation

```bash
#!/bin/bash
# parallel-validation.sh

# Split files for parallel processing
find . -name "*.py" -o -name "*.js" | split -n l/4 - /tmp/files_

# Run validation in parallel
parallel -j 4 'cat {} | xargs mcp-standards validate --format json --output /tmp/results_{#}.json' ::: /tmp/files_*

# Merge results
mcp-standards report --merge /tmp/results_*.json --output final-report.json
```

### Incremental Validation

```yaml
# Only validate changed files in PR
- name: Get changed files
  id: changed-files
  uses: tj-actions/changed-files@v46.0.1
  with:
    files: |
      **/*.py
      **/*.js
      **/*.ts

- name: Validate changed files
  if: steps.changed-files.outputs.any_changed == 'true'
  run: |
    echo "${{ steps.changed-files.outputs.all_changed_files }}" | \
    xargs mcp-standards validate --fail-on error
```

## Reporting and Notifications

### Slack Notifications

```yaml
# GitHub Actions with Slack
- name: Notify Slack
  if: failure()
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    text: |
      MCP Standards Validation Failed
      Repository: ${{ github.repository }}
      Branch: ${{ github.ref }}
      Commit: ${{ github.sha }}
      
      View details: ${{ github.server_url }}/${{ github.repository }}/actions/runs/${{ github.run_id }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

### Email Reports

```groovy
// Jenkins email template
emailext (
    subject: "Standards Report: ${env.JOB_NAME} - ${env.BUILD_NUMBER}",
    body: '''
        <h2>MCP Standards Validation Report</h2>
        
        <h3>Summary</h3>
        <ul>
            <li>Total Issues: ${ISSUES_COUNT}</li>
            <li>Errors: ${ERROR_COUNT}</li>
            <li>Warnings: ${WARNING_COUNT}</li>
        </ul>
        
        <h3>Details</h3>
        ${FILE, path="standards-report.html"}
        
        <p>View full report: <a href="${BUILD_URL}">Build #${BUILD_NUMBER}</a></p>
    ''',
    mimeType: 'text/html',
    to: '${DEFAULT_RECIPIENTS}',
    attachmentsPattern: 'standards-report.*'
)
```

### Dashboard Integration

```yaml
# Grafana Dashboard metrics
- name: Push metrics to Prometheus
  if: always()
  run: |
    cat <<EOF | curl --data-binary @- http://prometheus-pushgateway:9091/metrics/job/mcp-standards/instance/${{ github.repository }}
    # TYPE mcp_validation_errors gauge
    mcp_validation_errors{repository="${{ github.repository }}",branch="${{ github.ref }}"} $(jq '.summary.errors' validation.json)
    # TYPE mcp_validation_warnings gauge  
    mcp_validation_warnings{repository="${{ github.repository }}",branch="${{ github.ref }}"} $(jq '.summary.warnings' validation.json)
    # TYPE mcp_validation_duration_seconds gauge
    mcp_validation_duration_seconds{repository="${{ github.repository }}",branch="${{ github.ref }}"} $(jq '.duration' validation.json)
    EOF
```

## Best Practices

1. **Cache Everything**: Cache standards, dependencies, and validation results
2. **Fail Fast**: Run quick validations first, comprehensive checks later
3. **Incremental Checks**: Only validate changed files in PRs
4. **Parallel Processing**: Split large codebases for parallel validation
5. **Clear Reporting**: Generate actionable reports with specific fixes
6. **Baseline Comparison**: Track improvement over time
7. **Integration with existing tools**: Combine with ESLint, Pylint, etc.