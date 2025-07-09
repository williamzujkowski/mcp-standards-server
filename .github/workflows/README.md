# GitHub Actions Workflows

This directory contains all the GitHub Actions workflows for the MCP Standards Server project. Below is a comprehensive overview of each workflow and its purpose.

## Workflows Overview

### 1. CI (Continuous Integration) - `ci.yml`
**Triggers:** Push to main/develop, Pull requests, Manual dispatch

**Purpose:** Main CI pipeline that ensures code quality and functionality
- **Linting:** Runs Ruff, Black, mypy, and yamllint
- **Testing:** Multi-OS (Ubuntu, Windows, macOS) and multi-Python version (3.10, 3.11, 3.12) testing
- **E2E Tests:** Full end-to-end testing with Redis and frontend build
- **Build:** Package building and validation
- **Docker:** Container image building and testing
- **Standards Validation:** Ensures all standards files are consistent

### 2. Release - `release.yml`
**Triggers:** Push of version tags (v*.*.*), Manual dispatch

**Purpose:** Automated release pipeline
- **Version Validation:** Ensures semantic versioning compliance
- **Build & Test:** Full test suite before release
- **GitHub Release:** Creates release with auto-generated changelog
- **PyPI Publishing:** Publishes to Python Package Index
- **Docker Publishing:** Pushes images to Docker Hub and GitHub Container Registry
- **Documentation Updates:** Updates version numbers across the codebase

### 3. Security - `security.yml`
**Triggers:** Push to main/develop, Pull requests, Weekly schedule, Manual dispatch

**Purpose:** Comprehensive security scanning
- **Dependency Check:** Safety, pip-audit vulnerability scanning
- **Code Scanning:** Bandit security analysis
- **Semgrep:** Advanced security patterns detection
- **Container Scanning:** Trivy vulnerability scanner for Docker images
- **License Compliance:** Checks for restrictive licenses
- **Secrets Scanning:** TruffleHog and pattern-based secret detection

### 4. Benchmark - `benchmark.yml`
**Triggers:** Pull requests, Push to main, Daily schedule, Manual dispatch

**Purpose:** Performance monitoring and regression detection
- **Performance Benchmarks:** Runs all performance test suites
- **Memory Benchmarks:** Memory usage profiling and leak detection
- **Load Testing:** Stress testing with various user loads
- **Comparison:** Automatic performance comparison for PRs
- **Historical Tracking:** Stores benchmark results for trend analysis

### 5. PR Validation - `pr-validation.yml`
**Triggers:** Pull request events (opened, edited, synchronized)

**Purpose:** Automated PR quality checks
- **Title Validation:** Ensures conventional commit format
- **Size Labeling:** Adds size labels (XS, S, M, L, XL)
- **Breaking Change Detection:** Identifies and labels breaking changes
- **File Checks:** Ensures tests are added with source changes
- **Auto-labeling:** Applies labels based on changed files

### 6. Documentation - `docs.yml`
**Triggers:** Push to main (docs changes), Pull requests (docs changes), Manual dispatch

**Purpose:** Documentation building and deployment
- **Build Docs:** Generates documentation with MkDocs
- **Link Checking:** Validates all documentation links
- **Deploy to GitHub Pages:** Publishes documentation site
- **Docstring Coverage:** Ensures code documentation quality
- **Changelog Generation:** Automatic changelog updates

### 7. Maintenance - `maintenance.yml`
**Triggers:** Weekly schedule (Monday 2 AM UTC), Manual dispatch

**Purpose:** Automated repository maintenance
- **Stale Issues/PRs:** Closes inactive issues and pull requests
- **Artifact Cleanup:** Removes old workflow artifacts
- **Cache Cleanup:** Manages GitHub Actions cache storage
- **Dependency Updates:** Reports outdated dependencies
- **Security Audits:** Triggers weekly security scans

## Configuration Files

### `.github/dependabot.yml`
Configures automated dependency updates for:
- Python packages (weekly)
- npm packages (weekly)
- GitHub Actions (weekly)
- Docker base images (weekly)

### `.github/labeler.yml`
Defines automatic labeling rules based on file changes:
- `documentation`: Docs and markdown files
- `python`: Python source and config files
- `tests`: Test files
- `ci`: CI/CD configuration
- `docker`: Container-related files
- `frontend`/`backend`: Web UI components
- `security`: Security-related files
- `performance`: Benchmark and optimization files

### `.github/cliff.toml`
Configures git-cliff for automatic changelog generation with:
- Conventional commit parsing
- Grouped sections (Features, Bug Fixes, etc.)
- GitHub issue/PR linking
- Semantic versioning support

### `.yamllint.yml`
YAML linting rules for consistent formatting:
- 2-space indentation
- 120 character line limit
- Quote style enforcement
- Proper spacing rules

## Required Secrets

The following secrets need to be configured in the repository settings:

- `CODECOV_TOKEN`: For coverage reporting
- `DOCKERHUB_USERNAME`: Docker Hub username
- `DOCKERHUB_TOKEN`: Docker Hub access token
- `PYPI_API_TOKEN`: PyPI publishing token (or use Trusted Publishing)

## Workflow Badges

Add these badges to your README.md:

```markdown
[![CI](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/ci.yml)
[![Security](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/security.yml/badge.svg)](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/security.yml)
[![Documentation](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/docs.yml/badge.svg)](https://github.com/YOUR_USERNAME/mcp-standards-server/actions/workflows/docs.yml)
```

## Best Practices

1. **Always test workflows** in a feature branch before merging to main
2. **Use environment variables** for configuration values
3. **Cache dependencies** to speed up workflow runs
4. **Set appropriate timeouts** to prevent hanging jobs
5. **Use job dependencies** to ensure proper execution order
6. **Add status checks** to PR protection rules
7. **Monitor workflow usage** to stay within GitHub Actions limits

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Check branch protection rules
   - Verify file paths in trigger conditions
   - Ensure proper YAML syntax

2. **Permission errors**
   - Check repository settings for Actions permissions
   - Verify GITHUB_TOKEN permissions in workflow
   - Add necessary permissions to job definitions

3. **Cache misses**
   - Review cache key strategy
   - Check for changes in dependency files
   - Verify cache size limits

4. **Test failures in CI**
   - Check for environment-specific issues
   - Review service container logs
   - Ensure all dependencies are installed

## Maintenance

- Review and update workflows quarterly
- Monitor for deprecated Actions versions
- Keep dependencies up to date
- Archive old workflow runs periodically
- Review security scan results weekly