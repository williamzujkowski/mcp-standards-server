# Contributing to MCP Standards Server

Thank you for your interest in contributing to MCP Standards Server! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project follows a standard code of conduct. Please be respectful and professional in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/mcp-standards-server.git
   cd mcp-standards-server
   ```
3. **Set up development environment**:
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

## Development Process

### 1. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes

#### Code Standards

- **Python Style**: Follow PEP 8, enforced by ruff
- **Type Hints**: Required for all new code
- **Docstrings**: Required for all public functions/classes
- **NIST Annotations**: Required for security-relevant code

#### NIST Control Annotations

When adding security-relevant code, include NIST control annotations:

```python
# @nist-controls: AC-3, AU-2
# @evidence: Role-based access control with audit logging
# @oscal-component: api-gateway
def authorize_request(user, resource):
    """Authorize user access to resource"""
    # Implementation
```

### 3. Write Tests

- Maintain test coverage above 80%
- Write unit tests for new functionality
- Include integration tests for complex features
- Test file naming: `test_<module_name>.py`

### 4. Run Quality Checks

```bash
# Format code
ruff format src tests

# Lint code
ruff check src tests

# Type checking
mypy src

# Run tests with coverage
pytest --cov=src --cov-report=term

# Run security scan
mcp-standards scan
```

### 5. Update Documentation

- Update README.md if adding new features
- Add/update docstrings
- Update CLAUDE.md for LLM-specific instructions
- Add examples if introducing new functionality

### 6. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git commit -m "feat: add support for Ruby language analyzer

- Implement RubyAnalyzer class with AST parsing
- Add pattern detection for Rails security features
- Include tests with 95% coverage
- Update documentation

@nist-controls: SA-11, SA-15"
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Pull Request Guidelines

### PR Title Format

Use conventional commits format:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `test:` Test additions/modifications
- `refactor:` Code refactoring
- `perf:` Performance improvements
- `chore:` Maintenance tasks

### PR Description Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Coverage maintained above 80%
- [ ] Security scan passes

## NIST Controls
List any NIST controls implemented or affected

## Checklist
- [ ] Code follows project style
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] NIST annotations added where appropriate
```

## Areas for Contribution

### High Priority
- Import standards from williamzujkowski/standards repository
- Implement remaining CLI commands (generate, validate)
- Add MCP resource providers
- Create MCP prompt templates

### Medium Priority
- REST API implementation
- Additional language analyzers (Ruby, PHP, C++, Rust)
- Enhanced error reporting
- Performance optimizations

### Documentation
- Improve user guides
- Add more examples
- Create video tutorials
- Translate documentation

### Testing
- Increase test coverage
- Add performance tests
- Create end-to-end tests
- Test on different platforms

## Architecture Decisions

When proposing significant changes:

1. Create an ADR (Architecture Decision Record) in `docs/architecture/decisions/`
2. Use template: `docs/architecture/decisions/template.md`
3. Discuss in issue before implementing

## Security Considerations

All contributions must:
- Follow secure coding practices
- Include appropriate NIST control annotations
- Pass security scanning
- Not introduce vulnerabilities
- Handle errors securely (no information leakage)

## Questions?

- Open an issue for bugs or feature requests
- Start a discussion for general questions
- Check existing issues/PRs before creating new ones

## Recognition

Contributors will be recognized in:
- Release notes
- Contributors file
- Project documentation

Thank you for contributing to making development more secure and compliant!