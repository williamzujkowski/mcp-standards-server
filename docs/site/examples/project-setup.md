# Project Setup Examples

Real-world examples of setting up MCP Standards Server for different types of projects.

## Web Application Projects

### React + TypeScript Web App

**Project Structure:**
```
my-react-app/
├── src/
│   ├── components/
│   ├── pages/
│   └── utils/
├── public/
├── package.json
├── tsconfig.json
└── .mcp-standards.json
```

**Setup Commands:**
```bash
# 1. Initialize project
npx create-react-app my-react-app --template typescript
cd my-react-app

# 2. Get applicable standards
mcp-standards query applicable \
  --project-type web_application \
  --framework react \
  --language typescript

# 3. Create project configuration
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "web_application",
  "framework": "react",
  "language": "typescript",
  "version": "1.0.0",
  "standards": {
    "required": [
      "react-18-patterns",
      "typescript-strict",
      "web-accessibility-wcag",
      "security-web-app"
    ],
    "optional": [
      "performance-web",
      "seo-optimization"
    ]
  },
  "validation": {
    "severity": "warning",
    "autoFix": true,
    "excludePatterns": [
      "build/**",
      "node_modules/**",
      "**/*.test.ts",
      "**/*.spec.ts"
    ]
  },
  "compliance": {
    "frameworks": ["WCAG-2.1-AA"],
    "reporting": true
  }
}
EOF

# 4. Setup VS Code integration
mkdir -p .vscode
cat > .vscode/settings.json << 'EOF'
{
  "mcpStandards.enabled": true,
  "mcpStandards.serverUrl": "http://localhost:8080",
  "mcpStandards.enableRealTimeValidation": true,
  "mcpStandards.autoFixOnSave": true,
  "mcpStandards.projectConfigFile": ".mcp-standards.json"
}
EOF

# 5. Setup GitHub Actions
mkdir -p .github/workflows
cat > .github/workflows/standards.yml << 'EOF'
name: Standards Validation

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
    
    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'
        cache: 'npm'
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        npm ci
        pip install mcp-standards-server
    
    - name: Sync Standards
      run: mcp-standards sync
    
    - name: Validate Code
      run: |
        mcp-standards validate \
          --format sarif \
          --output standards-results.sarif
    
    - name: Upload SARIF
      uses: github/codeql-action/upload-sarif@v2
      if: always()
      with:
        sarif_file: standards-results.sarif
EOF

# 6. Setup pre-commit hooks
npm install --save-dev husky
npx husky install
npx husky add .husky/pre-commit "mcp-standards validate --fix --severity error ."

# 7. Initial validation
mcp-standards validate .
```

**Expected Output:**
```
✅ Standards validation completed
📝 Found 3 applicable standards
⚠️  12 warnings found
✅ 8 issues auto-fixed
📈 Compliance: WCAG-2.1-AA (87%)
```

### Vue.js + Composition API

**Setup:**
```bash
# Project creation
npm create vue@latest my-vue-app
cd my-vue-app

# Standards configuration
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "web_application",
  "framework": "vue",
  "language": "typescript",
  "standards": {
    "required": [
      "vue-3-composition",
      "typescript-strict",
      "web-accessibility-wcag"
    ]
  },
  "validation": {
    "severity": "warning",
    "rules": {
      "vue-composition-api": "error",
      "accessibility-alt-text": "error"
    }
  }
}
EOF

# Validate setup
mcp-standards validate --standard vue-3-composition .
```

## API Projects

### FastAPI Python API

**Project Structure:**
```
my-api/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   └── dependencies.py
│   ├── core/
│   ├── models/
│   └── main.py
├── tests/
├── requirements.txt
└── .mcp-standards.json
```

**Setup:**
```bash
# 1. Create project
mkdir my-api && cd my-api
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn

# 2. Configure standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "api",
  "framework": "fastapi",
  "language": "python",
  "apiVersion": "v1",
  "standards": {
    "required": [
      "api-design-restful",
      "python-pep8",
      "security-api",
      "openapi-3.0"
    ],
    "optional": [
      "performance-api",
      "monitoring-observability"
    ]
  },
  "validation": {
    "severity": "error",
    "apiSpecific": {
      "validateOpenAPI": true,
      "requireAuthentication": true,
      "enforceRateLimit": true
    }
  },
  "compliance": {
    "frameworks": ["OWASP-API-Top-10"],
    "reporting": true
  }
}
EOF

# 3. Setup development environment
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
sqlalchemy==2.0.23
alembic==1.13.0
EOF

pip install -r requirements.txt

# 4. Create basic API structure
mkdir -p app/api/v1 app/core app/models tests

# 5. Validate API design
mcp-standards validate \
  --standard api-design-restful \
  --format openapi \
  .
```

### GraphQL API with Node.js

**Setup:**
```bash
# Project initialization
npm init -y
npm install apollo-server-express graphql express
npm install -D @types/node typescript ts-node

# Standards configuration
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "api",
  "framework": "apollo-graphql",
  "language": "typescript",
  "standards": {
    "required": [
      "graphql-best-practices",
      "typescript-strict",
      "security-graphql",
      "api-design-graphql"
    ]
  },
  "validation": {
    "graphqlSpecific": {
      "enforceDepthLimit": true,
      "requireAuth": true,
      "validateSchema": true
    }
  }
}
EOF

# Validate GraphQL schema
mcp-standards validate --standard graphql-best-practices ./schema/
```

## Mobile Applications

### React Native App

**Setup:**
```bash
# Create React Native project
npx react-native init MyMobileApp --template react-native-template-typescript
cd MyMobileApp

# Configure standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "mobile_application",
  "platform": "react-native",
  "language": "typescript",
  "targetPlatforms": ["ios", "android"],
  "standards": {
    "required": [
      "react-native-patterns",
      "mobile-accessibility",
      "mobile-performance",
      "typescript-strict"
    ],
    "platform": {
      "ios": ["ios-human-interface"],
      "android": ["material-design"]
    }
  },
  "validation": {
    "mobileSpecific": {
      "validateAccessibility": true,
      "checkPerformance": true,
      "enforceOfflineSupport": false
    }
  }
}
EOF

# Platform-specific validation
mcp-standards validate \
  --platform ios \
  --standard mobile-accessibility \
  ./src/
```

### Flutter App

**Setup:**
```bash
# Create Flutter project
flutter create my_flutter_app
cd my_flutter_app

# Configure standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "mobile_application",
  "framework": "flutter",
  "language": "dart",
  "standards": {
    "required": [
      "flutter-best-practices",
      "dart-effective",
      "mobile-accessibility"
    ]
  },
  "validation": {
    "flutterSpecific": {
      "enforceStatelessWidgets": true,
      "validatePubspec": true
    }
  }
}
EOF

# Validate Dart code
mcp-standards validate --language dart ./lib/
```

## Desktop Applications

### Electron App

**Setup:**
```bash
# Create Electron app
npm init -y
npm install electron
npm install -D electron-builder

# Configure standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "desktop_application",
  "framework": "electron",
  "language": "typescript",
  "standards": {
    "required": [
      "electron-security",
      "desktop-patterns",
      "typescript-strict"
    ]
  },
  "validation": {
    "electronSpecific": {
      "enforceContextIsolation": true,
      "validateSecurityHeaders": true,
      "checkNodeIntegration": true
    }
  }
}
EOF

# Security validation
mcp-standards validate \
  --standard electron-security \
  --severity error \
  .
```

## Microservices Projects

### Docker + Kubernetes Microservice

**Project Structure:**
```
microservice/
├── src/
├── tests/
├── Dockerfile
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
└── .mcp-standards.json
```

**Setup:**
```bash
# Configure microservice standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "microservice",
  "deploymentTarget": "kubernetes",
  "language": "go",
  "standards": {
    "required": [
      "microservices-patterns",
      "go-effective",
      "docker-best-practices",
      "kubernetes-security",
      "observability-tracing"
    ]
  },
  "validation": {
    "microserviceSpecific": {
      "validateDockerfile": true,
      "checkK8sManifests": true,
      "enforceHealthChecks": true,
      "requireMetrics": true
    }
  },
  "compliance": {
    "frameworks": ["NIST-800-190"],
    "containerSecurity": true
  }
}
EOF

# Validate containerization
mcp-standards validate \
  --standard docker-best-practices \
  ./Dockerfile

# Validate Kubernetes manifests
mcp-standards validate \
  --standard kubernetes-security \
  ./k8s/
```

## Library/Package Projects

### Python Package

**Setup:**
```bash
# Create package structure
mkdir my-python-package && cd my-python-package

# Configure package standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "library",
  "language": "python",
  "packageManager": "pip",
  "distributionTarget": "pypi",
  "standards": {
    "required": [
      "python-packaging",
      "python-pep8",
      "library-design",
      "documentation-sphinx"
    ]
  },
  "validation": {
    "librarySpecific": {
      "validateSetupPy": true,
      "checkDocstrings": true,
      "enforceTyping": true,
      "validateExamples": true
    }
  },
  "documentation": {
    "required": true,
    "format": "sphinx",
    "examples": true
  }
}
EOF

# Package structure validation
mcp-standards validate \
  --standard python-packaging \
  --check-structure \
  .
```

### NPM Package

**Setup:**
```bash
# Initialize npm package
npm init -y

# Configure package standards
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "library",
  "language": "typescript",
  "packageManager": "npm",
  "distributionTarget": "npm",
  "standards": {
    "required": [
      "npm-packaging",
      "typescript-library",
      "library-design"
    ]
  },
  "validation": {
    "librarySpecific": {
      "validatePackageJson": true,
      "checkExports": true,
      "enforceTypings": true
    }
  }
}
EOF

# Package validation
mcp-standards validate \
  --standard npm-packaging \
  ./package.json
```

## Multi-Language Projects

### Full-Stack Application

**Project Structure:**
```
full-stack-app/
├── frontend/          # React TypeScript
├── backend/           # Python FastAPI
├── mobile/            # React Native
├── shared/            # Shared types/utilities
└── .mcp-standards.json
```

**Setup:**
```bash
# Root configuration
cat > .mcp-standards.json << 'EOF'
{
  "projectType": "full_stack_application",
  "architecture": "monorepo",
  "components": {
    "frontend": {
      "framework": "react",
      "language": "typescript",
      "path": "./frontend"
    },
    "backend": {
      "framework": "fastapi",
      "language": "python",
      "path": "./backend"
    },
    "mobile": {
      "framework": "react-native",
      "language": "typescript",
      "path": "./mobile"
    }
  },
  "standards": {
    "global": [
      "security-comprehensive",
      "api-consistency"
    ],
    "frontend": [
      "react-18-patterns",
      "web-accessibility-wcag"
    ],
    "backend": [
      "api-design-restful",
      "python-pep8"
    ],
    "mobile": [
      "react-native-patterns",
      "mobile-accessibility"
    ]
  },
  "validation": {
    "crossComponent": {
      "enforceApiConsistency": true,
      "validateSharedTypes": true
    }
  }
}
EOF

# Component-specific validation
mcp-standards validate ./frontend --component frontend
mcp-standards validate ./backend --component backend
mcp-standards validate ./mobile --component mobile

# Cross-component validation
mcp-standards validate \
  --cross-component \
  --standard api-consistency \
  .
```

## Team Configuration Templates

### Development Team Standard Config

```json
{
  "team": {
    "name": "Development Team",
    "standards": {
      "baseline": [
        "security-basic",
        "code-quality",
        "documentation-minimal"
      ],
      "language": {
        "python": ["python-pep8", "python-typing"],
        "javascript": ["javascript-es6", "react-patterns"],
        "typescript": ["typescript-strict"]
      }
    },
    "validation": {
      "preCommit": true,
      "severity": "warning",
      "autoFix": true
    },
    "reporting": {
      "weekly": true,
      "compliance": true
    }
  }
}
```

### Enterprise Configuration

```json
{
  "organization": {
    "name": "Enterprise Corp",
    "standards": {
      "mandatory": [
        "security-enterprise",
        "compliance-sox",
        "accessibility-508",
        "performance-strict"
      ],
      "governance": {
        "approvalRequired": true,
        "auditTrail": true,
        "exemptionProcess": true
      }
    },
    "validation": {
      "severity": "error",
      "blockDeployment": true,
      "requireSign off": true
    },
    "compliance": {
      "frameworks": [
        "SOX",
        "GDPR",
        "HIPAA",
        "ISO-27001"
      ],
      "reporting": {
        "frequency": "daily",
        "stakeholders": [
          "security-team@company.com",
          "compliance@company.com"
        ]
      }
    }
  }
}
```

## Validation Examples

### Successful Validation Output

```
✅ MCP Standards Validation Report

📁 Project: my-react-app (web_application)
📚 Standards Applied: 4 required, 2 optional
📅 Validation Time: 2024-01-15 14:30:00

✅ Standards Status:
  ✓ react-18-patterns      (98% compliant)
  ✓ typescript-strict      (100% compliant)
  ✓ web-accessibility-wcag (89% compliant)
  ✓ security-web-app       (95% compliant)

📈 Summary:
  ✓ Total Files Scanned: 47
  ✓ Issues Found: 3 warnings
  ✓ Auto-Fixed: 8 issues
  ✓ Coverage: 94%

⚠️  Remaining Issues:
  1. src/components/UserProfile.tsx:23
     [accessibility] Missing alt text for profile image
     💡 Suggestion: Add descriptive alt attribute

  2. src/utils/api.ts:45
     [security] Hardcoded API endpoint
     💡 Suggestion: Use environment variables

  3. src/pages/Dashboard.tsx:67
     [performance] Large bundle size detected
     💡 Suggestion: Implement code splitting

📈 Compliance Report:
  ✓ WCAG 2.1 AA: 89% (Target: 95%)
  ✓ OWASP Top 10: 95% (Target: 98%)

🔗 Next Steps:
  1. Fix accessibility issues for WCAG compliance
  2. Review security configuration
  3. Consider performance optimization

Validation completed in 3.2 seconds
```

### Failed Validation Output

```
❌ MCP Standards Validation Failed

📁 Project: legacy-api (api)
📚 Standards Applied: 3 required
📅 Validation Time: 2024-01-15 14:35:00

❌ Critical Issues Found: 5 errors

🔴 ERRORS (Blocking):
  1. src/auth.py:12
     [security] SQL injection vulnerability
     💡 Use parameterized queries

  2. src/api/users.py:34
     [security] Unvalidated user input
     💡 Add input validation

  3. requirements.txt:8
     [security] Vulnerable dependency: requests==2.20.0
     💡 Update to requests>=2.25.0

  4. src/config.py:5
     [security] Database credentials in source code
     💡 Use environment variables

  5. src/api/admin.py:89
     [security] Missing authentication check
     💡 Add @requires_auth decorator

⚠️  WARNINGS:
  - 12 code style violations (auto-fixable)
  - 3 documentation issues

📈 Compliance Status:
  ❌ OWASP API Top 10: 45% (Target: 95%)
  ❌ PCI DSS: Failed (Critical security issues)

🚫 Deployment blocked due to security violations

🔧 Recommended Actions:
  1. Fix all security errors immediately
  2. Update vulnerable dependencies
  3. Implement proper authentication
  4. Add input validation
  5. Move secrets to environment variables

Validation completed in 1.8 seconds
```

---

For more examples and advanced configurations, see:
- [Configuration Guide](../guides/configuration.md)
- [Workflows Guide](../guides/workflows.md)
- [API Reference](../api/mcp-tools.md)
