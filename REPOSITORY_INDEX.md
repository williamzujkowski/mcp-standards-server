# MCP Standards Server - Repository Index

## 🏗️ Repository Overview

**Project**: MCP Standards Server  
**Description**: Model Context Protocol server for NIST 800-53r5 compliance checking and standards enforcement  
**Version**: 0.1.0  
**Architecture**: Three-tier hybrid vector store (Redis + FAISS + ChromaDB)  
**Test Coverage**: 77% (targeting 80%)  
**NIST Controls**: 200+ across 20 families  
**Standards Imported**: 17 from williamzujkowski/standards  

## 📁 Core Components

### MCP Server Layer
- **`src/server.py`** - Main MCP server with 8 tools, 20 resources, 5 prompts
- **`src/core/mcp/`** - MCP protocol handlers and models

### Three-Tier Hybrid Vector Store
- **`src/core/standards/hybrid_vector_store.py`** - Orchestrator (650 lines)
- **`src/core/standards/chromadb_tier.py`** - Persistent storage (350 lines)  
- **`src/core/standards/tiered_storage_strategy.py`** - Caching intelligence (455 lines)
- **`src/core/standards/engine.py`** - Standards engine (850 lines)

### Code Analyzers
- **`src/analyzers/python_analyzer.py`** - Python + Django/Flask/FastAPI (450 lines)
- **`src/analyzers/javascript_analyzer.py`** - JS/TS + React/Angular/Vue (400 lines)
- **`src/analyzers/go_analyzer.py`** - Go + Gin/Fiber/gRPC
- **`src/analyzers/java_analyzer.py`** - Java + Spring/JPA
- **`src/analyzers/terraform_analyzer.py`** - IaC for AWS/Azure/GCP (350 lines)
- **`src/analyzers/dockerfile_analyzer.py`** - Docker security analysis
- **`src/analyzers/k8s_analyzer.py`** - Kubernetes manifest validation

## 🎯 Performance Architecture

### Tier 1: Redis Query Cache
- **Performance**: <0.1ms
- **Purpose**: Instant response for repeated queries
- **Implementation**: `src/core/standards/engine.py`

### Tier 2: FAISS Hot Cache  
- **Performance**: <1ms
- **Purpose**: Ultra-fast search for top 1000 standards
- **Implementation**: `src/core/standards/hybrid_vector_store.py`
- **Features**: LRU eviction, access tracking

### Tier 3: ChromaDB Persistent Storage
- **Performance**: 10-50ms
- **Purpose**: Full corpus with metadata filtering
- **Implementation**: `src/core/standards/chromadb_tier.py`
- **Features**: Persistent embeddings, rich metadata

## 🛠️ CLI Interface

**Main CLI**: `src/cli/main.py` (700 lines, 10 commands)

### Commands
1. **`init`** - Initialize project with NIST compliance structure
2. **`scan`** - Scan code for NIST control implementations  
3. **`ssp`** - Generate OSCAL System Security Plans
4. **`validate`** - Validate against specific controls
5. **`coverage`** - Generate control coverage reports
6. **`generate`** - Generate secure code templates
7. **`standards`** - Manage standards (list/import/update/search)
8. **`cache`** - Manage three-tier cache system
9. **`server`** - Start MCP server
10. **`version`** - Show version information

## 🔍 MCP Tools

**Server**: `src/server.py` exposes 8 tools for Claude integration:

1. **`load_standards`** - Load standards by query with token optimization
2. **`analyze_code`** - Analyze code for NIST compliance  
3. **`suggest_controls`** - Get control recommendations
4. **`generate_template`** - Generate secure code templates
5. **`validate_compliance`** - Validate against requirements
6. **`scan_with_llm`** - Enhanced LLM-powered scanning
7. **`semantic_search`** - Natural language standards search
8. **`cache_stats`** - View three-tier cache performance

## 📊 NIST 800-53r5 Coverage

### Control Families (20 total)
- **AC**: Access Control (25 controls)
- **AU**: Audit and Accountability (16 controls)
- **SC**: System and Communications Protection (45 controls)
- **SI**: System and Information Integrity (23 controls)
- **IA**: Identification and Authentication (12 controls)
- **CM**: Configuration Management (12 controls)
- **CP**: Contingency Planning (13 controls)
- **CA**: Assessment and Monitoring (9 controls)
- **[17 more families...]**

### High-Priority Controls
- **IA-2(1)**: Multi-factor Authentication
- **SC-8/SC-13**: Encryption in Transit and at Rest
- **SI-10**: Input Validation and SQL Injection Prevention
- **AU-2/AU-3**: Security Event Logging with Context
- **AC-3/AC-6**: Access Control with Least Privilege

## 🧪 Testing Infrastructure

### Test Organization
- **Unit Tests**: 90 tests across all modules
- **Integration Tests**: 5 tests for end-to-end workflows  
- **E2E Tests**: 2 tests for complete MCP integration
- **Current Coverage**: 77% (target: 80%)

### Key Test Files
- **`tests/unit/core/standards/test_hybrid_vector_store.py`** - 27 tests, 64% coverage
- **`tests/unit/core/standards/test_tiered_storage_strategy.py`** - 31 tests, 98% coverage
- **`tests/unit/analyzers/test_python_analyzer.py`** - Framework detection tests

## 📚 Documentation

### Primary Documentation
- **`README.md`** - Main project overview with architecture diagrams
- **`REPOSITORY_INDEX.md`** - This comprehensive file index
- **`REPOSITORY_INDEX.json`** - LLM-optimized metadata for all files

### User Guides
- **`docs/USAGE_GUIDE.md`** - Complete usage guide (767 lines) with all CLI commands
- **`docs/QUICK_REFERENCE.md`** - Command cheat sheet (230 lines) for quick lookup
- **`docs/CLAUDE_INTEGRATION.md`** - Claude CLI setup guide (442 lines) with platform-specific instructions
- **`docs/guides/quickstart.md`** - Quick start guide for new users

### Developer Documentation
- **`CLAUDE.md`** - LLM logic router with decision trees and workflows
- **`TODO_ANALYZERS.md`** - Status tracking and implementation guide
- **`IMPLEMENTATION_SUMMARY.md`** - Technical implementation summary and architecture decisions
- **`CONTRIBUTING.md`** - Guidelines for contributing to the project
- **`docs/analyzers-implementation.md`** - Detailed analyzer implementation documentation
- **`docs/enhanced-control-detection.md`** - Enhanced NIST control detection patterns

### API Documentation
- **`docs/api/mcp-tools.md`** - MCP tools API documentation and specifications
- **`docs/nist/controls.md`** - NIST 800-53r5 controls reference and mapping

### Project Documentation
- **`CHANGELOG.md`** - Project changelog and version history
- **`SECURITY.md`** - Security policy and vulnerability reporting
- **`Project_plan.md`** - Original project planning and roadmap
- **`docs/DOCUMENTATION_STATUS.md`** - Documentation status and completeness tracking

### Specialized Documentation
- **`docs/llm-optimization.md`** - LLM optimization strategies and token reduction techniques
- **`docs/standards-versioning.md`** - Standards versioning and management documentation
- **`docs/analyzers/iac-analyzer-specs.md`** - Infrastructure as Code analyzer specifications

### Section Documentation
- **`docs/README.md`** - Documentation directory index and navigation
- **`tests/README.md`** - Testing framework and guidelines
- **`examples/README.md`** - Examples directory overview and usage
- **`examples/python-api/README.md`** - Python Flask API example with NIST compliance
- **`examples/javascript-frontend/README.md`** - JavaScript frontend example with security patterns
- **`examples/python-flask-api/README.md`** - Python Flask API example implementation

## 🔧 Configuration

### Core Dependencies (`pyproject.toml`)
- **MCP**: `mcp>=0.1.0` - Model Context Protocol
- **Vector Stores**: `faiss-cpu>=1.7.4`, `chromadb>=0.4.22`
- **Caching**: `redis>=5.0.0`
- **AI/ML**: `tiktoken>=0.5.0`, `sentence-transformers>=2.2.0`
- **Analysis**: `tree-sitter>=0.20.0`, `tree-sitter-languages>=1.10.0`

### Environment Configuration
- **`REDIS_URL`**: Redis connection (required)
- **`STANDARDS_PATH`**: Standards directory path
- **`FAISS_HOT_CACHE_SIZE`**: Hot cache capacity (default: 1000)
- **`CHROMADB_PERSIST_PATH`**: ChromaDB storage location

## 🚀 CI/CD Pipeline

### GitHub Actions (`.github/workflows/`)
- **`ci.yml`** - Main CI with testing, linting, security
- **`nist-compliance.yml`** - NIST compliance checking
- **`standards-compliance.yml`** - Standards validation
- **`release.yml`** - Automated releases

### Quality Tools
- **Linting**: `ruff` - Code formatting and style
- **Type Checking**: `mypy` - Static type analysis  
- **Security**: `trivy` - Vulnerability scanning
- **Testing**: `pytest` - Test framework with coverage

## 📈 Performance Metrics

### Query Performance
- **Redis Cache Hits**: <0.1ms
- **FAISS Searches**: <1ms
- **ChromaDB Searches**: 10-50ms
- **Cache Hit Rate**: ~85%

### Analysis Performance  
- **Python Files**: ~50ms per file
- **JavaScript Files**: ~40ms per file
- **Terraform Files**: ~60ms per file

## 🔗 Integration Points

### Claude CLI Integration
- **Configuration**: Documented in `docs/CLAUDE_INTEGRATION.md`
- **MCP Protocol**: Version 1.0
- **Tools Exposed**: 8 tools for compliance workflows
- **Resources**: 20+ dynamic standards resources
- **Prompts**: 5 specialized compliance prompts

### Development Workflow
- **Pre-commit Hooks**: Automated compliance validation
- **Git Integration**: NIST annotation tracking
- **VS Code Support**: Settings and workflow integration (planned)

## 🏗️ Standards Data

### Imported Standards (`data/standards/`)
- **17 standards** from williamzujkowski/standards
- **Master Documents**: UNIFIED_STANDARDS.yaml, MODERN_SECURITY_STANDARDS.yaml
- **Index**: standards_index.json with metadata
- **Import Script**: `scripts/import_standards.py`

### Content Organization
- **Core Standards**: Security, development, infrastructure
- **Specialized**: Cloud-native, DevOps, data engineering
- **Compliance**: Legal, cost optimization, project management

## 🎯 LLM Optimization Features

### Token Reduction (`src/core/standards/token_optimizer.py`)
- **Target**: 90% token reduction
- **Strategies**: Summarization, essential-only, hierarchical
- **Implementation**: Three optimization algorithms

### Semantic Search (`src/core/standards/semantic_search.py`)
- **Embeddings**: sentence-transformers
- **Query Types**: Natural language, NIST notation, mixed
- **Performance**: Integrated with three-tier architecture

---

## 🔍 Quick File Lookup

### Need to find...?
- **MCP Tools Implementation** → `src/server.py`
- **Vector Store Logic** → `src/core/standards/hybrid_vector_store.py`
- **Python Analysis** → `src/analyzers/python_analyzer.py`
- **CLI Commands** → `src/cli/main.py`
- **NIST Patterns** → `src/analyzers/enhanced_patterns.py`
- **Cache Management** → `src/core/standards/tiered_storage_strategy.py`
- **OSCAL Generation** → `src/core/compliance/oscal_handler.py`
- **Installation Guide** → `README.md` or `docs/USAGE_GUIDE.md`
- **Troubleshooting** → `docs/CLAUDE_INTEGRATION.md`

This index provides comprehensive metadata for LLM consumption while maintaining human readability for development workflows.