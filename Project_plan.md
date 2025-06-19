# MCP Standards Server - Technical Implementation Plan

## Overview

This technical implementation plan provides detailed instructions for an LLM to build the MCP Standards Server with bidirectional NIST compliance capabilities. It incorporates standards from the williamzujkowski/standards repository.

## Phase 0: Project Foundation (Week 1)

### 0.1 Repository Structure Setup

Create the following directory structure following [KNOWLEDGE_MANAGEMENT_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/KNOWLEDGE_MANAGEMENT_STANDARDS.md):

```bash
mcp-standards-server/
├── .github/
│   ├── workflows/
│   │   ├── standards-compliance.yml    # From standards repo
│   │   ├── nist-compliance.yml         # Check control coverage
│   │   └── release.yml                 # Semantic versioning
│   └── ISSUE_TEMPLATE/
├── docs/
│   ├── architecture/
│   │   ├── decisions/                  # ADRs
│   │   └── diagrams/                   # System diagrams
│   ├── api/                            # OpenAPI specs
│   └── nist/                           # Control mappings
├── src/
│   ├── core/
│   │   ├── mcp/                        # MCP protocol implementation
│   │   ├── standards/                  # Standards engine
│   │   └── compliance/                 # NIST compliance engine
│   ├── analyzers/                      # Code analysis modules
│   │   ├── python/
│   │   ├── javascript/
│   │   ├── go/
│   │   └── java/
│   ├── generators/                     # Code generation
│   └── api/                           # REST/GraphQL endpoints
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── examples/                           # Usage examples
├── templates/                          # NIST-tagged templates
└── scripts/                           # Setup and utilities
```

### 0.2 Technology Stack Implementation

Following [CODING_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/CODING_STANDARDS.md), implement:

```python
# pyproject.toml
[tool.poetry]
name = "mcp-standards-server"
version = "0.1.0"
description = "MCP server for LLM-driven NIST compliance"

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.104.0"
uvicorn = "^0.24.0"
pydantic = "^2.4.0"
sqlalchemy = "^2.0.0"
redis = "^5.0.0"
tree-sitter = "^0.20.0"  # For AST parsing
oscal = "^1.0.0"         # OSCAL processing
rich = "^13.6.0"         # CLI output
typer = "^0.9.0"         # CLI framework

[tool.poetry.group.dev.dependencies]
pytest = "^7.4.0"
black = "^23.10.0"
ruff = "^0.1.0"
mypy = "^1.6.0"
pre-commit = "^3.5.0"

[tool.black]
line-length = 88

[tool.ruff]
select = ["E", "F", "I", "N", "UP", "S", "B", "A", "C4", "DTZ", "T10", "ISC", "RET", "SLF", "SIM", "ARG"]
line-length = 88

[tool.mypy]
strict = true
```

### 0.3 Core MCP Protocol Implementation

Implement the MCP server base following [MODERN_SECURITY_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/MODERN_SECURITY_STANDARDS.md):

```python
# src/core/mcp/server.py
"""
MCP Standards Server - Core Protocol Implementation
@nist-controls: AC-4, SC-8, SC-13
@evidence: Secure communication protocol with encryption
"""
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import json
import asyncio
from cryptography.fernet import Fernet
import logging

from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configure structured logging per OBSERVABILITY_STANDARDS.md
logging.basicConfig(
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s", "context": %(context)s}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class MCPMessage(BaseModel):
    """Base MCP message structure"""
    id: str = Field(..., description="Unique message ID")
    method: str = Field(..., description="RPC method name")
    params: Dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(..., description="Unix timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "msg_123",
                "method": "load_standards",
                "params": {"query": "CS:api + SEC:*"},
                "timestamp": 1234567890.123
            }
        }

@dataclass
class ComplianceContext:
    """
    Tracks compliance context for requests
    @nist-controls: AU-2, AU-3, AU-12
    @evidence: Comprehensive audit logging
    """
    user_id: str
    organization_id: str
    session_id: str
    request_id: str
    timestamp: float
    ip_address: str
    user_agent: str

class MCPHandler(ABC):
    """Abstract handler for MCP methods"""
    
    @abstractmethod
    async def handle(self, message: MCPMessage, context: ComplianceContext) -> Dict[str, Any]:
        """Handle MCP message"""
        pass

class MCPServer:
    """
    Main MCP Server implementation
    @nist-controls: AC-2, AC-3, AC-4, AC-6
    @evidence: Role-based access control and secure session management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.handlers: Dict[str, MCPHandler] = {}
        self.sessions: Dict[str, ComplianceContext] = {}
        self.app = FastAPI(title="MCP Standards Server")
        self._setup_routes()
        self._setup_security()
        
    def _setup_security(self):
        """
        Configure security measures
        @nist-controls: IA-2, IA-5, SC-8
        @evidence: Multi-factor authentication ready, encrypted communications
        """
        self.security = HTTPBearer()
        self.encryption_key = Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
    async def authenticate(self, credentials: HTTPAuthorizationCredentials) -> ComplianceContext:
        """
        Authenticate and create compliance context
        @nist-controls: IA-2, IA-8, AU-2
        @evidence: User authentication with audit logging
        """
        # Verify JWT token (implementation details omitted)
        # Create compliance context
        context = ComplianceContext(
            user_id="authenticated_user",
            organization_id="org_123",
            session_id="session_abc",
            request_id="req_xyz",
            timestamp=asyncio.get_event_loop().time(),
            ip_address="127.0.0.1",
            user_agent="MCP-Client/1.0"
        )
        
        # Audit log the authentication
        logger.info(
            "User authenticated",
            extra={"context": context.__dict__}
        )
        
        return context
    
    def register_handler(self, method: str, handler: MCPHandler):
        """Register method handler"""
        self.handlers[method] = handler
        
    async def handle_message(self, message: MCPMessage, context: ComplianceContext) -> Dict[str, Any]:
        """
        Route message to appropriate handler
        @nist-controls: AC-4, AU-2
        @evidence: Information flow enforcement and audit logging
        """
        if message.method not in self.handlers:
            raise ValueError(f"Unknown method: {message.method}")
            
        # Audit log the request
        logger.info(
            f"Handling MCP method: {message.method}",
            extra={"context": {**context.__dict__, "method": message.method}}
        )
        
        # Handle the message
        result = await self.handlers[message.method].handle(message, context)
        
        # Audit log the response
        logger.info(
            f"MCP method completed: {message.method}",
            extra={"context": {**context.__dict__, "method": message.method, "success": True}}
        )
        
        return result
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.websocket("/mcp")
        async def mcp_websocket(
            websocket: WebSocket,
            credentials: HTTPAuthorizationCredentials = Depends(self.security)
        ):
            """
            WebSocket endpoint for MCP protocol
            @nist-controls: SC-8, SC-13
            @evidence: TLS encryption for data in transit
            """
            context = await self.authenticate(credentials)
            await websocket.accept()
            
            try:
                while True:
                    data = await websocket.receive_text()
                    message = MCPMessage.parse_raw(data)
                    
                    try:
                        result = await self.handle_message(message, context)
                        await websocket.send_json({
                            "id": message.id,
                            "result": result,
                            "error": None
                        })
                    except Exception as e:
                        await websocket.send_json({
                            "id": message.id,
                            "result": None,
                            "error": str(e)
                        })
                        
            except Exception as e:
                logger.error(f"WebSocket error: {e}", extra={"context": context.__dict__})
            finally:
                await websocket.close()
```

### 0.4 Standards Engine Foundation

Implement the standards loading engine based on [CLAUDE.md](https://github.com/williamzujkowski/standards/blob/main/CLAUDE.md):

```python
# src/core/standards/engine.py
"""
Standards Engine - Intelligent loading and caching
@nist-controls: AC-4, SC-28, SI-12
@evidence: Information flow control and secure caching
"""
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
import yaml
import json
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime, timedelta

import redis
from pydantic import BaseModel

class StandardType(Enum):
    """Standard document types from CLAUDE.md"""
    CS = "coding"
    TS = "testing"
    SEC = "security"
    FE = "frontend"
    DE = "data_engineering"
    CN = "cloud_native"
    OBS = "observability"
    LEG = "legal_compliance"
    PM = "project_management"
    KM = "knowledge_management"
    WD = "web_design"
    EVT = "event_driven"
    DOP = "devops"
    CONT = "content"
    SEO = "seo"
    COST = "cost_optimization"
    GH = "github"
    TOOL = "tools"
    UNIFIED = "unified"

@dataclass
class StandardSection:
    """Represents a section of a standard"""
    id: str
    type: StandardType
    section: str
    content: str
    tokens: int
    version: str
    last_updated: datetime
    dependencies: List[str]
    nist_controls: Set[str]

class NaturalLanguageMapper:
    """
    Maps natural language queries to standards
    Based on mappings in CLAUDE.md
    """
    
    def __init__(self):
        self.mappings = {
            "secure api": ["CS:api", "SEC:api", "TS:integration"],
            "react app": ["FE:react", "WD:*", "TS:jest", "CS:javascript"],
            "microservices": ["CN:microservices", "EVT:*", "OBS:distributed"],
            "ci/cd pipeline": ["DOP:cicd", "GH:actions", "TS:*"],
            "database optimization": ["CS:performance", "DE:optimization", "OBS:metrics"],
            "documentation system": ["KM:*", "CS:documentation", "DOP:automation"],
            "nist compliance": ["SEC:*", "LEG:compliance", "OBS:logging", "TS:security"],
            "fedramp": ["SEC:fedramp", "LEG:fedramp", "CN:govcloud", "OBS:continuous-monitoring"],
            # Add more mappings
        }
        
    def map_query(self, query: str) -> List[str]:
        """Map natural language query to standard notations"""
        query_lower = query.lower()
        matched_standards = []
        
        # Check for exact matches
        for key, standards in self.mappings.items():
            if key in query_lower:
                matched_standards.extend(standards)
                
        # Check for partial matches
        if not matched_standards:
            for key, standards in self.mappings.items():
                words = key.split()
                if any(word in query_lower for word in words):
                    matched_standards.extend(standards)
                    
        return list(set(matched_standards))  # Remove duplicates

class StandardsEngine:
    """
    Core engine for loading and managing standards
    @nist-controls: AC-3, AC-4, CM-7
    @evidence: Access control and least functionality
    """
    
    def __init__(self, standards_path: Path, redis_client: Optional[redis.Redis] = None):
        self.standards_path = standards_path
        self.cache = redis_client or {}  # Use dict if no Redis
        self.nl_mapper = NaturalLanguageMapper()
        self.loaded_standards: Dict[str, StandardSection] = {}
        self._load_schema()
        
    def _load_schema(self):
        """Load standards schema"""
        schema_path = self.standards_path / "standards-schema.yaml"
        with open(schema_path) as f:
            self.schema = yaml.safe_load(f)
            
    def parse_query(self, query: str) -> List[str]:
        """
        Parse various query formats
        @nist-controls: SI-10
        @evidence: Input validation for queries
        """
        # Remove @ prefix if present
        query = query.strip().lstrip('@')
        
        # Natural language query
        if ':' not in query and not query.startswith('load'):
            return self.nl_mapper.map_query(query)
            
        # Direct notation (CS:api, SEC:*, etc.)
        if ':' in query:
            parts = []
            for part in query.split('+'):
                part = part.strip()
                if part:
                    parts.append(part)
            return parts
            
        # Load command
        if query.startswith('load'):
            # Extract the query part
            query = query[4:].strip()
            return self.parse_query(query)
            
        return []
    
    async def load_standards(
        self,
        query: str,
        context: Optional[str] = None,
        version: str = "latest",
        token_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load standards based on query
        @nist-controls: AC-4, SC-28
        @evidence: Information flow control with caching
        """
        # Parse the query
        standard_refs = self.parse_query(query)
        
        # Add context-based standards
        if context:
            context_refs = self._analyze_context(context)
            standard_refs.extend(context_refs)
            
        # Load each standard
        loaded = []
        total_tokens = 0
        
        for ref in standard_refs:
            sections = await self._load_standard_sections(ref, version)
            
            for section in sections:
                if token_limit and total_tokens + section.tokens > token_limit:
                    # Apply token optimization
                    section = self._optimize_for_tokens(section, token_limit - total_tokens)
                    
                loaded.append(section)
                total_tokens += section.tokens
                
                if token_limit and total_tokens >= token_limit:
                    break
                    
        return {
            "standards": [self._section_to_dict(s) for s in loaded],
            "metadata": {
                "version": version,
                "token_count": total_tokens,
                "refs_loaded": list(set(s.id for s in loaded))
            }
        }
    
    async def _load_standard_sections(self, ref: str, version: str) -> List[StandardSection]:
        """Load sections for a standard reference"""
        # Check cache first
        cache_key = f"standard:{ref}:{version}"
        
        if isinstance(self.cache, redis.Redis):
            cached = self.cache.get(cache_key)
            if cached:
                return json.loads(cached)
                
        # Parse reference (e.g., "CS:api" or "SEC:*")
        parts = ref.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid standard reference: {ref}")
            
        std_type = parts[0]
        section = parts[1]
        
        # Load from file
        sections = []
        
        if section == '*':
            # Load all sections
            sections = await self._load_all_sections(std_type, version)
        else:
            # Load specific section
            section_data = await self._load_section(std_type, section, version)
            if section_data:
                sections.append(section_data)
                
        # Cache the result
        if isinstance(self.cache, redis.Redis) and sections:
            self.cache.setex(
                cache_key,
                timedelta(hours=24),
                json.dumps([self._section_to_dict(s) for s in sections])
            )
            
        return sections
    
    def _optimize_for_tokens(self, section: StandardSection, max_tokens: int) -> StandardSection:
        """
        Optimize section for token limit
        Following token reduction strategies from CLAUDE.md
        """
        if section.tokens <= max_tokens:
            return section
            
        # Create condensed version
        condensed_content = self._create_summary(section.content, max_tokens)
        
        return StandardSection(
            id=section.id,
            type=section.type,
            section=section.section,
            content=condensed_content,
            tokens=self._count_tokens(condensed_content),
            version=section.version,
            last_updated=section.last_updated,
            dependencies=section.dependencies,
            nist_controls=section.nist_controls
        )
    
    def _section_to_dict(self, section: StandardSection) -> Dict[str, Any]:
        """Convert section to dictionary"""
        return {
            "id": section.id,
            "type": section.type.value,
            "section": section.section,
            "content": section.content,
            "tokens": section.tokens,
            "version": section.version,
            "last_updated": section.last_updated.isoformat(),
            "dependencies": section.dependencies,
            "nist_controls": list(section.nist_controls)
        }
```

## Phase 1: Core Compliance Features (Weeks 2-4)

### 1.1 NIST Control Mapping Engine

Implement control mapping based on [COMPLIANCE_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/COMPLIANCE_STANDARDS.md):

```python
# src/core/compliance/nist_mapper.py
"""
NIST Control Mapping Engine
@nist-controls: CA-2, CA-7, RA-5
@evidence: Security assessment and continuous monitoring
"""
from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import yaml
import re
from collections import defaultdict

from tree_sitter import Language, Parser
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_go as tsgo
import tree_sitter_java as tsjava

@dataclass
class NISTControl:
    """NIST 800-53r5 control representation"""
    id: str
    title: str
    family: str
    description: str
    implementation_guidance: str
    related_controls: List[str]
    
@dataclass
class ControlMapping:
    """Maps code pattern to NIST control"""
    pattern_name: str
    control_ids: List[str]
    confidence: float
    evidence_template: str
    code_indicators: List[str]
    
@dataclass
class CodeAnnotation:
    """Represents a NIST annotation in code"""
    file_path: str
    line_number: int
    control_ids: List[str]
    evidence: Optional[str]
    component: Optional[str]
    confidence: float

class NISTControlMapper:
    """
    Maps code patterns to NIST controls
    @nist-controls: PM-5, SA-11, SA-15
    @evidence: System inventory and security testing
    """
    
    def __init__(self, controls_path: Path):
        self.controls_path = controls_path
        self.controls: Dict[str, NISTControl] = {}
        self.mappings: Dict[str, ControlMapping] = {}
        self.parsers: Dict[str, Parser] = {}
        self._load_controls()
        self._load_mappings()
        self._init_parsers()
        
    def _load_controls(self):
        """Load NIST 800-53r5 controls"""
        controls_file = self.controls_path / "nist-800-53r5.json"
        with open(controls_file) as f:
            data = json.load(f)
            for control in data["controls"]:
                self.controls[control["id"]] = NISTControl(**control)
                
    def _load_mappings(self):
        """Load code pattern to control mappings"""
        # Core security pattern mappings
        self.mappings = {
            # Authentication patterns
            "mfa_authentication": ControlMapping(
                pattern_name="Multi-Factor Authentication",
                control_ids=["IA-2", "IA-2(1)", "IA-2(2)"],
                confidence=0.95,
                evidence_template="Implements MFA using {method}",
                code_indicators=["mfa", "two_factor", "2fa", "totp", "authenticator"]
            ),
            "password_policy": ControlMapping(
                pattern_name="Password Complexity",
                control_ids=["IA-5", "IA-5(1)"],
                confidence=0.90,
                evidence_template="Enforces password policy: {requirements}",
                code_indicators=["password_complexity", "min_length", "require_special"]
            ),
            "session_management": ControlMapping(
                pattern_name="Session Control",
                control_ids=["AC-12", "SC-10"],
                confidence=0.85,
                evidence_template="Session timeout after {duration}",
                code_indicators=["session_timeout", "idle_timeout", "expire_session"]
            ),
            
            # Access control patterns
            "rbac": ControlMapping(
                pattern_name="Role-Based Access Control",
                control_ids=["AC-2", "AC-3", "AC-6"],
                confidence=0.92,
                evidence_template="RBAC implementation with {roles}",
                code_indicators=["role", "permission", "authorize", "can", "policy"]
            ),
            "least_privilege": ControlMapping(
                pattern_name="Least Privilege",
                control_ids=["AC-6", "CM-7"],
                confidence=0.88,
                evidence_template="Least privilege enforced for {resource}",
                code_indicators=["minimum_permissions", "least_privilege", "deny_by_default"]
            ),
            
            # Encryption patterns
            "encryption_transit": ControlMapping(
                pattern_name="Encryption in Transit",
                control_ids=["SC-8", "SC-8(1)", "SC-13"],
                confidence=0.95,
                evidence_template="TLS {version} encryption for data in transit",
                code_indicators=["tls", "https", "ssl", "encrypt_transport"]
            ),
            "encryption_rest": ControlMapping(
                pattern_name="Encryption at Rest",
                control_ids=["SC-28", "SC-28(1)"],
                confidence=0.93,
                evidence_template="AES-{bits} encryption for data at rest",
                code_indicators=["encrypt_data", "aes", "encryption_key", "kms"]
            ),
            
            # Logging patterns
            "audit_logging": ControlMapping(
                pattern_name="Audit Logging",
                control_ids=["AU-2", "AU-3", "AU-12"],
                confidence=0.90,
                evidence_template="Comprehensive audit logging for {events}",
                code_indicators=["audit_log", "log_event", "security_log", "track"]
            ),
            "log_retention": ControlMapping(
                pattern_name="Log Retention",
                control_ids=["AU-11", "SI-12"],
                confidence=0.87,
                evidence_template="Logs retained for {duration}",
                code_indicators=["retention_policy", "archive_logs", "log_rotation"]
            ),
            
            # Input validation patterns
            "input_validation": ControlMapping(
                pattern_name="Input Validation",
                control_ids=["SI-10", "SI-15"],
                confidence=0.91,
                evidence_template="Input validation for {data_types}",
                code_indicators=["validate", "sanitize", "escape", "filter_input"]
            ),
            "sql_injection_prevention": ControlMapping(
                pattern_name="SQL Injection Prevention",
                control_ids=["SI-10", "SC-18"],
                confidence=0.94,
                evidence_template="Parameterized queries prevent SQL injection",
                code_indicators=["prepared_statement", "parameterized", "bind_param"]
            ),
            
            # Error handling patterns
            "secure_error_handling": ControlMapping(
                pattern_name="Secure Error Handling",
                control_ids=["SI-11", "AU-5"],
                confidence=0.86,
                evidence_template="Secure error handling without info disclosure",
                code_indicators=["generic_error", "hide_stacktrace", "error_handler"]
            ),
            
            # Add more patterns as needed
        }
        
    def _init_parsers(self):
        """Initialize language parsers"""
        # Python parser
        PY_LANGUAGE = Language(tspython.language(), "python")
        self.parsers["python"] = Parser()
        self.parsers["python"].set_language(PY_LANGUAGE)
        
        # JavaScript/TypeScript parser
        JS_LANGUAGE = Language(tsjavascript.language(), "javascript")
        self.parsers["javascript"] = Parser()
        self.parsers["javascript"].set_language(JS_LANGUAGE)
        self.parsers["typescript"] = self.parsers["javascript"]
        
        # Go parser
        GO_LANGUAGE = Language(tsgo.language(), "go")
        self.parsers["go"] = Parser()
        self.parsers["go"].set_language(GO_LANGUAGE)
        
        # Java parser
        JAVA_LANGUAGE = Language(tsjava.language(), "java")
        self.parsers["java"] = Parser()
        self.parsers["java"].set_language(JAVA_LANGUAGE)
        
    def analyze_code(self, code: str, language: str, file_path: str = "") -> List[CodeAnnotation]:
        """
        Analyze code for NIST control implementations
        @nist-controls: SA-11, RA-5
        @evidence: Static code analysis for security controls
        """
        annotations = []
        
        # First, check for explicit @nist annotations
        explicit_annotations = self._extract_explicit_annotations(code, file_path)
        annotations.extend(explicit_annotations)
        
        # Then, use AST analysis for implicit patterns
        if language in self.parsers:
            implicit_annotations = self._analyze_with_ast(code, language, file_path)
            annotations.extend(implicit_annotations)
        
        # Pattern matching for common security implementations
        pattern_annotations = self._match_patterns(code, file_path)
        annotations.extend(pattern_annotations)
        
        # Deduplicate and merge annotations
        return self._merge_annotations(annotations)
    
    def _extract_explicit_annotations(self, code: str, file_path: str) -> List[CodeAnnotation]:
        """Extract @nist-controls annotations from code"""
        annotations = []
        
        # Regex for @nist-controls annotations
        pattern = r'@nist-controls:\s*([A-Z]{2}-\d+(?:\(\d+\))?(?:\s*,\s*[A-Z]{2}-\d+(?:\(\d+\))?)*)'
        evidence_pattern = r'@evidence:\s*(.+?)(?=\n|$)'
        component_pattern = r'@oscal-component:\s*(.+?)(?=\n|$)'
        
        lines = code.split('\n')
        for i, line in enumerate(lines):
            # Find control annotations
            control_match = re.search(pattern, line)
            if control_match:
                controls = [c.strip() for c in control_match.group(1).split(',')]
                
                # Look for evidence in nearby lines
                evidence = None
                component = None
                
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    evidence_match = re.search(evidence_pattern, lines[j])
                    if evidence_match:
                        evidence = evidence_match.group(1).strip()
                        
                    component_match = re.search(component_pattern, lines[j])
                    if component_match:
                        component = component_match.group(1).strip()
                
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=i + 1,
                    control_ids=controls,
                    evidence=evidence,
                    component=component,
                    confidence=1.0  # Explicit annotations have full confidence
                ))
        
        return annotations
    
    def _analyze_with_ast(self, code: str, language: str, file_path: str) -> List[CodeAnnotation]:
        """Analyze code using AST for implicit patterns"""
        annotations = []
        parser = self.parsers[language]
        tree = parser.parse(bytes(code, "utf8"))
        
        # Language-specific analysis
        if language == "python":
            annotations.extend(self._analyze_python_ast(tree, code, file_path))
        elif language in ["javascript", "typescript"]:
            annotations.extend(self._analyze_javascript_ast(tree, code, file_path))
        elif language == "go":
            annotations.extend(self._analyze_go_ast(tree, code, file_path))
        elif language == "java":
            annotations.extend(self._analyze_java_ast(tree, code, file_path))
            
        return annotations
    
    def _match_patterns(self, code: str, file_path: str) -> List[CodeAnnotation]:
        """Match code against known security patterns"""
        annotations = []
        code_lower = code.lower()
        
        for pattern_name, mapping in self.mappings.items():
            # Check if any indicators are present
            indicators_found = []
            for indicator in mapping.code_indicators:
                if indicator in code_lower:
                    indicators_found.append(indicator)
                    
            if indicators_found:
                # Find the line number of the first indicator
                lines = code.split('\n')
                line_number = 1
                for i, line in enumerate(lines):
                    if any(ind in line.lower() for ind in indicators_found):
                        line_number = i + 1
                        break
                        
                # Create annotation
                evidence = mapping.evidence_template.format(
                    method=", ".join(indicators_found),
                    requirements="complex passwords required",
                    duration="30 minutes",
                    roles="admin, user, guest",
                    resource="API endpoints",
                    version="1.3",
                    bits="256",
                    events="authentication, authorization, data access",
                    data_types="user input, API parameters"
                )
                
                annotations.append(CodeAnnotation(
                    file_path=file_path,
                    line_number=line_number,
                    control_ids=mapping.control_ids,
                    evidence=evidence,
                    component=pattern_name.replace("_", "-"),
                    confidence=mapping.confidence
                ))
                
        return annotations
    
    def _merge_annotations(self, annotations: List[CodeAnnotation]) -> List[CodeAnnotation]:
        """Merge and deduplicate annotations"""
        # Group by file and line
        grouped = defaultdict(list)
        for ann in annotations:
            key = (ann.file_path, ann.line_number)
            grouped[key].append(ann)
            
        # Merge annotations at the same location
        merged = []
        for (file_path, line_number), group in grouped.items():
            # Combine control IDs
            all_controls = set()
            evidence_parts = []
            components = set()
            max_confidence = 0.0
            
            for ann in group:
                all_controls.update(ann.control_ids)
                if ann.evidence:
                    evidence_parts.append(ann.evidence)
                if ann.component:
                    components.add(ann.component)
                max_confidence = max(max_confidence, ann.confidence)
                
            merged.append(CodeAnnotation(
                file_path=file_path,
                line_number=line_number,
                control_ids=sorted(list(all_controls)),
                evidence="; ".join(evidence_parts) if evidence_parts else None,
                component=", ".join(sorted(components)) if components else None,
                confidence=max_confidence
            ))
            
        return sorted(merged, key=lambda x: (x.file_path, x.line_number))
```

### 1.2 Code Analysis Engine

Implement multi-language code analysis following [TESTING_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/TESTING_STANDARDS.md):

```python
# src/analyzers/base.py
"""
Base code analyzer with language-specific implementations
@nist-controls: SA-11, SA-15, CA-7
@evidence: Continuous code analysis and security testing
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import ast
import re
from dataclasses import dataclass

from tree_sitter import Node

from ..core.compliance.nist_mapper import CodeAnnotation, NISTControlMapper

@dataclass
class SecurityPattern:
    """Represents a security pattern in code"""
    pattern_type: str
    location: str
    line_number: int
    confidence: float
    details: Dict[str, Any]
    suggested_controls: List[str]

class BaseAnalyzer(ABC):
    """
    Abstract base class for language analyzers
    @nist-controls: PM-5, SA-11
    @evidence: Systematic security analysis across languages
    """
    
    def __init__(self, control_mapper: NISTControlMapper):
        self.control_mapper = control_mapper
        self.security_patterns: List[SecurityPattern] = []
        
    @abstractmethod
    def analyze_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze a single file for NIST controls"""
        pass
        
    @abstractmethod
    def analyze_project(self, project_path: Path) -> Dict[str, List[CodeAnnotation]]:
        """Analyze entire project"""
        pass
        
    @abstractmethod
    def suggest_controls(self, code: str) -> List[str]:
        """Suggest NIST controls for given code"""
        pass
        
    def find_security_patterns(self, code: str, file_path: str) -> List[SecurityPattern]:
        """Find common security patterns in code"""
        patterns = []
        
        # Authentication patterns
        if self._has_authentication(code):
            patterns.append(SecurityPattern(
                pattern_type="authentication",
                location=file_path,
                line_number=self._find_pattern_line(code, "auth"),
                confidence=0.8,
                details={"type": "basic"},
                suggested_controls=["IA-2", "IA-5", "AC-7"]
            ))
            
        # Encryption patterns
        if self._has_encryption(code):
            patterns.append(SecurityPattern(
                pattern_type="encryption",
                location=file_path,
                line_number=self._find_pattern_line(code, "encrypt"),
                confidence=0.9,
                details={"algorithms": self._find_crypto_algorithms(code)},
                suggested_controls=["SC-8", "SC-13", "SC-28"]
            ))
            
        # Access control patterns
        if self._has_access_control(code):
            patterns.append(SecurityPattern(
                pattern_type="access_control",
                location=file_path,
                line_number=self._find_pattern_line(code, "permission"),
                confidence=0.85,
                details={"type": "rbac"},
                suggested_controls=["AC-2", "AC-3", "AC-6"]
            ))
            
        # Logging patterns
        if self._has_logging(code):
            patterns.append(SecurityPattern(
                pattern_type="logging",
                location=file_path,
                line_number=self._find_pattern_line(code, "log"),
                confidence=0.9,
                details={"type": "security"},
                suggested_controls=["AU-2", "AU-3", "AU-12"]
            ))
            
        return patterns
    
    def _has_authentication(self, code: str) -> bool:
        """Check if code has authentication patterns"""
        auth_keywords = [
            "authenticate", "login", "signin", "auth",
            "credential", "password", "token", "jwt",
            "oauth", "saml", "ldap", "mfa", "2fa"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in auth_keywords)
    
    def _has_encryption(self, code: str) -> bool:
        """Check if code has encryption patterns"""
        crypto_keywords = [
            "encrypt", "decrypt", "cipher", "aes", "rsa",
            "sha", "hash", "crypto", "tls", "ssl", "https",
            "certificate", "key", "kms"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in crypto_keywords)
    
    def _has_access_control(self, code: str) -> bool:
        """Check if code has access control patterns"""
        ac_keywords = [
            "permission", "authorize", "role", "rbac", "abac",
            "policy", "grant", "deny", "access", "privilege",
            "can", "cannot", "allow", "forbidden"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in ac_keywords)
    
    def _has_logging(self, code: str) -> bool:
        """Check if code has logging patterns"""
        log_keywords = [
            "log", "audit", "trace", "monitor", "event",
            "track", "record", "journal", "syslog"
        ]
        code_lower = code.lower()
        return any(keyword in code_lower for keyword in log_keywords)
    
    def _find_pattern_line(self, code: str, pattern: str) -> int:
        """Find line number where pattern first appears"""
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if pattern.lower() in line.lower():
                return i + 1
        return 1
    
    def _find_crypto_algorithms(self, code: str) -> List[str]:
        """Find cryptographic algorithms mentioned in code"""
        algorithms = []
        crypto_patterns = {
            "AES": r"aes[-_]?\d{3}",
            "RSA": r"rsa[-_]?\d{4}",
            "SHA": r"sha[-_]?\d{3}",
            "ECDSA": r"ecdsa|ec[-_]?dsa",
            "HMAC": r"hmac",
            "PBKDF2": r"pbkdf2",
            "Bcrypt": r"bcrypt",
            "Argon2": r"argon2"
        }
        
        code_lower = code.lower()
        for name, pattern in crypto_patterns.items():
            if re.search(pattern, code_lower):
                algorithms.append(name)
                
        return algorithms

# src/analyzers/python_analyzer.py
"""
Python-specific code analyzer
@nist-controls: SA-11, SA-15
@evidence: Python static analysis for security controls
"""
import ast
from typing import List, Dict, Any
from pathlib import Path

from .base import BaseAnalyzer, SecurityPattern
from ..core.compliance.nist_mapper import CodeAnnotation

class PythonAnalyzer(BaseAnalyzer):
    """
    Analyzes Python code for NIST control implementations
    @nist-controls: SA-11, CA-7
    @evidence: Python-specific security analysis
    """
    
    def analyze_file(self, file_path: Path) -> List[CodeAnnotation]:
        """Analyze Python file for NIST controls"""
        with open(file_path, 'r') as f:
            code = f.read()
            
        # Use control mapper for basic analysis
        annotations = self.control_mapper.analyze_code(
            code, "python", str(file_path)
        )
        
        # Add Python-specific analysis
        try:
            tree = ast.parse(code)
            python_annotations = self._analyze_ast(tree, code, str(file_path))
            annotations.extend(python_annotations)
        except SyntaxError:
            # If we can't parse, fall back to pattern matching
            pass
            
        return annotations
    
    def _analyze_ast(self, tree: ast.AST, code: str, file_path: str) -> List[CodeAnnotation]:
        """Analyze Python AST for security patterns"""
        annotations = []
        
        for node in ast.walk(tree):
            # Check for authentication decorators
            if isinstance(node, ast.FunctionDef):
                for decorator in node.decorator_list:
                    if self._is_auth_decorator(decorator):
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=node.lineno,
                            control_ids=["IA-2", "AC-3"],
                            evidence=f"Authentication required for {node.name}",
                            component="authentication",
                            confidence=0.9
                        ))
                        
            # Check for encryption usage
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ['cryptography', 'Crypto', 'hashlib', 'secrets']:
                        annotations.append(CodeAnnotation(
                            file_path=file_path,
                            line_number=node.lineno,
                            control_ids=["SC-13"],
                            evidence=f"Cryptographic library imported: {alias.name}",
                            component="encryption",
                            confidence=0.8
                        ))
                        
            # Check for logging setup
            elif isinstance(node, ast.Call):
                if self._is_logging_call(node):
                    annotations.append(CodeAnnotation(
                        file_path=file_path,
                        line_number=node.lineno,
                        control_ids=["AU-2", "AU-3"],
                        evidence="Logging configured",
                        component="logging",
                        confidence=0.85
                    ))
                    
        return annotations
    
    def _is_auth_decorator(self, node: ast.AST) -> bool:
        """Check if decorator is authentication-related"""
        auth_decorators = [
            'login_required', 'authenticate', 'requires_auth',
            'authorized', 'permission_required', 'jwt_required'
        ]
        
        if isinstance(node, ast.Name):
            return node.id in auth_decorators
        elif isinstance(node, ast.Attribute):
            return node.attr in auth_decorators
            
        return False
    
    def _is_logging_call(self, node: ast.Call) -> bool:
        """Check if call is logging-related"""
        if isinstance(node.func, ast.Attribute):
            return node.func.attr in ['getLogger', 'basicConfig', 'info', 'warning', 'error']
        elif isinstance(node.func, ast.Name):
            return node.func.id in ['logging', 'logger']
            
        return False
    
    def analyze_project(self, project_path: Path) -> Dict[str, List[CodeAnnotation]]:
        """Analyze entire Python project"""
        results = {}
        
        for py_file in project_path.rglob("*.py"):
            # Skip virtual environments and cache
            if any(part in py_file.parts for part in ['venv', '__pycache__', '.env']):
                continue
                
            annotations = self.analyze_file(py_file)
            if annotations:
                results[str(py_file)] = annotations
                
        return results
    
    def suggest_controls(self, code: str) -> List[str]:
        """Suggest NIST controls for Python code"""
        suggestions = []
        patterns = self.find_security_patterns(code, "temp.py")
        
        for pattern in patterns:
            suggestions.extend(pattern.suggested_controls)
            
        # Python-specific suggestions
        if "django" in code.lower() or "flask" in code.lower():
            suggestions.extend(["AC-3", "AC-4", "SC-8"])  # Web framework controls
            
        if "boto3" in code.lower() or "azure" in code.lower():
            suggestions.extend(["AC-2", "AU-2", "SC-28"])  # Cloud SDK controls
            
        if "sqlalchemy" in code.lower() or "psycopg" in code.lower():
            suggestions.extend(["SC-28", "SI-10"])  # Database controls
            
        return list(set(suggestions))  # Remove duplicates
```

### 1.3 OSCAL Integration

Implement OSCAL support following federal standards:

```python
# src/core/compliance/oscal_handler.py
"""
OSCAL (Open Security Controls Assessment Language) Handler
@nist-controls: CA-2, CA-7, PM-31
@evidence: OSCAL-compliant documentation and assessment
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import json
import uuid
from dataclasses import dataclass, asdict

from pydantic import BaseModel, Field

@dataclass
class OSCALComponent:
    """OSCAL Component Definition"""
    uuid: str
    type: str
    title: str
    description: str
    props: List[Dict[str, str]]
    control_implementations: List[Dict[str, Any]]
    
@dataclass
class OSCALControlImplementation:
    """OSCAL Control Implementation"""
    uuid: str
    source: str
    description: str
    implemented_requirements: List[Dict[str, Any]]
    
class OSCALHandler:
    """
    Handles OSCAL format conversion and generation
    @nist-controls: CA-2, PM-31
    @evidence: Standardized security documentation
    """
    
    def __init__(self):
        self.components: Dict[str, OSCALComponent] = {}
        
    def create_component_from_annotations(
        self,
        component_name: str,
        annotations: List['CodeAnnotation'],
        metadata: Dict[str, Any]
    ) -> OSCALComponent:
        """
        Create OSCAL component from code annotations
        @nist-controls: CA-2, SA-4
        @evidence: Automated component documentation
        """
        component_uuid = str(uuid.uuid4())
        
        # Group annotations by control
        control_groups = {}
        for ann in annotations:
            for control_id in ann.control_ids:
                if control_id not in control_groups:
                    control_groups[control_id] = []
                control_groups[control_id].append(ann)
                
        # Create control implementations
        control_implementations = []
        
        for control_id, annotations_list in control_groups.items():
            implementation = {
                "uuid": str(uuid.uuid4()),
                "control-id": control_id,
                "description": self._generate_implementation_description(
                    control_id, annotations_list
                ),
                "evidence": [
                    {
                        "description": ann.evidence or "Implementation in code",
                        "link": {
                            "href": f"file://{ann.file_path}#L{ann.line_number}",
                            "text": f"{ann.file_path}:{ann.line_number}"
                        }
                    }
                    for ann in annotations_list
                ]
            }
            control_implementations.append(implementation)
            
        component = OSCALComponent(
            uuid=component_uuid,
            type="software",
            title=component_name,
            description=metadata.get("description", f"Component: {component_name}"),
            props=[
                {"name": "version", "value": metadata.get("version", "1.0.0")},
                {"name": "last-modified", "value": datetime.utcnow().isoformat()},
                {"name": "compliance-scan-date", "value": datetime.utcnow().isoformat()}
            ],
            control_implementations=control_implementations
        )
        
        self.components[component_uuid] = component
        return component
    
    def generate_ssp_content(
        self,
        system_name: str,
        components: List[OSCALComponent],
        profile: str = "NIST_SP-800-53_rev5_MODERATE"
    ) -> Dict[str, Any]:
        """
        Generate SSP (System Security Plan) content
        @nist-controls: CA-2, PM-31
        @evidence: Automated SSP generation
        """
        ssp = {
            "system-security-plan": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"System Security Plan for {system_name}",
                    "last-modified": datetime.utcnow().isoformat(),
                    "version": "1.0.0",
                    "oscal-version": "1.0.0"
                },
                "import-profile": {
                    "href": f"#{profile}"
                },
                "system-characteristics": {
                    "system-name": system_name,
                    "description": f"Automated SSP for {system_name}",
                    "security-sensitivity-level": "moderate",
                    "system-information": {
                        "information-types": [
                            {
                                "title": "System Data",
                                "categorizations": [
                                    {
                                        "system": "https://doi.org/10.6028/NIST.SP.800-60v2r1",
                                        "information-type-ids": ["C.3.5.8"]
                                    }
                                ]
                            }
                        ]
                    },
                    "security-impact-level": {
                        "security-objective-confidentiality": "moderate",
                        "security-objective-integrity": "moderate",
                        "security-objective-availability": "moderate"
                    },
                    "status": {
                        "state": "operational"
                    },
                    "authorization-boundary": {
                        "description": "System authorization boundary"
                    }
                },
                "system-implementation": {
                    "components": [
                        self._component_to_oscal_format(comp)
                        for comp in components
                    ]
                },
                "control-implementation": {
                    "description": "Control implementation for the system",
                    "implemented-requirements": self._merge_control_implementations(components)
                }
            }
        }
        
        return ssp
    
    def _generate_implementation_description(
        self,
        control_id: str,
        annotations: List['CodeAnnotation']
    ) -> str:
        """Generate implementation description from annotations"""
        descriptions = []
        
        # Collect unique evidence statements
        evidence_statements = set()
        for ann in annotations:
            if ann.evidence:
                evidence_statements.add(ann.evidence)
                
        if evidence_statements:
            descriptions.append(
                f"Control {control_id} is implemented through: " +
                "; ".join(sorted(evidence_statements))
            )
        else:
            descriptions.append(
                f"Control {control_id} is implemented in the codebase"
            )
            
        # Add file references
        files = sorted(set(ann.file_path for ann in annotations))
        if files:
            descriptions.append(
                f"Implementation found in: {', '.join(files[:5])}" +
                (" and others" if len(files) > 5 else "")
            )
            
        return " ".join(descriptions)
    
    def _component_to_oscal_format(self, component: OSCALComponent) -> Dict[str, Any]:
        """Convert component to OSCAL format"""
        return {
            "uuid": component.uuid,
            "type": component.type,
            "title": component.title,
            "description": component.description,
            "props": component.props,
            "status": {
                "state": "operational"
            }
        }
    
    def _merge_control_implementations(
        self,
        components: List[OSCALComponent]
    ) -> List[Dict[str, Any]]:
        """Merge control implementations from all components"""
        merged = {}
        
        for component in components:
            for impl in component.control_implementations:
                control_id = impl["control-id"]
                
                if control_id not in merged:
                    merged[control_id] = {
                        "uuid": str(uuid.uuid4()),
                        "control-id": control_id,
                        "description": "",
                        "by-components": []
                    }
                    
                merged[control_id]["by-components"].append({
                    "component-uuid": component.uuid,
                    "uuid": impl["uuid"],
                    "description": impl["description"],
                    "evidence": impl.get("evidence", [])
                })
                
        return list(merged.values())
    
    def export_to_file(self, ssp: Dict[str, Any], output_path: Path):
        """
        Export SSP to OSCAL JSON file
        @nist-controls: AU-10, SI-7
        @evidence: Integrity-protected export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(ssp, f, indent=2)
            
        # Generate checksum for integrity
        import hashlib
        with open(output_path, 'rb') as f:
            checksum = hashlib.sha256(f.read()).hexdigest()
            
        checksum_path = output_path.with_suffix('.sha256')
        checksum_path.write_text(f"{checksum}  {output_path.name}\n")
        
        return output_path, checksum_path
```

## Phase 2: CLI and Integration Tools (Weeks 5-6)

### 2.1 CLI Implementation

Create a powerful CLI following [DEVOPS_PLATFORM_STANDARDS.md](https://github.com/williamzujkowski/standards/blob/main/DEVOPS_PLATFORM_STANDARDS.md):

```python
# src/cli/main.py
"""
MCP Standards CLI - Command-line interface
@nist-controls: AC-3, AU-2, SI-10
@evidence: Secure CLI with audit logging
"""
import typer
from pathlib import Path
from typing import Optional, List
import json
import yaml
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
import asyncio

from ..core.mcp.server import MCPServer
from ..core.standards.engine import StandardsEngine
from ..core.compliance.nist_mapper import NISTControlMapper
from ..analyzers.python_analyzer import PythonAnalyzer
from ..analyzers.javascript_analyzer import JavaScriptAnalyzer
from ..analyzers.go_analyzer import GoAnalyzer
from ..core.compliance.oscal_handler import OSCALHandler

app = typer.Typer(
    name="mcp-standards",
    help="MCP Standards Server - NIST compliance for modern development",
    add_completion=False
)
console = Console()

# Configuration
CONFIG_FILE = Path.home() / ".mcp-standards" / "config.yaml"

@app.command()
def init(
    project_path: Path = typer.Argument(Path.cwd(), help="Project path to initialize"),
    profile: str = typer.Option("moderate", help="NIST profile (low/moderate/high)"),
    language: str = typer.Option(None, help="Primary language (auto-detect if not specified)")
):
    """
    Initialize MCP standards for a project
    @nist-controls: CM-2, CM-3
    @evidence: Configuration management
    """
    console.print(f"[bold green]Initializing MCP Standards for {project_path}[/bold green]")
    
    # Create config directory
    config_dir = project_path / ".mcp-standards"
    config_dir.mkdir(exist_ok=True)
    
    # Detect language if not specified
    if not language:
        language = _detect_language(project_path)
        console.print(f"Detected language: [cyan]{language}[/cyan]")
    
    # Create project config
    config = {
        "version": "1.0.0",
        "profile": profile,
        "language": language,
        "standards": {
            "enabled": _get_default_standards(language),
            "custom": []
        },
        "compliance": {
            "frameworks": ["NIST-800-53r5"],
            "scan_exclude": ["node_modules", "venv", ".git", "dist", "build"]
        }
    }
    
    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create hooks
    _setup_git_hooks(project_path)
    
    # Create VS Code settings
    _setup_vscode(project_path)
    
    console.print("[bold green]✓[/bold green] Project initialized successfully!")
    console.print(f"Configuration saved to: [dim]{config_file}[/dim]")
    console.print("\nNext steps:")
    console.print("  1. Run [cyan]mcp-standards scan[/cyan] to analyze existing code")
    console.print("  2. Run [cyan]mcp-standards server[/cyan] to start the MCP server")
    console.print("  3. Check [cyan].mcp-standards/config.yaml[/cyan] to customize settings")

@app.command()
def scan(
    path: Path = typer.Argument(Path.cwd(), help="Path to scan"),
    output_format: str = typer.Option("table", help="Output format (table/json/yaml/oscal)"),
    output_file: Optional[Path] = typer.Option(None, help="Output file (stdout if not specified)"),
    deep: bool = typer.Option(False, help="Perform deep analysis"),
    fix: bool = typer.Option(False, help="Apply automatic fixes where possible")
):
    """
    Scan codebase for NIST control implementations
    @nist-controls: CA-7, RA-5, SA-11
    @evidence: Continuous monitoring and vulnerability scanning
    """
    console.print(f"[bold]Scanning {path} for NIST controls...[/bold]")
    
    # Load configuration
    config = _load_config(path)
    
    # Initialize components
    control_mapper = NISTControlMapper(Path(__file__).parent.parent / "data" / "nist")
    
    # Select appropriate analyzer
    analyzers = {
        "python": PythonAnalyzer(control_mapper),
        "javascript": JavaScriptAnalyzer(control_mapper),
        "typescript": JavaScriptAnalyzer(control_mapper),
        "go": GoAnalyzer(control_mapper)
    }
    
    # Scan files
    all_annotations = {}
    file_count = 0
    control_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Scanning...", total=None)
        
        for language, analyzer in analyzers.items():
            if language == config.get("language") or deep:
                results = analyzer.analyze_project(path)
                all_annotations.update(results)
                file_count += len(results)
                control_count += sum(len(anns) for anns in results.values())
                progress.update(task, description=f"Scanned {file_count} files...")
    
    # Apply fixes if requested
    if fix and all_annotations:
        _apply_fixes(all_annotations)
        console.print("[bold green]✓[/bold green] Applied automatic fixes")
    
    # Display results
    if output_format == "table":
        _display_table_results(all_annotations)
    elif output_format == "json":
        _output_json_results(all_annotations, output_file)
    elif output_format == "yaml":
        _output_yaml_results(all_annotations, output_file)
    elif output_format == "oscal":
        _output_oscal_results(all_annotations, path, output_file)
    
    # Summary
    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Files scanned: [cyan]{file_count}[/cyan]")
    console.print(f"  Controls found: [cyan]{control_count}[/cyan]")
    console.print(f"  Unique controls: [cyan]{len(_get_unique_controls(all_annotations))}[/cyan]")
    
    # Recommendations
    if control_count == 0:
        console.print("\n[yellow]No NIST controls found![/yellow]")
        console.print("Consider:")
        console.print("  - Adding @nist-controls annotations to security-relevant code")
        console.print("  - Running with --deep for pattern-based detection")
        console.print("  - Checking the language detection in .mcp-standards/config.yaml")

@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development")
):
    """
    Start the MCP Standards Server
    @nist-controls: AC-3, SC-8, AU-2
    @evidence: Secure server with access control and encryption
    """
    console.print(f"[bold green]Starting MCP Standards Server[/bold green]")
    console.print(f"Host: [cyan]{host}:{port}[/cyan]")
    
    # Load configuration
    config = {
        "host": host,
        "port": port,
        "reload": reload,
        "ssl_cert": None,  # TODO: Add SSL support
        "ssl_key": None
    }
    
    # Start server
    import uvicorn
    uvicorn.run(
        "src.core.mcp.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

@app.command()
def generate(
    template: str = typer.Argument(..., help="Template name (e.g., 'api-endpoint', 'auth-module')"),
    output: Path = typer.Option(Path.cwd(), help="Output directory"),
    language: str = typer.Option(None, help="Target language"),
    controls: Optional[str] = typer.Option(None, help="Required NIST controls (comma-separated)")
):
    """
    Generate code from NIST-compliant templates
    @nist-controls: SA-3, SA-4, SA-8
    @evidence: Secure development with approved templates
    """
    console.print(f"[bold]Generating {template} template[/bold]")
    
    # Load template
    template_path = Path(__file__).parent.parent / "templates" / template
    if not template_path.exists():
        console.print(f"[red]Template '{template}' not found![/red]")
        console.print("Available templates:")
        _list_templates()
        return
    
    # Parse required controls
    required_controls = []
    if controls:
        required_controls = [c.strip() for c in controls.split(',')]
    
    # Generate from template
    generated_files = _generate_from_template(
        template_path, output, language, required_controls
    )
    
    console.print(f"[bold green]✓[/bold green] Generated {len(generated_files)} files:")
    for file in generated_files:
        console.print(f"  - {file}")

@app.command()
def validate(
    file: Path = typer.Argument(..., help="File to validate"),
    standard: str = typer.Option("all", help="Standard to validate against"),
    fix: bool = typer.Option(False, help="Apply fixes automatically")
):
    """
    Validate a file against standards
    @nist-controls: SA-11, SI-10
    @evidence: Input validation and code review
    """
    console.print(f"[bold]Validating {file}[/bold]")
    
    # Load standards engine
    standards_engine = StandardsEngine(
        Path(__file__).parent.parent / "standards"
    )
    
    # Load file content
    content = file.read_text()
    language = _detect_file_language(file)
    
    # Validate
    violations = []
    
    if standard == "all" or standard == "nist":
        # Check for NIST annotations
        if not "@nist-controls:" in content:
            violations.append({
                "type": "missing_nist_annotation",
                "message": "No NIST control annotations found",
                "severity": "warning"
            })
    
    # Display results
    if violations:
        console.print(f"[yellow]Found {len(violations)} violations:[/yellow]")
        for v in violations:
            console.print(f"  [{v['severity']}] {v['message']}")
        
        if fix:
            _apply_file_fixes(file, violations)
            console.print("[bold green]✓[/bold green] Applied fixes")
    else:
        console.print("[bold green]✓[/bold green] File is compliant!")

@app.command()
def ssp(
    output: Path = typer.Option(Path("ssp.json"), help="Output file for SSP"),
    format: str = typer.Option("oscal", help="Output format (oscal/docx/pdf)"),
    profile: str = typer.Option("moderate", help="NIST profile")
):
    """
    Generate System Security Plan (SSP) from code
    @nist-controls: CA-2, PM-31
    @evidence: Automated SSP generation
    """
    console.print("[bold]Generating System Security Plan...[/bold]")
    
    # Scan codebase first
    path = Path.cwd()
    config = _load_config(path)
    
    # Initialize components
    control_mapper = NISTControlMapper(Path(__file__).parent.parent / "data" / "nist")
    oscal_handler = OSCALHandler()
    
    # Analyze code
    all_annotations = {}
    analyzers = {
        "python": PythonAnalyzer(control_mapper),
        "javascript": JavaScriptAnalyzer(control_mapper),
        "go": GoAnalyzer(control_mapper)
    }
    
    for language, analyzer in analyzers.items():
        if language == config.get("language"):
            results = analyzer.analyze_project(path)
            all_annotations.update(results)
    
    # Create OSCAL components
    components = []
    for file_path, annotations in all_annotations.items():
        component_name = Path(file_path).stem
        component = oscal_handler.create_component_from_annotations(
            component_name, annotations, {"version": "1.0.0"}
        )
        components.append(component)
    
    # Generate SSP
    system_name = path.name
    ssp_content = oscal_handler.generate_ssp_content(
        system_name, components, f"NIST_SP-800-53_rev5_{profile.upper()}"
    )
    
    # Export
    if format == "oscal":
        oscal_handler.export_to_file(ssp_content, output)
        console.print(f"[bold green]✓[/bold green] SSP generated: {output}")
    else:
        console.print(f"[yellow]Format '{format}' not yet implemented[/yellow]")

# Helper functions
def _detect_language(path: Path) -> str:
    """Detect primary language of project"""
    extensions = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".java": "java"
    }
    
    counts = {}
    for ext, lang in extensions.items():
        count = len(list(path.rglob(f"*{ext}")))
        if count > 0:
            counts[lang] = count
    
    if counts:
        return max(counts, key=counts.get)
    return "python"  # Default

def _load_config(path: Path) -> dict:
    """Load project configuration"""
    config_file = path / ".mcp-standards" / "config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            return yaml.safe_load(f)
    return {}

def _display_table_results(annotations: Dict[str, List]):
    """Display scan results in a table"""
    table = Table(title="NIST Control Scan Results")
    table.add_column("File", style="cyan")
    table.add_column("Line", justify="right")
    table.add_column("Controls", style="green")
    table.add_column("Evidence", style="dim")
    
    for file_path, file_annotations in annotations.items():
        for ann in file_annotations:
            table.add_row(
                Path(file_path).name,
                str(ann.line_number),
                ", ".join(ann.control_ids),
                ann.evidence or ""
            )
    
    console.print(table)

def _get_unique_controls(annotations: Dict[str, List]) -> set:
    """Get unique control IDs from annotations"""
    controls = set()
    for file_annotations in annotations.values():
        for ann in file_annotations:
            controls.update(ann.control_ids)
    return controls

if __name__ == "__main__":
    app()