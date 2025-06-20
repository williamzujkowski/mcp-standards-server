# Enhanced NIST Control Detection

## Overview

The MCP Standards Server now includes comprehensive detection for all 20 NIST 800-53 rev5 control families, with over 200 specific control patterns that can be automatically identified in code.

## Control Families Covered

### 1. Access Control (AC)
- **AC-2**: Account Management - Detects user account creation, modification, and deletion
- **AC-3**: Access Enforcement - Identifies RBAC, ABAC, and permission checks
- **AC-6**: Least Privilege - Finds implementations of minimal permissions
- **AC-7**: Unsuccessful Login Attempts - Detects account lockout mechanisms
- **AC-10**: Concurrent Session Control - Identifies session limiting
- **AC-12**: Session Termination - Finds automatic logout/timeout
- **AC-17**: Remote Access - Detects VPN, SSH, RDP configurations
- **AC-18**: Wireless Access - Identifies wireless security controls
- **AC-19**: Mobile Device Access - Finds MDM and BYOD policies
- **AC-23**: Data Mining Protection - Detects privacy filters

### 2. Audit and Accountability (AU)
- **AU-2**: Auditable Events - Identifies security event logging
- **AU-3**: Content of Audit Records - Detects comprehensive logging (who, what, when, where, outcome)
- **AU-4**: Audit Storage Capacity - Finds log rotation and archival
- **AU-5**: Response to Audit Processing Failures - Detects alert mechanisms
- **AU-6**: Audit Review, Analysis, and Reporting - Identifies SIEM integration
- **AU-8**: Time Stamps - Finds NTP and time synchronization
- **AU-9**: Protection of Audit Information - Detects tamper-proof logging
- **AU-11**: Audit Record Retention - Identifies retention policies
- **AU-12**: Audit Generation - Finds audit log generation

### 3. Configuration Management (CM)
- **CM-2**: Baseline Configuration - Detects configuration baselines
- **CM-3**: Configuration Change Control - Identifies change management
- **CM-4**: Security Impact Analysis - Finds impact assessment
- **CM-6**: Configuration Settings - Detects security hardening
- **CM-7**: Least Functionality - Identifies software whitelisting/blacklisting
- **CM-8**: Information System Component Inventory - Finds asset management
- **CM-11**: User-Installed Software - Detects installation restrictions

### 4. Contingency Planning (CP)
- **CP-2**: Contingency Plan - Identifies disaster recovery planning
- **CP-3**: Contingency Training - Detects DR training references
- **CP-4**: Contingency Plan Testing - Finds failover testing
- **CP-7**: Alternate Processing Site - Detects multi-site configurations
- **CP-8**: Telecommunications Services - Identifies redundant communications
- **CP-9**: Information System Backup - Finds backup implementations
- **CP-10**: Information System Recovery - Detects restore capabilities

### 5. Identification and Authentication (IA)
- **IA-2**: User Identification and Authentication - Basic authentication
- **IA-2(1)**: Multi-factor Authentication - Detects MFA/2FA implementations
- **IA-2(2)**: Network Access MFA - Finds network-level MFA
- **IA-2(5)**: Group Authentication - Identifies biometric authentication
- **IA-3**: Device Identification - Detects device certificates
- **IA-5**: Authenticator Management - Finds password policies
- **IA-5(1)**: Password-based Authentication - Detects password complexity
- **IA-5(2)**: PKI-based Authentication - Identifies certificate auth
- **IA-8**: Non-organizational User Authentication - Finds federated identity
- **IA-11**: Re-authentication - Detects session timeout/re-auth

### 6. Incident Response (IR)
- **IR-3**: Incident Response Testing - Identifies tabletop exercises
- **IR-4**: Incident Handling - Detects incident ticket creation
- **IR-4(1)**: Automated Incident Handling - Finds SOC integration
- **IR-4(4)**: Incident Correlation - Identifies forensic collection
- **IR-5**: Incident Monitoring - Detects incident tracking
- **IR-6**: Incident Reporting - Finds reporting mechanisms
- **IR-9**: Information Spillage Response - Identifies spillage handling

### 7. Maintenance (MA)
- **MA-2**: Controlled Maintenance - Detects maintenance windows
- **MA-2(1)**: Record Content - Finds maintenance logging
- **MA-4**: Nonlocal Maintenance - Identifies remote maintenance
- **MA-5**: Maintenance Personnel - Detects authorized maintenance

### 8. Media Protection (MP)
- **MP-2**: Media Access - Identifies removable media controls
- **MP-3**: Media Marking - Detects classification labels
- **MP-4**: Media Storage - Finds secure storage references
- **MP-5**: Media Transport - Identifies secure transport
- **MP-6**: Media Sanitization - Detects secure wipe/destruction
- **MP-7**: Media Use - Finds USB control policies

### 9. Physical and Environmental Protection (PE)
- **PE-2**: Physical Access Authorizations - Badge readers, access cards
- **PE-3**: Physical Access Control - Identifies turnstiles, barriers
- **PE-6**: Monitoring Physical Access - Detects CCTV, surveillance
- **PE-17**: Alternate Work Site - Finds datacenter security

### 10. Personnel Security (PS)
- **PS-3**: Personnel Screening - Background checks
- **PS-4**: Personnel Termination - Offboarding processes
- **PS-5**: Personnel Transfer - Role change procedures
- **PS-6**: Access Agreements - NDAs, confidentiality

### 11. Risk Assessment (RA)
- **RA-3**: Risk Assessment - Threat modeling
- **RA-5**: Vulnerability Scanning - Security scans
- **RA-5(1)**: Update Tool Capability - Scan tool updates
- **RA-8**: Privacy Impact Assessment - PIA references
- **RA-9**: Criticality Analysis - BCP/BIA

### 12. System and Communications Protection (SC)
- **SC-5**: Denial of Service Protection - Rate limiting, DDoS protection
- **SC-7**: Boundary Protection - Firewalls, DMZ, perimeter security
- **SC-8**: Transmission Confidentiality - TLS/SSL, HTTPS, VPN
- **SC-10**: Network Disconnect - Session termination
- **SC-12**: Cryptographic Key Management - HSM, key management
- **SC-13**: Cryptographic Protection - Encryption algorithms
- **SC-20/21/22**: Secure DNS - DNSSEC, DoH
- **SC-23**: Session Authenticity - Session tokens
- **SC-24**: Fail in Known State - Graceful degradation
- **SC-26**: Honeypots - Deception technology
- **SC-27**: Platform Verification - Secure boot
- **SC-28**: Protection at Rest - Disk/database encryption
- **SC-32**: Network Partitioning - VLANs, segmentation
- **SC-39**: Process Isolation - Sandboxing, containers

### 13. System and Information Integrity (SI)
- **SI-2**: Flaw Remediation - Patch management
- **SI-3**: Malicious Code Protection - Antivirus/antimalware
- **SI-4**: Information System Monitoring - IDS/IPS
- **SI-5**: Security Alerts - SIEM integration
- **SI-7**: Software Integrity - File integrity monitoring
- **SI-8**: Spam Protection - Email filtering
- **SI-10**: Information Input Validation - Input sanitization
- **SI-11**: Error Handling - Secure error messages
- **SI-12**: Information Management - Data retention
- **SI-13**: Predictive Failure Analysis - Health monitoring
- **SI-14**: Non-persistence - Ephemeral systems
- **SI-15**: Information Output Filtering - XSS protection
- **SI-16**: Memory Protection - ASLR, DEP, stack canaries

### 14. Supply Chain Risk Management (SR)
- **SR-1/2/3**: Supply Chain Risk Management Policy
- **SR-4**: Supply Chain Inventory - SBOM generation
- **SR-6**: Supplier Assessments - Vendor audits
- **SR-9/10**: Tamper Resistance - Integrity seals
- **SR-11**: Component Authenticity - Counterfeit prevention

### 15. Privacy (PT)
- **PT-1**: Privacy Notice - Privacy policies
- **PT-2**: Privacy Consent - Opt-in/opt-out
- **PT-3**: Data Minimization - Minimal collection
- **PT-4/5**: PII Processing - Personal data handling
- **PT-7**: Privacy by Design - Privacy engineering

## Pattern Detection Methods

### 1. Explicit Annotations
```python
# @nist-controls: AC-3, AU-2
# @evidence: Role-based access control with audit logging
@require_permission('admin')
def sensitive_operation():
    audit_log.info("Admin operation performed")
```

### 2. Decorator Pattern Detection
The system recognizes common security decorators:
- `@login_required`, `@authenticated` → IA-2
- `@require_permission`, `@roles_allowed` → AC-3
- `@ratelimit`, `@throttle` → SC-5
- `@cache`, `@memoize` → SC-5(2)

### 3. Function Name Analysis
Security-related function names are automatically detected:
- `validate_*`, `sanitize_*` → SI-10
- `encrypt_*`, `decrypt_*` → SC-13
- `backup_*`, `restore_*` → CP-9
- `audit_*`, `log_*` → AU-2

### 4. Import Statement Analysis
Security library imports are recognized:
- Cryptography libraries → SC-13, SC-28
- Authentication libraries (JWT, OAuth) → IA-2, IA-8
- Logging frameworks → AU-2, AU-3
- Security tools (Bandit, Safety) → SA-11, RA-5

### 5. Code Pattern Matching
Complex patterns using regex:
- Session timeout configuration → AC-12, IA-11
- Rate limiting implementation → SC-5
- Input validation and parameterized queries → SI-10
- Error handling without info disclosure → SI-11

### 6. AST-Based Analysis
Deep code analysis using Abstract Syntax Trees:
- Class inheritance patterns
- Security configuration detection
- Hardcoded credential warnings
- Control flow analysis

## Using the Coverage Command

### Basic Usage
```bash
# Analyze current directory
mcp-standards coverage

# Analyze specific directory
mcp-standards coverage /path/to/project

# Generate JSON report
mcp-standards coverage --output-format json --output-file coverage.json

# Generate HTML report
mcp-standards coverage --output-format html --output-file coverage.html
```

### Understanding the Report

The coverage report includes:

1. **Executive Summary**
   - Total unique controls implemented
   - File coverage statistics
   - Overall compliance posture

2. **Control Family Coverage**
   - Percentage of controls implemented per family
   - Visual status indicators
   - Gap analysis

3. **High Confidence Controls**
   - Controls with explicit implementation
   - Controls with high-confidence pattern matches

4. **Suggested Additional Controls**
   - Related controls based on what's implemented
   - Helps achieve comprehensive coverage

5. **Detailed Control List**
   - All implemented controls by family
   - Brief descriptions of each control

## Best Practices

### 1. Use Explicit Annotations
Always prefer explicit `@nist-controls` annotations for critical security functions:
```python
# @nist-controls: AC-3, AU-2, SI-10
# @evidence: Input validation with RBAC and audit logging
# @oscal-component: api-gateway
def process_user_request(user, request_data):
    validate_input(request_data)  # SI-10
    check_permissions(user)       # AC-3
    audit_log.info(f"Request processed for {user}")  # AU-2
```

### 2. Implement Control Families Together
Related controls often work together:
- Authentication (IA-2) + Password Policy (IA-5) + Account Lockout (AC-7)
- Encryption in Transit (SC-8) + Encryption at Rest (SC-28) + Key Management (SC-12)
- Input Validation (SI-10) + Error Handling (SI-11) + Output Filtering (SI-15)

### 3. Focus on High-Impact Controls
Prioritize controls marked as "high severity":
- Multi-factor Authentication (IA-2(1))
- Input Validation (SI-10)
- Encryption (SC-8, SC-13, SC-28)
- Patch Management (SI-2)
- Media Sanitization (MP-6)

### 4. Regular Coverage Reviews
Run coverage reports regularly:
```bash
# Add to CI/CD pipeline
mcp-standards coverage --output-format json --output-file coverage.json
if [ $(jq '.summary.total_controls' coverage.json) -lt 50 ]; then
    echo "Insufficient control coverage"
    exit 1
fi
```

## Control Relationships

The system understands control dependencies and will suggest related controls:

- **AC-2** (Account Management) suggests:
  - AC-3 (Access Enforcement)
  - AC-6 (Least Privilege)
  - AU-2 (Audit Events)

- **AU-2** (Audit Events) suggests:
  - AU-3 (Content of Audit Records)
  - AU-4 (Audit Storage Capacity)
  - AU-12 (Audit Generation)

- **SC-8** (Transmission Confidentiality) suggests:
  - SC-13 (Cryptographic Protection)
  - SC-28 (Protection at Rest)

## Integration with Development Workflow

### 1. Pre-commit Hook
```bash
#!/bin/bash
# .git/hooks/pre-commit
coverage=$(mcp-standards coverage --output-format json | jq '.summary.total_controls')
if [ $coverage -lt 30 ]; then
    echo "Error: Insufficient NIST control coverage ($coverage controls)"
    echo "Run 'mcp-standards coverage' for details"
    exit 1
fi
```

### 2. VS Code Integration
Add to `.vscode/tasks.json`:
```json
{
    "label": "Check NIST Coverage",
    "type": "shell",
    "command": "mcp-standards coverage",
    "presentation": {
        "reveal": "always",
        "panel": "new"
    }
}
```

### 3. CI/CD Pipeline
```yaml
# .github/workflows/compliance.yml
- name: Check NIST Control Coverage
  run: |
    mcp-standards coverage --output-format json --output-file coverage.json
    python -c "
    import json
    with open('coverage.json') as f:
        data = json.load(f)
    if data['summary']['total_controls'] < 40:
        raise SystemExit('Insufficient control coverage')
    "