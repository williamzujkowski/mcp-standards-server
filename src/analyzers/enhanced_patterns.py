"""
Enhanced NIST Control Pattern Detection
@nist-controls: SA-11, SA-15, CA-7, PM-5
@evidence: Comprehensive pattern detection for NIST 800-53 rev5 controls
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class ControlPattern:
    """Enhanced control pattern with more context"""
    pattern_regex: str
    control_ids: list[str]
    evidence_template: str
    confidence: float
    pattern_type: str
    severity: str = "medium"  # low, medium, high


class EnhancedNISTPatterns:
    """
    Enhanced NIST 800-53 rev5 control pattern detection
    Based on comprehensive analysis of all 20 control families
    """

    def __init__(self):
        self.patterns = self._initialize_patterns()

    def _initialize_patterns(self) -> dict[str, list[ControlPattern]]:
        """Initialize comprehensive pattern detection for all NIST families"""
        return {
            # Access Control (AC) Family - Enhanced
            "access_control": [
                ControlPattern(
                    r"@(RolesAllowed|PreAuthorize|Secured|RequiresPermission)",
                    ["AC-2", "AC-3", "AC-6"],
                    "Role-based access control via {match}",
                    0.95, "decorator"
                ),
                ControlPattern(
                    r"(checkPermission|hasRole|isAuthorized|canAccess)\s*\(",
                    ["AC-3", "AC-6"],
                    "Permission checking implementation",
                    0.90, "function_call"
                ),
                ControlPattern(
                    r"principle\s*of\s*least\s*privilege|least_privilege|minimal_permissions",
                    ["AC-6", "AC-6(1)", "CM-7"],
                    "Least privilege implementation",
                    0.85, "comment"
                ),
                ControlPattern(
                    r"concurrent\s*session|session_limit|max_sessions",
                    ["AC-10"],
                    "Concurrent session control",
                    0.88, "configuration"
                ),
                ControlPattern(
                    r"lock(out|_out)|account_lock|failed_attempts",
                    ["AC-7"],
                    "Account lockout mechanism",
                    0.92, "security_feature"
                ),
                ControlPattern(
                    r"remote_access|vpn_|ssh_config|rdp_",
                    ["AC-17", "AC-17(1)"],
                    "Remote access control",
                    0.85, "configuration"
                ),
                ControlPattern(
                    r"wireless_(security|auth)|wpa[23]|802\.1x",
                    ["AC-18", "AC-18(1)"],
                    "Wireless access control",
                    0.87, "configuration"
                ),
                ControlPattern(
                    r"mobile_device|mdm_|byod_policy",
                    ["AC-19", "AC-19(5)"],
                    "Mobile device access control",
                    0.85, "configuration"
                ),
                ControlPattern(
                    r"data_mining|information_sharing|privacy_filter",
                    ["AC-23"],
                    "Data mining and disclosure protection",
                    0.82, "privacy"
                ),
            ],

            # Audit and Accountability (AU) Family - Enhanced
            "audit_accountability": [
                ControlPattern(
                    r"audit_log|security_log|compliance_log",
                    ["AU-2", "AU-3", "AU-12"],
                    "Security audit logging",
                    0.92, "logging"
                ),
                ControlPattern(
                    r"log\.(info|warn|error|critical|security)",
                    ["AU-2", "AU-3"],
                    "Structured logging implementation",
                    0.88, "logging"
                ),
                ControlPattern(
                    r"(who|what|when|where|outcome)\s*(logging|logged)",
                    ["AU-3", "AU-3(1)"],
                    "Comprehensive audit record content",
                    0.90, "logging"
                ),
                ControlPattern(
                    r"log_retention|archive_logs|log_rotation",
                    ["AU-4", "AU-11"],
                    "Audit log storage and retention",
                    0.88, "configuration"
                ),
                ControlPattern(
                    r"alert|notification|alarm|security_event",
                    ["AU-5", "AU-5(1)", "AU-5(2)"],
                    "Audit processing failure alerts",
                    0.85, "monitoring"
                ),
                ControlPattern(
                    r"(tamper|integrity)[_-]?(proof|protection)|log[_-]?sign|sign[_-]?log",
                    ["AU-9", "AU-9(2)", "AU-9(3)"],
                    "Audit information protection",
                    0.92, "security_feature"
                ),
                ControlPattern(
                    r"time(stamp|_stamp)|ntp_|time_sync",
                    ["AU-8", "AU-8(1)"],
                    "Time stamp implementation",
                    0.90, "configuration"
                ),
                ControlPattern(
                    r"audit_reduction|log_analysis|siem_",
                    ["AU-6", "AU-6(1)", "AU-7"],
                    "Audit review and analysis tools",
                    0.85, "monitoring"
                ),
            ],

            # Configuration Management (CM) Family - Enhanced
            "configuration_management": [
                ControlPattern(
                    r"baseline_config|configuration_baseline|approved_config",
                    ["CM-2", "CM-2(2)", "CM-3"],
                    "Configuration baseline management",
                    0.88, "configuration"
                ),
                ControlPattern(
                    r"change_control|change_management|version_control",
                    ["CM-3", "CM-3(1)", "CM-4"],
                    "Configuration change control",
                    0.90, "process"
                ),
                ControlPattern(
                    r"security_impact|impact_analysis|change_review",
                    ["CM-4", "CM-4(1)"],
                    "Security impact analysis",
                    0.85, "process"
                ),
                ControlPattern(
                    r"(whitelist|allowlist|approved_software|software_restriction)",
                    ["CM-7", "CM-7(1)", "CM-7(2)"],
                    "Least functionality - software restrictions",
                    0.92, "security_feature"
                ),
                ControlPattern(
                    r"(blacklist|blocklist|prohibited_software|deny_list)",
                    ["CM-7(3)", "CM-7(4)"],
                    "Unauthorized software prevention",
                    0.90, "security_feature"
                ),
                ControlPattern(
                    r"inventory|asset_management|cmdb",
                    ["CM-8", "CM-8(1)", "CM-8(3)"],
                    "Information system component inventory",
                    0.85, "management"
                ),
                ControlPattern(
                    r"config_settings|security_settings|hardening",
                    ["CM-6", "CM-6(1)"],
                    "Configuration settings management",
                    0.88, "configuration"
                ),
                ControlPattern(
                    r"user_installed|software_installation|install_permission",
                    ["CM-11", "CM-11(1)"],
                    "User-installed software restrictions",
                    0.85, "security_feature"
                ),
            ],

            # Contingency Planning (CP) Family
            "contingency_planning": [
                ControlPattern(
                    r"backup|restore|recovery_point|rpo_|rto_",
                    ["CP-9", "CP-9(1)", "CP-10"],
                    "Information system backup",
                    0.90, "backup"
                ),
                ControlPattern(
                    r"disaster_recovery|dr_plan|business_continuity",
                    ["CP-2", "CP-2(1)", "CP-2(3)"],
                    "Contingency plan implementation",
                    0.85, "planning"
                ),
                ControlPattern(
                    r"alternate_(site|location|processing)|failover",
                    ["CP-7", "CP-7(1)", "CP-8"],
                    "Alternate processing site",
                    0.87, "infrastructure"
                ),
                ControlPattern(
                    r"contingency_training|dr_training|recovery_training",
                    ["CP-3"],
                    "Contingency training",
                    0.82, "training"
                ),
                ControlPattern(
                    r"contingency_test|dr_test|recovery_test|failover_test",
                    ["CP-4", "CP-4(1)"],
                    "Contingency plan testing",
                    0.85, "testing"
                ),
            ],

            # Identification and Authentication (IA) Family - Enhanced
            "identification_authentication": [
                ControlPattern(
                    r"multi[_-]?factor|mfa|2fa|totp|authenticator_app",
                    ["IA-2(1)", "IA-2(2)", "IA-2(6)", "IA-2(8)"],
                    "Multi-factor authentication implementation",
                    0.95, "authentication", "high"
                ),
                ControlPattern(
                    r"biometric|fingerprint|face_recognition|iris_scan",
                    ["IA-2(5)", "IA-3"],
                    "Biometric authentication",
                    0.90, "authentication"
                ),
                ControlPattern(
                    r"password_(policy|complexity|strength|requirements)",
                    ["IA-5", "IA-5(1)"],
                    "Password policy enforcement",
                    0.92, "authentication"
                ),
                ControlPattern(
                    r"password_(history|reuse)|previous_passwords",
                    ["IA-5(1)(e)"],
                    "Password history enforcement",
                    0.88, "authentication"
                ),
                ControlPattern(
                    r"certificate|pki|x509|cert_auth",
                    ["IA-5(2)", "IA-8(1)"],
                    "PKI-based authentication",
                    0.90, "authentication"
                ),
                ControlPattern(
                    r"device_(identifier|authentication|certificate)",
                    ["IA-3"],
                    "Device identification and authentication",
                    0.85, "authentication"
                ),
                ControlPattern(
                    r"session_timeout|idle_timeout|inactivity_logout|session_lifetime|permanent_session",
                    ["IA-11", "AC-12"],
                    "Re-authentication for sessions",
                    0.88, "session_management"
                ),
                ControlPattern(
                    r"federated|saml|oauth|openid|sso",
                    ["IA-8", "IA-8(2)", "IA-8(4)"],
                    "Federated identity management",
                    0.90, "authentication"
                ),
            ],

            # Incident Response (IR) Family
            "incident_response": [
                ControlPattern(
                    r"incident_(response|report|ticket)|security_incident",
                    ["IR-4", "IR-5", "IR-6"],
                    "Incident handling implementation",
                    0.85, "incident_management"
                ),
                ControlPattern(
                    r"incident_track|incident_monitor|soc_|security_operations",
                    ["IR-4(1)", "IR-5(1)"],
                    "Incident monitoring and tracking",
                    0.85, "monitoring"
                ),
                ControlPattern(
                    r"forensic|evidence_collection|chain_of_custody",
                    ["IR-4(4)", "AU-9(2)"],
                    "Forensic data collection",
                    0.87, "forensics"
                ),
                ControlPattern(
                    r"incident_test|tabletop|incident_simulation",
                    ["IR-3", "IR-3(2)"],
                    "Incident response testing",
                    0.82, "testing"
                ),
                ControlPattern(
                    r"spillage|data_spillage|information_spillage",
                    ["IR-9", "IR-9(1)"],
                    "Information spillage response",
                    0.85, "incident_management"
                ),
            ],

            # Maintenance (MA) Family
            "maintenance": [
                ControlPattern(
                    r"maintenance_window|scheduled_maintenance|maintenance_mode",
                    ["MA-2", "MA-2(2)"],
                    "Controlled maintenance implementation",
                    0.85, "maintenance"
                ),
                ControlPattern(
                    r"maintenance_log|maintenance_record|service_record",
                    ["MA-2(1)", "MA-5(1)"],
                    "Maintenance records",
                    0.85, "logging"
                ),
                ControlPattern(
                    r"remote_maintenance|remote_diagnostic|remote_admin",
                    ["MA-4", "MA-4(1)", "MA-4(3)"],
                    "Nonlocal maintenance controls",
                    0.87, "remote_access"
                ),
                ControlPattern(
                    r"maintenance_personnel|authorized_maintenance|service_provider",
                    ["MA-5", "MA-5(1)"],
                    "Maintenance personnel authorization",
                    0.82, "access_control"
                ),
            ],

            # Media Protection (MP) Family
            "media_protection": [
                ControlPattern(
                    r"media_access|removable_media|usb_control",
                    ["MP-2", "MP-7"],
                    "Media access restrictions",
                    0.85, "access_control"
                ),
                ControlPattern(
                    r"media_marking|classification_label|sensitivity_label",
                    ["MP-3"],
                    "Media marking for sensitivity",
                    0.82, "data_classification"
                ),
                ControlPattern(
                    r"media_storage|secure_storage|locked_cabinet",
                    ["MP-4"],
                    "Media storage controls",
                    0.82, "physical_security"
                ),
                ControlPattern(
                    r"media_transport|secure_transport|courier|encrypted_transport",
                    ["MP-5", "MP-5(4)"],
                    "Media transport security",
                    0.85, "data_protection"
                ),
                ControlPattern(
                    r"(media_sanitiz|secure_wipe|data_destruction|degauss|shred)",
                    ["MP-6", "MP-6(1)", "MP-6(2)"],
                    "Media sanitization and disposal",
                    0.90, "data_protection", "high"
                ),
            ],

            # Physical and Environmental Protection (PE) Family
            "physical_environmental": [
                ControlPattern(
                    r"physical_access|badge_reader|access_card|turnstile",
                    ["PE-2", "PE-3", "PE-3(1)"],
                    "Physical access controls",
                    0.85, "physical_security"
                ),
                ControlPattern(
                    r"visitor_log|visitor_escort|visitor_badge",
                    ["PE-2(1)", "PE-3(2)"],
                    "Visitor access controls",
                    0.85, "physical_security"
                ),
                ControlPattern(
                    r"cctv|surveillance|security_camera|motion_detect",
                    ["PE-6", "PE-6(1)"],
                    "Physical access monitoring",
                    0.87, "monitoring"
                ),
                ControlPattern(
                    r"datacenter|server_room|secure_area|restricted_area",
                    ["PE-17"],
                    "Alternate work site security",
                    0.82, "physical_security"
                ),
            ],

            # Personnel Security (PS) Family
            "personnel_security": [
                ControlPattern(
                    r"background_check|security_clearance|personnel_screening",
                    ["PS-3", "PS-3(1)"],
                    "Personnel screening",
                    0.85, "hr_security"
                ),
                ControlPattern(
                    r"termination_process|offboarding|revoke_access|disable_account",
                    ["PS-4", "PS-4(1)", "PS-4(2)"],
                    "Personnel termination procedures",
                    0.88, "hr_security"
                ),
                ControlPattern(
                    r"personnel_transfer|role_change|access_review",
                    ["PS-5"],
                    "Personnel transfer procedures",
                    0.82, "hr_security"
                ),
                ControlPattern(
                    r"nda|non_disclosure|confidentiality_agreement",
                    ["PS-6", "PS-6(1)"],
                    "Access agreements",
                    0.85, "hr_security"
                ),
            ],

            # Risk Assessment (RA) Family
            "risk_assessment": [
                ControlPattern(
                    r"risk_assessment|threat_model|risk_analysis",
                    ["RA-3", "RA-3(1)"],
                    "Risk assessment implementation",
                    0.85, "risk_management"
                ),
                ControlPattern(
                    r"vulnerability_scan|security_scan|vuln_assessment",
                    ["RA-5", "RA-5(1)", "RA-5(2)"],
                    "Vulnerability scanning",
                    0.90, "vulnerability_management", "high"
                ),
                ControlPattern(
                    r"threat_intelligence|threat_feed|ioc_|indicator",
                    ["RA-3(2)", "PM-16"],
                    "Threat intelligence integration",
                    0.85, "threat_management"
                ),
                ControlPattern(
                    r"privacy_impact|pia_|privacy_assessment",
                    ["RA-8"],
                    "Privacy impact assessment",
                    0.85, "privacy"
                ),
                ControlPattern(
                    r"criticality_analysis|bcp_|business_impact",
                    ["RA-9"],
                    "Criticality analysis",
                    0.82, "risk_management"
                ),
            ],

            # System and Communications Protection (SC) Family - Enhanced
            "system_communications": [
                ControlPattern(
                    r"boundary_protection|firewall|dmz|perimeter",
                    ["SC-7", "SC-7(3)", "SC-7(4)"],
                    "Boundary protection implementation",
                    0.90, "network_security"
                ),
                ControlPattern(
                    r"encrypt.*transit|tls|ssl|https|ipsec|vpn",
                    ["SC-8", "SC-8(1)", "SC-13"],
                    "Transmission confidentiality and integrity",
                    0.95, "encryption", "high"
                ),
                ControlPattern(
                    r"encrypt.*rest|disk_encrypt|database_encrypt|file_encrypt",
                    ["SC-28", "SC-28(1)"],
                    "Protection of information at rest",
                    0.92, "encryption", "high"
                ),
                ControlPattern(
                    r"network_segment|vlan|subnet|micro_segment",
                    ["SC-7(13)", "SC-32"],
                    "Network segmentation",
                    0.87, "network_security"
                ),
                ControlPattern(
                    r"honeypot|honeynet|deception",
                    ["SC-26", "SC-35"],
                    "Honeypots/deception technology",
                    0.85, "deception"
                ),
                ControlPattern(
                    r"covert_channel|side_channel|timing_attack",
                    ["SC-31", "SC-31(1)"],
                    "Covert channel analysis",
                    0.82, "advanced_security"
                ),
                ControlPattern(
                    r"secure_boot|trusted_boot|measured_boot",
                    ["SC-27"],
                    "Platform verification",
                    0.88, "boot_security"
                ),
                ControlPattern(
                    r"process_isolation|sandbox|container_isolation",
                    ["SC-39", "SC-2"],
                    "Process isolation",
                    0.90, "isolation"
                ),
                ControlPattern(
                    r"cryptographic_module|hsm|tpm|secure_enclave",
                    ["SC-13", "SC-12"],
                    "Cryptographic key management",
                    0.92, "encryption"
                ),
                ControlPattern(
                    r"session_lock|screen_lock|auto_lock",
                    ["SC-10"],
                    "Network disconnect/session termination",
                    0.85, "session_management"
                ),
                ControlPattern(
                    r"dos_protection|ddos_|rate_limit|throttl",
                    ["SC-5", "SC-5(1)", "SC-5(2)"],
                    "Denial of service protection",
                    0.88, "availability"
                ),
                ControlPattern(
                    r"secure_dns|dnssec|dns_over_https",
                    ["SC-20", "SC-21", "SC-22"],
                    "Secure name/address resolution",
                    0.85, "network_security"
                ),
                ControlPattern(
                    r"fail_secure|failover|graceful_degradation",
                    ["SC-24"],
                    "Fail in known state",
                    0.85, "resilience"
                ),
            ],

            # System and Information Integrity (SI) Family - Enhanced
            "system_information_integrity": [
                ControlPattern(
                    r"patch_management|update_management|patch_Tuesday",
                    ["SI-2", "SI-2(2)", "SI-2(3)"],
                    "Flaw remediation and patching",
                    0.90, "vulnerability_management", "high"
                ),
                ControlPattern(
                    r"antivirus|anti[_-]?malware|malware_scan|virus_scan",
                    ["SI-3", "SI-3(1)", "SI-3(2)"],
                    "Malicious code protection",
                    0.92, "malware_protection", "high"
                ),
                ControlPattern(
                    r"ids|ips|intrusion_detection|snort|suricata",
                    ["SI-4", "SI-4(1)", "SI-4(2)"],
                    "Information system monitoring - IDS/IPS",
                    0.90, "monitoring"
                ),
                ControlPattern(
                    r"security_alert|incident_alert|siem|security_event",
                    ["SI-4(5)", "SI-5"],
                    "Security alerts and advisories",
                    0.87, "monitoring"
                ),
                ControlPattern(
                    r"file_integrity|fim|tripwire|aide|checksum_verify",
                    ["SI-7", "SI-7(1)", "SI-7(6)"],
                    "Software and information integrity",
                    0.90, "integrity"
                ),
                ControlPattern(
                    r"input_validat|sanitiz|escape|parameteriz|prepared_statement",
                    ["SI-10", "SI-10(1)"],
                    "Information input validation",
                    0.95, "input_validation", "high"
                ),
                ControlPattern(
                    r"error[_\s]?handl|exception[_\s]?handl|error_message|stack_trace|except\s+\w+Exception",
                    ["SI-11"],
                    "Error handling",
                    0.88, "error_handling"
                ),
                ControlPattern(
                    r"memory_protect|aslr|dep|nx_bit|stack_canary",
                    ["SI-16"],
                    "Memory protection",
                    0.87, "memory_security"
                ),
                ControlPattern(
                    r"output_filter|content_filter|xss_filter|response_filter",
                    ["SI-15"],
                    "Information output filtering",
                    0.88, "output_validation"
                ),
                ControlPattern(
                    r"spam_filter|email_filter|phishing_detect",
                    ["SI-8", "SI-8(1)", "SI-8(2)"],
                    "Spam and phishing protection",
                    0.85, "email_security"
                ),
                ControlPattern(
                    r"(retention_policy|data_retention|archive_policy|purge_data)",
                    ["SI-12"],
                    "Information management and retention",
                    0.85, "data_management"
                ),
                ControlPattern(
                    r"predictive_failure|smart_monitor|health_check",
                    ["SI-13", "SI-13(1)"],
                    "Predictive failure analysis",
                    0.82, "monitoring"
                ),
                ControlPattern(
                    r"non_persistence|volatile|ephemeral|stateless",
                    ["SI-14", "SI-14(1)"],
                    "Non-persistence implementation",
                    0.85, "architecture"
                ),
            ],

            # Supply Chain Risk Management (SR) Family
            "supply_chain": [
                ControlPattern(
                    r"supply_chain|vendor_risk|third_party_risk",
                    ["SR-1", "SR-2", "SR-3"],
                    "Supply chain risk management",
                    0.85, "risk_management"
                ),
                ControlPattern(
                    r"software_bill|sbom|component_inventory",
                    ["SR-4", "SR-4(1)"],
                    "Supply chain inventory",
                    0.87, "asset_management"
                ),
                ControlPattern(
                    r"vendor[_\s]?(assessment|risk|audit)|supplier[_\s]?audit|third[_\s]?party[_\s]?(audit|risk)",
                    ["SR-6", "SR-6(1)"],
                    "Supplier assessments",
                    0.82, "vendor_management"
                ),
                ControlPattern(
                    r"tamper_evident|tamper_proof|integrity_seal",
                    ["SR-9", "SR-10"],
                    "Tamper resistance and detection",
                    0.85, "physical_security"
                ),
                ControlPattern(
                    r"component_authentic|genuine|counterfeit_prevent",
                    ["SR-11", "SR-11(1)"],
                    "Component authenticity",
                    0.85, "supply_chain_security"
                ),
            ],

            # Privacy Authorization (PT) Family
            "privacy": [
                ControlPattern(
                    r"privacy_notice|privacy_policy|data_collection_notice",
                    ["PT-1", "PT-2"],
                    "Privacy notices",
                    0.85, "privacy"
                ),
                ControlPattern(
                    r"consent|opt_in|opt_out|user_choice|privacy_preference",
                    ["PT-2", "PT-2(1)"],
                    "Privacy consent implementation",
                    0.88, "privacy"
                ),
                ControlPattern(
                    r"data_minimization|minimal_collection|necessary_data",
                    ["PT-3", "PT-3(1)"],
                    "Data minimization",
                    0.85, "privacy"
                ),
                ControlPattern(
                    r"privacy_engineering|privacy_by_design|pbd",
                    ["PT-7"],
                    "Privacy engineering",
                    0.82, "privacy"
                ),
                ControlPattern(
                    r"pii|personally_identifiable|personal_data",
                    ["PT-3", "PT-4", "PT-5"],
                    "PII processing controls",
                    0.90, "privacy"
                ),
            ],
        }

    def get_patterns_for_code(self, code: str) -> list[tuple[ControlPattern, re.Match]]:
        """Get all matching patterns for a piece of code"""
        matches = []
        code.lower()

        for _category, patterns in self.patterns.items():
            for pattern in patterns:
                # Try case-insensitive match first
                match = re.search(pattern.pattern_regex, code, re.IGNORECASE | re.MULTILINE)
                if match:
                    matches.append((pattern, match))

        return matches

    def get_unique_controls(self, code: str) -> set[str]:
        """Get unique set of applicable controls for code"""
        controls = set()
        matches = self.get_patterns_for_code(code)

        for pattern, _ in matches:
            controls.update(pattern.control_ids)

        return controls

    def get_evidence_statements(self, code: str) -> list[dict[str, Any]]:
        """Generate evidence statements for detected patterns"""
        evidence = []
        matches = self.get_patterns_for_code(code)

        for pattern, match in matches:
            evidence.append({
                "controls": pattern.control_ids,
                "evidence": pattern.evidence_template.format(match=match.group(0)),
                "confidence": pattern.confidence,
                "pattern_type": pattern.pattern_type,
                "severity": pattern.severity,
                "line_match": match.group(0)
            })

        return evidence

    def suggest_missing_controls(self, detected_controls: set[str]) -> dict[str, list[str]]:
        """Suggest related controls that might be missing"""
        suggestions = {}

        # Control relationships and dependencies
        control_relationships = {
            "AC-2": ["AC-3", "AC-6", "AU-2"],  # Account mgmt requires access enforcement
            "AC-3": ["AC-4", "AC-6"],  # Access enforcement relates to info flow
            "AU-2": ["AU-3", "AU-4", "AU-12"],  # Audit events need content and storage
            "SC-8": ["SC-13", "SC-28"],  # Transit encryption needs crypto and at-rest
            "SI-10": ["SI-11", "SI-15"],  # Input validation needs error handling
            "IA-2": ["IA-5", "AC-7"],  # Authentication needs password policy
            "CM-2": ["CM-3", "CM-4"],  # Baseline needs change control
            "RA-5": ["SI-2", "SI-3"],  # Vuln scanning needs patching
            "IR-4": ["IR-5", "IR-6"],  # Incident handling needs tracking
            "CP-9": ["CP-10", "MP-6"],  # Backup needs restore testing
        }

        for control in detected_controls:
            control.split('-')[0]
            related = control_relationships.get(control, [])
            missing = [c for c in related if c not in detected_controls]
            if missing:
                suggestions[control] = missing

        return suggestions

    def get_control_family_coverage(self, detected_controls: set[str]) -> dict[str, float]:
        """Calculate coverage percentage for each control family"""
        # Approximate counts of implementable controls per family
        family_totals = {
            "AC": 25, "AU": 16, "CM": 12, "CP": 13, "IA": 12,
            "IR": 10, "MA": 6, "MP": 8, "PE": 20, "PS": 8,
            "RA": 10, "SC": 45, "SI": 23, "SR": 12, "PT": 8
        }

        family_counts: dict[str, int] = defaultdict(int)
        for control in detected_controls:
            family = control.split('-')[0]
            family_counts[family] += 1

        coverage = {}
        for family, total in family_totals.items():
            implemented = family_counts.get(family, 0)
            coverage[family] = (implemented / total) * 100 if total > 0 else 0

        return coverage
