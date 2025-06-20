"""
Tests for enhanced NIST control pattern detection
@nist-controls: SA-11, CA-7
@evidence: Testing comprehensive control detection
"""
import pytest
from src.analyzers.enhanced_patterns import EnhancedNISTPatterns, ControlPattern


class TestEnhancedNISTPatterns:
    """Test enhanced pattern detection"""
    
    @pytest.fixture
    def patterns(self):
        """Create pattern detector instance"""
        return EnhancedNISTPatterns()
    
    def test_access_control_patterns(self, patterns):
        """Test AC family pattern detection"""
        code = """
        @RolesAllowed(['admin', 'user'])
        def sensitive_function():
            checkPermission('write')
            if hasRole('admin'):
                # Apply least privilege principle
                grant_minimal_permissions()
        """
        
        matches = patterns.get_patterns_for_code(code)
        control_ids = patterns.get_unique_controls(code)
        
        assert len(matches) > 0
        assert "AC-2" in control_ids
        assert "AC-3" in control_ids
        assert "AC-6" in control_ids
    
    def test_authentication_patterns(self, patterns):
        """Test IA family pattern detection"""
        code = """
        from flask_mfa import enable_multi_factor
        
        def login():
            # Configure MFA with TOTP
            enable_multi_factor(method='totp')
            
            # Password policy enforcement
            if not check_password_complexity(password):
                return "Password does not meet requirements"
            
            # Session timeout after 30 minutes
            session.permanent = True
            app.permanent_session_lifetime = timedelta(minutes=30)
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "IA-2(1)" in control_ids  # MFA
        assert "IA-5" in control_ids      # Password policy
        assert "IA-11" in control_ids     # Re-authentication
    
    def test_encryption_patterns(self, patterns):
        """Test SC family pattern detection"""
        code = """
        import ssl
        from cryptography.fernet import Fernet
        
        # Configure TLS for data in transit
        context = ssl.create_default_context()
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        
        # Encrypt data at rest
        def encrypt_database_field(data):
            key = Fernet.generate_key()
            f = Fernet(key)
            encrypted_data = f.encrypt(data.encode())
            return encrypted_data
        
        # Network segmentation
        VLAN_CONFIG = {
            'dmz': '10.0.1.0/24',
            'internal': '192.168.1.0/24'
        }
        """
        
        control_ids = patterns.get_unique_controls(code)
        evidence = patterns.get_evidence_statements(code)
        
        assert "SC-8" in control_ids      # Transit encryption
        assert "SC-28" in control_ids     # Rest encryption
        assert "SC-7(13)" in control_ids  # Network segmentation
        
        # Check evidence quality
        assert any(e['severity'] == 'high' for e in evidence)
    
    def test_audit_patterns(self, patterns):
        """Test AU family pattern detection"""
        code = """
        import logging
        from datetime import datetime
        
        # Configure security audit logging
        audit_log = logging.getLogger('security_audit')
        
        def log_security_event(who, what, when, where, outcome):
            # Comprehensive audit record content
            audit_log.critical({
                'user': who,
                'action': what,
                'timestamp': when,
                'location': where,
                'result': outcome
            })
        
        # Log retention policy - 90 days
        LOG_RETENTION_DAYS = 90
        
        # Tamper-proof logging with digital signatures
        def sign_log_entry(entry):
            return hmac.new(key, entry.encode(), hashlib.sha256).hexdigest()
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "AU-2" in control_ids   # Audit events
        assert "AU-3" in control_ids   # Content of records
        assert "AU-11" in control_ids  # Retention
        assert "AU-9" in control_ids   # Protection of info
    
    def test_system_integrity_patterns(self, patterns):
        """Test SI family pattern detection"""
        code = """
        # Input validation
        def validate_user_input(data):
            # Sanitize and escape user input
            cleaned = sanitize_html(data)
            validated = escape_special_chars(cleaned)
            
            # Use parameterized queries
            cursor.execute(
                "SELECT * FROM users WHERE id = ?", 
                (user_id,)
            )
            
        # Error handling without information disclosure
        try:
            process_request()
        except Exception as e:
            # Log detailed error internally
            logger.error(f"Error: {e}")
            # Return generic error to user
            return "An error occurred", 500
            
        # Patch management
        def check_security_updates():
            return patch_management_system.get_pending_updates()
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "SI-10" in control_ids  # Input validation
        assert "SI-11" in control_ids  # Error handling
        assert "SI-2" in control_ids   # Flaw remediation
    
    def test_incident_response_patterns(self, patterns):
        """Test IR family pattern detection"""
        code = """
        class IncidentResponse:
            def report_security_incident(self, incident_data):
                # Create incident ticket
                ticket = create_incident_ticket(
                    severity=incident_data['severity'],
                    description=incident_data['description']
                )
                
                # Track in SOC system
                soc_dashboard.add_incident(ticket)
                
                # Collect forensic evidence
                evidence = collect_forensic_data(
                    system_snapshot=True,
                    memory_dump=True,
                    maintain_chain_of_custody=True
                )
                
            def handle_data_spillage(self):
                # Information spillage response
                isolate_affected_systems()
                identify_spillage_extent()
                sanitize_spillage_data()
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "IR-4" in control_ids   # Incident handling
        assert "IR-4(4)" in control_ids  # Forensics
        assert "IR-9" in control_ids   # Spillage response
    
    def test_contingency_planning_patterns(self, patterns):
        """Test CP family pattern detection"""
        code = """
        # Backup configuration
        BACKUP_CONFIG = {
            'frequency': 'daily',
            'retention': '30_days',
            'rpo': '4_hours',  # Recovery point objective
            'rto': '2_hours'   # Recovery time objective
        }
        
        def perform_backup():
            # Create backup with versioning
            backup_id = create_versioned_backup(
                data=production_data,
                timestamp=datetime.now()
            )
            
            # Test restore capability
            test_restore_from_backup(backup_id)
            
            # Replicate to alternate site
            replicate_to_dr_site(backup_id)
        
        # Disaster recovery plan
        DR_PLAN = {
            'primary_site': 'us-east-1',
            'alternate_site': 'us-west-2',
            'failover_threshold': '99.9% uptime'
        }
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "CP-9" in control_ids   # Backup
        assert "CP-10" in control_ids  # Recovery
        assert "CP-7" in control_ids   # Alternate site
    
    def test_privacy_patterns(self, patterns):
        """Test PT family pattern detection"""
        code = """
        # Privacy notice implementation
        def display_privacy_notice():
            return render_template('privacy_policy.html')
        
        # User consent management
        def process_user_consent(user_id, consent_choices):
            # Allow opt-in/opt-out for data collection
            update_privacy_preferences(
                user_id=user_id,
                marketing_emails=consent_choices.get('marketing', False),
                analytics_tracking=consent_choices.get('analytics', False)
            )
        
        # Data minimization
        def collect_user_data(required_only=True):
            # Collect only necessary data
            if required_only:
                return get_minimal_user_data()
            
        # PII handling
        def process_personally_identifiable_info(pii_data):
            # Encrypt PII before storage
            encrypted_pii = encrypt_sensitive_data(pii_data)
            store_with_retention_policy(encrypted_pii)
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "PT-1" in control_ids  # Privacy notice
        assert "PT-2" in control_ids  # Consent
        assert "PT-3" in control_ids  # Data minimization
    
    def test_supply_chain_patterns(self, patterns):
        """Test SR family pattern detection"""
        code = """
        # Software Bill of Materials
        def generate_sbom():
            components = analyze_dependencies()
            return create_software_bill_of_materials(components)
        
        # Vendor risk assessment
        def assess_third_party_risk(vendor_id):
            risk_score = calculate_vendor_risk_score(vendor_id)
            audit_report = get_latest_vendor_audit(vendor_id)
            return combine_risk_assessments(risk_score, audit_report)
        
        # Component authenticity verification
        def verify_component_authenticity(component):
            # Check against known good hashes
            if not verify_hash(component.hash, TRUSTED_HASHES):
                raise CounterfeitComponentError()
        """
        
        control_ids = patterns.get_unique_controls(code)
        
        assert "SR-4" in control_ids   # SBOM
        assert "SR-6" in control_ids   # Vendor assessment
        assert "SR-11" in control_ids  # Component authenticity
    
    def test_control_family_coverage(self, patterns):
        """Test coverage analysis"""
        detected_controls = {
            "AC-2", "AC-3", "AU-2", "AU-3", 
            "SC-8", "SC-13", "SI-10", "IA-2"
        }
        
        coverage = patterns.get_control_family_coverage(detected_controls)
        
        assert "AC" in coverage
        assert "AU" in coverage
        assert coverage["AC"] > 0
        assert coverage["AU"] > 0
    
    def test_missing_control_suggestions(self, patterns):
        """Test control relationship suggestions"""
        detected_controls = {"AC-2", "AU-2", "SC-8"}
        
        suggestions = patterns.suggest_missing_controls(detected_controls)
        
        # AC-2 should suggest AC-3 and AC-6
        assert "AC-2" in suggestions
        assert "AC-3" in suggestions["AC-2"]
        
        # AU-2 should suggest AU-3
        assert "AU-2" in suggestions
        assert "AU-3" in suggestions["AU-2"]
        
        # SC-8 should suggest SC-13
        assert "SC-8" in suggestions
        assert "SC-13" in suggestions["SC-8"]
    
    def test_high_severity_patterns(self, patterns):
        """Test detection of high-severity security patterns"""
        code = """
        # Multiple high-severity patterns
        
        # 1. Media sanitization
        def secure_disk_wipe(device):
            perform_dod_5220_wipe(device)
            verify_data_destruction(device)
        
        # 2. Multi-factor authentication
        @require_mfa
        def admin_panel():
            pass
        
        # 3. Input validation for SQL injection
        def query_database(user_input):
            # Parameterized query prevents SQL injection
            cursor.execute("SELECT * FROM data WHERE id = %s", (user_input,))
        
        # 4. Encryption in transit
        app.config['SESSION_COOKIE_SECURE'] = True
        app.config['SESSION_COOKIE_HTTPONLY'] = True
        app.config['PREFERRED_URL_SCHEME'] = 'https'
        """
        
        evidence = patterns.get_evidence_statements(code)
        high_severity = [e for e in evidence if e['severity'] == 'high']
        
        assert len(high_severity) >= 3
        
        # Verify specific high-severity controls detected
        control_ids = patterns.get_unique_controls(code)
        assert "MP-6" in control_ids   # Media sanitization
        assert "IA-2(1)" in control_ids  # MFA
        assert "SI-10" in control_ids  # Input validation
        assert "SC-8" in control_ids   # Transit encryption