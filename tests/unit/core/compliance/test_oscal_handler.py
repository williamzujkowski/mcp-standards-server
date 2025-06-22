"""
Comprehensive tests for oscal_handler module
@nist-controls: SA-11, CA-7
@evidence: OSCAL handler testing
"""

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest

from src.analyzers.base import CodeAnnotation
from src.core.compliance.oscal_handler import (
    OSCALComponent,
    OSCALControlImplementation,
    OSCALHandler,
    OSCALMetadata,
)


class TestOSCALComponent:
    """Test OSCALComponent dataclass"""

    def test_component_creation(self):
        """Test creating an OSCAL component"""
        component = OSCALComponent(
            uuid="test-uuid",
            type="software",
            title="Test Component",
            description="Test description",
            props=[{"name": "version", "value": "1.0"}],
            control_implementations=[]
        )

        assert component.uuid == "test-uuid"
        assert component.type == "software"
        assert component.title == "Test Component"
        assert component.description == "Test description"
        assert len(component.props) == 1
        assert component.props[0]["name"] == "version"
        assert len(component.control_implementations) == 0

    def test_component_with_implementations(self):
        """Test component with control implementations"""
        implementations = [
            {
                "uuid": "impl-1",
                "control-id": "AC-3",
                "description": "Access control implementation"
            }
        ]

        component = OSCALComponent(
            uuid="test-uuid-2",
            type="software",
            title="Test Component 2",
            description="Test description 2",
            props=[],
            control_implementations=implementations
        )

        assert len(component.control_implementations) == 1
        assert component.control_implementations[0]["control-id"] == "AC-3"


class TestOSCALControlImplementation:
    """Test OSCALControlImplementation dataclass"""

    def test_control_implementation_creation(self):
        """Test creating control implementation"""
        impl = OSCALControlImplementation(
            uuid="impl-uuid",
            source="NIST_SP-800-53_rev5",
            description="Implementation description",
            implemented_requirements=[]
        )

        assert impl.uuid == "impl-uuid"
        assert impl.source == "NIST_SP-800-53_rev5"
        assert impl.description == "Implementation description"
        assert len(impl.implemented_requirements) == 0


class TestOSCALMetadata:
    """Test OSCALMetadata model"""

    def test_metadata_creation_minimal(self):
        """Test creating metadata with minimal fields"""
        metadata = OSCALMetadata(title="Test SSP")

        assert metadata.title == "Test SSP"
        assert metadata.version == "1.0.0"
        assert metadata.oscal_version == "1.0.0"
        assert isinstance(metadata.last_modified, str)
        assert len(metadata.authors) == 0
        assert len(metadata.props) == 0

    def test_metadata_creation_full(self):
        """Test creating metadata with all fields"""
        metadata = OSCALMetadata(
            title="Complete SSP",
            last_modified="2024-01-01T00:00:00Z",
            version="2.0.0",
            oscal_version="1.1.0",
            authors=[{"name": "Test Author"}],
            props=[{"name": "category", "value": "system"}]
        )

        assert metadata.title == "Complete SSP"
        assert metadata.last_modified == "2024-01-01T00:00:00Z"
        assert metadata.version == "2.0.0"
        assert metadata.oscal_version == "1.1.0"
        assert len(metadata.authors) == 1
        assert metadata.authors[0]["name"] == "Test Author"
        assert len(metadata.props) == 1


class TestOSCALHandler:
    """Test OSCALHandler class"""

    @pytest.fixture
    def handler(self):
        """Create handler instance"""
        return OSCALHandler()

    @pytest.fixture
    def sample_annotations(self):
        """Create sample code annotations"""
        return [
            CodeAnnotation(
                control_ids=["AC-3", "AC-6"],
                evidence="Role-based access control implementation",
                confidence=0.95,
                file_path="/app/auth.py",
                line_number=42,
                component="auth"
            ),
            CodeAnnotation(
                control_ids=["AU-2", "AU-3"],
                evidence="Comprehensive audit logging",
                confidence=0.90,
                file_path="/app/logging.py",
                line_number=15,
                component="logging"
            ),
            CodeAnnotation(
                control_ids=["AC-3"],  # Duplicate control
                evidence="Additional access control",
                confidence=0.85,
                file_path="/app/permissions.py",
                line_number=100,
                component="permissions"
            )
        ]

    def test_handler_initialization(self, handler):
        """Test handler initialization"""
        assert handler.components == {}
        assert handler.nist_catalog_url == "https://csrc.nist.gov/extensions/oscal/catalog/nist_rev5_800-53_catalog.json"

    @patch('uuid.uuid4')
    def test_create_component_from_annotations(self, mock_uuid, handler, sample_annotations):
        """Test creating component from annotations"""
        # Mock UUID generation
        mock_uuid.side_effect = [
            "component-uuid",
            "impl-1-uuid",
            "impl-2-uuid",
            "impl-3-uuid"
        ]

        metadata = {
            "description": "Test application",
            "version": "1.2.3",
            "component_type": "web-app"
        }

        component = handler.create_component_from_annotations(
            "TestApp",
            sample_annotations,
            metadata
        )

        assert component.uuid == "component-uuid"
        assert component.type == "software"
        assert component.title == "TestApp"
        assert component.description == "Test application"

        # Check props
        props_dict = {p["name"]: p["value"] for p in component.props}
        assert props_dict["version"] == "1.2.3"
        assert props_dict["component-type"] == "web-app"
        assert "last-modified" in props_dict
        assert "compliance-scan-date" in props_dict

        # Check control implementations
        assert len(component.control_implementations) == 3  # AC-3, AC-6, AU-2/AU-3

        # Find AC-3 implementation (should have 2 evidences)
        ac3_impl = next(
            impl for impl in component.control_implementations
            if impl["control-id"] == "AC-3"
        )
        assert len(ac3_impl["evidence"]) == 2  # Two annotations for AC-3

        # Check evidence structure
        evidence = ac3_impl["evidence"][0]
        assert "description" in evidence
        assert "link" in evidence
        assert evidence["link"]["href"].startswith("file://")
        assert evidence["props"][0]["name"] == "confidence"

        # Component should be stored
        assert "component-uuid" in handler.components
        assert handler.components["component-uuid"] == component

    def test_create_component_minimal(self, handler):
        """Test creating component with minimal data"""
        annotations = [
            CodeAnnotation(
                control_ids=["SC-8"],
                evidence=None,  # No evidence text
                confidence=0.7,
                file_path="/app/crypto.py",
                line_number=10,
                component=None
            )
        ]

        component = handler.create_component_from_annotations(
            "MinimalApp",
            annotations,
            {}  # Empty metadata
        )

        assert component.title == "MinimalApp"
        assert component.description == "Component: MinimalApp"
        assert len(component.control_implementations) == 1

        # Check default evidence
        impl = component.control_implementations[0]
        assert impl["evidence"][0]["description"] == "Implementation in code"

    @patch('uuid.uuid4')
    def test_generate_ssp_content(self, mock_uuid, handler):
        """Test generating SSP content"""
        # Create test components
        components = [
            OSCALComponent(
                uuid="comp-1",
                type="software",
                title="Web App",
                description="Web application component",
                props=[{"name": "version", "value": "1.0"}],
                control_implementations=[
                    {
                        "uuid": "impl-1",
                        "control-id": "AC-3",
                        "description": "Access control",
                        "props": [],
                        "evidence": []
                    }
                ]
            ),
            OSCALComponent(
                uuid="comp-2",
                type="software",
                title="API Server",
                description="API server component",
                props=[],
                control_implementations=[
                    {
                        "uuid": "impl-2",
                        "control-id": "AU-2",
                        "description": "Audit logging",
                        "props": [],
                        "evidence": []
                    }
                ]
            )
        ]

        # Store components in handler
        for comp in components:
            handler.components[comp.uuid] = comp

        # Mock UUID for SSP
        mock_uuid.return_value = "ssp-uuid"

        metadata = {
            "version": "2.0.0",
            "system_id": "test-system",
            "description": "Test system for compliance",
            "cloud_model": "private",
            "service_model": "iaas",
            "sensitivity": "high",
            "confidentiality_impact": "high",
            "integrity_impact": "moderate",
            "availability_impact": "low",
            "status": "development",
            "parties": [{"name": "Test Org"}],
            "users": [{"uuid": "user-1", "title": "Admin Users"}]
        }

        ssp = handler.generate_ssp_content(
            "Test System",
            components,
            "NIST_SP-800-53_rev5_HIGH",
            metadata
        )

        # Verify SSP structure
        assert "system-security-plan" in ssp
        ssp_content = ssp["system-security-plan"]

        # Check metadata
        assert ssp_content["uuid"] == "ssp-uuid"
        assert ssp_content["metadata"]["title"] == "System Security Plan for Test System"
        assert ssp_content["metadata"]["version"] == "2.0.0"
        assert len(ssp_content["metadata"]["roles"]) == 3
        assert ssp_content["metadata"]["parties"] == [{"name": "Test Org"}]

        # Check import profile
        assert ssp_content["import-profile"]["href"] == "#NIST_SP-800-53_rev5_HIGH"
        assert len(ssp_content["import-profile"]["include-controls"]) == 2  # AC-3, AU-2

        # Check system characteristics
        sys_char = ssp_content["system-characteristics"]
        assert sys_char["system-name"] == "Test System"
        assert sys_char["system-ids"][0]["id"] == "test-system"
        assert sys_char["description"] == "Test system for compliance"
        assert sys_char["security-sensitivity-level"] == "high"

        # Check cloud properties
        cloud_props = {p["name"]: p["value"] for p in sys_char["props"]}
        assert cloud_props["cloud-deployment-model"] == "private"
        assert cloud_props["cloud-service-model"] == "iaas"

        # Check impacts
        info_type = sys_char["system-information"]["information-types"][0]
        assert info_type["confidentiality-impact"]["base"] == "high"
        assert info_type["integrity-impact"]["base"] == "moderate"
        assert info_type["availability-impact"]["base"] == "low"

        # Check security impact level
        impact = sys_char["security-impact-level"]
        assert impact["security-objective-confidentiality"] == "moderate"  # Uses default

        # Check status
        assert sys_char["status"]["state"] == "development"

        # Check system implementation
        sys_impl = ssp_content["system-implementation"]
        assert len(sys_impl["users"]) == 1
        assert len(sys_impl["components"]) == 2

        # Check control implementation
        ctrl_impl = ssp_content["control-implementation"]
        assert "description" in ctrl_impl
        assert "implemented-requirements" in ctrl_impl

        # Check back matter
        assert len(ssp_content["back-matter"]["resources"]) > 0

    def test_generate_ssp_minimal(self, handler):
        """Test generating SSP with minimal data"""
        component = OSCALComponent(
            uuid="minimal-comp",
            type="software",
            title="Minimal Component",
            description="Minimal test",
            props=[],
            control_implementations=[]
        )

        ssp = handler.generate_ssp_content(
            "Minimal System",
            [component]
        )

        ssp_content = ssp["system-security-plan"]

        # Check defaults are applied
        assert ssp_content["metadata"]["version"] == "1.0.0"
        assert ssp_content["import-profile"]["href"] == "#NIST_SP-800-53_rev5_MODERATE"

        sys_char = ssp_content["system-characteristics"]
        assert sys_char["security-sensitivity-level"] == "moderate"
        assert sys_char["status"]["state"] == "operational"

        # Check default descriptions
        assert "authorization boundary includes all components" in sys_char["authorization-boundary"]["description"]
        assert "three-tier architecture" in sys_char["network-architecture"]["description"]
        assert "encryption at rest and in transit" in sys_char["data-flow"]["description"]

    def test_generate_implementation_description(self, handler, sample_annotations):
        """Test generating implementation description"""
        # Test with multiple annotations for same control
        ac3_annotations = [ann for ann in sample_annotations if "AC-3" in ann.control_ids]

        description = handler._generate_implementation_description(
            "AC-3",
            ac3_annotations
        )

        # Should include evidence from both annotations
        assert "Role-based access control" in description
        assert "Additional access control" in description
        assert "auth.py:42" in description
        assert "permissions.py:100" in description

    def test_get_all_control_ids(self, handler):
        """Test getting all control IDs from components"""
        components = [
            OSCALComponent(
                uuid="c1",
                type="software",
                title="C1",
                description="Component 1",
                props=[],
                control_implementations=[
                    {"control-id": "AC-3"},
                    {"control-id": "AU-2"}
                ]
            ),
            OSCALComponent(
                uuid="c2",
                type="software",
                title="C2",
                description="Component 2",
                props=[],
                control_implementations=[
                    {"control-id": "SC-8"},
                    {"control-id": "AC-3"}  # Duplicate
                ]
            )
        ]

        control_ids = handler._get_all_control_ids(components)

        # Should have unique control IDs
        assert len(control_ids) == 3
        assert "AC-3" in control_ids
        assert "AU-2" in control_ids
        assert "SC-8" in control_ids

    def test_component_to_oscal_format(self, handler):
        """Test converting component to OSCAL format"""
        component = OSCALComponent(
            uuid="test-comp",
            type="software",
            title="Test Component",
            description="Test description",
            props=[{"name": "version", "value": "1.0"}],
            control_implementations=[]
        )

        oscal_comp = handler._component_to_oscal_format(component)

        assert oscal_comp["uuid"] == "test-comp"
        assert oscal_comp["type"] == "software"
        assert oscal_comp["title"] == "Test Component"
        assert oscal_comp["description"] == "Test description"
        assert oscal_comp["props"] == component.props
        assert oscal_comp["control-implementations"] == []

    def test_generate_inventory_items(self, handler):
        """Test generating inventory items"""
        components = [
            OSCALComponent(
                uuid="web-comp",
                type="software",
                title="Web Server",
                description="Web server component",
                props=[
                    {"name": "version", "value": "2.4.41"},
                    {"name": "component-type", "value": "web-server"}
                ],
                control_implementations=[]
            ),
            OSCALComponent(
                uuid="db-comp",
                type="software",
                title="Database",
                description="Database component",
                props=[
                    {"name": "version", "value": "8.0.23"},
                    {"name": "component-type", "value": "database"}
                ],
                control_implementations=[]
            )
        ]

        inventory_items = handler._generate_inventory_items(components)

        assert len(inventory_items) == 2

        # Check first item
        item1 = inventory_items[0]
        assert "uuid" in item1
        assert item1["description"] == "Web Server"
        assert item1["implemented-components"][0]["component-uuid"] == "web-comp"

        props_dict = {p["name"]: p["value"] for p in item1["props"]}
        assert props_dict["version"] == "2.4.41"
        assert props_dict["asset-type"] == "software"

        # Check second item
        item2 = inventory_items[1]
        assert item2["description"] == "Database"
        assert item2["implemented-components"][0]["component-uuid"] == "db-comp"

    def test_merge_control_implementations(self, handler):
        """Test merging control implementations from components"""
        components = [
            OSCALComponent(
                uuid="comp1",
                type="software",
                title="Component 1",
                description="First component",
                props=[],
                control_implementations=[
                    {
                        "uuid": "impl-1",
                        "control-id": "AC-3",
                        "description": "Access control from comp1",
                        "props": [{"name": "status", "value": "implemented"}],
                        "evidence": []
                    },
                    {
                        "uuid": "impl-2",
                        "control-id": "AU-2",
                        "description": "Audit from comp1",
                        "props": [],
                        "evidence": []
                    }
                ]
            ),
            OSCALComponent(
                uuid="comp2",
                type="software",
                title="Component 2",
                description="Second component",
                props=[],
                control_implementations=[
                    {
                        "uuid": "impl-3",
                        "control-id": "AC-3",  # Same control as comp1
                        "description": "Access control from comp2",
                        "props": [],
                        "evidence": []
                    },
                    {
                        "uuid": "impl-4",
                        "control-id": "SC-8",
                        "description": "Encryption from comp2",
                        "props": [],
                        "evidence": []
                    }
                ]
            )
        ]

        merged = handler._merge_control_implementations(components)

        # Should have 3 unique controls (AC-3, AU-2, SC-8)
        assert len(merged) == 3

        # Find AC-3 (should have statements from both components)
        ac3_impl = next(impl for impl in merged if impl["control-id"] == "AC-3")
        assert len(ac3_impl["statements"]) == 2
        assert ac3_impl["statements"][0]["description"] == "Access control from comp1"
        assert ac3_impl["statements"][1]["description"] == "Access control from comp2"

        # Check other controls
        au2_impl = next(impl for impl in merged if impl["control-id"] == "AU-2")
        assert len(au2_impl["statements"]) == 1

        sc8_impl = next(impl for impl in merged if impl["control-id"] == "SC-8")
        assert len(sc8_impl["statements"]) == 1

    def test_export_component_definition(self, handler):
        """Test exporting component definition"""
        # Create and store components
        component = OSCALComponent(
            uuid="export-comp",
            type="software",
            title="Export Test",
            description="Component for export test",
            props=[{"name": "test", "value": "true"}],
            control_implementations=[]
        )
        handler.components["export-comp"] = component

        # Test with metadata
        metadata = OSCALMetadata(
            title="Test Component Definition",
            version="1.5.0"
        )

        result = handler.export_component_definition(
            [component],
            metadata,
            include_validation=False
        )

        assert "component-definition" in result
        comp_def = result["component-definition"]

        # Check metadata
        assert comp_def["metadata"]["title"] == "Test Component Definition"
        assert comp_def["metadata"]["version"] == "1.5.0"
        assert comp_def["metadata"]["oscal-version"] == "1.0.0"

        # Check components
        assert len(comp_def["components"]) == 1
        assert comp_def["components"][0]["uuid"] == "export-comp"

        # Check back matter
        assert "back-matter" in comp_def

    def test_validate_ssp_structure(self, handler):
        """Test SSP structure validation"""
        # Valid SSP
        valid_ssp = {
            "system-security-plan": {
                "uuid": "test-uuid",
                "metadata": {"title": "Test"},
                "import-profile": {"href": "#profile"},
                "system-characteristics": {"system-name": "Test"},
                "system-implementation": {"components": []},
                "control-implementation": {"implemented-requirements": []}
            }
        }

        is_valid, errors = handler.validate_ssp_structure(valid_ssp)
        assert is_valid is True
        assert len(errors) == 0

        # Invalid SSP - missing required field
        invalid_ssp = {
            "system-security-plan": {
                "uuid": "test-uuid",
                "metadata": {"title": "Test"}
                # Missing other required fields
            }
        }

        is_valid, errors = handler.validate_ssp_structure(invalid_ssp)
        assert is_valid is False
        assert len(errors) > 0
        assert any("import-profile" in error for error in errors)

    def test_add_checksum(self, handler):
        """Test adding checksum to document"""
        document = {
            "system-security-plan": {
                "uuid": "test",
                "metadata": {"title": "Test SSP"}
            }
        }

        doc_with_checksum = handler._add_checksum(document)

        # Should have checksum in metadata
        assert "document-hash" in doc_with_checksum["system-security-plan"]["metadata"]

        hash_info = doc_with_checksum["system-security-plan"]["metadata"]["document-hash"]
        assert hash_info["algorithm"] == "SHA-256"
        assert len(hash_info["value"]) == 64  # SHA-256 hex length

    def test_generate_description_with_frameworks(self, handler):
        """Test description generation with framework info"""
        annotations = [
            CodeAnnotation(
                control_ids=["AC-3"],
                evidence="RBAC implementation",
                confidence=0.9,
                file_path="/app/auth.py",
                line_number=10,
                component="django"
            ),
            CodeAnnotation(
                control_ids=["AC-3"],
                evidence="Permission checks",
                confidence=0.85,
                file_path="/app/views.py",
                line_number=50,
                component="django"
            )
        ]

        description = handler._generate_implementation_description("AC-3", annotations)

        # Should mention frameworks
        assert "django" in description or "Django" in description
        assert "auth.py" in description
        assert "views.py" in description


class TestIntegration:
    """Integration tests for OSCAL handler"""

    def test_full_workflow(self):
        """Test complete workflow from annotations to SSP"""
        handler = OSCALHandler()

        # Create annotations simulating real scan results
        annotations = [
            CodeAnnotation(
                control_ids=["AC-2", "AC-3", "AC-6"],
                evidence="User account management with RBAC",
                confidence=0.92,
                file_path="/app/auth/users.py",
                line_number=25,
                component="django"
            ),
            CodeAnnotation(
                control_ids=["AU-2", "AU-3", "AU-4"],
                evidence="Comprehensive audit logging with rotation",
                confidence=0.88,
                file_path="/app/audit/logger.py",
                line_number=100,
                component="python-logging"
            ),
            CodeAnnotation(
                control_ids=["SC-8", "SC-13"],
                evidence="TLS encryption for data in transit",
                confidence=0.95,
                file_path="/app/security/crypto.py",
                line_number=50,
                component="ssl"
            ),
            CodeAnnotation(
                control_ids=["IA-2", "IA-2(1)"],
                evidence="Multi-factor authentication implementation",
                confidence=0.90,
                file_path="/app/auth/mfa.py",
                line_number=75,
                component="pyotp"
            )
        ]

        # Create components
        auth_component = handler.create_component_from_annotations(
            "Authentication Service",
            [ann for ann in annotations if "auth" in str(ann.file_path)],
            {
                "description": "Handles user authentication and authorization",
                "version": "2.1.0",
                "component_type": "authentication"
            }
        )

        audit_component = handler.create_component_from_annotations(
            "Audit Service",
            [ann for ann in annotations if "audit" in str(ann.file_path)],
            {
                "description": "Provides comprehensive audit logging",
                "version": "1.5.0",
                "component_type": "logging"
            }
        )

        security_component = handler.create_component_from_annotations(
            "Security Service",
            [ann for ann in annotations if "security" in str(ann.file_path)],
            {
                "description": "Handles encryption and security controls",
                "version": "3.0.0",
                "component_type": "security"
            }
        )

        # Generate SSP
        ssp = handler.generate_ssp_content(
            "Enterprise Application System",
            [auth_component, audit_component, security_component],
            "NIST_SP-800-53_rev5_HIGH",
            {
                "version": "1.0.0",
                "system_id": "enterprise-app",
                "description": "Enterprise application with strong security controls",
                "cloud_model": "private",
                "service_model": "saas",
                "sensitivity": "high",
                "confidentiality_impact": "high",
                "integrity_impact": "high",
                "availability_impact": "moderate",
                "status": "operational",
                "auth_boundary": "All application components within the private cloud",
                "parties": [
                    {
                        "uuid": str(uuid.uuid4()),
                        "type": "organization",
                        "name": "Enterprise Corp"
                    }
                ]
            }
        )

        # Validate generated SSP
        assert "system-security-plan" in ssp
        ssp_content = ssp["system-security-plan"]

        # Check all components are included
        assert len(ssp_content["system-implementation"]["components"]) == 3

        # Check all controls are captured
        control_ids = {
            ctrl["control-id"]
            for ctrl in ssp_content["control-implementation"]["implemented-requirements"]
        }
        expected_controls = {"AC-2", "AC-3", "AC-6", "AU-2", "AU-3", "AU-4",
                           "SC-8", "SC-13", "IA-2", "IA-2(1)"}
        assert control_ids == expected_controls

        # Validate structure
        is_valid, errors = handler.validate_ssp_structure(ssp)
        assert is_valid is True
        assert len(errors) == 0

        # Test export with validation
        comp_def = handler.export_component_definition(
            [auth_component, audit_component, security_component],
            OSCALMetadata(
                title="Enterprise Application Components",
                version="1.0.0",
                authors=[{"name": "Security Team"}]
            ),
            include_validation=True
        )

        assert "component-definition" in comp_def
        assert "document-hash" in comp_def["component-definition"]["metadata"]

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        handler = OSCALHandler()

        # Test with empty annotations
        component = handler.create_component_from_annotations(
            "Empty Component",
            [],
            {}
        )
        assert len(component.control_implementations) == 0

        # Test with invalid metadata
        ssp = handler.generate_ssp_content(
            "Test System",
            [],
            "INVALID_PROFILE",
            {"invalid_field": "value"}
        )
        # Should still generate valid SSP with defaults
        assert "system-security-plan" in ssp

        # Test validation with malformed SSP
        malformed = {"wrong-key": {}}
        is_valid, errors = handler.validate_ssp_structure(malformed)
        assert is_valid is False
        assert len(errors) > 0
