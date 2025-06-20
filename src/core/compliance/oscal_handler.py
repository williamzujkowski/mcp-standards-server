"""
OSCAL (Open Security Controls Assessment Language) Handler
@nist-controls: CA-2, CA-7, PM-31
@evidence: OSCAL-compliant documentation and assessment
"""
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...analyzers.base import CodeAnnotation


@dataclass
class OSCALComponent:
    """OSCAL Component Definition"""
    uuid: str
    type: str
    title: str
    description: str
    props: list[dict[str, str]]
    control_implementations: list[dict[str, Any]]


@dataclass
class OSCALControlImplementation:
    """OSCAL Control Implementation"""
    uuid: str
    source: str
    description: str
    implemented_requirements: list[dict[str, Any]]


class OSCALMetadata(BaseModel):
    """OSCAL document metadata"""
    title: str
    last_modified: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    version: str = "1.0.0"
    oscal_version: str = "1.0.0"
    authors: list[dict[str, str]] = Field(default_factory=list)
    props: list[dict[str, str]] = Field(default_factory=list)


class OSCALHandler:
    """
    Handles OSCAL format conversion and generation
    @nist-controls: CA-2, PM-31
    @evidence: Standardized security documentation
    """

    def __init__(self):
        self.components: dict[str, OSCALComponent] = {}
        self.nist_catalog_url = "https://csrc.nist.gov/extensions/oscal/catalog/nist_rev5_800-53_catalog.json"

    def create_component_from_annotations(
        self,
        component_name: str,
        annotations: list['CodeAnnotation'],
        metadata: dict[str, Any]
    ) -> OSCALComponent:
        """
        Create OSCAL component from code annotations
        @nist-controls: CA-2, SA-4
        @evidence: Automated component documentation
        """
        component_uuid = str(uuid.uuid4())

        # Group annotations by control
        control_groups: dict[str, list[CodeAnnotation]] = {}
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
                "props": [
                    {
                        "name": "implementation-status",
                        "value": "implemented"
                    },
                    {
                        "name": "implementation-type",
                        "value": "code"
                    }
                ],
                "evidence": [
                    {
                        "description": ann.evidence or "Implementation in code",
                        "link": {
                            "href": f"file://{ann.file_path}#L{ann.line_number}",
                            "text": f"{ann.file_path}:{ann.line_number}"
                        },
                        "props": [
                            {
                                "name": "confidence",
                                "value": str(ann.confidence)
                            }
                        ]
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
                {"name": "last-modified", "value": datetime.now(UTC).isoformat()},
                {"name": "compliance-scan-date", "value": datetime.now(UTC).isoformat()},
                {"name": "component-type", "value": metadata.get("component_type", "application")}
            ],
            control_implementations=control_implementations
        )

        self.components[component_uuid] = component
        return component

    def generate_ssp_content(
        self,
        system_name: str,
        components: list[OSCALComponent],
        profile: str = "NIST_SP-800-53_rev5_MODERATE",
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate SSP (System Security Plan) content
        @nist-controls: CA-2, PM-31
        @evidence: Automated SSP generation
        """
        if metadata is None:
            metadata = {}

        ssp = {
            "system-security-plan": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": f"System Security Plan for {system_name}",
                    "last-modified": datetime.now(UTC).isoformat(),
                    "version": metadata.get("version", "1.0.0"),
                    "oscal-version": "1.0.0",
                    "roles": [
                        {
                            "id": "system-owner",
                            "title": "System Owner"
                        },
                        {
                            "id": "developer",
                            "title": "Developer"
                        },
                        {
                            "id": "security-officer",
                            "title": "Information System Security Officer"
                        }
                    ],
                    "parties": metadata.get("parties", []),
                    "responsible-parties": metadata.get("responsible_parties", [])
                },
                "import-profile": {
                    "href": f"#{profile}",
                    "include-controls": [
                        {"with-id": control_id}
                        for control_id in self._get_all_control_ids(components)
                    ]
                },
                "system-characteristics": {
                    "system-ids": [
                        {
                            "id": metadata.get("system_id", system_name.lower().replace(" ", "-")),
                            "identifier-type": "internal"
                        }
                    ],
                    "system-name": system_name,
                    "description": metadata.get("description", f"System Security Plan for {system_name}"),
                    "props": [
                        {
                            "name": "cloud-deployment-model",
                            "value": metadata.get("cloud_model", "hybrid")
                        },
                        {
                            "name": "cloud-service-model",
                            "value": metadata.get("service_model", "saas")
                        }
                    ],
                    "security-sensitivity-level": metadata.get("sensitivity", "moderate"),
                    "system-information": {
                        "information-types": [
                            {
                                "uuid": str(uuid.uuid4()),
                                "title": "System Data",
                                "description": "System operational and user data",
                                "categorizations": [
                                    {
                                        "system": "https://doi.org/10.6028/NIST.SP.800-60v2r1",
                                        "information-type-ids": metadata.get("info_types", ["C.3.5.8"])
                                    }
                                ],
                                "confidentiality-impact": {
                                    "base": metadata.get("confidentiality_impact", "moderate")
                                },
                                "integrity-impact": {
                                    "base": metadata.get("integrity_impact", "moderate")
                                },
                                "availability-impact": {
                                    "base": metadata.get("availability_impact", "moderate")
                                }
                            }
                        ]
                    },
                    "security-impact-level": {
                        "security-objective-confidentiality": metadata.get("confidentiality", "moderate"),
                        "security-objective-integrity": metadata.get("integrity", "moderate"),
                        "security-objective-availability": metadata.get("availability", "moderate")
                    },
                    "status": {
                        "state": metadata.get("status", "operational"),
                        "remarks": metadata.get("status_remarks", "System is actively maintained and monitored")
                    },
                    "authorization-boundary": {
                        "description": metadata.get("auth_boundary",
                            "The authorization boundary includes all components of the application, "
                            "including web servers, application servers, databases, and supporting infrastructure.")
                    },
                    "network-architecture": {
                        "description": metadata.get("network_desc",
                            "The system follows a standard three-tier architecture with "
                            "web, application, and data tiers separated by network segmentation.")
                    },
                    "data-flow": {
                        "description": metadata.get("data_flow",
                            "Data flows from users through the web tier, processed by the application tier, "
                            "and stored in the data tier with appropriate encryption at rest and in transit.")
                    }
                },
                "system-implementation": {
                    "users": metadata.get("users", [
                        {
                            "uuid": str(uuid.uuid4()),
                            "title": "System Users",
                            "description": "Authorized system users",
                            "role-ids": ["user"]
                        }
                    ]),
                    "components": [
                        self._component_to_oscal_format(comp)
                        for comp in components
                    ],
                    "inventory-items": self._generate_inventory_items(components)
                },
                "control-implementation": {
                    "description": "This section describes how controls are implemented for the system",
                    "implemented-requirements": self._merge_control_implementations(components)
                },
                "back-matter": {
                    "resources": [
                        {
                            "uuid": str(uuid.uuid4()),
                            "title": "NIST SP 800-53 Revision 5",
                            "rlinks": [
                                {
                                    "href": "https://csrc.nist.gov/publications/detail/sp/800-53/rev-5/final"
                                }
                            ]
                        }
                    ]
                }
            }
        }

        return ssp

    def _generate_implementation_description(
        self,
        control_id: str,
        annotations: list['CodeAnnotation']
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
                f"Control {control_id} is implemented in the codebase as identified by automated scanning"
            )

        # Add file references
        files = sorted({ann.file_path for ann in annotations})
        if files:
            descriptions.append(
                f"Implementation found in: {', '.join(files[:5])}" +
                (" and others" if len(files) > 5 else "")
            )

        # Add confidence information
        avg_confidence = sum(ann.confidence for ann in annotations) / len(annotations)
        descriptions.append(f"Average confidence: {avg_confidence:.0%}")

        return " ".join(descriptions)

    def _component_to_oscal_format(self, component: OSCALComponent) -> dict[str, Any]:
        """Convert component to OSCAL format"""
        return {
            "uuid": component.uuid,
            "type": component.type,
            "title": component.title,
            "description": component.description,
            "props": component.props,
            "status": {
                "state": "operational"
            },
            "responsible-roles": [
                {
                    "role-id": "developer",
                    "party-uuids": []
                }
            ]
        }

    def _merge_control_implementations(
        self,
        components: list[OSCALComponent]
    ) -> list[dict[str, Any]]:
        """Merge control implementations from all components"""
        merged = {}

        for component in components:
            for impl in component.control_implementations:
                control_id = impl["control-id"]

                if control_id not in merged:
                    merged[control_id] = {
                        "uuid": str(uuid.uuid4()),
                        "control-id": control_id,
                        "description": "This control is implemented by the following components:",
                        "props": [
                            {
                                "name": "implementation-status",
                                "value": "implemented"
                            }
                        ],
                        "by-components": []
                    }

                merged[control_id]["by-components"].append({
                    "component-uuid": component.uuid,
                    "uuid": impl["uuid"],
                    "description": impl["description"],
                    "props": impl.get("props", []),
                    "links": [
                        {
                            "href": f"#component_{component.uuid}",
                            "rel": "component"
                        }
                    ],
                    "evidence": impl.get("evidence", [])
                })

        return list(merged.values())

    def _get_all_control_ids(self, components: list[OSCALComponent]) -> list[str]:
        """Get all unique control IDs from components"""
        control_ids = set()
        for component in components:
            for impl in component.control_implementations:
                control_ids.add(impl["control-id"])
        return sorted(control_ids)

    def _generate_inventory_items(self, components: list[OSCALComponent]) -> list[dict[str, Any]]:
        """Generate inventory items from components"""
        items = []
        for component in components:
            items.append({
                "uuid": str(uuid.uuid4()),
                "description": f"Software component: {component.title}",
                "props": [
                    {
                        "name": "asset-type",
                        "value": "software"
                    },
                    {
                        "name": "asset-id",
                        "value": component.uuid
                    }
                ],
                "implemented-components": [
                    {
                        "component-uuid": component.uuid
                    }
                ]
            })
        return items

    def export_to_file(self, content: dict[str, Any], output_path: Path, format: str = "json"):
        """
        Export OSCAL content to file with integrity checking
        @nist-controls: AU-10, SI-7
        @evidence: Integrity-protected export
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(content, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

        # Generate checksum for integrity
        with open(output_path, 'rb') as f:
            file_content = f.read()
            checksum = hashlib.sha256(file_content).hexdigest()

        checksum_path = output_path.with_suffix(f'.{format}.sha256')
        checksum_path.write_text(f"{checksum}  {output_path.name}\n")

        return output_path, checksum_path

    def validate_oscal_document(self, document: dict[str, Any]) -> list[str]:
        """
        Validate OSCAL document structure
        @nist-controls: SI-10
        @evidence: Input validation for OSCAL documents
        """
        errors = []

        # Check for required top-level keys
        if "system-security-plan" not in document and "component-definition" not in document:
            errors.append("Document must contain either 'system-security-plan' or 'component-definition'")

        # Validate metadata
        if "system-security-plan" in document:
            ssp = document["system-security-plan"]
            if "metadata" not in ssp:
                errors.append("SSP must contain 'metadata'")
            if "system-characteristics" not in ssp:
                errors.append("SSP must contain 'system-characteristics'")
            if "control-implementation" not in ssp:
                errors.append("SSP must contain 'control-implementation'")

        return errors

    def generate_component_definition(
        self,
        components: list[OSCALComponent],
        metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Generate OSCAL component definition
        @nist-controls: CA-2, SA-4
        @evidence: Component-based security documentation
        """
        if metadata is None:
            metadata = {}

        comp_def = {
            "component-definition": {
                "uuid": str(uuid.uuid4()),
                "metadata": {
                    "title": metadata.get("title", "Component Definition"),
                    "last-modified": datetime.now(UTC).isoformat(),
                    "version": metadata.get("version", "1.0.0"),
                    "oscal-version": "1.0.0"
                },
                "components": [
                    {
                        "uuid": comp.uuid,
                        "type": comp.type,
                        "title": comp.title,
                        "description": comp.description,
                        "props": comp.props,
                        "control-implementations": [
                            {
                                "uuid": str(uuid.uuid4()),
                                "source": self.nist_catalog_url,
                                "description": "NIST 800-53 Rev 5 Control Implementation",
                                "implemented-requirements": [
                                    {
                                        "uuid": impl["uuid"],
                                        "control-id": impl["control-id"],
                                        "description": impl["description"],
                                        "props": impl.get("props", [])
                                    }
                                    for impl in comp.control_implementations
                                ]
                            }
                        ]
                    }
                    for comp in components
                ]
            }
        }

        return comp_def
