"""
Standards Validator

Validation framework for generated standards documents.
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from .metadata import StandardMetadata


class StandardsValidator:
    """Validator for standards documents."""
    
    def __init__(self):
        """Initialize the validator."""
        self.nist_controls = self._load_nist_controls()
        self.compliance_frameworks = self._load_compliance_frameworks()
        
    def _load_nist_controls(self) -> List[str]:
        """Load NIST control IDs."""
        # Common NIST SP 800-53 controls
        return [
            "AC-1", "AC-2", "AC-3", "AC-4", "AC-5", "AC-6", "AC-7", "AC-8",
            "AU-1", "AU-2", "AU-3", "AU-4", "AU-5", "AU-6", "AU-7", "AU-8",
            "CA-1", "CA-2", "CA-3", "CA-4", "CA-5", "CA-6", "CA-7", "CA-8",
            "CM-1", "CM-2", "CM-3", "CM-4", "CM-5", "CM-6", "CM-7", "CM-8",
            "CP-1", "CP-2", "CP-3", "CP-4", "CP-5", "CP-6", "CP-7", "CP-8",
            "IA-1", "IA-2", "IA-3", "IA-4", "IA-5", "IA-6", "IA-7", "IA-8",
            "IR-1", "IR-2", "IR-3", "IR-4", "IR-5", "IR-6", "IR-7", "IR-8",
            "MA-1", "MA-2", "MA-3", "MA-4", "MA-5", "MA-6", "MA-7", "MA-8",
            "MP-1", "MP-2", "MP-3", "MP-4", "MP-5", "MP-6", "MP-7", "MP-8",
            "PE-1", "PE-2", "PE-3", "PE-4", "PE-5", "PE-6", "PE-7", "PE-8",
            "PL-1", "PL-2", "PL-3", "PL-4", "PL-5", "PL-6", "PL-7", "PL-8",
            "PS-1", "PS-2", "PS-3", "PS-4", "PS-5", "PS-6", "PS-7", "PS-8",
            "RA-1", "RA-2", "RA-3", "RA-4", "RA-5", "RA-6", "RA-7", "RA-8",
            "SA-1", "SA-2", "SA-3", "SA-4", "SA-5", "SA-6", "SA-7", "SA-8",
            "SC-1", "SC-2", "SC-3", "SC-4", "SC-5", "SC-6", "SC-7", "SC-8",
            "SI-1", "SI-2", "SI-3", "SI-4", "SI-5", "SI-6", "SI-7", "SI-8"
        ]
    
    def _load_compliance_frameworks(self) -> List[str]:
        """Load compliance framework names."""
        return [
            "NIST", "ISO-27001", "SOC2", "PCI-DSS", "HIPAA", "GDPR",
            "CCPA", "FedRAMP", "FISMA", "COSO", "COBIT", "ITIL"
        ]
    
    def validate_standard(self, content: str, metadata: StandardMetadata) -> Dict[str, Any]:
        """
        Validate a generated standard document.
        
        Args:
            content: The standard document content
            metadata: Standard metadata
            
        Returns:
            Validation results
        """
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "checks": {}
        }
        
        # Validate metadata
        metadata_validation = metadata.validate()
        if not metadata_validation["valid"]:
            results["valid"] = False
            results["errors"].extend(metadata_validation["errors"])
        results["warnings"].extend(metadata_validation["warnings"])
        
        # Validate content structure
        structure_check = self._validate_structure(content)
        results["checks"]["structure"] = structure_check
        if not structure_check["valid"]:
            results["valid"] = False
            results["errors"].extend(structure_check["errors"])
        results["warnings"].extend(structure_check["warnings"])
        
        # Validate NIST controls
        nist_check = self._validate_nist_controls(content, metadata)
        results["checks"]["nist_controls"] = nist_check
        if not nist_check["valid"]:
            results["warnings"].extend(nist_check["warnings"])
        
        # Validate compliance frameworks
        compliance_check = self._validate_compliance_frameworks(content, metadata)
        results["checks"]["compliance"] = compliance_check
        if not compliance_check["valid"]:
            results["warnings"].extend(compliance_check["warnings"])
        
        # Validate cross-references
        xref_check = self._validate_cross_references(content)
        results["checks"]["cross_references"] = xref_check
        if not xref_check["valid"]:
            results["warnings"].extend(xref_check["warnings"])
        
        # Validate completeness
        completeness_check = self._validate_completeness(content, metadata)
        results["checks"]["completeness"] = completeness_check
        if not completeness_check["valid"]:
            results["warnings"].extend(completeness_check["warnings"])
        
        return results
    
    def _validate_structure(self, content: str) -> Dict[str, Any]:
        """Validate document structure."""
        errors = []
        warnings = []
        
        # Check for required sections
        required_sections = [
            "# ",  # Title
            "## Purpose",
            "## Scope", 
            "## Implementation",
            "## Compliance"
        ]
        
        for section in required_sections:
            if section not in content:
                errors.append(f"Missing required section: {section.strip('#').strip()}")
        
        # Check for proper heading hierarchy
        headings = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        for i, (level, title) in enumerate(headings):
            if i == 0 and len(level) != 1:
                errors.append("Document must start with a top-level heading")
            elif i > 0:
                prev_level = len(headings[i-1][0])
                curr_level = len(level)
                if curr_level > prev_level + 1:
                    warnings.append(f"Heading hierarchy skip detected at: {title}")
        
        # Check for proper markdown formatting
        if content.count('```') % 2 != 0:
            errors.append("Unmatched code blocks detected")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def _validate_nist_controls(self, content: str, metadata: StandardMetadata) -> Dict[str, Any]:
        """Validate NIST controls integration."""
        warnings = []
        
        # Check if NIST controls in metadata are referenced in content
        for control in metadata.nist_controls:
            if control not in self.nist_controls:
                warnings.append(f"Unknown NIST control: {control}")
            elif control not in content:
                warnings.append(f"NIST control {control} in metadata but not referenced in content")
        
        # Check for NIST controls mentioned in content but not in metadata
        mentioned_controls = re.findall(r'NIST[- ]([A-Z]{2}-\d+)', content)
        for control in mentioned_controls:
            if control not in metadata.nist_controls:
                warnings.append(f"NIST control {control} mentioned in content but not in metadata")
        
        return {
            "valid": True,
            "warnings": warnings
        }
    
    def _validate_compliance_frameworks(self, content: str, metadata: StandardMetadata) -> Dict[str, Any]:
        """Validate compliance framework integration."""
        warnings = []
        
        # Check if compliance frameworks in metadata are referenced in content
        for framework in metadata.compliance_frameworks:
            if framework not in self.compliance_frameworks:
                warnings.append(f"Unknown compliance framework: {framework}")
            elif framework not in content:
                warnings.append(f"Compliance framework {framework} in metadata but not referenced in content")
        
        return {
            "valid": True,
            "warnings": warnings
        }
    
    def _validate_cross_references(self, content: str) -> Dict[str, Any]:
        """Validate cross-references within the document."""
        warnings = []
        
        # Find all internal references
        internal_refs = re.findall(r'\[([^\]]+)\]\(#([^)]+)\)', content)
        
        # Find all heading anchors
        headings = re.findall(r'^#+\s+(.+)$', content, re.MULTILINE)
        heading_anchors = [self._heading_to_anchor(h) for h in headings]
        
        # Check if all internal references point to valid anchors
        for ref_text, anchor in internal_refs:
            if anchor not in heading_anchors:
                warnings.append(f"Broken internal reference: {ref_text} -> #{anchor}")
        
        return {
            "valid": True,
            "warnings": warnings
        }
    
    def _heading_to_anchor(self, heading: str) -> str:
        """Convert heading to anchor."""
        # Simple anchor conversion (lowercase, replace spaces with hyphens)
        return re.sub(r'[^a-z0-9-]', '', heading.lower().replace(' ', '-'))
    
    def _validate_completeness(self, content: str, metadata: StandardMetadata) -> Dict[str, Any]:
        """Validate document completeness."""
        warnings = []
        
        # Check for placeholder content
        placeholders = ['TODO', 'TBD', 'FIXME', '{{', '}}']
        for placeholder in placeholders:
            if placeholder in content:
                warnings.append(f"Placeholder content detected: {placeholder}")
        
        # Check for minimum content length
        if len(content) < 1000:
            warnings.append("Document content is very short")
        
        # Check for examples if technical standard
        if metadata.type == "technical" and "```" not in content:
            warnings.append("Technical standard should include code examples")
        
        # Check for implementation guide references
        if metadata.implementation_guides and "implementation" not in content.lower():
            warnings.append("Implementation guides specified but not referenced in content")
        
        return {
            "valid": True,
            "warnings": warnings
        }