"""
Base Standards Generator

Core generator class that orchestrates the standards generation process.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime

from .engine import TemplateEngine
from .metadata import StandardMetadata
from .validator import StandardsValidator
from .quality_assurance import QualityAssuranceSystem


class StandardsGenerator:
    """Main standards generator class that coordinates all generation components."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the standards generator.
        
        Args:
            templates_dir: Path to templates directory. If None, uses default.
        """
        self.templates_dir = templates_dir or self._get_default_templates_dir()
        self.engine = TemplateEngine(self.templates_dir)
        self.validator = StandardsValidator()
        self.qa_system = QualityAssuranceSystem()
        
    def _get_default_templates_dir(self) -> str:
        """Get default templates directory path."""
        return os.path.join(os.path.dirname(__file__), "../../templates")
    
    def generate_standard(
        self,
        template_name: str,
        metadata: Dict[str, Any],
        output_path: str,
        validate: bool = True,
        preview: bool = False
    ) -> Dict[str, Any]:
        """
        Generate a standard document.
        
        Args:
            template_name: Name of the template to use
            metadata: Standard metadata dictionary
            output_path: Path where to save the generated standard
            validate: Whether to validate the generated standard
            preview: If True, return preview without saving
            
        Returns:
            Dictionary containing generation results and validation info
        """
        # Validate metadata
        std_metadata = StandardMetadata.from_dict(metadata)
        std_metadata.validate()
        
        # Generate content
        content = self.engine.render_template(template_name, std_metadata.to_dict())
        
        # Validate generated content if requested
        validation_results = {}
        if validate:
            validation_results = self.validator.validate_standard(content, std_metadata)
            
        # Quality assurance check
        qa_results = self.qa_system.assess_standard(content, std_metadata)
        
        # Save or return preview
        if preview:
            return {
                "content": content,
                "metadata": std_metadata.to_dict(),
                "validation": validation_results,
                "quality_assessment": qa_results,
                "preview": True
            }
        else:
            # Save to file
            self._save_standard(content, output_path, std_metadata)
            
            return {
                "output_path": output_path,
                "metadata": std_metadata.to_dict(),
                "validation": validation_results,
                "quality_assessment": qa_results,
                "success": True
            }
    
    def _save_standard(self, content: str, output_path: str, metadata: StandardMetadata):
        """Save the standard to file with metadata."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save main content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        # Save metadata
        metadata_path = output_path.replace('.md', '.yaml')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            yaml.dump(metadata.to_dict(), f, default_flow_style=False)
    
    def list_templates(self) -> List[Dict[str, Any]]:
        """List available templates with their metadata."""
        return self.engine.list_templates()
    
    def get_template_schema(self, template_name: str) -> Dict[str, Any]:
        """Get the schema for a specific template."""
        return self.engine.get_template_schema(template_name)
    
    def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate a template file."""
        return self.engine.validate_template(template_name)
    
    def create_custom_template(
        self,
        template_name: str,
        base_template: str,
        customizations: Dict[str, Any]
    ) -> str:
        """Create a custom template based on an existing one."""
        return self.engine.create_custom_template(template_name, base_template, customizations)