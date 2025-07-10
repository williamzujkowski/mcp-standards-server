"""
Base Standards Generator

Core generator class that orchestrates the standards generation process.
"""

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import yaml

from .engine import TemplateEngine
from .metadata import StandardMetadata
from .quality_assurance import QualityAssuranceSystem
from .validator import StandardsValidator


@dataclass
class GenerationResult:
    """Result of standard generation."""

    standard: str
    metadata: dict[str, Any]
    warnings: list[str]
    quality_score: float


class StandardsGenerator:
    """Main standards generator class that coordinates all generation components."""

    def __init__(self, templates_dir: str | None = None) -> None:
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
        context: dict[str, Any],
        title: str,
        domain: str | None = None,
        output_path: str | None = None,
        validate: bool = True,
        preview: bool = True,
    ) -> GenerationResult:
        """
        Generate a standard document.

        Args:
            template_name: Name of the template to use
            context: Generation context dictionary
            title: Standard title
            domain: Domain/category for the standard
            output_path: Path where to save the generated standard
            validate: Whether to validate the generated standard
            preview: If True, return preview without saving

        Returns:
            Dictionary containing generation results and validation info
        """
        # Create metadata from context
        metadata = {
            "title": title,
            "domain": domain or "general",
            "created_date": datetime.utcnow().isoformat(),
            "author": context.get("author", "MCP Standards Generator"),
            **context,
        }

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

        # Extract warnings and quality score
        warnings = validation_results.get("warnings", [])
        if qa_results:
            warnings.extend(qa_results.get("warnings", []))

        quality_score = qa_results.get("overall_score", 0.8) if qa_results else 0.8

        # Save to file if output_path is provided
        if output_path and not preview:
            self._save_standard(content, output_path, std_metadata)

        return GenerationResult(
            standard=content,
            metadata=std_metadata.to_dict(),
            warnings=warnings,
            quality_score=quality_score,
        )

    def _save_standard(
        self, content: str, output_path: str, metadata: StandardMetadata
    ) -> None:
        """Save the standard to file with metadata."""
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save main content
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Save metadata
        metadata_path = output_path.replace(".md", ".yaml")
        with open(metadata_path, "w", encoding="utf-8") as f:
            yaml.dump(metadata.to_dict(), f, default_flow_style=False)

    def list_templates(self, domain: str | None = None) -> list[dict[str, Any]]:
        """List available templates with their metadata."""
        templates = self.engine.list_templates()

        if domain:
            # Filter templates by domain if specified
            filtered_templates = []
            for template in templates:
                template_domain = template.get("domain", "general")
                if (
                    template_domain == domain
                    or domain in template.get("name", "").lower()
                ):
                    filtered_templates.append(template)
            return filtered_templates

        return templates

    def get_template_schema(self, template_name: str) -> dict[str, Any]:
        """Get the schema for a specific template."""
        return self.engine.get_template_schema(template_name)

    def validate_template(self, template_name: str) -> dict[str, Any]:
        """Validate a template file."""
        return self.engine.validate_template(template_name)

    def create_custom_template(
        self, template_name: str, base_template: str, customizations: dict[str, Any]
    ) -> str:
        """Create a custom template based on an existing one."""
        return self.engine.create_custom_template(
            template_name, base_template, customizations
        )
