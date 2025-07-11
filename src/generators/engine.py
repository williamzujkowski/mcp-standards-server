"""
Template Engine

Jinja2-based template engine for generating standards documents.
"""

import json
from pathlib import Path
from typing import Any, cast

import yaml
from jinja2 import Environment, FileSystemLoader, meta, select_autoescape
from jinja2.exceptions import TemplateNotFound, TemplateSyntaxError


class TemplateEngine:
    """Jinja2-based template engine for standards generation."""

    def __init__(self, templates_dir: str) -> None:
        """
        Initialize the template engine.

        Args:
            templates_dir: Path to templates directory
        """
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Add custom filters
        self._add_custom_filters()

    def _add_custom_filters(self) -> None:
        """Add custom Jinja2 filters."""

        def format_nist_control(control_id: str) -> str:
            """Format NIST control ID for display."""
            return f"NIST-{control_id.upper()}"

        def format_compliance_level(level: str) -> str:
            """Format compliance level for display."""
            level_map = {
                "low": "Low Impact",
                "moderate": "Moderate Impact",
                "high": "High Impact",
            }
            return level_map.get(level.lower(), level)

        def format_risk_level(level: str) -> str:
            """Format risk level for display."""
            return level.upper()

        def generate_toc(sections: list[str]) -> str:
            """Generate table of contents."""
            toc = []
            for i, section in enumerate(sections, 1):
                toc.append(f"{i}. {section}")
            return "\n".join(toc)

        def format_version(version: str) -> str:
            """Format version number."""
            if not version.startswith("v"):
                return f"v{version}"
            return version

        # Register filters
        self.env.filters["format_nist_control"] = format_nist_control
        self.env.filters["format_compliance_level"] = format_compliance_level
        self.env.filters["format_risk_level"] = format_risk_level
        self.env.filters["generate_toc"] = generate_toc
        self.env.filters["format_version"] = format_version

    def render_template(self, template_name: str, context: dict[str, Any]) -> str:
        """
        Render a template with given context.

        Args:
            template_name: Name of the template file
            context: Template context variables

        Returns:
            Rendered template content
        """
        try:
            template = self.env.get_template(template_name)
            result = template.render(**context)
            return str(result)
        except TemplateNotFound:
            raise ValueError(f"Template '{template_name}' not found")
        except TemplateSyntaxError as e:
            raise ValueError(f"Template syntax error in '{template_name}': {e}")

    def list_templates(self) -> list[dict[str, Any]]:
        """List available templates with metadata."""
        templates = []

        # Scan templates directory
        for template_path in self.templates_dir.rglob("*.j2"):
            relative_path = template_path.relative_to(self.templates_dir)
            template_name = str(relative_path)

            # Load template metadata if exists
            metadata_path = template_path.with_suffix(".yaml")
            metadata: dict[str, Any] = {}
            if metadata_path.exists():
                with open(metadata_path, encoding="utf-8") as f:
                    metadata = yaml.safe_load(f) or {}

            templates.append(
                {
                    "name": template_name,
                    "path": str(template_path),
                    "category": self._get_template_category(template_name),
                    "description": metadata.get("description", ""),
                    "version": metadata.get("version", "1.0.0"),
                    "author": metadata.get("author", ""),
                    "tags": metadata.get("tags", []),
                    "variables": self._get_template_variables(template_name),
                }
            )

        return templates

    def _get_template_category(self, template_name: str) -> str:
        """Determine template category from path."""
        parts = template_name.split("/")
        if len(parts) > 1:
            return parts[0]
        return "base"

    def _get_template_variables(self, template_name: str) -> list[str]:
        """Extract template variables using Jinja2 meta."""
        try:
            # Get the source directly from the loader
            loader = self.env.get_or_select_template(template_name).environment.loader
            if loader is None:
                return []
            source, _, _ = loader.get_source(self.env, template_name)
            ast = self.env.parse(source)
            return list(meta.find_undeclared_variables(ast))
        except Exception:
            return []

    def get_template_schema(self, template_name: str) -> dict[str, Any]:
        """Get schema for a specific template."""
        template_path = self.templates_dir / template_name
        schema_path = template_path.with_suffix(".schema.json")

        if schema_path.exists():
            with open(schema_path, encoding="utf-8") as f:
                data = json.load(f)
                return cast(dict[str, Any], data)

        # Generate basic schema from template variables
        variables = self._get_template_variables(template_name)
        return {
            "type": "object",
            "properties": {var: {"type": "string"} for var in variables},
            "required": variables,
        }

    def validate_template(self, template_name: str) -> dict[str, Any]:
        """Validate a template file."""
        try:
            # Try to parse the template
            if self.env.loader is None:
                return {"valid": False, "errors": ["No template loader available"]}
            source, _, _ = self.env.loader.get_source(self.env, template_name)
            self.env.parse(source)

            return {
                "valid": True,
                "template": template_name,
                "variables": self._get_template_variables(template_name),
                "message": "Template is valid",
            }
        except TemplateNotFound:
            return {
                "valid": False,
                "template": template_name,
                "error": "Template not found",
                "message": f"Template '{template_name}' does not exist",
            }
        except TemplateSyntaxError as e:
            return {
                "valid": False,
                "template": template_name,
                "error": "Syntax error",
                "message": str(e),
                "line": e.lineno,
            }
        except Exception as e:
            return {
                "valid": False,
                "template": template_name,
                "error": "Unknown error",
                "message": str(e),
            }

    def create_custom_template(
        self, template_name: str, base_template: str, customizations: dict[str, Any]
    ) -> str:
        """Create a custom template based on an existing one."""
        # Load base template content
        if self.env.loader is None:
            raise ValueError("No template loader available")
        base_content, _, _ = self.env.loader.get_source(self.env, base_template)

        # Apply customizations
        custom_content = self._apply_customizations(base_content, customizations)

        # Save custom template
        custom_path = self.templates_dir / "custom" / template_name
        custom_path.parent.mkdir(parents=True, exist_ok=True)

        with open(custom_path, "w", encoding="utf-8") as f:
            f.write(custom_content)

        return str(custom_path)

    def _apply_customizations(
        self, base_content: str, customizations: dict[str, Any]
    ) -> str:
        """Apply customizations to base template content."""
        content = base_content

        # Apply section replacements
        if "section_replacements" in customizations:
            for section, replacement in customizations["section_replacements"].items():
                content = content.replace(f"{{% block {section} %}}", replacement)

        # Apply variable defaults
        if "variable_defaults" in customizations:
            defaults = customizations["variable_defaults"]
            for var, default in defaults.items():
                content = content.replace(
                    f"{{{{ {var} }}}}", f"{{{{ {var} | default('{default}') }}}}"
                )

        return content
