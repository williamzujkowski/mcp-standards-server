"""
Standards Generation CLI Commands

Command-line interface for generating standards using templates.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from generators import StandardMetadata, StandardsGenerator


@click.group()
def generate() -> None:
    """Generate standards from templates."""
    pass


@generate.command()
@click.option("--template", "-t", help="Template name to use")
@click.option("--domain", "-d", help="Domain-specific template")
@click.option("--output", "-o", help="Output file path")
@click.option("--title", help="Standard title")
@click.option("--version", default="1.0.0", help="Standard version")
@click.option("--author", help="Standard author")
@click.option("--description", help="Standard description")
@click.option("--interactive", "-i", is_flag=True, help="Interactive mode")
@click.option("--preview", "-p", is_flag=True, help="Preview mode (no file output)")
@click.option(
    "--validate", is_flag=True, default=True, help="Validate generated standard"
)
@click.option("--config", "-c", help="Configuration file path")
def standard(
    template: str | None,
    domain: str | None,
    output: str | None,
    title: str | None,
    version: str,
    author: str | None,
    description: str | None,
    interactive: bool,
    preview: bool,
    validate: bool,
    config: str | None,
) -> None:
    """Generate a standard document from template."""

    try:
        generator = StandardsGenerator()

        # Load configuration if provided
        if config:
            with open(config) as f:
                if config.endswith(".yaml") or config.endswith(".yml"):
                    config_data = yaml.safe_load(f)
                else:
                    config_data = json.load(f)
        else:
            config_data = {}

        # Interactive mode
        if interactive:
            metadata = _interactive_standard_creation(generator, domain)
        else:
            # Build metadata from parameters and config
            metadata = {
                "title": title or config_data.get("title"),
                "version": version or config_data.get("version", "1.0.0"),
                "author": author or config_data.get("author"),
                "description": description or config_data.get("description"),
                "domain": domain or config_data.get("domain", "general"),
                "type": config_data.get("type", "technical"),
                "created_date": datetime.now().isoformat(),
                "updated_date": datetime.now().isoformat(),
                **config_data,
            }

        # Validate required fields
        if not metadata.get("title"):
            click.echo("Error: Title is required", err=True)
            sys.exit(1)

        # Determine template
        if not template:
            if domain:
                template = f"domains/{domain}.j2"
            else:
                template = f"standards/{metadata.get('type', 'base')}.j2"

        # Determine output path
        if not output and not preview:
            safe_title = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in metadata["title"]
            )
            output = f"{safe_title.lower()}_standard.md"

        # Generate standard
        result = generator.generate_standard(
            template_name=template,
            metadata=metadata,
            output_path=output or "",
            validate=validate,
            preview=preview,
        )

        if preview:
            click.echo("=== PREVIEW ===")
            click.echo(result["content"])
            click.echo("\n=== METADATA ===")
            click.echo(yaml.dump(result["metadata"], default_flow_style=False))
        else:
            click.echo(f"Standard generated successfully: {result['output_path']}")

        # Show validation results
        if validate and "validation" in result:
            validation = result["validation"]
            if validation["valid"]:
                click.echo("✓ Validation passed")
            else:
                click.echo("✗ Validation failed:")
                for error in validation["errors"]:
                    click.echo(f"  - {error}")

            if validation["warnings"]:
                click.echo("Warnings:")
                for warning in validation["warnings"]:
                    click.echo(f"  - {warning}")

        # Show quality assessment
        if "quality_assessment" in result:
            qa = result["quality_assessment"]
            click.echo(f"Quality Score: {qa['overall_score']}/100")

            if qa["recommendations"]:
                click.echo("Recommendations:")
                for rec in qa["recommendations"][:5]:  # Show top 5
                    click.echo(f"  - {rec}")

    except Exception as e:
        click.echo(f"Error generating standard: {e}", err=True)
        sys.exit(1)


@generate.command()
def list_templates() -> None:
    """List available templates."""

    try:
        generator = StandardsGenerator()
        templates = generator.list_templates()

        click.echo("Available Templates:")
        click.echo("=" * 50)

        # Group by category
        categories: dict[str, list[dict[str, Any]]] = {}
        for template in templates:
            category = template["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(template)

        for category, category_templates in categories.items():
            click.echo(f"\n{category.upper()}:")
            for template in category_templates:
                click.echo(f"  {template['name']}")
                if template["description"]:
                    click.echo(f"    Description: {template['description']}")
                if template["tags"]:
                    click.echo(f"    Tags: {', '.join(template['tags'])}")
                click.echo()

    except Exception as e:
        click.echo(f"Error listing templates: {e}", err=True)
        sys.exit(1)


@generate.command()
@click.argument("template_name")
def template_info(template_name: str) -> None:
    """Get detailed information about a template."""

    try:
        generator = StandardsGenerator()

        # Get template schema
        schema = generator.get_template_schema(template_name)

        # Validate template
        validation = generator.validate_template(template_name)

        click.echo(f"Template: {template_name}")
        click.echo("=" * 50)

        click.echo(f"Valid: {'✓' if validation['valid'] else '✗'}")
        if not validation["valid"]:
            click.echo(f"Error: {validation.get('error', 'Unknown error')}")
            click.echo(f"Message: {validation.get('message', '')}")

        if "variables" in validation:
            click.echo("\nRequired Variables:")
            for var in validation["variables"]:
                click.echo(f"  - {var}")

        if schema:
            click.echo("\nSchema:")
            click.echo(yaml.dump(schema, default_flow_style=False))

    except Exception as e:
        click.echo(f"Error getting template info: {e}", err=True)
        sys.exit(1)


@generate.command()
@click.option("--template", "-t", required=True, help="Template to customize")
@click.option("--name", "-n", required=True, help="Custom template name")
@click.option("--config", "-c", help="Customization configuration file")
@click.option("--interactive", "-i", is_flag=True, help="Interactive customization")
def customize(template: str, name: str, config: str | None, interactive: bool) -> None:
    """Create a custom template based on an existing one."""

    try:
        generator = StandardsGenerator()

        if interactive:
            customizations = _interactive_template_customization()
        elif config:
            with open(config) as f:
                if config.endswith(".yaml") or config.endswith(".yml"):
                    customizations = yaml.safe_load(f)
                else:
                    customizations = json.load(f)
        else:
            click.echo(
                "Error: Either --config or --interactive must be specified", err=True
            )
            sys.exit(1)

        # Create custom template
        custom_path = generator.create_custom_template(name, template, customizations)

        click.echo(f"Custom template created: {custom_path}")

    except Exception as e:
        click.echo(f"Error creating custom template: {e}", err=True)
        sys.exit(1)


@generate.command()
@click.argument("standard_file")
@click.option("--report", "-r", help="Output report file")
def validate_standard(standard_file: str, report: str | None) -> None:
    """Validate an existing standard document."""

    try:
        # Read the standard file
        with open(standard_file) as f:
            content = f.read()

        # Try to find corresponding metadata file
        metadata_file = standard_file.replace(".md", ".yaml")
        if os.path.exists(metadata_file):
            with open(metadata_file) as f:
                metadata_dict = yaml.safe_load(f)
            metadata = StandardMetadata.from_dict(metadata_dict)
        else:
            # Create minimal metadata for validation
            metadata = StandardMetadata(
                title=Path(standard_file).stem,
                version="1.0.0",
                domain="general",
                type="technical",
            )

        # Validate
        from generators.quality_assurance import QualityAssuranceSystem
        from generators.validator import StandardsValidator

        validator = StandardsValidator()
        qa_system = QualityAssuranceSystem()

        validation_results = validator.validate_standard(content, metadata)
        qa_results = qa_system.assess_standard(content, metadata)

        # Display results
        click.echo(f"Validation Results for: {standard_file}")
        click.echo("=" * 50)

        if validation_results["valid"]:
            click.echo("✓ Validation passed")
        else:
            click.echo("✗ Validation failed")
            for error in validation_results["errors"]:
                click.echo(f"  Error: {error}")

        if validation_results["warnings"]:
            click.echo("Warnings:")
            for warning in validation_results["warnings"]:
                click.echo(f"  - {warning}")

        click.echo(f"\nQuality Score: {qa_results['overall_score']}/100")
        click.echo("\nScore Breakdown:")
        for metric, score in qa_results["scores"].items():
            click.echo(f"  {metric}: {score:.1f}")

        if qa_results["recommendations"]:
            click.echo("\nRecommendations:")
            for rec in qa_results["recommendations"][:10]:
                click.echo(f"  - {rec}")

        # Save report if requested
        if report:
            report_data = {
                "standard_file": standard_file,
                "validation_results": validation_results,
                "quality_assessment": qa_results,
                "generated_at": datetime.now().isoformat(),
            }

            with open(report, "w") as f:
                if report.endswith(".yaml") or report.endswith(".yml"):
                    yaml.dump(report_data, f, default_flow_style=False)
                else:
                    json.dump(report_data, f, indent=2)

            click.echo(f"\nReport saved to: {report}")

    except Exception as e:
        click.echo(f"Error validating standard: {e}", err=True)
        sys.exit(1)


def _interactive_standard_creation(
    generator: StandardsGenerator, domain: str | None = None
) -> dict[str, Any]:
    """Interactive standard creation wizard."""

    click.echo("=== Interactive Standard Creation ===")

    # Basic information
    title = click.prompt("Standard title")
    version = click.prompt("Version", default="1.0.0")
    author = click.prompt("Author")
    description = click.prompt("Description")

    # Domain and type
    if not domain:
        available_domains = ["general", "ai_ml", "blockchain", "iot", "gaming", "api"]
        domain = click.prompt(
            "Domain", type=click.Choice(available_domains), default="general"
        )

    standard_types = ["technical", "compliance", "process", "architecture"]
    standard_type = click.prompt(
        "Standard type", type=click.Choice(standard_types), default="technical"
    )

    # Risk and maturity
    risk_levels = ["low", "moderate", "high"]
    risk_level = click.prompt(
        "Risk level", type=click.Choice(risk_levels), default="moderate"
    )

    maturity_levels = ["planning", "developing", "testing", "production", "deprecated"]
    maturity_level = click.prompt(
        "Maturity level", type=click.Choice(maturity_levels), default="developing"
    )

    # Optional fields
    metadata = {
        "title": title,
        "version": version,
        "author": author,
        "description": description,
        "domain": domain,
        "type": standard_type,
        "risk_level": risk_level,
        "maturity_level": maturity_level,
        "created_date": datetime.now().isoformat(),
        "updated_date": datetime.now().isoformat(),
        "tags": [],
        "nist_controls": [],
        "compliance_frameworks": [],
        "implementation_guides": [],
        "dependencies": [],
    }

    # Tags
    if click.confirm("Add tags?"):
        tags = click.prompt("Tags (comma-separated)").split(",")
        metadata["tags"] = [tag.strip() for tag in tags if tag.strip()]

    # NIST controls
    if click.confirm("Add NIST controls?"):
        controls = click.prompt("NIST controls (comma-separated)").split(",")
        metadata["nist_controls"] = [
            control.strip() for control in controls if control.strip()
        ]

    # Compliance frameworks
    if click.confirm("Add compliance frameworks?"):
        frameworks = click.prompt("Compliance frameworks (comma-separated)").split(",")
        metadata["compliance_frameworks"] = [
            fw.strip() for fw in frameworks if fw.strip()
        ]

    # Domain-specific fields
    if domain == "ai_ml":
        if click.confirm("Add ML frameworks?"):
            frameworks = click.prompt("ML frameworks (comma-separated)").split(",")
            metadata["ml_frameworks"] = [fw.strip() for fw in frameworks if fw.strip()]

    elif domain == "blockchain":
        if click.confirm("Add blockchain networks?"):
            networks = click.prompt("Blockchain networks (comma-separated)").split(",")
            metadata["blockchain_networks"] = [
                net.strip() for net in networks if net.strip()
            ]

    return metadata


def _interactive_template_customization() -> dict[str, Any]:
    """Interactive template customization wizard."""

    click.echo("=== Interactive Template Customization ===")

    customizations: dict[str, Any] = {}

    # Section replacements
    if click.confirm("Customize sections?"):
        customizations["section_replacements"] = {}

        sections = ["purpose", "scope", "implementation", "compliance", "monitoring"]
        for section in sections:
            if click.confirm(f"Customize {section} section?"):
                content = click.prompt(f"New {section} content", type=str)
                customizations["section_replacements"][section] = content

    # Variable defaults
    if click.confirm("Set variable defaults?"):
        customizations["variable_defaults"] = {}

        while True:
            var_name = click.prompt("Variable name (or 'done' to finish)")
            if var_name.lower() == "done":
                break

            var_default = click.prompt(f"Default value for {var_name}")
            customizations["variable_defaults"][var_name] = var_default

    return customizations


if __name__ == "__main__":
    generate()
