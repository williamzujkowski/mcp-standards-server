#!/usr/bin/env python3
"""
MCP Standards Server CLI

Main command-line interface for managing, syncing, and generating standards.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from core.standards.sync import StandardsSynchronizer, sync_standards

from .__version__ import __version__


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=level, format=format_str, handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for CLI."""
    parser = argparse.ArgumentParser(
        prog="mcp-standards",
        description="MCP Standards Server - Manage and sync development standards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )

    parser.add_argument(
        "-c", "--config", type=Path, help="Path to sync configuration file"
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Sync command
    sync_parser = subparsers.add_parser(
        "sync", help="Synchronize standards from GitHub repository"
    )
    sync_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force sync even if files are up to date",
    )
    sync_parser.add_argument(
        "--check", action="store_true", help="Check for updates without downloading"
    )

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show sync status and statistics"
    )
    status_parser.add_argument(
        "--json", action="store_true", help="Output status in JSON format"
    )

    # Cache command
    cache_parser = subparsers.add_parser("cache", help="Manage local cache")
    cache_parser.add_argument(
        "--clear", action="store_true", help="Clear all cached files"
    )
    cache_parser.add_argument("--list", action="store_true", help="List cached files")

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Show or validate configuration"
    )
    config_parser.add_argument(
        "--validate", action="store_true", help="Validate configuration file"
    )
    config_parser.add_argument(
        "--show", action="store_true", help="Show current configuration"
    )

    # Generate command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate standards from templates"
    )
    generate_parser.add_argument("--template", "-t", help="Template name to use")
    generate_parser.add_argument("--domain", "-d", help="Domain-specific template")
    generate_parser.add_argument("--output", "-o", help="Output file path")
    generate_parser.add_argument("--title", help="Standard title")
    generate_parser.add_argument("--version", default="1.0.0", help="Standard version")
    generate_parser.add_argument("--author", help="Standard author")
    generate_parser.add_argument("--description", help="Standard description")
    generate_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive mode"
    )
    generate_parser.add_argument(
        "--preview", "-p", action="store_true", help="Preview mode (no file output)"
    )
    generate_parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate generated standard",
    )
    generate_parser.add_argument("--config-file", help="Configuration file path")

    # Generate subcommands
    generate_subparsers = generate_parser.add_subparsers(
        dest="generate_command", help="Generate subcommands"
    )

    # List templates
    generate_subparsers.add_parser("list-templates", help="List available templates")

    # Template info
    template_info_parser = generate_subparsers.add_parser(
        "template-info", help="Get template information"
    )
    template_info_parser.add_argument("template_name", help="Template name")

    # Customize template
    customize_parser = generate_subparsers.add_parser(
        "customize", help="Create custom template"
    )
    customize_parser.add_argument(
        "--template", "-t", required=True, help="Base template"
    )
    customize_parser.add_argument(
        "--name", "-n", required=True, help="Custom template name"
    )
    customize_parser.add_argument("--config", help="Customization config file")
    customize_parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactive customization"
    )

    # Validate standard
    validate_parser = generate_subparsers.add_parser(
        "validate", help="Validate existing standard"
    )
    validate_parser.add_argument("standard_file", help="Standard file to validate")
    validate_parser.add_argument("--report", "-r", help="Output report file")

    return parser


def cmd_sync(args: argparse.Namespace) -> int:
    """Handle sync command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)

    if args.check:
        # Check for updates
        print("Checking for updates...")
        updates = synchronizer.check_updates()

        if updates["outdated_files"]:
            print(f"\nOutdated files ({len(updates['outdated_files'])}):")
            for file in updates["outdated_files"]:
                age_hours = file["age_hours"]
                print(f"  - {file['path']} (last synced {age_hours:.1f} hours ago)")
        else:
            print("\nAll files are up to date!")

        print(f"\nTotal cached files: {updates['total_cached']}")
        print(f"Cache TTL: {updates['cache_ttl_hours']} hours")

        return 0

    # Perform sync
    print("Starting standards synchronization...")
    if args.force:
        print("Force sync enabled - all files will be re-downloaded")

    try:
        result = sync_standards(force=args.force, config_path=args.config)

        # Display results
        print(f"\nSync completed with status: {result.status.value}")
        print(f"Duration: {result.duration.total_seconds():.2f} seconds")
        print(f"Files synced: {len(result.synced_files)}/{result.total_files}")

        if result.synced_files:
            print("\nSynced files:")
            for file in result.synced_files:
                print(f"  - {file.path} ({file.size} bytes)")

        if result.failed_files:
            print(f"\nFailed files ({len(result.failed_files)}):")
            for path, error in result.failed_files:
                print(f"  - {path}: {error}")

        if result.message:
            print(f"\n{result.message}")

        return 0 if result.status.value in ["success", "partial"] else 1

    except Exception as e:
        print(f"Error during sync: {e}", file=sys.stderr)
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Handle status command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)
    status = synchronizer.get_sync_status()

    if args.json:
        # JSON output
        print(json.dumps(status, indent=2, default=str))
    else:
        # Human-readable output
        print("MCP Standards Server - Sync Status\n")
        print(f"Total files cached: {status['total_files']}")
        print(f"Total cache size: {status['total_size_mb']:.2f} MB")

        # Rate limit info
        rate_limit = status["rate_limit"]
        print("\nGitHub API Rate Limit:")
        print(f"  Remaining: {rate_limit['remaining']}/{rate_limit['limit']}")
        if rate_limit["reset_time"]:
            print(f"  Resets at: {rate_limit['reset_time']}")

        # Recent syncs
        if status["last_sync_times"]:
            print("\nRecent syncs:")
            recent = sorted(
                [(p, t) for p, t in status["last_sync_times"].items() if t],
                key=lambda x: x[1],
                reverse=True,
            )[:5]

            for path, sync_time in recent:
                print(f"  - {path}: {sync_time}")

        # Configuration summary
        config = status["config"]
        repo = config.get("repository", {})
        print(f"\nRepository: {repo.get('owner')}/{repo.get('repo')}")
        print(f"Branch: {repo.get('branch')}")
        print(f"Path: {repo.get('path')}")

    return 0


def cmd_cache(args: argparse.Namespace) -> int:
    """Handle cache command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)

    if args.clear:
        print("Clearing cache...")
        synchronizer.clear_cache()
        print("Cache cleared successfully!")
        return 0

    if args.list:
        cached_files = synchronizer.get_cached_standards()

        if cached_files:
            print(f"Cached files ({len(cached_files)}):\n")
            for file_path in sorted(cached_files):
                size = file_path.stat().st_size / 1024  # KB
                print(f"  - {file_path.name} ({size:.1f} KB)")
        else:
            print("No cached files found.")

        return 0

    # Default: show cache info
    status = synchronizer.get_sync_status()
    print(f"Cache location: {synchronizer.cache_dir}")
    print(f"Total files: {status['total_files']}")
    print(f"Total size: {status['total_size_mb']:.2f} MB")

    return 0


def cmd_config(args: argparse.Namespace) -> int:
    """Handle config command."""
    synchronizer = StandardsSynchronizer(config_path=args.config)

    if args.validate:
        # Validate configuration
        try:
            # Check required fields
            required = ["repository", "sync", "cache"]
            missing = [field for field in required if field not in synchronizer.config]

            if missing:
                print(f"Invalid configuration: missing fields {missing}")
                return 1

            # Validate repository config
            repo = synchronizer.config["repository"]
            repo_required = ["owner", "repo", "branch", "path"]
            repo_missing = [field for field in repo_required if field not in repo]

            if repo_missing:
                print(f"Invalid repository configuration: missing {repo_missing}")
                return 1

            print("Configuration is valid!")
            return 0

        except Exception as e:
            print(f"Configuration validation failed: {e}")
            return 1

    if args.show:
        # Show configuration
        import yaml

        print("Current configuration:\n")
        print(yaml.dump(synchronizer.config, default_flow_style=False))
        return 0

    # Default: show config file path
    print(f"Configuration file: {synchronizer.config_path}")
    if synchronizer.config_path.exists():
        print("Status: Found")
    else:
        print("Status: Using default configuration")

    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Handle generate command."""
    try:
        # Import generate commands
        from cli.commands.generate import (
            StandardMetadata,
            StandardsGenerator,
            _interactive_standard_creation,
            _interactive_template_customization,
        )
        from generators.quality_assurance import QualityAssuranceSystem
        from generators.validator import StandardsValidator

        generator = StandardsGenerator()

        # Handle subcommands
        if args.generate_command == "list-templates":
            templates = generator.list_templates()

            print("Available Templates:")
            print("=" * 50)

            # Group by category
            categories: dict[str, list[dict[str, Any]]] = {}
            for template in templates:
                category = template["category"]
                if category not in categories:
                    categories[category] = []
                categories[category].append(template)

            for category, category_templates in categories.items():
                print(f"\n{category.upper()}:")
                for template in category_templates:
                    print(f"  {template['name']}")
                    if template["description"]:
                        print(f"    Description: {template['description']}")
                    if template["tags"]:
                        print(f"    Tags: {', '.join(template['tags'])}")
                    print()

            return 0

        elif args.generate_command == "template-info":
            # Get template schema
            schema = generator.get_template_schema(args.template_name)

            # Validate template
            validation = generator.validate_template(args.template_name)

            print(f"Template: {args.template_name}")
            print("=" * 50)

            print(f"Valid: {'✓' if validation['valid'] else '✗'}")
            if not validation["valid"]:
                print(f"Error: {validation.get('error', 'Unknown error')}")
                print(f"Message: {validation.get('message', '')}")

            if "variables" in validation:
                print("\nRequired Variables:")
                for var in validation["variables"]:
                    print(f"  - {var}")

            if schema:
                import yaml

                print("\nSchema:")
                print(yaml.dump(schema, default_flow_style=False))

            return 0

        elif args.generate_command == "customize":
            if args.interactive:
                customizations = _interactive_template_customization()
            elif args.config:
                import yaml

                with open(args.config) as f:
                    if args.config.endswith(".yaml") or args.config.endswith(".yml"):
                        customizations = yaml.safe_load(f)
                    else:
                        customizations = json.load(f)
            else:
                print(
                    "Error: Either --config or --interactive must be specified",
                    file=sys.stderr,
                )
                return 1

            # Create custom template
            custom_path = generator.create_custom_template(
                args.name, args.template, customizations
            )

            print(f"Custom template created: {custom_path}")
            return 0

        elif args.generate_command == "validate":
            # Read the standard file
            with open(args.standard_file) as f:
                content = f.read()

            # Try to find corresponding metadata file
            metadata_file = args.standard_file.replace(".md", ".yaml")
            if Path(metadata_file).exists():
                import yaml

                with open(metadata_file) as f:
                    metadata_dict = yaml.safe_load(f)
                metadata = StandardMetadata.from_dict(metadata_dict)
            else:
                # Create minimal metadata for validation
                metadata = StandardMetadata(
                    title=Path(args.standard_file).stem,
                    version="1.0.0",
                    domain="general",
                    type="technical",
                )

            # Validate
            validator = StandardsValidator()
            qa_system = QualityAssuranceSystem()

            validation_results = validator.validate_standard(content, metadata)
            qa_results = qa_system.assess_standard(content, metadata)

            # Display results
            print(f"Validation Results for: {args.standard_file}")
            print("=" * 50)

            if validation_results["valid"]:
                print("✓ Validation passed")
            else:
                print("✗ Validation failed")
                for error in validation_results["errors"]:
                    print(f"  Error: {error}")

            if validation_results["warnings"]:
                print("Warnings:")
                for warning in validation_results["warnings"]:
                    print(f"  - {warning}")

            print(f"\nQuality Score: {qa_results['overall_score']}/100")
            print("\nScore Breakdown:")
            for metric, score in qa_results["scores"].items():
                print(f"  {metric}: {score:.1f}")

            if qa_results["recommendations"]:
                print("\nRecommendations:")
                for rec in qa_results["recommendations"][:10]:
                    print(f"  - {rec}")

            # Save report if requested
            if args.report:
                report_data = {
                    "standard_file": args.standard_file,
                    "validation_results": validation_results,
                    "quality_assessment": qa_results,
                    "generated_at": datetime.now().isoformat(),
                }

                import yaml

                with open(args.report, "w") as f:
                    if args.report.endswith(".yaml") or args.report.endswith(".yml"):
                        yaml.dump(report_data, f, default_flow_style=False)
                    else:
                        json.dump(report_data, f, indent=2)

                print(f"\nReport saved to: {args.report}")

            return 0

        else:
            # Main generate command
            # Load configuration if provided
            if args.config_file:
                import yaml

                with open(args.config_file) as f:
                    if args.config_file.endswith(".yaml") or args.config_file.endswith(
                        ".yml"
                    ):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
            else:
                config_data = {}

            # Interactive mode
            if args.interactive:
                metadata = _interactive_standard_creation(generator, args.domain)
            else:
                # Build metadata from parameters and config
                metadata = {
                    "title": args.title or config_data.get("title"),
                    "version": args.version or config_data.get("version", "1.0.0"),
                    "author": args.author or config_data.get("author"),
                    "description": args.description or config_data.get("description"),
                    "domain": args.domain or config_data.get("domain", "general"),
                    "type": config_data.get("type", "technical"),
                    "created_date": datetime.now().isoformat(),
                    "updated_date": datetime.now().isoformat(),
                    **config_data,
                }

            # Validate required fields
            if not metadata.get("title"):
                print("Error: Title is required", file=sys.stderr)
                return 1

            # Determine template
            template = args.template
            if not template:
                if args.domain:
                    template = f"domains/{args.domain}.j2"
                else:
                    template = f"standards/{metadata.get('type', 'base')}.j2"

            # Determine output path
            output = args.output
            if not output and not args.preview:
                safe_title = "".join(
                    c if c.isalnum() or c in ("-", "_") else "_"
                    for c in metadata["title"]
                )
                output = f"{safe_title.lower()}_standard.md"

            # Generate standard
            result = generator.generate_standard(
                template_name=template,
                metadata=metadata,
                output_path=output or "",
                validate=args.validate,
                preview=args.preview,
            )

            if args.preview:
                print("=== PREVIEW ===")
                print(result["content"])
                print("\n=== METADATA ===")
                import yaml

                print(yaml.dump(result["metadata"], default_flow_style=False))
            else:
                print(f"Standard generated successfully: {result['output_path']}")

            # Show validation results
            if args.validate and "validation" in result:
                validation = result["validation"]
                if validation["valid"]:
                    print("✓ Validation passed")
                else:
                    print("✗ Validation failed:")
                    for error in validation["errors"]:
                        print(f"  - {error}")

                if validation["warnings"]:
                    print("Warnings:")
                    for warning in validation["warnings"]:
                        print(f"  - {warning}")

            # Show quality assessment
            if "quality_assessment" in result:
                qa = result["quality_assessment"]
                print(f"Quality Score: {qa['overall_score']}/100")

                if qa["recommendations"]:
                    print("Recommendations:")
                    for rec in qa["recommendations"][:5]:  # Show top 5
                        print(f"  - {rec}")

            return 0

    except Exception as e:
        print(f"Error in generate command: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main entry point for CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Handle commands
    if args.command == "sync":
        return cmd_sync(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "cache":
        return cmd_cache(args)
    elif args.command == "config":
        return cmd_config(args)
    elif args.command == "generate":
        return cmd_generate(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
