#!/usr/bin/env python3
"""
Enhanced CLI for MCP Standards Server with improved help and autocomplete.
"""

import argparse
import os
import sys
from pathlib import Path


# ANSI color codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    YELLOW = "\033[93m"  # Add YELLOW alias for WARNING
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def colored(text: str, color: str) -> str:
    """Return colored text if colors are enabled."""
    if os.environ.get("NO_COLOR"):
        return text
    return f"{color}{text}{Colors.ENDC}"


class ImprovedHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Custom help formatter with better formatting and examples."""

    def _format_action(self, action: argparse.Action) -> str:
        # Get the original help text
        help_text = super()._format_action(action)

        # Add color to option names
        if action.option_strings:
            for option in action.option_strings:
                help_text = help_text.replace(option, colored(option, Colors.CYAN))

        return help_text


def create_enhanced_parser() -> argparse.ArgumentParser:
    """Create enhanced argument parser with detailed help."""

    # Main description with examples
    description = f"""
{colored('MCP Standards Server', Colors.BOLD + Colors.BLUE)} - Manage and validate development standards

{colored('OVERVIEW', Colors.BOLD)}
  The MCP Standards Server provides tools for managing, syncing, and validating
  code against development standards. It integrates with IDEs, CI/CD pipelines,
  and development workflows.

{colored('QUICK START', Colors.BOLD)}
  {colored('$', Colors.GREEN)} mcp-standards config --init     # Initialize configuration
  {colored('$', Colors.GREEN)} mcp-standards sync              # Download standards
  {colored('$', Colors.GREEN)} mcp-standards validate .        # Validate current directory
  {colored('$', Colors.GREEN)} mcp-standards serve             # Start MCP server

{colored('COMMON WORKFLOWS', Colors.BOLD)}
  {colored('New Project:', Colors.CYAN)}
    mcp-standards query --project-type web --framework react > standards.md

  {colored('CI/CD Integration:', Colors.CYAN)}
    mcp-standards validate . --format junit --fail-on error

  {colored('Auto-fix Issues:', Colors.CYAN)}
    mcp-standards validate src/ --fix
"""

    epilog = f"""
{colored('EXAMPLES', Colors.BOLD)}
  {colored('# Initialize a new project', Colors.GREEN)}
  mcp-standards config --init
  mcp-standards sync
  mcp-standards query --project-type api --language python

  {colored('# Validate and fix code', Colors.GREEN)}
  mcp-standards validate src/ --fix --dry-run  # Preview changes
  mcp-standards validate src/ --fix            # Apply fixes

  {colored('# CI/CD pipeline usage', Colors.GREEN)}
  mcp-standards validate . --format sarif --output results.sarif

  {colored('# Start MCP server for IDE integration', Colors.GREEN)}
  mcp-standards serve --port 3000

{colored('CONFIGURATION', Colors.BOLD)}
  Config files are loaded in order:
  1. /etc/mcp-standards/config.yaml       (system)
  2. ~/.config/mcp-standards/config.yaml  (user)
  3. ./.mcp-standards.yaml                (project)
  4. Environment variables (MCP_STANDARDS_*)
  5. Command line options

{colored('MORE INFORMATION', Colors.BOLD)}
  Documentation: https://mcp-standards.dev/docs
  GitHub: https://github.com/williamzujkowski/mcp-standards-server

  Use 'mcp-standards COMMAND --help' for command-specific help.
"""

    parser = argparse.ArgumentParser(
        prog="mcp-standards",
        description=description,
        epilog=epilog,
        formatter_class=ImprovedHelpFormatter,
        add_help=False,  # We'll add custom help
    )

    # Custom help action
    parser.add_argument(
        "-h", "--help", action="help", help="Show this help message and exit"
    )

    # Global options group
    global_opts = parser.add_argument_group(colored("GLOBAL OPTIONS", Colors.BOLD))

    global_opts.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output for debugging",
    )

    global_opts.add_argument(
        "-c",
        "--config",
        type=Path,
        metavar="FILE",
        help="Use custom configuration file",
    )

    global_opts.add_argument(
        "--no-color", action="store_true", help="Disable colored output"
    )

    global_opts.add_argument(
        "--json", action="store_true", help="Output in JSON format (where applicable)"
    )

    global_opts.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0",
        help="Show program version",
    )

    # Commands
    subparsers = parser.add_subparsers(
        title=colored("COMMANDS", Colors.BOLD),
        dest="command",
        help="Available commands",
        metavar="",
    )

    # Sync command
    sync_parser = subparsers.add_parser(
        "sync",
        help="Synchronize standards from repository",
        description=create_sync_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_sync_arguments(sync_parser)

    # Status command
    status_parser = subparsers.add_parser(
        "status",
        help="Show sync status and statistics",
        description=create_status_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_status_arguments(status_parser)

    # Cache command
    cache_parser = subparsers.add_parser(
        "cache",
        help="Manage local cache",
        description=create_cache_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_cache_arguments(cache_parser)

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Show or manage configuration",
        description=create_config_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_config_arguments(config_parser)

    # Query command
    query_parser = subparsers.add_parser(
        "query",
        help="Query standards based on context",
        description=create_query_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_query_arguments(query_parser)

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate code against standards",
        description=create_validate_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_validate_arguments(validate_parser)

    # Serve command
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start MCP server",
        description=create_serve_description(),
        formatter_class=ImprovedHelpFormatter,
    )
    add_serve_arguments(serve_parser)

    return parser


def create_sync_description() -> str:
    """Create detailed description for sync command."""
    return f"""
{colored('Synchronize standards from GitHub repository', Colors.BOLD)}

Downloads standards files from the configured repository and caches them locally.
Supports incremental updates, force sync, and checking for updates.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Basic sync', Colors.GREEN)}
  mcp-standards sync

  {colored('# Check for updates without downloading', Colors.GREEN)}
  mcp-standards sync --check

  {colored('# Force re-download all files', Colors.GREEN)}
  mcp-standards sync --force

  {colored('# Sync only specific files', Colors.GREEN)}
  mcp-standards sync --include "*.yaml" --exclude "*.draft.*"
"""


def add_sync_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for sync command."""
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force sync all files (ignore cache)"
    )

    parser.add_argument(
        "--check", action="store_true", help="Check for updates without downloading"
    )

    parser.add_argument(
        "--include",
        metavar="PATTERN",
        action="append",
        help="Include files matching pattern (can be repeated)",
    )

    parser.add_argument(
        "--exclude",
        metavar="PATTERN",
        action="append",
        help="Exclude files matching pattern (can be repeated)",
    )

    parser.add_argument(
        "--parallel",
        type=int,
        metavar="N",
        default=5,
        help="Number of parallel downloads (default: 5)",
    )


def create_status_description() -> str:
    """Create detailed description for status command."""
    return f"""
{colored('Show synchronization status and statistics', Colors.BOLD)}

Displays information about cached standards, sync history, rate limits,
and system health.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Show basic status', Colors.GREEN)}
  mcp-standards status

  {colored('# Get JSON output for scripting', Colors.GREEN)}
  mcp-standards status --json | jq '.rate_limit'

  {colored('# Show detailed file listing', Colors.GREEN)}
  mcp-standards status --detailed

  {colored('# Check system health', Colors.GREEN)}
  mcp-standards status --check-health
"""


def add_status_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for status command."""
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed information including all files",
    )

    parser.add_argument(
        "--check-health",
        action="store_true",
        help="Perform health checks and report issues",
    )

    parser.add_argument(
        "--summary", action="store_true", help="Show only summary information"
    )


def create_cache_description() -> str:
    """Create detailed description for cache command."""
    return f"""
{colored('Manage local standards cache', Colors.BOLD)}

Tools for managing cached standards files including listing, clearing,
analyzing, and importing/exporting cache contents.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# List cached files', Colors.GREEN)}
  mcp-standards cache --list

  {colored('# Clear outdated files only', Colors.GREEN)}
  mcp-standards cache --clear-outdated

  {colored('# Export cache for backup', Colors.GREEN)}
  mcp-standards cache --export backup.tar.gz

  {colored('# Analyze cache usage', Colors.GREEN)}
  mcp-standards cache --analyze
"""


def add_cache_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for cache command."""
    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument(
        "--list", action="store_true", help="List all cached files"
    )

    action_group.add_argument(
        "--clear", action="store_true", help="Clear all cached files"
    )

    action_group.add_argument(
        "--clear-outdated", action="store_true", help="Clear only outdated files"
    )

    action_group.add_argument(
        "--analyze", action="store_true", help="Analyze cache usage and statistics"
    )

    parser.add_argument(
        "--export", metavar="PATH", help="Export cache to file or directory"
    )

    parser.add_argument(
        "--import",
        metavar="PATH",
        dest="import_path",
        help="Import cache from file or directory",
    )


def create_config_description() -> str:
    """Create detailed description for config command."""
    return f"""
{colored('Manage MCP Standards configuration', Colors.BOLD)}

View, validate, and modify configuration settings. Supports multiple
configuration sources and formats.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Initialize new configuration', Colors.GREEN)}
  mcp-standards config --init

  {colored('# Show current configuration', Colors.GREEN)}
  mcp-standards config --show

  {colored('# Set a configuration value', Colors.GREEN)}
  mcp-standards config --set sync.cache_ttl_hours 48

  {colored('# Validate configuration', Colors.GREEN)}
  mcp-standards config --validate
"""


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for config command."""
    action_group = parser.add_mutually_exclusive_group()

    action_group.add_argument(
        "--init", action="store_true", help="Initialize new configuration interactively"
    )

    action_group.add_argument(
        "--show", action="store_true", help="Display current configuration"
    )

    action_group.add_argument(
        "--validate", action="store_true", help="Validate configuration file"
    )

    action_group.add_argument(
        "--edit", action="store_true", help="Open configuration in editor"
    )

    parser.add_argument("--get", metavar="KEY", help="Get specific configuration value")

    parser.add_argument(
        "--set", nargs=2, metavar=("KEY", "VALUE"), help="Set configuration value"
    )


def create_query_description() -> str:
    """Create detailed description for query command."""
    return f"""
{colored('Query standards based on project context', Colors.BOLD)}

Find applicable standards based on project type, frameworks, languages,
and requirements. Supports semantic search for natural language queries.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Query for React project', Colors.GREEN)}
  mcp-standards query --project-type web --framework react

  {colored('# Natural language query', Colors.GREEN)}
  mcp-standards query --semantic "How to implement authentication?"

  {colored('# Export standards as markdown', Colors.GREEN)}
  mcp-standards query --project-type api --format markdown > standards.md

  {colored('# Use context file', Colors.GREEN)}
  mcp-standards query --context .mcp-context.json
"""


def add_query_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for query command."""
    parser.add_argument(
        "--project-type",
        choices=["web-application", "api", "cli", "library", "mobile", "desktop"],
        help="Type of project",
    )

    parser.add_argument(
        "--framework", action="append", help="Frameworks used (can be repeated)"
    )

    parser.add_argument(
        "--language", action="append", help="Programming languages (can be repeated)"
    )

    parser.add_argument(
        "--requirements", action="append", help="Special requirements (can be repeated)"
    )

    parser.add_argument("--semantic", metavar="QUERY", help="Natural language query")

    parser.add_argument(
        "--context", type=Path, metavar="FILE", help="Load context from JSON file"
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "yaml", "markdown"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--detailed", action="store_true", help="Include full standard content"
    )


def create_validate_description() -> str:
    """Create detailed description for validate command."""
    return f"""
{colored('Validate code against MCP standards', Colors.BOLD)}

Check code files for standards compliance, with options for auto-fixing
issues and various output formats for CI/CD integration.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Validate current directory', Colors.GREEN)}
  mcp-standards validate .

  {colored('# Auto-fix issues', Colors.GREEN)}
  mcp-standards validate src/ --fix

  {colored('# Preview fixes without applying', Colors.GREEN)}
  mcp-standards validate src/ --fix --dry-run

  {colored('# CI/CD integration', Colors.GREEN)}
  mcp-standards validate . --format junit --output results.xml
"""


def add_validate_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for validate command."""
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Paths to validate (default: current directory)",
    )

    parser.add_argument(
        "--fix", action="store_true", help="Automatically fix issues where possible"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without applying",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "junit", "sarif"],
        default="text",
        help="Output format (default: text)",
    )

    parser.add_argument(
        "--severity",
        choices=["error", "warning", "info"],
        default="info",
        help="Minimum severity to report (default: info)",
    )

    parser.add_argument(
        "--fail-on",
        choices=["error", "warning", "info"],
        default="error",
        help="Exit with error if issues at this level (default: error)",
    )

    parser.add_argument(
        "--output", type=Path, metavar="FILE", help="Write results to file"
    )


def create_serve_description() -> str:
    """Create detailed description for serve command."""
    return f"""
{colored('Start MCP Standards Server', Colors.BOLD)}

Launch the Model Context Protocol server for IDE integration and
programmatic access to standards functionality.

{colored('EXAMPLES:', Colors.BOLD)}
  {colored('# Start server on default port', Colors.GREEN)}
  mcp-standards serve

  {colored('# Start on custom port', Colors.GREEN)}
  mcp-standards serve --port 8080

  {colored('# Start in stdio mode for tool integration', Colors.GREEN)}
  mcp-standards serve --stdio

  {colored('# Start as daemon', Colors.GREEN)}
  mcp-standards serve --daemon
"""


def add_serve_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for serve command."""
    parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )

    parser.add_argument(
        "--port", type=int, default=3000, help="Port to listen on (default: 3000)"
    )

    parser.add_argument(
        "--stdio", action="store_true", help="Run in stdio mode for direct integration"
    )

    parser.add_argument(
        "--daemon", action="store_true", help="Run as background daemon"
    )

    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level (default: info)",
    )


def suggest_command(invalid_cmd: str, commands: list[str]) -> str | None:
    """Suggest a command based on similarity."""
    from difflib import get_close_matches

    matches = get_close_matches(invalid_cmd, commands, n=1, cutoff=0.6)
    return matches[0] if matches else None


def main() -> int:
    """Enhanced main entry point."""
    # Handle NO_COLOR environment variable
    if os.environ.get("NO_COLOR"):
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    parser = create_enhanced_parser()

    # Get all valid commands for suggestion

    # Parse arguments
    args = parser.parse_args()

    # If no command provided, show help
    if not args.command:
        parser.print_help()

        # Show quick start hint
        print(f"\n{colored('💡 Quick start:', Colors.YELLOW)}")
        print(
            f"   Try '{colored('mcp-standards config --init', Colors.GREEN)}' to get started"
        )
        print(
            f"   Or  '{colored('mcp-standards --help', Colors.GREEN)}' for more information\n"
        )
        return 1

    # Command handling would go here
    # For now, just print what would be executed
    print(f"{colored('Would execute:', Colors.GREEN)} {args.command}")
    print(f"{colored('With args:', Colors.CYAN)} {args}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
