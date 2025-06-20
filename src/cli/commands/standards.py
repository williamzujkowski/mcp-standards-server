"""
Standards management CLI commands
@nist-controls: CM-2, CM-3, CM-4
@evidence: CLI interface for standards version management
"""
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ...core.standards.versioning import (
    StandardsVersionManager,
    UpdateConfiguration,
    UpdateFrequency,
    VersioningStrategy,
)

app = typer.Typer(help="Standards version management commands")
console = Console()


@app.command()
def version(
    standard_id: str = typer.Argument(..., help="Standard ID to check version"),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
):
    """
    Check version of a standard
    @nist-controls: CM-2
    @evidence: Version inquiry command
    """
    try:
        manager = StandardsVersionManager(standards_dir)
        
        # Get version history
        versions = manager.get_version_history(standard_id)
        latest = manager.get_latest_version(standard_id)
        
        if not versions:
            console.print(f"[yellow]No version history found for {standard_id}[/yellow]")
            return
        
        # Display version table
        table = Table(title=f"Version History: {standard_id}")
        table.add_column("Version", style="cyan")
        table.add_column("Created", style="green")
        table.add_column("Author", style="yellow")
        table.add_column("Strategy", style="magenta")
        table.add_column("Latest", style="bold")
        
        for v in versions:
            is_latest = v.version == latest
            table.add_row(
                v.version,
                v.created_at.strftime("%Y-%m-%d %H:%M"),
                v.author or "unknown",
                v.strategy.value,
                "✓" if is_latest else ""
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def update(
    source_url: str = typer.Option(
        "https://raw.githubusercontent.com/williamzujkowski/standards/main",
        "--source",
        "-s",
        help="Source repository URL"
    ),
    standards: list[str] = typer.Option(
        None,
        "--standard",
        "-std",
        help="Specific standards to update (can be repeated)"
    ),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
    backup: bool = typer.Option(True, "--backup/--no-backup", help="Create backup before update"),
    validate: bool = typer.Option(True, "--validate/--no-validate", help="Validate updates"),
):
    """
    Update standards from remote source
    @nist-controls: CM-3, CM-4
    @evidence: Controlled standards update process
    """
    console.print(f"[cyan]Updating standards from {source_url}...[/cyan]")
    
    try:
        # Configure update
        config = UpdateConfiguration(
            source_url=source_url,
            backup_enabled=backup,
            validation_required=validate,
            allowed_sources=[source_url]
        )
        
        manager = StandardsVersionManager(standards_dir, config=config)
        
        # Run async update
        import asyncio
        report = asyncio.run(manager.update_from_source(source_url, standards))
        
        # Display results
        console.print("\n[bold]Update Report:[/bold]")
        console.print(f"Timestamp: {report['timestamp']}")
        console.print(f"Source: {report['source']}")
        
        if report.get("updated"):
            console.print(f"\n[green]✓ Updated ({len(report['updated'])}):[/green]")
            for item in report["updated"]:
                console.print(f"  - {item['standard']} → {item['version']}")
        
        if report.get("failed"):
            console.print(f"\n[red]✗ Failed ({len(report['failed'])}):[/red]")
            for item in report["failed"]:
                console.print(f"  - {item['standard']}: {item['reason']}")
        
        if report.get("skipped"):
            console.print(f"\n[yellow]⚠ Skipped ({len(report['skipped'])}):[/yellow]")
            for std in report["skipped"]:
                console.print(f"  - {std}")
        
        if report.get("error"):
            console.print(f"\n[red]Error: {report['error']}[/red]")
            raise typer.Exit(1)
        
    except Exception as e:
        console.print(f"[red]Update failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def compare(
    standard_id: str = typer.Argument(..., help="Standard ID to compare"),
    version1: str = typer.Argument(..., help="First version"),
    version2: str = typer.Argument(..., help="Second version"),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
):
    """
    Compare two versions of a standard
    @nist-controls: CM-3
    @evidence: Version comparison capability
    """
    try:
        manager = StandardsVersionManager(standards_dir)
        
        # Run async comparison
        import asyncio
        diff = asyncio.run(manager.compare_versions(standard_id, version1, version2))
        
        # Display comparison
        console.print(f"\n[bold]Version Comparison: {standard_id}[/bold]")
        console.print(f"Old: {diff.old_version} → New: {diff.new_version}")
        console.print(f"Impact Level: [{'red' if diff.impact_level == 'high' else 'yellow'}]{diff.impact_level}[/]")
        console.print(f"Breaking Changes: [{'red' if diff.breaking_changes else 'green'}]{diff.breaking_changes}[/]")
        
        if diff.added_sections:
            console.print(f"\n[green]+ Added Sections ({len(diff.added_sections)}):[/green]")
            for section in diff.added_sections:
                console.print(f"  + {section}")
        
        if diff.removed_sections:
            console.print(f"\n[red]- Removed Sections ({len(diff.removed_sections)}):[/red]")
            for section in diff.removed_sections:
                console.print(f"  - {section}")
        
        if diff.modified_sections:
            console.print(f"\n[yellow]~ Modified Sections ({len(diff.modified_sections)}):[/yellow]")
            for section in diff.modified_sections:
                console.print(f"  ~ {section}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def rollback(
    standard_id: str = typer.Argument(..., help="Standard ID to rollback"),
    target_version: str = typer.Argument(..., help="Version to rollback to"),
    reason: str = typer.Option(..., "--reason", "-r", help="Reason for rollback"),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
):
    """
    Rollback a standard to a previous version
    @nist-controls: CM-3
    @evidence: Controlled rollback functionality
    """
    # Confirm rollback
    if not typer.confirm(f"Rollback {standard_id} to version {target_version}?"):
        console.print("[yellow]Rollback cancelled[/yellow]")
        return
    
    try:
        manager = StandardsVersionManager(standards_dir)
        
        # Run async rollback
        import asyncio
        version = asyncio.run(manager.rollback_version(standard_id, target_version, reason))
        
        console.print(f"[green]✓ Rollback successful![/green]")
        console.print(f"New version: {version.version}")
        console.print(f"Reason: {reason}")
        
    except Exception as e:
        console.print(f"[red]Rollback failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def schedule(
    frequency: UpdateFrequency = typer.Option(
        UpdateFrequency.MONTHLY,
        "--frequency",
        "-f",
        help="Update frequency"
    ),
    enable: bool = typer.Option(True, "--enable/--disable", help="Enable auto-updates"),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
):
    """
    Configure automatic updates schedule
    @nist-controls: CM-3
    @evidence: Update scheduling configuration
    """
    try:
        config = UpdateConfiguration(
            update_frequency=frequency,
            auto_update=enable
        )
        
        manager = StandardsVersionManager(standards_dir, config=config)
        
        # Schedule updates
        import asyncio
        asyncio.run(manager.schedule_updates())
        
        if enable:
            console.print(f"[green]✓ Auto-updates enabled: {frequency.value}[/green]")
        else:
            console.print("[yellow]Auto-updates disabled[/yellow]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def create_version(
    standard_id: str = typer.Argument(..., help="Standard ID"),
    changelog: str = typer.Option(..., "--changelog", "-c", help="Version changelog"),
    author: str = typer.Option(None, "--author", "-a", help="Version author"),
    strategy: VersioningStrategy = typer.Option(
        VersioningStrategy.SEMANTIC,
        "--strategy",
        "-s",
        help="Versioning strategy"
    ),
    standards_dir: Path = typer.Option(
        Path("data/standards"),
        "--dir",
        "-d",
        help="Standards directory"
    ),
):
    """
    Create a new version of a standard
    @nist-controls: CM-2
    @evidence: Manual version creation
    """
    try:
        manager = StandardsVersionManager(standards_dir)
        
        # Load current content
        std_files = list(standards_dir.glob(f"*{standard_id}*.yaml"))
        if not std_files:
            console.print(f"[red]Standard {standard_id} not found[/red]")
            raise typer.Exit(1)
        
        import yaml
        with open(std_files[0]) as f:
            content = yaml.safe_load(f)
        
        # Create version
        import asyncio
        version = asyncio.run(
            manager.create_version(
                standard_id,
                content,
                author=author,
                changelog=changelog,
                strategy=strategy
            )
        )
        
        console.print(f"[green]✓ Version created: {version.version}[/green]")
        console.print(f"Author: {author or 'unknown'}")
        console.print(f"Changelog: {changelog}")
        console.print(f"Strategy: {strategy.value}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()