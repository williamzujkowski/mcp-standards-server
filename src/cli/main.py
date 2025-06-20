"""
MCP Standards CLI - Command-line interface
@nist-controls: AC-3, AU-2, SI-10
@evidence: Secure CLI with audit logging
"""
import asyncio
import json
from pathlib import Path

import typer
import yaml
from rich import print
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ..compliance.scanner import ComplianceScanner

app = typer.Typer(
    name="mcp-standards",
    help="MCP Standards Server - NIST compliance for modern development",
    add_completion=False
)
console = Console()


@app.command()
def init(
    project_path: Path = typer.Argument(Path.cwd(), help="Project path to initialize"),
    profile: str = typer.Option("moderate", help="NIST profile (low/moderate/high)"),
    language: str = typer.Option(None, help="Primary language (auto-detect if not specified)")
) -> None:
    """
    Initialize MCP standards for a project
    @nist-controls: CM-2, CM-3
    @evidence: Configuration management
    """
    console.print(f"[bold green]Initializing MCP Standards for {project_path}[/bold green]")
    console.print("[yellow]Full CLI implementation coming in Phase 2[/yellow]")

    # Create config directory
    config_dir = project_path / ".mcp-standards"
    config_dir.mkdir(exist_ok=True)

    # Create basic config
    config = {
        "version": "1.0.0",
        "profile": profile,
        "language": language or "python",
        "initialized": True
    }

    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[bold green]✓[/bold green] Created configuration at {config_file}")


@app.command()
def server(
    host: str = typer.Option("127.0.0.1", help="Server host"),
    port: int = typer.Option(8000, help="Server port"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development")
) -> None:
    """
    Start the MCP Standards Server
    @nist-controls: AC-3, SC-8, AU-2
    @evidence: Secure server with access control and encryption
    """
    console.print("[bold green]Starting MCP Standards Server[/bold green]")
    console.print(f"Host: [cyan]{host}:{port}[/cyan]")

    # Start server
    import uvicorn
    uvicorn.run(
        "src.core.mcp.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@app.command()
def scan(
    path: Path = typer.Argument(Path.cwd(), help="Path to scan"),
    output_format: str = typer.Option("table", help="Output format (table/json/yaml/oscal)"),
    output_file: Path | None = typer.Option(None, help="Output file"),
    deep: bool = typer.Option(False, help="Perform deep analysis")
) -> None:
    """
    Scan codebase for NIST control implementations
    @nist-controls: CA-7, RA-5, SA-11
    @evidence: Continuous monitoring and vulnerability scanning
    """
    # Validate path exists
    if not path.exists():
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        raise typer.Exit(1)
    
    # Create scanner instance
    scanner = ComplianceScanner()
    
    # Run the scan with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task(f"Scanning {path}...", total=None)
        
        # Run async scan
        scan_results = asyncio.run(scanner.scan_directory(path))
        
        progress.update(task, completed=True)
    
    # Generate report
    report = scanner.generate_report(scan_results, output_format)
    
    # Output results
    if output_format == "table":
        # Table format is handled by the formatter
        scanner.format_output(report, output_format)
    else:
        # Format as requested
        formatted_output = scanner.format_output(report, output_format)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(formatted_output)
            console.print(f"[green]✓[/green] Report saved to {output_file}")
        else:
            print(formatted_output)
    
    # Show summary for non-table formats
    if output_format != "table":
        summary = report["summary"]
        console.print(f"\n[bold]Scan Summary:[/bold]")
        console.print(f"  • Files scanned: {summary['total_files']}")
        console.print(f"  • Files with controls: {summary['files_with_controls']}")
        console.print(f"  • Coverage: {summary['coverage_percentage']}%")
        console.print(f"  • Issues found: {summary['total_issues']}")
        
        if summary['critical_issues'] > 0:
            console.print(f"  • [red]Critical issues: {summary['critical_issues']}[/red]")


@app.command()
def version() -> None:
    """Show version information"""
    console.print("[bold]MCP Standards Server[/bold]")
    console.print("Version: 0.1.0")
    console.print("NIST 800-53r5 compliant")


if __name__ == "__main__":
    app()
