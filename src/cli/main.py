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
from rich.syntax import Syntax
from rich.table import Table

from ..analyzers import GoAnalyzer, JavaAnalyzer, JavaScriptAnalyzer, PythonAnalyzer
from ..compliance.scanner import ComplianceScanner
from ..core.compliance.oscal_handler import OSCALHandler
from .commands import standards as standards_cmd

app = typer.Typer(
    name="mcp-standards",
    help="MCP Standards Server - NIST compliance for modern development",
    add_completion=False
)
console = Console()

# Add standards sub-command
app.add_typer(standards_cmd.app, name="standards", help="Standards version management")


@app.command()
def init(
    project_path: Path = typer.Argument(Path.cwd(), help="Project path to initialize"),
    profile: str = typer.Option("moderate", help="NIST profile (low/moderate/high)"),
    language: str = typer.Option(None, help="Primary language (auto-detect if not specified)"),
    setup_hooks: bool = typer.Option(True, help="Setup Git hooks for compliance")
) -> None:
    """
    Initialize MCP standards for a project
    @nist-controls: CM-2, CM-3
    @evidence: Configuration management
    """
    console.print(f"[bold green]Initializing MCP Standards for {project_path}[/bold green]")

    # Create project directory if it doesn't exist
    project_path.mkdir(parents=True, exist_ok=True)

    # Create config directory
    config_dir = project_path / ".mcp-standards"
    config_dir.mkdir(exist_ok=True)

    # Create compliance directory
    compliance_dir = project_path / "compliance"
    compliance_dir.mkdir(exist_ok=True)

    # Create enhanced config
    config = {
        "version": "1.0.0",
        "profile": profile,
        "language": language or "python",
        "initialized": True,
        "scanning": {
            "include_patterns": ["*.py", "*.js", "*.ts", "*.go", "*.java"],
            "exclude_patterns": ["node_modules/**", "venv/**", ".git/**", "dist/**"]
        },
        "compliance": {
            "required_controls": ["AC-3", "AU-2", "IA-2", "SC-8", "SI-10"],
            "nist_profile": profile
        }
    }

    config_file = config_dir / "config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    console.print(f"[bold green]✓[/bold green] Created configuration at {config_file}")

    # Create compliance documentation
    compliance_readme = compliance_dir / "README.md"
    if not compliance_readme.exists():
        readme_content = f"""# Compliance Documentation

This project follows NIST 800-53r5 {profile} profile.

## Getting Started

1. Run `mcp-standards scan` to analyze your code
2. Generate SSP with `mcp-standards ssp`
3. Review and update control implementations

## Required Controls

{chr(10).join(f"- {control}" for control in config["compliance"]["required_controls"])}

"""
        compliance_readme.write_text(readme_content)
        console.print("[bold green]✓[/bold green] Created compliance documentation")

    # Setup Git hooks if requested
    if setup_hooks and (project_path / ".git").exists():
        _setup_git_hooks(project_path)
        console.print("[bold green]✓[/bold green] Git hooks configured")
    elif setup_hooks:
        console.print("[yellow]Warning: Not a Git repository, skipping hooks setup[/yellow]")

    console.print("\n[bold green]Initialization complete![/bold green]")
    console.print("Next steps:")
    console.print("  1. Review configuration in .mcp-standards/config.yaml")
    console.print("  2. Run 'mcp-standards scan' to analyze your code")
    console.print("  3. Generate SSP with 'mcp-standards ssp'")


def _setup_git_hooks(project_path: Path) -> None:
    """
    Setup Git hooks for compliance checking
    @nist-controls: CM-3, SA-15
    @evidence: Automated compliance verification
    """
    hooks_dir = project_path / ".git" / "hooks"

    # Pre-commit hook
    pre_commit_hook = hooks_dir / "pre-commit"
    hook_content = """#!/bin/bash
# Pre-commit hook for NIST compliance checking
echo "Running NIST compliance validation..."

if command -v mcp-standards &> /dev/null; then
    mcp-standards validate --output-format json > /tmp/compliance-check.json
    if [ $? -ne 0 ]; then
        echo "❌ Compliance validation failed!"
        exit 1
    fi
    echo "✅ Compliance validation passed!"
else
    echo "Warning: mcp-standards not found"
fi
"""
    pre_commit_hook.write_text(hook_content)
    pre_commit_hook.chmod(0o755)


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
    deep: bool = typer.Option(False, help="Perform deep analysis")  # noqa: ARG001
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
        console.print("\n[bold]Scan Summary:[/bold]")
        console.print(f"  • Files scanned: {summary['total_files']}")
        console.print(f"  • Files with controls: {summary['files_with_controls']}")
        console.print(f"  • Coverage: {summary['coverage_percentage']}%")
        console.print(f"  • Issues found: {summary['total_issues']}")

        if summary['critical_issues'] > 0:
            console.print(f"  • [red]Critical issues: {summary['critical_issues']}[/red]")


@app.command()
def ssp(
    output: Path = typer.Option(Path("ssp.json"), help="Output file for SSP"),
    format: str = typer.Option("oscal", help="Output format (oscal/json)"),
    profile: str = typer.Option("moderate", help="NIST profile (low/moderate/high)"),
    path: Path = typer.Argument(Path.cwd(), help="Path to analyze")
) -> None:
    """
    Generate System Security Plan (SSP) from code
    @nist-controls: CA-2, PM-31
    @evidence: Automated SSP generation
    """
    console.print("[bold]Generating System Security Plan...[/bold]")

    # Check if path exists
    if not path.exists():
        console.print(f"[red]Error: Path '{path}' does not exist[/red]")
        raise typer.Exit(1)

    # Initialize components
    oscal_handler = OSCALHandler()

    # Initialize analyzers
    analyzers = {
        '.py': PythonAnalyzer(),
        '.js': JavaScriptAnalyzer(),
        '.jsx': JavaScriptAnalyzer(),
        '.ts': JavaScriptAnalyzer(),
        '.tsx': JavaScriptAnalyzer(),
        '.go': GoAnalyzer(),
        '.java': JavaAnalyzer()
    }

    # Analyze code
    all_annotations = {}
    components = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("Analyzing codebase...", total=None)

        # Scan for files
        for file_path in path.rglob('*'):
            if file_path.is_file() and file_path.suffix in analyzers:
                # Skip common directories
                skip_dirs = {'node_modules', 'venv', '.git', 'dist', 'build', '__pycache__'}
                if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                    continue

                analyzer = analyzers[file_path.suffix]
                annotations = analyzer.analyze_file(file_path)

                if annotations:
                    all_annotations[str(file_path)] = annotations
                    progress.update(task, description=f"Analyzed {file_path.name}")

        progress.update(task, description="Creating OSCAL components...")

        # Create OSCAL components from annotations
        for file_path, annotations in all_annotations.items():
            component_name = Path(file_path).stem
            component = oscal_handler.create_component_from_annotations(
                component_name,
                annotations,
                {
                    "version": "1.0.0",
                    "description": f"Component from {Path(file_path).name}",
                    "component_type": "software"
                }
            )
            components.append(component)

        progress.update(task, description="Generating SSP...")

        # Generate SSP
        system_name = path.name or "System"
        ssp_metadata = {
            "version": "1.0.0",
            "description": f"System Security Plan for {system_name}",
            "sensitivity": profile,
            "system_id": system_name.lower().replace(" ", "-"),
            "status": "operational",
            "confidentiality_impact": profile,
            "integrity_impact": profile,
            "availability_impact": profile
        }

        ssp_content = oscal_handler.generate_ssp_content(
            system_name,
            components,
            f"NIST_SP-800-53_rev5_{profile.upper()}",
            ssp_metadata
        )

        # Export
        if format == "oscal":
            output_path, checksum_path = oscal_handler.export_to_file(ssp_content, output, "json")
            console.print(f"[green]✓[/green] SSP generated: {output_path}")
            console.print(f"[green]✓[/green] Checksum: {checksum_path}")
        else:
            with open(output, 'w') as f:
                json.dump(ssp_content, f, indent=2)
            console.print(f"[green]✓[/green] SSP generated: {output}")

        progress.update(task, completed=True)

    # Show summary
    console.print("\n[bold]SSP Generation Summary:[/bold]")
    console.print(f"  • Files analyzed: {len(all_annotations)}")
    console.print(f"  • Components created: {len(components)}")
    console.print(f"  • Profile: {profile}")

    # Show control coverage
    all_controls = set()
    for annotations in all_annotations.values():
        for ann in annotations:
            all_controls.update(ann.control_ids)

    console.print(f"  • Unique controls found: {len(all_controls)}")
    if all_controls:
        console.print(f"  • Control families: {', '.join(sorted({c.split('-')[0] for c in all_controls}))}")


@app.command()
def generate(
    template_type: str = typer.Argument(
        ...,
        help="Type of template to generate (api, auth, logging, encryption, database)"
    ),
    language: str = typer.Option(
        "python",
        help="Programming language",
        show_choices=True
    ),
    output: Path = typer.Option(
        None,
        help="Output file path (defaults to stdout)"
    ),
    controls: str = typer.Option(
        "",
        help="Comma-separated NIST controls to implement"
    )
) -> None:
    """
    Generate NIST-compliant code templates
    @nist-controls: SA-11, SA-15
    @evidence: Secure code generation
    """
    from ..core.templates import TemplateGenerator

    console.print(f"[bold]Generating {template_type} template...[/bold]")

    # Parse controls
    control_list = [c.strip() for c in controls.split(",") if c.strip()] if controls else []

    # Initialize generator
    generator = TemplateGenerator()

    try:
        # Generate template
        template_content = generator.generate(
            template_type=template_type,
            language=language,
            controls=control_list
        )

        # Output
        if output:
            output.write_text(template_content)
            console.print(f"[green]✓[/green] Template written to: {output}")
        else:
            console.print("\n[dim]--- Generated Template ---[/dim]\n")
            console.print(Syntax(template_content, language, theme="monokai"))

        # Show implemented controls
        if control_list:
            console.print(f"\n[bold]Implemented controls:[/bold] {', '.join(control_list)}")

    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from e


@app.command()
def validate(
    path: Path = typer.Argument(Path.cwd(), help="File or directory to validate"),
    profile: str = typer.Option("moderate", help="NIST profile (low/moderate/high)"),
    controls: str = typer.Option("", help="Specific controls to check (comma-separated)"),
    output_format: str = typer.Option("table", help="Output format (table/json/yaml)")
) -> None:
    """
    Validate code against NIST compliance requirements
    @nist-controls: CA-2, CA-7
    @evidence: Continuous monitoring and assessment
    """
    from ..analyzers import get_analyzer_for_file

    console.print(f"[bold]Validating {path} against {profile} profile...[/bold]\n")

    # Parse controls
    control_list = [c.strip() for c in controls.split(",") if c.strip()] if controls else []

    results = {"files": {}, "summary": {"total_files": 0, "compliant_files": 0, "controls_found": set()}}

    # Scan files
    files_to_scan = []
    if path.is_file():
        files_to_scan = [path]
    else:
        files_to_scan = list(path.rglob("*.py")) + list(path.rglob("*.js")) + list(path.rglob("*.go")) + list(path.rglob("*.java"))

    with Progress() as progress:
        task = progress.add_task("Validating...", total=len(files_to_scan))

        for file_path in files_to_scan:
            progress.update(task, description=f"Validating {file_path.name}")

            # Skip common directories
            skip_dirs = {'node_modules', 'venv', '.git', 'dist', 'build', '__pycache__'}
            if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                continue

            analyzer = get_analyzer_for_file(file_path)
            if analyzer:
                annotations = analyzer.analyze_file(file_path)
                file_controls = set()
                for ann in annotations:
                    file_controls.update(ann.control_ids)

                results["files"][str(file_path)] = {
                    "controls": list(file_controls),
                    "annotations": len(annotations),
                    "compliant": bool(file_controls)
                }
                results["summary"]["controls_found"].update(file_controls)
                if file_controls:
                    results["summary"]["compliant_files"] += 1
                results["summary"]["total_files"] += 1

            progress.advance(task)

    # Convert set to list for JSON serialization
    results["summary"]["controls_found"] = sorted(results["summary"]["controls_found"])

    # Output results
    if output_format == "json":
        console.print_json(data=results)
    elif output_format == "yaml":
        import yaml
        console.print(yaml.dump(results, default_flow_style=False))
    else:
        # Table format
        table = Table(title="Validation Results")
        table.add_column("File", style="cyan")
        table.add_column("Controls", style="green")
        table.add_column("Annotations", style="yellow")
        table.add_column("Status", style="bold")

        for file_path, file_data in results["files"].items():
            status = "[green]✓[/green]" if file_data["compliant"] else "[red]✗[/red]"
            table.add_row(
                Path(file_path).name,
                ", ".join(file_data["controls"][:3]) + ("..." if len(file_data["controls"]) > 3 else ""),
                str(file_data["annotations"]),
                status
            )

        console.print(table)

        # Summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  • Total files: {results['summary']['total_files']}")
        console.print(f"  • Compliant files: {results['summary']['compliant_files']}")
        console.print(f"  • Unique controls: {len(results['summary']['controls_found'])}")

        if control_list:
            missing = set(control_list) - set(results['summary']['controls_found'])
            if missing:
                console.print(f"  • [red]Missing controls:[/red] {', '.join(sorted(missing))}")


@app.command()
def coverage(
    path: Path = typer.Argument(Path.cwd(), help="Path to analyze"),
    output_format: str = typer.Option("markdown", help="Output format (markdown/json/html)"),
    output_file: Path | None = typer.Option(None, help="Output file (stdout if not specified)"),
    detailed: bool = typer.Option(False, help="Include detailed file-level analysis")
) -> None:
    """
    Generate NIST control coverage report
    @nist-controls: CA-7, AU-6, PM-31
    @evidence: Automated coverage analysis and reporting
    """
    from ..analyzers.control_coverage_report import ControlCoverageReporter

    console.print(f"[bold]Analyzing NIST control coverage for {path}[/bold]\n")

    # Initialize reporter
    reporter = ControlCoverageReporter()

    # Initialize analyzers
    analyzers = {
        'python': PythonAnalyzer(),
        'javascript': JavaScriptAnalyzer(),
        'go': GoAnalyzer(),
        'java': JavaAnalyzer()
    }

    # Analyze project
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console
    ) as progress:
        task = progress.add_task("Analyzing project...", total=None)
        metrics = reporter.analyze_project(path, analyzers)
        progress.update(task, completed=True)

    # Generate report
    report = reporter.generate_report(metrics, output_format)

    # Output report
    if output_file:
        output_file.write_text(report)
        console.print(f"[green]✓[/green] Coverage report saved to {output_file}")
    else:
        if output_format == "markdown":
            # Pretty print markdown to console
            from rich.markdown import Markdown
            console.print(Markdown(report))
        else:
            print(report)

    # Summary statistics
    if output_file or output_format != "markdown":
        console.print("\n[bold]Summary:[/bold]")
        console.print(f"  • Total controls: [cyan]{metrics.total_controls_detected}[/cyan]")
        console.print(f"  • Control families: [cyan]{len(metrics.control_families)}[/cyan]")
        console.print(f"  • High confidence: [green]{len(metrics.high_confidence_controls)}[/green]")

        if metrics.suggested_missing_controls:
            console.print(f"  • [yellow]Suggested additions: {sum(len(v) for v in metrics.suggested_missing_controls.values())}[/yellow]")


@app.command()
def version() -> None:
    """Show version information"""
    console.print("[bold]MCP Standards Server[/bold]")
    console.print("Version: 0.1.0")
    console.print("NIST 800-53r5 compliant")


@app.command()
def cache(
    action: str = typer.Argument(..., help="Action: status, clear, optimize"),
    tier: str = typer.Option(None, "--tier", "-t", help="Specific tier: redis, faiss, stats, or all"),
    force: bool = typer.Option(False, "--force", "-f", help="Force action without confirmation")
) -> None:
    """
    Manage hybrid vector store cache
    @nist-controls: SI-12, AU-12
    @evidence: Cache management with audit logging
    """
    from ..core.standards.engine import StandardsEngine
    from ..core.redis_client import get_redis_client
    
    async def run_cache_command() -> None:
        # Initialize engine with hybrid search
        engine = StandardsEngine(
            standards_path=Path("data/standards"),
            redis_client=get_redis_client(),
            enable_hybrid_search=True
        )
        
        if action == "status":
            # Get tier statistics
            stats = await engine.get_tier_stats()
            
            console.print("[bold]Hybrid Vector Store Status[/bold]")
            console.print(f"Enabled: {stats['hybrid_search_enabled']}")
            
            if stats['hybrid_search_enabled'] and 'storage_tiers' in stats:
                tiers = stats['storage_tiers'].get('tiers', {})
                
                # FAISS tier
                if 'faiss' in tiers:
                    faiss = tiers['faiss']
                    console.print("\n[bold cyan]FAISS Hot Cache:[/bold cyan]")
                    console.print(f"  Items: {faiss.get('size', 0)}/{faiss.get('capacity', 1000)}")
                    console.print(f"  Utilization: {faiss.get('utilization', 0):.1%}")
                    console.print(f"  Hit Rate: {faiss.get('hit_rate', 0):.1%}")
                    console.print(f"  Avg Latency: {faiss.get('avg_latency_ms', 0):.2f}ms")
                
                # Redis tier
                if 'redis' in tiers:
                    redis = tiers['redis']
                    console.print("\n[bold red]Redis Query Cache:[/bold red]")
                    console.print(f"  Connected: {redis.get('redis_connected', False)}")
                    console.print(f"  Hit Rate: {redis.get('hit_rate', 0):.1%}")
                    console.print(f"  Avg Latency: {redis.get('avg_latency_ms', 0):.2f}ms")
                
                # ChromaDB tier
                if 'chromadb' in tiers:
                    chroma = tiers['chromadb']
                    console.print("\n[bold green]ChromaDB Persistent Storage:[/bold green]")
                    console.print(f"  Status: {chroma.get('status', 'unknown')}")
                    console.print(f"  Documents: {chroma.get('total_documents', 0)}")
                    console.print(f"  Hit Rate: {chroma.get('hit_rate', 0):.1%}")
                
                # Access patterns
                if 'access_patterns' in stats:
                    patterns = stats['access_patterns']
                    console.print("\n[bold]Access Patterns:[/bold]")
                    console.print(f"  Documents Tracked: {patterns.get('total_documents_tracked', 0)}")
                    console.print(f"  Above Threshold: {patterns.get('documents_above_threshold', 0)}")
                    
        elif action == "clear":
            if not force:
                confirm = typer.confirm(f"Clear cache for tier '{tier or 'all'}'?")
                if not confirm:
                    console.print("[yellow]Cancelled[/yellow]")
                    return
            
            results = await engine.clear_cache(tier)
            
            if 'status' in results and results['status'] == 'not_enabled':
                console.print("[red]Hybrid search not enabled[/red]")
            else:
                for cleared_tier in results.get('cleared', []):
                    console.print(f"[green]✓[/green] Cleared {cleared_tier} cache")
                    
        elif action == "optimize":
            console.print("[bold]Running tier optimization...[/bold]")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("Optimizing...", total=None)
                
                results = await engine.optimize_tiers()
                
                progress.update(task, completed=True)
            
            if 'status' in results and results['status'] == 'not_enabled':
                console.print("[red]Hybrid search not enabled[/red]")
            else:
                console.print(f"\n[bold]Optimization Results:[/bold]")
                console.print(f"Cache utilization before: {results.get('cache_utilization_before', 0):.1%}")
                console.print(f"Cache utilization after: {results.get('cache_utilization_after', 0):.1%}")
                console.print(f"Evictions: {len(results.get('evictions', []))}")
                console.print(f"Promotions: {len(results.get('promotions', []))}")
                console.print(f"Duration: {results.get('duration_ms', 0):.1f}ms")
                
                # Show top evictions/promotions
                if results.get('evictions'):
                    console.print("\n[yellow]Top Evictions:[/yellow]")
                    for eviction in results['evictions'][:5]:
                        console.print(f"  - {eviction['document_id']} (score: {eviction['score']:.3f})")
                
                if results.get('promotions'):
                    console.print("\n[green]Top Promotions:[/green]")
                    for promotion in results['promotions'][:5]:
                        console.print(f"  + {promotion['document_id']} (score: {promotion['score']:.3f})")
        else:
            console.print(f"[red]Unknown action: {action}[/red]")
            console.print("Available actions: status, clear, optimize")
    
    # Run async command
    asyncio.run(run_cache_command())


if __name__ == "__main__":
    app()
