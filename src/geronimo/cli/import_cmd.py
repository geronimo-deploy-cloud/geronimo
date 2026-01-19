"""Import command for Geronimo CLI.

Imports existing UV-managed FastAPI projects into the Geronimo deployment framework.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from geronimo.analyzers import DeterministicAnalyzer
from geronimo.config.loader import save_config
from geronimo.config.schema import (
    DeploymentConfig,
    EnvironmentConfig,
    GeronimoConfig,
    InfrastructureConfig,
    ModelConfig,
    MonitoringConfig,
    ProjectConfig,
    RuntimeConfig,
    ScalingConfig,
)
from geronimo.scanners import ProjectScanner

console = Console()


def import_project(
    project_path: str = typer.Argument(
        ...,
        help="Path to the existing UV-managed FastAPI project.",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Deployment name (defaults to project name from pyproject.toml).",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for geronimo.yaml (defaults to project directory).",
    ),
    services_dir: Optional[str] = typer.Option(
        None,
        "--services",
        "-s",
        help="Path to services directory (relative to project root).",
    ),
    model_dir: Optional[str] = typer.Option(
        None,
        "--models",
        "-m",
        help="Path to model artifacts directory (relative to project root).",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactive mode for resolving ambiguities (future feature).",
    ),
) -> None:
    """Import an existing UV-managed FastAPI project.

    Analyzes your existing project structure and generates a geronimo.yaml
    configuration file for deployment. Detects:

    \b
    - ML framework from pyproject.toml dependencies
    - API endpoints and their schemas
    - Service modules and preprocessing chains
    - Model artifacts
    """
    console.print("\n[bold blue]Importing project...[/bold blue]\n")

    # Convert to Path
    project_path = Path(project_path).resolve()

    try:
        # Scan the project
        console.print("[dim]Scanning project...[/dim]")
        scanner = ProjectScanner(project_path)
        scan = scanner.scan()

        # Display scan results
        _display_scan_results(scan)

        # Analyze the project
        console.print("\n[dim]Analyzing project structure...[/dim]")
        analyzer = DeterministicAnalyzer()
        result = analyzer.analyze(scan)

        # Display analysis results
        _display_analysis_results(result)

        # Generate configuration
        console.print("\n[dim]Generating configuration...[/dim]")
        config = _generate_config(
            scan=scan,
            result=result,
            name_override=name,
        )

        # Determine output path
        output_path = Path(output) if output else project_path
        config_file = output_path / "geronimo.yaml"

        # Check for existing config
        if config_file.exists():
            overwrite = typer.confirm(
                f"geronimo.yaml already exists at {config_file}. Overwrite?"
            )
            if not overwrite:
                console.print("[yellow]Import cancelled.[/yellow]")
                raise typer.Exit(code=0)

        # Save configuration
        save_config(config, config_file)

        # Generate SDK wrappers
        console.print("\n[dim]Generating SDK wrappers...[/dim]")
        from geronimo.analyzers.sdk_wrapper import SDKWrapperGenerator

        wrapper_gen = SDKWrapperGenerator(project_path)
        sdk_result = wrapper_gen.analyze()

        # Create geronimo_sdk directory
        sdk_dir = output_path / "geronimo_sdk"
        sdk_dir.mkdir(exist_ok=True)

        # Write generated files
        (sdk_dir / "__init__.py").write_text('"""Geronimo SDK wrappers."""\n')
        (sdk_dir / "features.py").write_text(sdk_result.feature_set_code)
        (sdk_dir / "data_sources.py").write_text(sdk_result.data_sources_code)
        (sdk_dir / "model.py").write_text(sdk_result.model_code)
        (sdk_dir / "endpoint.py").write_text(sdk_result.endpoint_code)
        (sdk_dir / "IMPORT_SUMMARY.md").write_text(wrapper_gen.generate_summary())

        console.print(f"  ✓ Generated SDK wrappers in [cyan]geronimo_sdk/[/cyan]")
        console.print(f"  ✓ {len(sdk_result.detected_patterns)} patterns detected")
        console.print(f"  ✓ {len(sdk_result.todos)} TODO items created")

        # Display summary
        _display_summary(config_file, result, sdk_result)

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        raise typer.Exit(code=1)


def _display_scan_results(scan) -> None:
    """Display project scan results."""
    console.print(f"  ✓ Found [cyan]pyproject.toml[/cyan] (UV managed)")
    console.print(f"  ✓ Project: [bold]{scan.project_name}[/bold]")
    console.print(f"  ✓ Python: {scan.python_version}")
    console.print(f"  ✓ Dependencies: {len(scan.dependencies)} packages")

    if scan.src_package:
        console.print(f"  ✓ Source package: [cyan]{scan.src_package.name}[/cyan]")
    if scan.services_dir:
        console.print(f"  ✓ Services directory: [cyan]services/[/cyan]")
    if scan.api_dir:
        console.print(f"  ✓ API directory: [cyan]api/[/cyan]")
    if scan.model_artifacts:
        console.print(f"  ✓ Model artifacts: {len(scan.model_artifacts)} files")


def _display_analysis_results(result) -> None:
    """Display analysis results."""
    console.print(f"\n  Framework: [bold]{result.detected_framework.value}[/bold]")
    console.print(f"  Endpoints: {len(result.endpoints)}")
    console.print(f"  Services: {len(result.services)}")
    console.print(f"  Preprocessing steps: {len(result.preprocessing_chain)}")
    console.print(f"  Confidence: {result.confidence:.0%}")

    # Show endpoints table if any found
    if result.endpoints:
        console.print("\n[bold]Detected Endpoints:[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Method")
        table.add_column("Path")
        table.add_column("Handler")

        for ep in result.endpoints[:5]:  # Limit to first 5
            table.add_row(ep.method, ep.path, ep.handler_function)

        if len(result.endpoints) > 5:
            table.add_row("...", f"+{len(result.endpoints) - 5} more", "")

        console.print(table)

    # Show warnings
    if result.warnings:
        console.print("\n[yellow]Warnings:[/yellow]")
        for warning in result.warnings:
            console.print(f"  • {warning}")

    # Show items needing review
    if result.needs_review:
        console.print("\n[bold yellow]⚠ Items needing review:[/bold yellow]")
        for item in result.needs_review:
            console.print(f"  • {item}")


def _generate_config(scan, result, name_override: str | None) -> GeronimoConfig:
    """Generate GeronimoConfig from analysis results."""
    project_name = name_override or scan.project_name

    # Estimate memory based on model size
    memory = 1024
    if result.model_artifacts:
        largest = max(a.size_bytes for a in result.model_artifacts)
        if largest > 500_000_000:  # > 500MB
            memory = 4096
        elif largest > 100_000_000:  # > 100MB
            memory = 2048

    # Determine model artifact path
    model_path = "models/model.joblib"
    primary = next((a for a in result.model_artifacts if a.is_primary), None)
    if primary:
        try:
            model_path = str(primary.path.relative_to(scan.root))
        except ValueError:
            model_path = str(primary.path)

    return GeronimoConfig(
        project=ProjectConfig(
            name=project_name,
            version="1.0.0",
            description=f"Imported from {scan.project_name}",
        ),
        model=ModelConfig(
            type="realtime",
            framework=result.detected_framework,
            artifact_path=model_path,
        ),
        runtime=RuntimeConfig(
            python_version=result.python_version,
            dependencies=result.dependencies,
        ),
        infrastructure=InfrastructureConfig(
            cpu=512,
            memory=memory,
            scaling=ScalingConfig(
                min_instances=1,
                max_instances=4,
            ),
        ),
        monitoring=MonitoringConfig(
            metrics=[
                "latency_p50",
                "latency_p99",
                "error_rate",
                "request_count",
            ],
            dashboard_enabled=True,
        ),
        deployment=DeploymentConfig(
            environments=[
                EnvironmentConfig(name="dev", auto_deploy=True),
                EnvironmentConfig(name="prod", approval_required=True),
            ],
        ),
    )


def _display_summary(config_file: Path, result, sdk_result=None) -> None:
    """Display import summary."""
    sdk_info = ""
    if sdk_result:
        high_todos = len([t for t in sdk_result.todos if t.priority.value == "HIGH"])
        sdk_info = (
            f"\nSDK wrappers: [cyan]geronimo_sdk/[/cyan]\n"
            f"Patterns detected: {len(sdk_result.detected_patterns)}\n"
            f"TODO items: {len(sdk_result.todos)} ({high_todos} HIGH priority)\n"
        )

    console.print(
        Panel.fit(
            f"[bold green]✓ Configuration generated![/bold green]\n\n"
            f"Config file: [cyan]{config_file}[/cyan]\n"
            f"{sdk_info}\n"
            f"Next steps:\n"
            f"  1. Review [cyan]geronimo_sdk/IMPORT_SUMMARY.md[/cyan] for TODOs\n"
            f"  2. Customize [cyan]geronimo_sdk/[/cyan] files\n"
            f"  3. Run [cyan]geronimo validate[/cyan]\n"
            f"  4. Run [cyan]geronimo generate all[/cyan]",
            border_style="green",
        )
    )

