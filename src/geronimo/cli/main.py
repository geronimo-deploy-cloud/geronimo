"""Main CLI entrypoint for Geronimo.

This module provides the primary CLI interface using Typer.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from geronimo import __version__
from geronimo.constants import MODULES

# Create the main Typer app
app = typer.Typer(
    name="geronimo",
    help="MLOps deployment platform - automate ML model deployments to AWS",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Console for rich output
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(
            Panel.fit(
                f"[bold blue]Geronimo[/bold blue] v{__version__}\n"
                "[dim]MLOps Deployment Platform[/dim]",
                border_style="blue",
            )
        )
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Geronimo - Automate ML model deployments to AWS.

    Generate production-ready Terraform, Docker, and CI/CD pipelines
    for your ML models with industry best practices built-in.
    """
    pass


# ============================================================================
# INIT Command
# ============================================================================


@app.command()
def init(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        prompt="Project name",
        help="Name of the ML project.",
    ),
    framework: str = typer.Option(
        "sklearn",
        "--framework",
        "-f",
        help="ML framework (sklearn, pytorch, tensorflow).",
    ),
    template: str = typer.Option(
        "realtime",
        "--template",
        "-t",
        help="Project type: 'realtime' (API endpoints), 'batch' (pipelines), or 'both'.",
    ),
    output_dir: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Output directory for the project.",
    ),
) -> None:
    """Initialize a new ML deployment project.

    Scaffolds a complete ML project with Geronimo SDK:

    Templates:
    - realtime: FastAPI endpoints with Endpoint class
    - batch: Metaflow pipelines with BatchPipeline class
    - both: Combined real-time and batch support
    """
    from geronimo.generators.project import ProjectGenerator

    # Validate template
    valid_templates = {"realtime", "batch", "both"}
    if template not in valid_templates:
        console.print(f"[bold red]Error:[/bold red] Invalid template '{template}'. Choose from: {valid_templates}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]Initializing project:[/bold blue] {name}")
    console.print(f"  Template: [cyan]{template}[/cyan]")

    generator = ProjectGenerator(
        project_name=name,
        framework=framework,
        output_dir=output_dir,
        template=template,
    )

    try:
        generator.generate()

        # SDK scaffolding is now handled by ProjectGenerator.generate()
        # which creates sdk/endpoint.py, sdk/pipeline.py, app.py, flow.py, etc.

        next_steps = [
            f"cd {name}",
            "uv sync",
        ]
        if template in ("realtime", "both"):
            next_steps.append(f"uvicorn {name.replace('-', '_')}.app:app --reload  # Run API server")
        if template in ("batch", "both"):
            next_steps.append(f"python -m {name.replace('-', '_')}.flow run  # Run batch pipeline")

        console.print(
            Panel.fit(
                f"[bold green]✓ Project '{name}' created successfully![/bold green]\n\n"
                f"Template: [cyan]{template}[/cyan]\n\n"
                f"Next steps:\n" + "\n".join(f"  {i+1}. {step}" for i, step in enumerate(next_steps)),
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)





# ============================================================================
# GENERATE Command Group
# ============================================================================

generate_app = typer.Typer(
    name="generate",
    help="Generate deployment artifacts (Terraform, Docker, pipelines).",
    no_args_is_help=True,
)
app.add_typer(generate_app, name="generate")

# Import and register keys CLI
from geronimo.cli.keys_cmd import keys_app
app.add_typer(keys_app, name="keys")

# Import and register auth CLI
from geronimo.cli.auth_cmd import auth_app
app.add_typer(auth_app, name="auth")

# Import and register config CLI
from geronimo.cli.config_cmd import config_app
app.add_typer(config_app, name="config")


@generate_app.command("terraform")
def generate_terraform(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_dir: str = typer.Option(
        "infrastructure",
        "--output",
        "-o",
        help="Output directory for Terraform files.",
    ),
) -> None:
    """Generate Terraform infrastructure files.

    Creates modular Terraform configuration for:
    - ECR repository
    - ECS Fargate task and service
    - Application Load Balancer
    - CloudWatch logging and monitoring
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.terraform import TerraformGenerator

    console.print("\n[bold blue]Generating Terraform...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = TerraformGenerator(config=config, output_dir=output_dir)
        files = generator.generate()

        console.print(f"[green]✓ Generated {len(files)} Terraform files:[/green]")
        for f in files:
            console.print(f"  • {f}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("dockerfile")
def generate_dockerfile(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_path: str = typer.Option(
        "Dockerfile",
        "--output",
        "-o",
        help="Output path for Dockerfile.",
    ),
) -> None:
    """Generate an optimized Dockerfile for ML serving.

    Creates a multi-stage Dockerfile with:
    - UV for fast dependency installation
    - Non-root user for security
    - Proper signal handling for graceful shutdown
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.docker import DockerGenerator

    console.print("\n[bold blue]Generating Dockerfile...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = DockerGenerator(config=config, output_path=output_path)
        generator.generate()

        console.print(f"[green]✓ Generated Dockerfile:[/green] {output_path}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("pipeline")
def generate_pipeline(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_path: str = typer.Option(
        "azure-pipelines.yaml",
        "--output",
        "-o",
        help="Output path for pipeline file.",
    ),
) -> None:
    """Generate CI/CD pipeline configuration.

    Creates Azure DevOps pipeline YAML with:
    - Build and test stage
    - Security scanning
    - Multi-environment deployments
    - Approval gates
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.pipeline import PipelineGenerator

    console.print("\n[bold blue]Generating pipeline...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = PipelineGenerator(config=config, output_path=output_path)
        generator.generate()

        console.print(f"[green]✓ Generated pipeline:[/green] {output_path}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("all")
def generate_all(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
) -> None:
    """Generate all deployment artifacts.

    Generates Terraform, Dockerfile, and CI/CD pipeline in one command.
    """
    console.print("\n[bold blue]Generating all artifacts...[/bold blue]")

    # Call individual generators
    generate_terraform(config_path=config_path, output_dir="infrastructure")
    generate_dockerfile(config_path=config_path, output_path="Dockerfile")
    generate_pipeline(config_path=config_path, output_path="azure-pipelines.yaml")

    console.print("\n[bold green]✓ All artifacts generated successfully![/bold green]")



# ============================================================================
# VALIDATE Command
# ============================================================================


@app.command()
def validate(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
) -> None:
    """Validate project configuration against deployment rules.

    Checks configuration for:
    - Required fields
    - Valid resource specifications
    - Compliance with deployment policies
    """
    from geronimo.config.loader import load_config
    from geronimo.validation.engine import ValidationEngine

    console.print("\n[bold blue]Validating configuration...[/bold blue]")

    try:
        config = load_config(config_path)
        engine = ValidationEngine()
        results = engine.validate(config)

        if results.is_valid:
            console.print(
                Panel.fit(
                    "[bold green]✓ Configuration is valid![/bold green]\n"
                    f"Checked {results.rules_checked} rules.",
                    border_style="green",
                )
            )
        else:
            console.print("[bold red]✗ Validation failed:[/bold red]")
            for error in results.errors:
                console.print(f"  [red]•[/red] {error}")
            raise typer.Exit(code=1)

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# NOTE: 'monitor' command removed - feature being redesigned



# ============================================================================
# DEPLOY Command Group
# ============================================================================

deploy_app = typer.Typer(
    name="deploy",
    help="Deploy infrastructure using Pulumi (requires: pip install geronimo[pulumi]).",
    no_args_is_help=True,
)
app.add_typer(deploy_app, name="deploy")


@deploy_app.command("up")
def deploy_up(
    project: str = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project name.",
    ),
    target: str = typer.Option(
        "aws",
        "--target",
        "-t",
        help="Cloud target (aws, gcp, azure, cloud).",
    ),
    region: str = typer.Option(
        "us-east-1",
        "--region",
        "-r",
        help="Cloud region.",
    ),
    component: str = typer.Option(
        None,
        "--component",
        "-c",
        help="Specific component to deploy (artifacts, serving, batch).",
    ),
    stack: str = typer.Option(
        "dev",
        "--stack",
        "-s",
        help="Pulumi stack name.",
    ),
    wait: bool = typer.Option(
        True,
        "--wait/--no-wait",
        help="Wait for deployment to complete.",
    ),
) -> None:
    """Deploy infrastructure to the cloud.
    
    Requires Pulumi: pip install geronimo[pulumi]
    
    Examples:
        geronimo deploy up --project iris --target aws
        geronimo deploy up --project iris --component artifacts
    """
    from geronimo.deploy import DeploymentConfig, deploy
    from geronimo.deploy.targets import PulumiNotInstalledError
    
    console.print(f"\n[bold blue]Deploying {project} to {target}...[/bold blue]")
    
    try:
        config = DeploymentConfig(
            project=project,
            target=target,
            region=region,
            stack_name=stack,
        )
        
        result = deploy(config, component=component, wait=wait)
        
        console.print(
            Panel(
                f"[green]✓ Deployment complete![/green]\n\n"
                f"Stack: {stack}\n"
                f"Outputs:\n" + "\n".join(f"  {k}: {v}" for k, v in result.get("outputs", {}).items()),
                title="Deployment Success",
                border_style="green",
            )
        )
        
    except PulumiNotInstalledError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {e}")
        console.print("\nAlternatives:")
        console.print("  1. Install Pulumi: [cyan]pip install geronimo[pulumi][/cyan]")
        console.print("  2. Generate static IaC: [cyan]geronimo generate terraform[/cyan]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@deploy_app.command("destroy")
def deploy_destroy(
    project: str = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project name.",
    ),
    target: str = typer.Option(
        "aws",
        "--target",
        "-t",
        help="Cloud target.",
    ),
    stack: str = typer.Option(
        "dev",
        "--stack",
        "-s",
        help="Pulumi stack name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Destroy deployed infrastructure.
    
    Removes all resources created by 'deploy up'.
    """
    if not force:
        confirm = typer.confirm(f"Destroy all resources for {project}/{stack}?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()
    
    console.print(f"\n[bold red]Destroying {project}/{stack}...[/bold red]")
    
    try:
        from geronimo.deploy import destroy, DeploymentConfig
        
        config = DeploymentConfig(
            project=project,
            target=target,
            stack_name=stack,
        )
        
        destroy(config)
        
        console.print("[green]✓ Resources destroyed.[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@deploy_app.command("status")
def deploy_status(
    deployment_id: str = typer.Option(
        ...,
        "--deployment-id",
        "-d",
        help="Deployment ID.",
    ),
) -> None:
    """Get deployment status."""
    from geronimo.deploy_cloud.client import GeronimoCloudClient
    
    client = GeronimoCloudClient()
    try:
        status_data = client.get_status(deployment_id)
        console.print(f"[bold blue]Status for {deployment_id}:[/bold blue] {status_data.get('status')}")
        if 'error' in status_data and status_data['error']:
            console.print(f"[red]Error:[/red] {status_data['error']}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@deploy_app.command("logs")
def deploy_logs(
    deployment_id: str = typer.Option(
        ...,
        "--deployment-id",
        "-d",
        help="Deployment ID.",
    ),
) -> None:
    """Get deployment logs."""
    from geronimo.deploy_cloud.client import GeronimoCloudClient
    
    client = GeronimoCloudClient()
    try:
        logs = client.get_logs(deployment_id)
        console.print(f"[bold blue]Logs for {deployment_id}:[/bold blue]")
        console.print(logs)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# NOTE: 'import' command removed - feature being redesigned


# ============================================================================
# DOCS Functions (for maintainer scripts, not exposed in CLI)
# ============================================================================

docs_app = typer.Typer(
    name="docs",
    help="Generate API documentation.",
    no_args_is_help=True,
)
# NOTE: docs_app is intentionally NOT registered with app.add_typer()
# End users don't need this command. Maintainers can run docs generation via:
#   uv run python -m pdoc --output-directory docs/api geronimo
# Or use the functions below directly in scripts.


@docs_app.command("generate")
def docs_generate(
    output: str = typer.Option(
        "docs/api",
        "--output",
        "-o",
        help="Output directory for generated documentation.",
    ),
) -> None:
    """Generate API documentation for the Geronimo library.

    Uses pdoc to extract documentation from Python docstrings
    and generate static HTML files.
    """
    import subprocess
    import sys
    from pathlib import Path

    console.print("\n[bold blue]Generating API documentation...[/bold blue]")

    # Find the project root (where pyproject.toml is)
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent.parent  # cli -> geronimo -> src -> root

    # All public modules to document
    modules = MODULES

    output_dir = Path(output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "--output-directory",
        str(output_dir),
        *modules,
    ]

    # Set PYTHONPATH to include src
    src_path = project_root / "src"
    env = {**subprocess.os.environ, "PYTHONPATH": str(src_path)}

    try:
        result = subprocess.run(cmd, cwd=project_root, env=env, check=True)
        console.print(
            Panel.fit(
                f"[bold green]✓ Documentation generated![/bold green]\n\n"
                f"Output: [cyan]{output_dir.resolve()}[/cyan]\n"
                f"Modules: {len(modules)}",
                border_style="green",
            )
        )
    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]Error:[/bold red] pdoc failed with exit code {e.returncode}")
        console.print("[dim]Make sure pdoc is installed: pip install pdoc[/dim]")
        raise typer.Exit(code=1)
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] pdoc not found")
        console.print("[dim]Install with: pip install pdoc[/dim]")
        raise typer.Exit(code=1)


@docs_app.command("serve")
def docs_serve(
    port: int = typer.Option(
        8080,
        "--port",
        "-p",
        help="Port to serve documentation on.",
    ),
) -> None:
    """Start a live-reloading documentation server.

    Opens a local development server that automatically
    rebuilds documentation when source files change.
    """
    import subprocess
    import sys
    from pathlib import Path

    console.print(f"\n[bold blue]Starting documentation server on port {port}...[/bold blue]")

    # Find the project root
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent.parent

    modules = MODULES

    cmd = [
        sys.executable,
        "-m",
        "pdoc",
        "--host",
        "localhost",
        "--port",
        str(port),
        *modules,
    ]

    src_path = project_root / "src"
    env = {**subprocess.os.environ, "PYTHONPATH": str(src_path)}

    console.print(f"[dim]Open http://localhost:{port} in your browser[/dim]")
    console.print("[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        subprocess.run(cmd, cwd=project_root, env=env)
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped.[/yellow]")


if __name__ == "__main__":
    app()

