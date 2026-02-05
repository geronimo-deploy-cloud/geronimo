"""Config CLI commands for managing global Geronimo settings.

Provides commands for:
- geronimo config init - Interactive setup wizard
- geronimo config set <key> <value> - Set individual values
- geronimo config show - View current config
- geronimo config reset - Reset to defaults
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from geronimo.config.user_config import (
    load_user_config,
    save_user_config,
    get_config_value,
    set_config_value,
    reset_user_config,
    USER_CONFIG_FILE,
    UserConfig,
    ArtifactConfig,
    DefaultsConfig,
)


config_app = typer.Typer(
    name="config",
    help="Manage global Geronimo configuration.",
    no_args_is_help=True,
)

console = Console()


@config_app.command("init")
def config_init():
    """Interactive setup wizard for first-time configuration.
    
    Sets up ~/.geronimo/config.yaml with your preferred defaults.
    """
    console.print("\n[bold blue]🚀 Geronimo Configuration Setup[/bold blue]\n")
    console.print("This wizard will configure your global Geronimo settings.\n")
    
    # Backend selection
    console.print("[bold]Artifact Storage Backend[/bold]")
    console.print("  [dim]local[/dim]  - Store artifacts in ~/.geronimo/artifacts")
    console.print("  [dim]s3[/dim]     - Store artifacts in your S3 bucket")
    console.print("  [dim]cloud[/dim]  - Store artifacts in Geronimo Cloud (requires auth)\n")
    
    backend = typer.prompt(
        "Select backend",
        default="local",
        type=str,
    )
    
    if backend not in ("local", "s3", "cloud"):
        console.print(f"[red]Invalid backend: {backend}. Using 'local'.[/red]")
        backend = "local"
    
    s3_bucket = None
    if backend == "s3":
        s3_bucket = typer.prompt(
            "S3 bucket name",
            default="ml-artifacts",
        )
    elif backend == "cloud":
        console.print("\n[yellow]Note:[/yellow] Cloud backend requires authentication.")
        console.print("Run [bold]geronimo auth login[/bold] to authenticate.\n")
    
    # Create config
    config = UserConfig(
        artifacts=ArtifactConfig(
            backend=backend,
            s3_bucket=s3_bucket,
        ),
        defaults=DefaultsConfig(),
    )
    
    save_user_config(config)
    
    console.print(Panel(
        f"Configuration saved to [cyan]{USER_CONFIG_FILE}[/cyan]\n\n"
        f"  Backend: [green]{backend}[/green]"
        + (f"\n  S3 Bucket: [green]{s3_bucket}[/green]" if s3_bucket else ""),
        title="✓ Setup Complete",
        border_style="green",
    ))
    
    console.print("\nNext steps:")
    console.print("  • Run [bold]geronimo init --name my-project[/bold] to create a project")
    console.print("  • Run [bold]geronimo config show[/bold] to view your settings\n")


@config_app.command("show")
def config_show():
    """Display current configuration settings."""
    config = load_user_config()
    
    table = Table(title="Geronimo Configuration", show_header=True)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    # Artifacts section
    table.add_row("artifacts.backend", config.artifacts.backend)
    table.add_row("artifacts.s3_bucket", config.artifacts.s3_bucket or "[dim]not set[/dim]")
    table.add_row("artifacts.base_path", config.artifacts.base_path)
    
    # Defaults section
    table.add_row("defaults.framework", config.defaults.framework)
    table.add_row("defaults.template", config.defaults.template)
    
    console.print()
    console.print(table)
    console.print(f"\n[dim]Config file: {USER_CONFIG_FILE}[/dim]\n")
    
    # Show auth status for cloud backend
    if config.artifacts.backend == "cloud":
        try:
            from geronimo.cli.auth_cmd import get_current_user
            user = get_current_user()
            if user:
                console.print(f"[green]✓ Authenticated as {user}[/green]\n")
            else:
                console.print("[yellow]⚠ Cloud backend requires authentication.[/yellow]")
                console.print("  Run [bold]geronimo auth login[/bold] to authenticate.\n")
        except Exception:
            console.print("[yellow]⚠ Could not check authentication status.[/yellow]\n")


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="Config key (e.g., artifacts.backend)"),
    value: str = typer.Argument(..., help="Value to set"),
):
    """Set a configuration value.
    
    Examples:
        geronimo config set artifacts.backend s3
        geronimo config set artifacts.s3_bucket my-ml-bucket
    """
    success = set_config_value(key, value)
    
    if success:
        console.print(f"[green]✓[/green] Set [cyan]{key}[/cyan] = [green]{value}[/green]")
    else:
        console.print(f"[red]✗[/red] Invalid key or value: {key}={value}")
        console.print("\nValid keys:")
        console.print("  artifacts.backend   - local, s3, or cloud")
        console.print("  artifacts.s3_bucket - S3 bucket name")
        console.print("  artifacts.base_path - Local storage path")
        console.print("  defaults.framework  - ML framework (sklearn, pytorch, etc.)")
        console.print("  defaults.template   - Project template (realtime, batch, both)")
        raise typer.Exit(1)


@config_app.command("reset")
def config_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Reset configuration to defaults."""
    if not force:
        confirm = typer.confirm("Reset all settings to defaults?")
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return
    
    reset_user_config()
    console.print("[green]✓[/green] Configuration reset to defaults.")
    console.print(f"[dim]Removed {USER_CONFIG_FILE}[/dim]")
