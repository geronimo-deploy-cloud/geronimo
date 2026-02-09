"""Authentication CLI commands."""

import typer
from rich.panel import Panel

from geronimo.deploy_cloud.client import GeronimoCloudClient
from geronimo.cli.utils import console, success, warning

auth_app = typer.Typer(
    name="auth",
    help="Manage Geronimo Cloud authentication for developers.",
    no_args_is_help=True,
)


@auth_app.command()
def login(
    token: str = typer.Option(
        None,
        "--token",
        "-t",
        help="Authentication token.",
        prompt="Enter your Geronimo Cloud token",
        hide_input=True,
    ),
) -> None:
    """Log in to Geronimo Cloud."""
    client = GeronimoCloudClient()
    
    try:
        console.print("[blue]Verifying token...[/blue]")
        user_data = client.login(token)
        
        console.print(
            Panel(
                f"[green]✓ Successfully logged in![/green]\n\n"
                f"User: [cyan]{user_data.get('email', 'unknown')}[/cyan]\n"
                f"Organization: [cyan]{user_data.get('org', 'default')}[/cyan]",
                title="Login Success",
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Login failed:[/bold red] {e}")
        raise typer.Exit(code=1)


@auth_app.command()
def logout() -> None:
    """Log out and clear credentials."""
    import shutil
    from pathlib import Path
    
    creds_dir = Path.home() / ".geronimo"
    if creds_dir.exists():
        shutil.rmtree(creds_dir)
        success("Credentials cleared. Logged out.")
    else:
        warning("No credentials found. Already logged out.")


@auth_app.command()
def status() -> None:
    """Check authentication status."""
    client = GeronimoCloudClient()
    
    if client.token:
        try:
            console.print(
                Panel(
                    "[green]✓ Authenticated[/green]\n"
                    "Token is present in credentials file.",
                    title="Auth Status",
                    border_style="green",
                )
            )
        except Exception:
             console.print("[red]✗ Token invalid or expired[/red]")
    else:
        console.print(
            Panel(
                "[yellow]Not authenticated[/yellow]\n"
                "Run [cyan]geronimo auth login[/cyan] to sign in.",
                title="Auth Status",
                border_style="yellow",
            )
        )
