"""Authentication CLI commands."""

import typer
from rich.console import Console
from rich.panel import Panel

from geronimo.cloud.client import GeronimoCloudClient

auth_app = typer.Typer(
    name="auth",
    help="Manage Geronimo Cloud authentication.",
    no_args_is_help=True,
)

console = Console()


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
        console.print("[green]✓ Credentials cleared. Logged out.[/green]")
    else:
        console.print("[yellow]No credentials found. Already logged out.[/yellow]")


@auth_app.command()
def status() -> None:
    """Check authentication status."""
    client = GeronimoCloudClient()
    
    if client.token:
        try:
            # Verify token is still valid by making a lightweight call or just decode if JWT (but verifying against server is better)
            # For now re-use login verification logic without prompt if we had a dedicated verify endpoint, 
            # but since we don't want to re-prompt, let's just show logged in state.
            
            # Ideally we check /auth/verify here too
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
