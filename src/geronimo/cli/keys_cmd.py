"""CLI commands for API key management."""

import typer
from rich.console import Console
from rich.table import Table

# Import directly to avoid fastapi dependency in CLI
from geronimo.serving.auth.keys import APIKeyManager

console = Console()

keys_app = typer.Typer(
    name="keys",
    help="Manage API keys for endpoint authentication.",
    no_args_is_help=True,
)


@keys_app.command("create")
def create_key(
    name: str = typer.Option(..., "--name", "-n", help="Name for the API key"),
    scopes: str = typer.Option(
        "predict",
        "--scopes",
        "-s",
        help="Comma-separated list of scopes",
    ),
    keys_file: str = typer.Option(
        ".geronimo/keys.json",
        "--keys-file",
        "-f",
        help="Path to keys file",
    ),
) -> None:
    """Create a new API key.

    The raw key is only displayed once - save it securely!
    """
    manager = APIKeyManager(keys_file)
    scope_list = [s.strip() for s in scopes.split(",")]

    raw_key, api_key = manager.create_key(name=name, scopes=scope_list)

    console.print("\n[bold green]✓ API key created successfully![/bold green]\n")
    console.print(f"  Name: [cyan]{api_key.name}[/cyan]")
    console.print(f"  ID: [dim]{api_key.key_id}[/dim]")
    console.print(f"  Scopes: [yellow]{', '.join(api_key.scopes)}[/yellow]")
    console.print()
    console.print("[bold yellow]⚠ Save this key - it won't be shown again:[/bold yellow]")
    console.print(f"\n  [bold]{raw_key}[/bold]\n")


@keys_app.command("list")
def list_keys(
    keys_file: str = typer.Option(
        ".geronimo/keys.json",
        "--keys-file",
        "-f",
        help="Path to keys file",
    ),
) -> None:
    """List all API keys."""
    manager = APIKeyManager(keys_file)
    keys = manager.list_keys()

    if not keys:
        console.print("[dim]No API keys found.[/dim]")
        return

    table = Table(title="API Keys")
    table.add_column("ID", style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Scopes", style="yellow")
    table.add_column("Created", style="dim")
    table.add_column("Status")

    for key in keys:
        status = "[green]active[/green]" if key.enabled else "[red]revoked[/red]"
        if key.expires_at:
            status += f" [dim](expires {key.expires_at.date()})[/dim]"

        table.add_row(
            key.key_id,
            key.name,
            ", ".join(key.scopes),
            key.created_at.strftime("%Y-%m-%d"),
            status,
        )

    console.print(table)


@keys_app.command("revoke")
def revoke_key(
    key_id: str = typer.Argument(..., help="ID of the key to revoke"),
    keys_file: str = typer.Option(
        ".geronimo/keys.json",
        "--keys-file",
        "-f",
        help="Path to keys file",
    ),
) -> None:
    """Revoke an API key (disable but keep record)."""
    manager = APIKeyManager(keys_file)

    if manager.revoke(key_id):
        console.print(f"[green]✓ Key {key_id} revoked[/green]")
    else:
        console.print(f"[red]✗ Key {key_id} not found[/red]")
        raise typer.Exit(code=1)


@keys_app.command("delete")
def delete_key(
    key_id: str = typer.Argument(..., help="ID of the key to delete"),
    keys_file: str = typer.Option(
        ".geronimo/keys.json",
        "--keys-file",
        "-f",
        help="Path to keys file",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation",
    ),
) -> None:
    """Permanently delete an API key."""
    manager = APIKeyManager(keys_file)

    key = manager.get_key(key_id)
    if not key:
        console.print(f"[red]✗ Key {key_id} not found[/red]")
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Permanently delete key '{key.name}' ({key_id})?")
        if not confirm:
            raise typer.Abort()

    manager.delete(key_id)
    console.print(f"[green]✓ Key {key_id} deleted[/green]")
