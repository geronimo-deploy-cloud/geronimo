"""CLI commands for API key management."""

import typer
from rich.table import Table

from geronimo.serving.auth.keys import APIKeyManager
from geronimo.cli.utils import console, success, error, warning, dim

keys_app = typer.Typer(
    name="keys",
    help="Manage Service to Service API keys for endpoint authentication",
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
        dim("No API keys found.")
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
        success(f"Key {key_id} revoked")
    else:
        error(f"Key {key_id} not found", exit_code=1)


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
        "-y",  # Changed from -f to avoid conflict with --keys-file
        help="Skip confirmation",
    ),
) -> None:
    """Permanently delete an API key."""
    manager = APIKeyManager(keys_file)

    key = manager.get_key(key_id)
    if not key:
        error(f"Key {key_id} not found", exit_code=1)

    if not force:
        confirm = typer.confirm(f"Permanently delete key '{key.name}' ({key_id})?")
        if not confirm:
            raise typer.Abort()

    manager.delete(key_id)
    success(f"Key {key_id} deleted")


@keys_app.command("sync")
def sync_keys(
    keys_file: str = typer.Option(
        ".geronimo/keys.json",
        "--keys-file",
        "-f",
        help="Path to keys file",
    ),
    key_ids: str = typer.Option(
        None,
        "--key-ids",
        "-k",
        help="Comma-separated key IDs to sync (default: all)",
    ),
    interactive: bool = typer.Option(
        False,
        "--interactive",
        "-i",
        help="Interactively select keys to sync",
    ),
) -> None:
    """Sync local API keys to Geronimo Cloud.
    
    Uploads your local API keys to Geronimo Cloud so they can be used
    for authenticating requests to cloud-deployed endpoints.
    
    Cloud-managed keys (created via dashboard) take precedence and
    won't be overwritten by synced keys.
    """
    from geronimo.deploy_cloud.client import GeronimoCloudClient
    
    manager = APIKeyManager(keys_file)
    all_keys = manager.list_keys()
    
    if not all_keys:
        dim("No local API keys found.")
        return
    
    # Filter keys based on options
    keys_to_sync = all_keys
    
    if key_ids:
        # Filter to specified key IDs
        requested_ids = {k.strip() for k in key_ids.split(",")}
        keys_to_sync = [k for k in all_keys if k.key_id in requested_ids]
        
        # Warn about missing keys
        found_ids = {k.key_id for k in keys_to_sync}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            warning(f"Keys not found: {', '.join(missing_ids)}")
        
        if not keys_to_sync:
            error("No matching keys found", exit_code=1)
    
    elif interactive:
        # Interactive selection
        console.print("\n[bold]Select keys to sync:[/bold]\n")
        keys_to_sync = []
        
        for key in all_keys:
            status = "[green]active[/green]" if key.enabled else "[red]revoked[/red]"
            console.print(f"  [dim]{key.key_id}[/dim] - [cyan]{key.name}[/cyan] ({status})")
            
            if typer.confirm("    Sync this key?", default=True):
                keys_to_sync.append(key)
        
        console.print()
        
        if not keys_to_sync:
            dim("No keys selected.")
            return
    
    # Convert to dicts for API
    keys_data = [key.to_dict() for key in keys_to_sync]
    
    # Sync to cloud
    try:
        client = GeronimoCloudClient()
        result = client.sync_keys(keys_data)
        
        synced = result.get("synced", 0)
        skipped = result.get("skipped", 0)
        
        console.print(f"\n[bold green]✓ Keys synced to Geronimo Cloud[/bold green]")
        console.print(f"  Synced: [green]{synced}[/green]")
        if skipped:
            console.print(f"  Skipped: [yellow]{skipped}[/yellow] (cloud-managed keys take precedence)")
        console.print()
        
    except RuntimeError as e:
        error(str(e), exit_code=1)
    except Exception as e:
        error(f"Failed to sync keys: {e}", exit_code=1)
