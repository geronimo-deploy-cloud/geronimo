"""CLI utility functions for consistent output formatting.

Provides styled output helpers, spinners, and common patterns used
across all CLI commands.
"""

from contextlib import contextmanager
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Shared console instance - use this instead of creating new Console() in each command
console = Console()


def success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def error(message: str, exit_code: Optional[int] = None) -> None:
    """Print an error message with red X.
    
    Args:
        message: Error message to display.
        exit_code: If provided, raises SystemExit with this code.
    """
    console.print(f"[red]✗[/red] {message}")
    if exit_code is not None:
        import typer
        raise typer.Exit(code=exit_code)


def warning(message: str) -> None:
    """Print a warning message with yellow icon."""
    console.print(f"[yellow]⚠[/yellow] {message}")


def info(message: str) -> None:
    """Print an info message with blue icon."""
    console.print(f"[blue]ℹ[/blue] {message}")


def dim(message: str) -> None:
    """Print dimmed text for secondary information."""
    console.print(f"[dim]{message}[/dim]")


def styled_panel(
    content: str,
    title: Optional[str] = None,
    style: str = "blue",
    border_style: Optional[str] = None,
) -> None:
    """Print content in a styled panel.
    
    Args:
        content: Panel content (can include rich markup).
        title: Optional panel title.
        style: Style for the content (default: blue).
        border_style: Border color (defaults to style).
    """
    console.print(Panel(
        content,
        title=title,
        border_style=border_style or style,
    ))


def create_table(
    title: str,
    columns: list[tuple[str, str]],
) -> Table:
    """Create a styled table with predefined column styles.
    
    Args:
        title: Table title.
        columns: List of (name, style) tuples for columns.
        
    Returns:
        Configured Table instance (call table.add_row() to add data).
        
    Example:
        table = create_table("API Keys", [
            ("ID", "dim"),
            ("Name", "cyan"),
            ("Status", "green"),
        ])
        table.add_row("key-1", "My Key", "active")
        console.print(table)
    """
    table = Table(title=title, show_header=True)
    for name, style in columns:
        table.add_column(name, style=style)
    return table


@contextmanager
def status_spinner(message: str):
    """Show a spinner while performing an operation.
    
    Args:
        message: Status message to display.
        
    Example:
        with status_spinner("Deploying project..."):
            deploy_project()
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description=message, total=None)
        yield


def confirm_action(
    message: str,
    default: bool = False,
) -> bool:
    """Prompt user for confirmation.
    
    Args:
        message: Confirmation prompt.
        default: Default value if user just presses Enter.
        
    Returns:
        True if user confirmed, False otherwise.
    """
    import typer
    return typer.confirm(message, default=default)


def prompt(
    message: str,
    default: Optional[str] = None,
    hide_input: bool = False,
) -> str:
    """Prompt user for input.
    
    Args:
        message: Input prompt.
        default: Default value if user just presses Enter.
        hide_input: If True, hide input (for passwords/tokens).
        
    Returns:
        User input string.
    """
    import typer
    return typer.prompt(message, default=default, hide_input=hide_input)
