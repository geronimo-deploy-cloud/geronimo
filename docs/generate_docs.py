#!/usr/bin/env python3
"""Generate API documentation for the Geronimo library using pdoc.

This script generates comprehensive HTML API documentation from Python
docstrings for all public modules in the Geronimo library.

Usage:
    python docs/generate_docs.py
    python docs/generate_docs.py --output ./custom/path
    python docs/generate_docs.py --serve  # Live preview on localhost:8080
"""

import argparse
import subprocess
import sys
from pathlib import Path

# All public modules to document
MODULES = [
    "geronimo.analyzers",
    "geronimo.artifacts",
    "geronimo.batch",
    "geronimo.cli",
    "geronimo.cloud",
    "geronimo.config",
    "geronimo.data",
    "geronimo.deploy",
    "geronimo.features",
    "geronimo.generators",
    "geronimo.mcp",
    "geronimo.models",
    "geronimo.monitoring",
    "geronimo.scanners",
    "geronimo.serving",
    "geronimo.validation",
]

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "api"


def generate_docs(output_dir: Path, serve: bool = False) -> int:
    """Generate API documentation using pdoc.

    Args:
        output_dir: Directory to write HTML documentation to.
        serve: If True, start a live-reloading server instead of generating files.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    # Ensure src is in the path for imports
    src_path = PROJECT_ROOT / "src"

    cmd = [
        sys.executable,
        "-m",
        "pdoc",
    ]

    if serve:
        # Live-reloading development server
        cmd.extend(["--host", "localhost", "--port", "8080"])
    else:
        # Generate static HTML
        cmd.extend(["--output-directory", str(output_dir)])

    # Add all modules
    cmd.extend(MODULES)

    print(f"Running: {' '.join(cmd)}")
    print(f"Source path: {src_path}")

    if not serve:
        print(f"Output directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)

    # Run pdoc with PYTHONPATH set to include src
    env = {"PYTHONPATH": str(src_path)}
    result = subprocess.run(cmd, cwd=PROJECT_ROOT, env={**subprocess.os.environ, **env})

    if result.returncode == 0 and not serve:
        print(f"\n✓ Documentation generated successfully!")
        print(f"  Open {output_dir / 'index.html'} in your browser to view.")
        print(f"\n  Module count: {len(MODULES)}")

    return result.returncode


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate API documentation for Geronimo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python docs/generate_docs.py
    python docs/generate_docs.py --output ./build/docs
    python docs/generate_docs.py --serve
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for generated docs (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--serve",
        "-s",
        action="store_true",
        help="Start a live-reloading development server instead of generating files",
    )

    args = parser.parse_args()

    return generate_docs(args.output, args.serve)


if __name__ == "__main__":
    sys.exit(main())
