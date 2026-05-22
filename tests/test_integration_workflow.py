"""Integration tests for the CLI workflow."""

import os
from pathlib import Path
from typer.testing import CliRunner

from geronimo.cli.main import app

def test_init_validate_generate_workflow(tmp_path: Path):
    """Test the full init -> validate -> generate all workflow."""
    runner = CliRunner()
    
    # Run init
    result = runner.invoke(app, [
        "init", 
        "--name", "test-project", 
        "--template", "realtime",
        "--output", str(tmp_path)
    ])
    assert result.exit_code == 0, f"Init failed: {result.output}"
    
    project_dir = tmp_path / "test-project"
    assert project_dir.exists()
    assert (project_dir / "geronimo.yaml").exists()
    
    # Change current working directory to the generated project
    orig_cwd = os.getcwd()
    os.chdir(project_dir)
    try:
        # Run validate
        result = runner.invoke(app, ["validate"])
        assert result.exit_code == 0, f"Validate failed: {result.output}"
        
        # Run generate all
        result = runner.invoke(app, ["generate", "all"])
        assert result.exit_code == 0, f"Generate failed: {result.output}"
        
        # Verify artifacts
        assert (project_dir / "infrastructure" / "main.tf").exists()
        assert (project_dir / "Dockerfile").exists()
        assert (project_dir / "azure-pipelines.yaml").exists()
    finally:
        # Back out of the dir to avoid issues with other tests
        os.chdir(orig_cwd)
