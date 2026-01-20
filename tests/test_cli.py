"""Tests for geronimo CLI commands."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from geronimo.cli.main import app


runner = CliRunner()


class TestCLIVersion:
    """Tests for version command."""

    def test_version_flag(self):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Geronimo" in result.output


class TestCLIInit:
    """Tests for init command."""

    def test_init_help(self):
        """Test init --help."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "Initialize" in result.output or "init" in result.output

    def test_init_creates_project(self, temp_dir):
        """Test init creates project structure."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "test-project",
                "--framework", "sklearn",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert (temp_dir / "test-project").exists()
        assert (temp_dir / "test-project" / "geronimo.yaml").exists()
        assert (temp_dir / "test-project" / "pyproject.toml").exists()

    def test_init_realtime_creates_sdk_files(self, temp_dir):
        """Test realtime template creates SDK endpoint and app.py."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "rt-test",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        project_dir = temp_dir / "rt-test"
        pkg_dir = project_dir / "src" / "rt_test"
        sdk_dir = pkg_dir / "sdk"
        
        # Check SDK files exist
        assert sdk_dir.exists()
        assert (sdk_dir / "endpoint.py").exists()
        assert (sdk_dir / "model.py").exists()
        assert (sdk_dir / "features.py").exists()
        assert (sdk_dir / "monitoring_config.py").exists()
        
        # Check app.py wrapper exists
        assert (pkg_dir / "app.py").exists()
        
        # Check endpoint has demo mode (not NotImplementedError)
        endpoint_content = (sdk_dir / "endpoint.py").read_text()
        assert "demo_mode" in endpoint_content
        assert "def initialize" in endpoint_content

    def test_init_batch_creates_sdk_files(self, temp_dir):
        """Test batch template creates SDK pipeline and flow.py."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "batch-test",
                "--template", "batch",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        project_dir = temp_dir / "batch-test"
        pkg_dir = project_dir / "src" / "batch_test"
        sdk_dir = pkg_dir / "sdk"
        
        # Check SDK files exist
        assert sdk_dir.exists()
        assert (sdk_dir / "pipeline.py").exists()
        assert (sdk_dir / "model.py").exists()
        assert (sdk_dir / "monitoring_config.py").exists()
        
        # Check flow.py wrapper exists
        assert (pkg_dir / "flow.py").exists()
        
        # Check pipeline has demo mode
        pipeline_content = (sdk_dir / "pipeline.py").read_text()
        assert "demo_mode" in pipeline_content
        
        # Check metaflow is in dependencies
        pyproject = (project_dir / "pyproject.toml").read_text()
        assert "metaflow" in pyproject

    def test_init_batch_includes_metaflow_dependency(self, temp_dir):
        """Test batch template includes metaflow in dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "mf-test",
                "--template", "batch",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = (temp_dir / "mf-test" / "pyproject.toml").read_text()
        assert "metaflow" in pyproject

    def test_init_realtime_includes_fastapi_dependency(self, temp_dir):
        """Test realtime template includes fastapi in dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "fa-test",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = (temp_dir / "fa-test" / "pyproject.toml").read_text()
        assert "fastapi" in pyproject
        assert "uvicorn" in pyproject



class TestCLIKeys:
    """Tests for keys commands."""

    def test_keys_help(self):
        """Test keys --help."""
        result = runner.invoke(app, ["keys", "--help"])
        assert result.exit_code == 0
        assert "create" in result.output
        assert "list" in result.output

    def test_keys_create(self, temp_dir):
        """Test creating an API key."""
        keys_file = temp_dir / "keys.json"
        result = runner.invoke(
            app,
            [
                "keys", "create",
                "--name", "test-key",
                "--scopes", "predict",
                "--keys-file", str(keys_file),
            ],
        )
        
        assert result.exit_code == 0
        assert "created successfully" in result.output
        assert "grn_" in result.output
        assert keys_file.exists()

    def test_keys_list(self, temp_dir):
        """Test listing API keys."""
        keys_file = temp_dir / "keys.json"
        
        # Create a key first
        runner.invoke(
            app,
            [
                "keys", "create",
                "--name", "list-test",
                "--keys-file", str(keys_file),
            ],
        )
        
        # List keys
        result = runner.invoke(
            app,
            ["keys", "list", "--keys-file", str(keys_file)],
        )
        
        assert result.exit_code == 0
        assert "list-test" in result.output


class TestCLIGenerate:
    """Tests for generate commands."""

    def test_generate_help(self):
        """Test generate --help."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "terraform" in result.output.lower() or "Generate" in result.output
