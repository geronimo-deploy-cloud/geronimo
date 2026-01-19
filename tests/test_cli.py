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
