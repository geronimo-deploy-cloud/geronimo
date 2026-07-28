"""Tests for geronimo CLI commands."""

import os
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from geronimo.cli.main import app


runner = CliRunner()


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape/color codes from terminal output."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m|\x1b\([0-9;]*H")
    return ansi_escape.sub("", text)


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

    def test_init_creates_project_with_target(self, temp_dir):
        """Test init with --target flag creates project structure."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "target-test",
                "--target", "aws",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert (temp_dir / "target-test").exists()
        assert (temp_dir / "target-test" / "pyproject.toml").exists()

    def test_init_invalid_target(self, temp_dir):
        """Test init with invalid target fails."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "bad-target",
                "--target", "invalid-target",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 1
        assert "Invalid target" in result.output

    def test_init_target_in_output(self, temp_dir):
        """Test init with target prints target in output."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "target-output-test",
                "--target", "gcp",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        assert "Target: gcp" in result.output

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

    def test_init_creates_experiments_directory(self, temp_dir):
        """Test init creates experiments/ alongside sdk/."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "exp-test",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )

        assert result.exit_code == 0
        project_dir = temp_dir / "exp-test"
        pkg_dir = project_dir / "src" / "exp_test"
        experiments_dir = pkg_dir / "experiments"

        assert experiments_dir.exists()
        assert (experiments_dir / "__init__.py").exists()
        init_content = (experiments_dir / "__init__.py").read_text()
        assert "ad-hoc" in init_content
        assert "excluded" in init_content or "production" in init_content

    def test_init_batch_creates_experiments_directory(self, temp_dir):
        """Test batch template also creates experiments/ alongside sdk/."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "exp-batch",
                "--template", "batch",
                "--output", str(temp_dir),
            ],
        )

        assert result.exit_code == 0
        project_dir = temp_dir / "exp-batch"
        pkg_dir = project_dir / "src" / "exp_batch"
        experiments_dir = pkg_dir / "experiments"

        assert experiments_dir.exists()
        init_content = (experiments_dir / "__init__.py").read_text()
        assert "ad-hoc" in init_content

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

    def test_init_batch_readme_content(self, temp_dir):
        """Test batch template generates batch-specific README."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "readme-batch",
                "--template", "batch",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        readme = (temp_dir / "readme-batch" / "README.md").read_text()
        
        # Should reference batch concepts
        assert "batch pipeline" in readme.lower()
        assert "flow" in readme
        assert "pipeline" in readme
        
        # Should NOT reference realtime concepts
        assert "uvicorn" not in readme
        assert "FastAPI" not in readme

    def test_init_realtime_readme_content(self, temp_dir):
        """Test realtime template generates realtime-specific README."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "readme-rt",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        readme = (temp_dir / "readme-rt" / "README.md").read_text()
        
        # Should reference realtime concepts
        assert "uvicorn" in readme
        assert "model serving" in readme.lower()
        
        # Should NOT reference batch concepts
        assert "batch pipeline" not in readme.lower()


class TestCLIPulumiDeps:
    """Tests for Pulumi dependency scaffolding based on deploy target."""

    def _get_pyproject_deps(self, temp_dir, project_name: str) -> str:
        """Helper to read pyproject.toml contents."""
        return (temp_dir / project_name / "pyproject.toml").read_text()

    def test_init_aws_target_includes_pulumi(self, temp_dir):
        """Test --target aws adds pulumi and pulumi-aws to dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "aws-pulumi-test",
                "--target", "aws",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "aws-pulumi-test")
        
        assert re.search(r'"pulumi>=3\.100\.0"', pyproject)
        assert re.search(r'"pulumi-aws>=9\.0\.0"', pyproject)

    def test_init_gcp_target_includes_pulumi(self, temp_dir):
        """Test --target gcp adds pulumi and pulumi-gcp to dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "gcp-pulumi-test",
                "--target", "gcp",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "gcp-pulumi-test")
        
        assert re.search(r'"pulumi>=3\.100\.0"', pyproject)
        assert re.search(r'"pulumi-gcp>=8\.0\.0"', pyproject)

    def test_init_azure_target_includes_pulumi(self, temp_dir):
        """Test --target azure adds pulumi and pulumi-azure-native to dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "azure-pulumi-test",
                "--target", "azure",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "azure-pulumi-test")
        
        assert re.search(r'"pulumi>=3\.100\.0"', pyproject)
        assert re.search(r'"pulumi-azure-native>=2\.0\.0"', pyproject)

    def test_init_gdc_target_excludes_pulumi(self, temp_dir):
        """Test --target gdc (default) does NOT include pulumi dependencies."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "gdc-no-pulumi",
                "--target", "gdc",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "gdc-no-pulumi")
        
        # Check for actual dependency declarations, not substring matches in project name
        pulumi_dep_pattern = re.compile(r'"pulumi[-\w]*[><=!]', re.IGNORECASE)
        assert not pulumi_dep_pattern.search(pyproject), "GDC target should not include pulumi dependencies"

    def test_init_default_target_excludes_pulumi(self, temp_dir):
        """Test omitting --target (defaults to gdc) does NOT include pulumi."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "default-no-pulumi",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "default-no-pulumi")
        
        # Check for actual dependency declarations, not substring matches in project name
        pulumi_dep_pattern = re.compile(r'"pulumi[-\w]*[><=!]', re.IGNORECASE)
        assert not pulumi_dep_pattern.search(pyproject), "Default (gdc) target should not include pulumi dependencies"

    def test_init_aws_batch_target_includes_pulumi(self, temp_dir):
        """Test --target aws with batch template includes pulumi deps."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "aws-batch-pulumi",
                "--target", "aws",
                "--template", "batch",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "aws-batch-pulumi")
        
        assert re.search(r'"pulumi>=3\.100\.0"', pyproject)
        assert re.search(r'"pulumi-aws>=9\.0\.0"', pyproject)
        # Also check batch deps are still present
        assert "metaflow" in pyproject

    def test_init_aws_both_template_includes_pulumi(self, temp_dir):
        """Test --target aws with 'both' template includes pulumi deps."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "aws-both-pulumi",
                "--target", "aws",
                "--template", "both",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "aws-both-pulumi")
        
        assert re.search(r'"pulumi>=3\.100\.0"', pyproject)
        assert re.search(r'"pulumi-aws>=9\.0\.0"', pyproject)
        # Also check both template deps
        assert "fastapi" in pyproject
        assert "metaflow" in pyproject

    def test_init_aws_target_version_pin_convention(self, temp_dir):
        """Test Pulumi deps use >= pinning convention matching other deps."""
        result = runner.invoke(
            app,
            [
                "init",
                "--name", "version-pin-test",
                "--target", "aws",
                "--template", "realtime",
                "--output", str(temp_dir),
            ],
        )
        
        assert result.exit_code == 0
        pyproject = self._get_pyproject_deps(temp_dir, "version-pin-test")
        
        # All deps should use >= pinning, not exact pins or unpinned
        # Find all dependency lines
        dep_lines = re.findall(r'"([^"]+)"', pyproject)
        for dep in dep_lines:
            # Skip project name and version strings
            if dep in ("aws-pulumi-test", "version-pin-test"):
                continue
            # Pulumi deps should use >=
            if "pulumi" in dep.lower():
                assert ">=" in dep, f"Pulumi dep '{dep}' should use >= pinning"




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


class TestCLIKeysSync:
    """Tests for keys sync command."""

    def test_keys_sync_help(self):
        """Test sync --help shows options."""
        result = runner.invoke(app, ["keys", "sync", "--help"])
        assert result.exit_code == 0
        output = _strip_ansi(result.output)
        assert "sync" in output.lower()
        assert "--key-ids" in output
        assert "--interactive" in output

    def test_keys_sync_no_keys(self, temp_dir):
        """Test sync with no local keys."""
        keys_file = temp_dir / "empty_keys.json"
        keys_file.write_text('{"keys": {}}')
        
        result = runner.invoke(
            app,
            ["keys", "sync", "--keys-file", str(keys_file)],
        )
        
        assert result.exit_code == 0
        assert "No local API keys found" in result.output

    def test_keys_sync_missing_key_ids(self, temp_dir):
        """Test sync with non-existent key IDs."""
        keys_file = temp_dir / "keys.json"
        
        # Create a key
        runner.invoke(
            app,
            ["keys", "create", "--name", "real-key", "--keys-file", str(keys_file)],
        )
        
        # Try to sync a non-existent key
        result = runner.invoke(
            app,
            ["keys", "sync", "--keys-file", str(keys_file), "--key-ids", "nonexistent"],
        )
        
        assert result.exit_code == 1
        assert "No matching keys found" in result.output

    def test_keys_sync_specific_keys(self, temp_dir, monkeypatch):
        """Test sync with specific key IDs."""
        from unittest.mock import MagicMock
        
        keys_file = temp_dir / "keys.json"
        
        # Create two keys
        runner.invoke(
            app,
            ["keys", "create", "--name", "key-1", "--keys-file", str(keys_file)],
        )
        runner.invoke(
            app,
            ["keys", "create", "--name", "key-2", "--keys-file", str(keys_file)],
        )
        
        # Get the key IDs
        import json
        data = json.loads(keys_file.read_text())
        key_ids = list(data["keys"].keys())
        
        # Mock the cloud client
        mock_instance = MagicMock()
        mock_instance.sync_keys.return_value = {"synced": 1, "skipped": 0}
        mock_client_class = MagicMock(return_value=mock_instance)
        
        monkeypatch.setattr("geronimo.gdc.client.GeronimoCloudClient", mock_client_class)
        
        # Sync only the first key
        result = runner.invoke(
            app,
            ["keys", "sync", "--keys-file", str(keys_file), "--key-ids", key_ids[0]],
        )
        
        assert result.exit_code == 0
        assert "Keys synced to Geronimo Cloud" in result.output
        
        # Verify only one key was synced
        call_args = mock_instance.sync_keys.call_args
        assert len(call_args[0][0]) == 1

    def test_keys_sync_not_authenticated(self, temp_dir, monkeypatch):
        """Test sync without authentication."""
        from unittest.mock import MagicMock
        
        keys_file = temp_dir / "keys.json"
        
        # Create a key
        runner.invoke(
            app,
            ["keys", "create", "--name", "auth-test", "--keys-file", str(keys_file)],
        )
        
        # Mock the cloud client to raise auth error
        mock_instance = MagicMock()
        mock_instance.sync_keys.side_effect = RuntimeError(
            "Not authenticated. Run 'geronimo auth login' first."
        )
        mock_client_class = MagicMock(return_value=mock_instance)
        
        monkeypatch.setattr("geronimo.gdc.client.GeronimoCloudClient", mock_client_class)
        
        result = runner.invoke(
            app,
            ["keys", "sync", "--keys-file", str(keys_file)],
        )
        
        assert result.exit_code == 1
        assert "Not authenticated" in result.output
        assert "geronimo auth login" in result.output


class TestCLIGenerate:
    """Tests for generate commands."""

    def test_generate_help(self):
        """Test generate --help."""
        result = runner.invoke(app, ["generate", "--help"])
        assert result.exit_code == 0
        assert "terraform" in result.output.lower() or "Generate" in result.output
