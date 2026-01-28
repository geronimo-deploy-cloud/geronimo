"""Main CLI entrypoint for Geronimo.

This module provides the primary CLI interface using Typer.
"""

import typer
from rich.console import Console
from rich.panel import Panel

from geronimo import __version__

# Create the main Typer app
app = typer.Typer(
    name="geronimo",
    help="MLOps deployment platform - automate ML model deployments to AWS",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Console for rich output
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(
            Panel.fit(
                f"[bold blue]Geronimo[/bold blue] v{__version__}\n"
                "[dim]MLOps Deployment Platform[/dim]",
                border_style="blue",
            )
        )
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Geronimo - Automate ML model deployments to AWS.

    Generate production-ready Terraform, Docker, and CI/CD pipelines
    for your ML models with industry best practices built-in.
    """
    pass


# ============================================================================
# INIT Command
# ============================================================================


@app.command()
def init(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        prompt="Project name",
        help="Name of the ML project.",
    ),
    framework: str = typer.Option(
        "sklearn",
        "--framework",
        "-f",
        help="ML framework (sklearn, pytorch, tensorflow).",
    ),
    template: str = typer.Option(
        "realtime",
        "--template",
        "-t",
        help="Project type: 'realtime' (API endpoints), 'batch' (pipelines), or 'both'.",
    ),
    output_dir: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Output directory for the project.",
    ),
) -> None:
    """Initialize a new ML deployment project.

    Scaffolds a complete ML project with Geronimo SDK:

    Templates:
    - realtime: FastAPI endpoints with Endpoint class
    - batch: Metaflow pipelines with BatchPipeline class
    - both: Combined real-time and batch support
    """
    from geronimo.generators.project import ProjectGenerator

    # Validate template
    valid_templates = {"realtime", "batch", "both"}
    if template not in valid_templates:
        console.print(f"[bold red]Error:[/bold red] Invalid template '{template}'. Choose from: {valid_templates}")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]Initializing project:[/bold blue] {name}")
    console.print(f"  Template: [cyan]{template}[/cyan]")

    generator = ProjectGenerator(
        project_name=name,
        framework=framework,
        output_dir=output_dir,
        template=template,
    )

    try:
        generator.generate()

        # SDK scaffolding is now handled by ProjectGenerator.generate()
        # which creates sdk/endpoint.py, sdk/pipeline.py, app.py, flow.py, etc.

        next_steps = [
            f"cd {name}",
            "uv sync",
        ]
        if template in ("realtime", "both"):
            next_steps.append(f"uvicorn {name.replace('-', '_')}.app:app --reload  # Run API server")
        if template in ("batch", "both"):
            next_steps.append(f"python -m {name.replace('-', '_')}.flow run  # Run batch pipeline")

        console.print(
            Panel.fit(
                f"[bold green]✓ Project '{name}' created successfully![/bold green]\n\n"
                f"Template: [cyan]{template}[/cyan]\n\n"
                f"Next steps:\n" + "\n".join(f"  {i+1}. {step}" for i, step in enumerate(next_steps)),
                border_style="green",
            )
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def _generate_sdk_scaffold(name: str, output_dir: str, template: str) -> None:
    """Generate Geronimo SDK scaffold files."""
    from pathlib import Path

    project_path = Path(output_dir) / name
    sdk_dir = project_path / "src" / name.replace("-", "_") / "sdk"
    sdk_dir.mkdir(parents=True, exist_ok=True)

    # SDK __init__.py
    (sdk_dir / "__init__.py").write_text('"""Geronimo SDK components."""\n')

    # Features file
    (sdk_dir / "features.py").write_text('''"""Feature definitions for the model."""

from geronimo.features import FeatureSet, Feature
# from sklearn.preprocessing import StandardScaler, OneHotEncoder


class ProjectFeatures(FeatureSet):
    """Define your features here.

    Example:
        age = Feature(dtype='numeric', transformer=StandardScaler())
        category = Feature(dtype='categorical', encoder=OneHotEncoder())
    """

    pass
''')

    # Data sources file
    (sdk_dir / "data_sources.py").write_text('''"""Data source definitions."""

from geronimo.data import DataSource, Query

# Example query-based source:
# training_data = DataSource(
#     name="training",
#     source="snowflake",
#     query=Query.from_file("queries/train.sql"),
# )

# Example file-based source:
# local_data = DataSource(name="local", source="file", path="data/train.csv")
''')

    # Model file
    (sdk_dir / "model.py").write_text(f'''"""Model definition."""

from geronimo.models import Model, HyperParams

# from .features import ProjectFeatures


class ProjectModel(Model):
    """Main model class."""

    name = "{name}"
    version = "1.0.0"
    # features = ProjectFeatures()

    def train(self, X, y, params: HyperParams) -> None:
        """Train the model."""
        # self.estimator = YourModel(**params.to_dict())
        # self.estimator.fit(X, y)
        raise NotImplementedError("Implement train() method")

    def predict(self, X):
        """Generate predictions."""
        # return self.estimator.predict(X)
        raise NotImplementedError("Implement predict() method")
''')

    # Endpoint or pipeline based on template
    if template in ("realtime", "both"):
        (sdk_dir / "endpoint.py").write_text(f'''"""Endpoint definition for real-time serving."""

from geronimo.serving import Endpoint

# from .model import ProjectModel


class PredictEndpoint(Endpoint):
    """Prediction endpoint."""

    # model_class = ProjectModel

    def preprocess(self, request: dict):
        """Preprocess incoming request."""
        # df = pd.DataFrame([request["data"]])
        # return self.model.features.transform(df)
        raise NotImplementedError("Implement preprocess() method")

    def postprocess(self, prediction):
        """Postprocess model output."""
        # return {{"score": float(prediction[0])}}
        raise NotImplementedError("Implement postprocess() method")
''')

    if template in ("batch", "both"):
        (sdk_dir / "pipeline.py").write_text(f'''"""Batch pipeline definition."""

from geronimo.batch import BatchPipeline, Schedule

# from .model import ProjectModel


class ScoringPipeline(BatchPipeline):
    """Batch scoring pipeline."""

    # model_class = ProjectModel
    schedule = Schedule.daily(hour=6)

    def run(self):
        """Main pipeline logic."""
        # data = self.model.features.data_source.load()
        # X = self.model.features.transform(data)
        # predictions = self.model.predict(X)
        # self.save_results(predictions)
        raise NotImplementedError("Implement run() method")
''')


# ============================================================================
# GENERATE Command Group
# ============================================================================

generate_app = typer.Typer(
    name="generate",
    help="Generate deployment artifacts (Terraform, Docker, pipelines).",
    no_args_is_help=True,
)
app.add_typer(generate_app, name="generate")

# Import and register keys CLI
from geronimo.cli.keys_cmd import keys_app
app.add_typer(keys_app, name="keys")


@generate_app.command("terraform")
def generate_terraform(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_dir: str = typer.Option(
        "infrastructure",
        "--output",
        "-o",
        help="Output directory for Terraform files.",
    ),
) -> None:
    """Generate Terraform infrastructure files.

    Creates modular Terraform configuration for:
    - ECR repository
    - ECS Fargate task and service
    - Application Load Balancer
    - CloudWatch logging and monitoring
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.terraform import TerraformGenerator

    console.print("\n[bold blue]Generating Terraform...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = TerraformGenerator(config=config, output_dir=output_dir)
        files = generator.generate()

        console.print(f"[green]✓ Generated {len(files)} Terraform files:[/green]")
        for f in files:
            console.print(f"  • {f}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("dockerfile")
def generate_dockerfile(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_path: str = typer.Option(
        "Dockerfile",
        "--output",
        "-o",
        help="Output path for Dockerfile.",
    ),
) -> None:
    """Generate an optimized Dockerfile for ML serving.

    Creates a multi-stage Dockerfile with:
    - UV for fast dependency installation
    - Non-root user for security
    - Proper signal handling for graceful shutdown
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.docker import DockerGenerator

    console.print("\n[bold blue]Generating Dockerfile...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = DockerGenerator(config=config, output_path=output_path)
        generator.generate()

        console.print(f"[green]✓ Generated Dockerfile:[/green] {output_path}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("pipeline")
def generate_pipeline(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_path: str = typer.Option(
        "azure-pipelines.yaml",
        "--output",
        "-o",
        help="Output path for pipeline file.",
    ),
) -> None:
    """Generate CI/CD pipeline configuration.

    Creates Azure DevOps pipeline YAML with:
    - Build and test stage
    - Security scanning
    - Multi-environment deployments
    - Approval gates
    """
    from geronimo.config.loader import load_config
    from geronimo.generators.pipeline import PipelineGenerator

    console.print("\n[bold blue]Generating pipeline...[/bold blue]")

    try:
        config = load_config(config_path)
        generator = PipelineGenerator(config=config, output_path=output_path)
        generator.generate()

        console.print(f"[green]✓ Generated pipeline:[/green] {output_path}")

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("batch")
def generate_batch(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
    output_dir: str = typer.Option(
        "batch",
        "--output",
        "-o",
        help="Output directory for batch job files.",
    ),
) -> None:
    """Generate Metaflow batch job infrastructure.

    Creates Metaflow flows and deployment configuration for:
    - AWS Step Functions (with Batch compute)
    - Astronomer Airflow (with K8s Pod Operators)
    """
    from pathlib import Path

    from geronimo.config.loader import load_config
    from geronimo.generators.metaflow import MetaflowGenerator

    console.print("\n[bold blue]Generating batch job artifacts...[/bold blue]")

    try:
        config = load_config(config_path)

        if not config.batch.enabled:
            console.print(
                "[yellow]Batch jobs not enabled. Set batch.enabled: true in geronimo.yaml[/yellow]"
            )
            return

        generator = MetaflowGenerator(
            project_name=config.project.name,
            batch_config=config.batch,
        )

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        generated = generator.generate(output_path)

        console.print(
            Panel(
                f"[green]Generated {len(generated)} batch job files in {output_dir}/[/green]",
                title="✓ Batch Generation Complete",
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@generate_app.command("all")
def generate_all(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
) -> None:
    """Generate all deployment artifacts.

    Generates Terraform, Dockerfile, CI/CD pipeline, and batch jobs in one command.
    """
    console.print("\n[bold blue]Generating all artifacts...[/bold blue]")

    # Call individual generators
    generate_terraform(config_path=config_path, output_dir="infrastructure")
    generate_dockerfile(config_path=config_path, output_path="Dockerfile")
    generate_pipeline(config_path=config_path, output_path="azure-pipelines.yaml")
    generate_batch(config_path=config_path, output_dir="batch")

    console.print("\n[bold green]✓ All artifacts generated successfully![/bold green]")



# ============================================================================
# VALIDATE Command
# ============================================================================


@app.command()
def validate(
    config_path: str = typer.Option(
        "geronimo.yaml",
        "--config",
        "-c",
        help="Path to geronimo.yaml configuration file.",
    ),
) -> None:
    """Validate project configuration against deployment rules.

    Checks configuration for:
    - Required fields
    - Valid resource specifications
    - Compliance with deployment policies
    """
    from geronimo.config.loader import load_config
    from geronimo.validation.engine import ValidationEngine

    console.print("\n[bold blue]Validating configuration...[/bold blue]")

    try:
        config = load_config(config_path)
        engine = ValidationEngine()
        results = engine.validate(config)

        if results.is_valid:
            console.print(
                Panel.fit(
                    "[bold green]✓ Configuration is valid![/bold green]\n"
                    f"Checked {results.rules_checked} rules.",
                    border_style="green",
                )
            )
        else:
            console.print("[bold red]✗ Validation failed:[/bold red]")
            for error in results.errors:
                console.print(f"  [red]•[/red] {error}")
            raise typer.Exit(code=1)

    except FileNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] Config file not found: {config_path}"
        )
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# MONITOR Command Group
# ============================================================================

monitor_app = typer.Typer(
    name="monitor",
    help="Model monitoring and drift detection.",
    no_args_is_help=True,
)
app.add_typer(monitor_app, name="monitor")


@monitor_app.command("capture-reference")
def capture_reference(
    data_path: str = typer.Argument(
        ...,
        help="Path to data file (CSV/Parquet/JSON) or SQL file for query input.",
    ),
    project_name: str = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project name.",
    ),
    input_type: str = typer.Option(
        "file",
        "--input-type",
        "-i",
        help="Input type: 'file' for data files or 'query' for SQL files.",
    ),
    source_system: str = typer.Option(
        None,
        "--source-system",
        "-s",
        help="Database source (snowflake/postgres/sqlserver). Required when input-type=query.",
    ),
    model_version: str = typer.Option(
        "1.0.0",
        "--version",
        "-v",
        help="Model version.",
    ),
    deployment_type: str = typer.Option(
        "realtime",
        "--type",
        "-t",
        help="Deployment type (realtime or batch).",
    ),
    sampling_rate: float = typer.Option(
        0.05,
        "--sampling-rate",
        help="Fraction of data to sample (0.001-1.0).",
    ),
    s3_bucket: str = typer.Option(
        "model-monitoring",
        "--s3-bucket",
        help="S3 bucket for storing snapshots.",
    ),
    output: str = typer.Option(
        "reference_snapshot.json",
        "--output",
        "-o",
        help="Output path for snapshot JSON.",
    ),
) -> None:
    """Capture a reference snapshot for drift detection.

    Supports two input types:
    - file: Reads CSV, Parquet, or JSON data files
    - query: Executes SQL from a .sql file against a database
    """
    import json
    from pathlib import Path

    import pandas as pd

    from geronimo.monitoring.snapshot import SnapshotService

    console.print("\n[bold blue]Capturing reference snapshot...[/bold blue]")

    try:
        data_file = Path(data_path)

        if input_type == "file":
            # Load from file
            if data_file.suffix == ".csv":
                data = pd.read_csv(data_file)
            elif data_file.suffix in [".parquet", ".pq"]:
                data = pd.read_parquet(data_file)
            elif data_file.suffix == ".json":
                data = pd.read_json(data_file)
            else:
                console.print(f"[bold red]Error:[/bold red] Unsupported format: {data_file.suffix}")
                raise typer.Exit(code=1)
            console.print(f"  ✓ Loaded {len(data)} rows from [cyan]{data_file.name}[/cyan]")

        elif input_type == "query":
            # Execute SQL query
            if not source_system:
                console.print(
                    "[bold red]Error:[/bold red] --source-system is required when input-type=query"
                )
                raise typer.Exit(code=1)

            if not data_file.suffix == ".sql":
                console.print(
                    f"[bold red]Error:[/bold red] Expected .sql file, got {data_file.suffix}"
                )
                raise typer.Exit(code=1)

            sql_query = data_file.read_text()
            console.print(f"  ✓ Loaded SQL from [cyan]{data_file.name}[/cyan]")
            console.print(f"  ✓ Connecting to [cyan]{source_system}[/cyan]...")

            # Execute based on source system
            data = _execute_query(sql_query, source_system)
            console.print(f"  ✓ Retrieved {len(data)} rows")

        else:
            console.print(f"[bold red]Error:[/bold red] Invalid input-type: {input_type}")
            raise typer.Exit(code=1)

        # Sample data based on sampling rate
        sample_size = max(1, int(len(data) * sampling_rate))
        if len(data) > sample_size:
            console.print(f"  ✓ Sampling {sample_size} rows ({sampling_rate:.1%})")

        service = SnapshotService(s3_bucket=s3_bucket)
        snapshot = service.capture_reference(
            data=data,
            project_name=project_name,
            model_version=model_version,
            deployment_type=deployment_type,
            sample_size=sample_size,
        )

        # Save snapshot metadata
        snapshot_dict = snapshot.model_dump(mode="json")
        # Add source info
        snapshot_dict["input_type"] = input_type
        if input_type == "query":
            snapshot_dict["source_system"] = source_system
            snapshot_dict["query_file"] = str(data_file)

        Path(output).write_text(json.dumps(snapshot_dict, indent=2, default=str))

        console.print(
            Panel(
                f"[green]Reference snapshot captured![/green]\n\n"
                f"Features: {len(snapshot.feature_statistics)}\n"
                f"Samples: {snapshot.sample_size}\n"
                f"Output: [cyan]{output}[/cyan]",
                title="✓ Snapshot Complete",
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


def _execute_query(sql: str, source_system: str) -> "pd.DataFrame":
    """Execute SQL query against specified database.

    Args:
        sql: SQL query string.
        source_system: Database type (snowflake/postgres/sqlserver).

    Returns:
        DataFrame with query results.
    """
    import pandas as pd

    if source_system == "snowflake":
        import snowflake.connector

        conn = snowflake.connector.connect(
            user=__import__("os").getenv("SNOWFLAKE_USER"),
            password=__import__("os").getenv("SNOWFLAKE_PASSWORD"),
            account=__import__("os").getenv("SNOWFLAKE_ACCOUNT"),
            warehouse=__import__("os").getenv("SNOWFLAKE_WAREHOUSE"),
            database=__import__("os").getenv("SNOWFLAKE_DATABASE"),
            schema=__import__("os").getenv("SNOWFLAKE_SCHEMA"),
        )
        return pd.read_sql(sql, conn)

    elif source_system == "postgres":
        import psycopg2

        conn = psycopg2.connect(__import__("os").getenv("POSTGRES_CONNECTION_STRING"))
        return pd.read_sql(sql, conn)

    elif source_system == "sqlserver":
        import pyodbc

        conn = pyodbc.connect(__import__("os").getenv("SQLSERVER_CONNECTION_STRING"))
        return pd.read_sql(sql, conn)

    else:
        raise ValueError(f"Unsupported source system: {source_system}")


@monitor_app.command("detect-drift")
def detect_drift(
    reference_path: str = typer.Argument(
        ...,
        help="Path to reference snapshot JSON.",
    ),
    current_data: str = typer.Argument(
        ...,
        help="Path to current data CSV/Parquet.",
    ),
    output: str = typer.Option(
        "drift_report.json",
        "--output",
        "-o",
        help="Output path for drift report.",
    ),
    threshold: float = typer.Option(
        0.1,
        "--threshold",
        help="Drift score threshold for alerting.",
    ),
) -> None:
    """Detect drift between reference snapshot and current data.

    Compares feature distributions and generates a drift report.
    """
    import json
    from datetime import datetime
    from pathlib import Path

    import pandas as pd

    from geronimo.monitoring.drift_models import DriftReport, FeatureDrift, ReferenceSnapshot
    from geronimo.monitoring.snapshot import SnapshotService

    console.print("\n[bold blue]Detecting drift...[/bold blue]")

    try:
        # Load reference
        ref_data = json.loads(Path(reference_path).read_text())
        reference = ReferenceSnapshot.model_validate(ref_data)

        # Load current data
        current_file = Path(current_data)
        if current_file.suffix == ".csv":
            data = pd.read_csv(current_file)
        else:
            data = pd.read_parquet(current_file)

        # Compute current stats
        service = SnapshotService()
        current_window = service.capture_window(
            data=data,
            project_name=reference.project_name,
            deployment_type=reference.deployment_type,
            window_start=datetime.utcnow(),
            window_end=datetime.utcnow(),
        )

        # Compare features
        feature_drift = {}
        drift_features = 0

        for name, ref_stats in reference.feature_statistics.items():
            if name not in current_window.feature_statistics:
                continue

            curr_stats = current_window.feature_statistics[name]
            drift_score = 0.0

            if ref_stats.mean is not None and curr_stats.mean is not None:
                # Simple mean shift as drift indicator
                if ref_stats.std and ref_stats.std > 0:
                    drift_score = abs(curr_stats.mean - ref_stats.mean) / ref_stats.std
                else:
                    drift_score = abs(curr_stats.mean - ref_stats.mean)

            drift_detected = drift_score > threshold
            if drift_detected:
                drift_features += 1

            feature_drift[name] = FeatureDrift(
                feature_name=name,
                drift_detected=drift_detected,
                drift_score=min(drift_score, 1.0),
                stattest_name="mean_shift",
                stattest_threshold=threshold,
                reference_mean=ref_stats.mean,
                current_mean=curr_stats.mean,
            )

        # Overall drift
        total_features = len(feature_drift)
        dataset_drift = (drift_features / total_features) > 0.3 if total_features > 0 else False
        overall_score = drift_features / total_features if total_features > 0 else 0.0

        report = DriftReport(
            id=str(__import__("uuid").uuid4()),
            project_name=reference.project_name,
            reference_id=reference.id,
            recent_window_id=current_window.id,
            created_at=datetime.utcnow(),
            dataset_drift=dataset_drift,
            drift_score=overall_score,
            feature_drift=feature_drift,
            alert_triggered=dataset_drift,
        )

        # Save report
        report_dict = report.model_dump(mode="json")
        Path(output).write_text(json.dumps(report_dict, indent=2, default=str))

        # Display results
        status = "[red]DRIFT DETECTED[/red]" if dataset_drift else "[green]No significant drift[/green]"
        console.print(
            Panel(
                f"{status}\n\n"
                f"Features with drift: {drift_features}/{total_features}\n"
                f"Overall score: {overall_score:.2%}\n"
                f"Report: [cyan]{output}[/cyan]",
                title="Drift Report",
            )
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# DEPLOY Command Group
# ============================================================================

deploy_app = typer.Typer(
    name="deploy",
    help="Deploy infrastructure using Pulumi (requires: pip install geronimo[pulumi]).",
    no_args_is_help=True,
)
app.add_typer(deploy_app, name="deploy")


@deploy_app.command("up")
def deploy_up(
    project: str = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project name.",
    ),
    target: str = typer.Option(
        "aws",
        "--target",
        "-t",
        help="Cloud target (aws, gcp, azure).",
    ),
    region: str = typer.Option(
        "us-east-1",
        "--region",
        "-r",
        help="Cloud region.",
    ),
    component: str = typer.Option(
        None,
        "--component",
        "-c",
        help="Specific component to deploy (artifacts, serving, batch).",
    ),
    stack: str = typer.Option(
        "dev",
        "--stack",
        "-s",
        help="Pulumi stack name.",
    ),
) -> None:
    """Deploy infrastructure to the cloud.
    
    Requires Pulumi: pip install geronimo[pulumi]
    
    Examples:
        geronimo deploy up --project iris --target aws
        geronimo deploy up --project iris --component artifacts
    """
    from geronimo.deploy import DeploymentConfig, deploy
    from geronimo.deploy.targets import PulumiNotInstalledError
    
    console.print(f"\n[bold blue]Deploying {project} to {target}...[/bold blue]")
    
    try:
        config = DeploymentConfig(
            project=project,
            target=target,
            region=region,
            stack_name=stack,
        )
        
        result = deploy(config, component=component)
        
        console.print(
            Panel(
                f"[green]✓ Deployment complete![/green]\n\n"
                f"Stack: {stack}\n"
                f"Outputs:\n" + "\n".join(f"  {k}: {v}" for k, v in result.get("outputs", {}).items()),
                title="Deployment Success",
                border_style="green",
            )
        )
        
    except PulumiNotInstalledError as e:
        console.print(f"[bold yellow]Warning:[/bold yellow] {e}")
        console.print("\nAlternatives:")
        console.print("  1. Install Pulumi: [cyan]pip install geronimo[pulumi][/cyan]")
        console.print("  2. Generate static IaC: [cyan]geronimo generate terraform[/cyan]")
        raise typer.Exit(code=1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@deploy_app.command("destroy")
def deploy_destroy(
    project: str = typer.Option(
        ...,
        "--project",
        "-p",
        help="Project name.",
    ),
    target: str = typer.Option(
        "aws",
        "--target",
        "-t",
        help="Cloud target.",
    ),
    stack: str = typer.Option(
        "dev",
        "--stack",
        "-s",
        help="Pulumi stack name.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Skip confirmation prompt.",
    ),
) -> None:
    """Destroy deployed infrastructure.
    
    Removes all resources created by 'deploy up'.
    """
    if not force:
        confirm = typer.confirm(f"Destroy all resources for {project}/{stack}?")
        if not confirm:
            console.print("[yellow]Aborted.[/yellow]")
            raise typer.Exit()
    
    console.print(f"\n[bold red]Destroying {project}/{stack}...[/bold red]")
    
    try:
        from geronimo.deploy.config import DeploymentConfig
        from geronimo.deploy.providers.aws import destroy_aws
        
        config = DeploymentConfig(
            project=project,
            target=target,
            stack_name=stack,
        )
        
        if target == "aws":
            destroy_aws(config)
        else:
            console.print(f"[yellow]Destroy not implemented for {target}[/yellow]")
            raise typer.Exit(code=1)
        
        console.print("[green]✓ Resources destroyed.[/green]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# ============================================================================
# IMPORT Command
# ============================================================================

# Register import command from separate module
from geronimo.cli.import_cmd import import_project

app.command(name="import")(import_project)


if __name__ == "__main__":
    app()
