"""MLflow backend for ArtifactStore.

Provides integration with MLflow for artifact storage and experiment tracking.
Requires: pip install geronimo[mlflow]
"""

import os
import pickle
import tempfile
from datetime import datetime
from typing import Any, Optional

from geronimo.artifacts.store import ArtifactMetadata


def _check_mlflow_available() -> None:
    """Check if MLflow is installed."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "MLflow is not installed. Install with: pip install geronimo[mlflow]"
        )


class MLflowArtifactStore:
    """MLflow-backed artifact store.

    Stores artifacts using MLflow's artifact storage and
    tracks experiments/runs for versioning.

    Example:
        ```python
        from geronimo.artifacts.mlflow_backend import MLflowArtifactStore

        # Set MLflow tracking URI (or use env var MLFLOW_TRACKING_URI)
        store = MLflowArtifactStore(
            project="credit-risk",
            version="1.2.0",
            tracking_uri="http://localhost:5000",
        )

        # Save artifacts (creates MLflow run)
        store.save("model", trained_model)
        store.save("encoder", fitted_encoder)

        # Load artifacts
        store = MLflowArtifactStore.load(project="credit-risk", version="1.2.0")
        model = store.get("model")
        ```
    """

    def __init__(
        self,
        project: str,
        version: str,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_id: Optional[str] = None,
    ):
        """Initialize MLflow artifact store.

        Args:
            project: Project name (used as experiment name if not specified).
            version: Version string (used as run name).
            tracking_uri: MLflow tracking server URI.
            experiment_name: MLflow experiment name.
            run_id: Existing run ID to use (for loading).
        """
        _check_mlflow_available()
        import mlflow

        self.project = project
        self.version = version
        self.experiment_name = experiment_name or project

        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self._run_id = run_id
        self._run = None
        self._metadata: dict[str, ArtifactMetadata] = {}

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self._experiment_id = experiment.experiment_id

    @classmethod
    def load(
        cls,
        project: str,
        version: str,
        tracking_uri: Optional[str] = None,
    ) -> "MLflowArtifactStore":
        """Load existing artifact store by finding matching run.

        Args:
            project: Project name.
            version: Version string.
            tracking_uri: MLflow tracking server URI.

        Returns:
            MLflowArtifactStore instance.
        """
        _check_mlflow_available()
        import mlflow

        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Find run by version tag
        experiment = mlflow.get_experiment_by_name(project)
        if experiment is None:
            raise ValueError(f"No experiment found for project: {project}")

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.version = '{version}'",
            max_results=1,
        )

        if len(runs) == 0:
            raise ValueError(f"No run found for version: {version}")

        run_id = runs.iloc[0]["run_id"]
        return cls(project=project, version=version, run_id=run_id)

    def save(
        self,
        name: str,
        artifact: Any,
        artifact_type: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> str:
        """Save an artifact to MLflow.

        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            artifact_type: Optional type hint.
            tags: Optional metadata tags.

        Returns:
            Artifact URI.
        """
        import mlflow

        artifact_type = artifact_type or type(artifact).__name__

        # Start run if not active
        if self._run is None:
            self._run = mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=f"{self.project}-{self.version}",
                tags={"version": self.version, "project": self.project},
            )
            self._run_id = self._run.info.run_id

        # Save artifact as pickle
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            temp_path = f.name

        try:
            mlflow.log_artifact(temp_path, artifact_path=name)
            artifact_uri = f"runs:/{self._run_id}/{name}/{os.path.basename(temp_path)}"
            size = os.path.getsize(temp_path)
        finally:
            os.unlink(temp_path)

        # Track metadata
        metadata = ArtifactMetadata(
            name=name,
            version=self.version,
            artifact_type=artifact_type,
            created_at=datetime.utcnow(),
            size_bytes=size,
            tags=tags or {},
        )
        self._metadata[name] = metadata

        # Log artifact metadata as params
        mlflow.log_param(f"{name}_type", artifact_type)

        return artifact_uri

    def get(self, name: str) -> Any:
        """Load an artifact from MLflow.

        Args:
            name: Artifact name.

        Returns:
            Deserialized artifact.
        """
        import mlflow

        if self._run_id is None:
            raise ValueError("No run ID available. Load store first.")

        # Download artifact
        client = mlflow.tracking.MlflowClient()
        local_path = client.download_artifacts(self._run_id, name)

        # Find pickle file
        import os

        pkl_files = [f for f in os.listdir(local_path) if f.endswith(".pkl")]
        if not pkl_files:
            raise KeyError(f"Artifact not found: {name}")

        with open(os.path.join(local_path, pkl_files[0]), "rb") as f:
            return pickle.load(f)

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to MLflow run.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step number.
        """
        import mlflow

        if self._run is None:
            raise ValueError("No active run. Save an artifact first to start a run.")

        mlflow.log_metrics(metrics, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow run.

        Args:
            params: Dictionary of parameter name to value.
        """
        import mlflow

        if self._run is None:
            raise ValueError("No active run. Save an artifact first to start a run.")

        mlflow.log_params(params)

    def end_run(self) -> None:
        """End the MLflow run."""
        import mlflow

        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def __repr__(self) -> str:
        return f"MLflowArtifactStore({self.project}@{self.version})"
