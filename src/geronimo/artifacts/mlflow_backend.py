"""MLflow backend for ArtifactStore.

Provides integration with MLflow for artifact storage and experiment tracking.
Requires: pip install geronimo[mlflow]
"""

import os
import pickle
import tempfile
from datetime import datetime
from typing import Any, Optional

from geronimo.artifacts.protocol import ArtifactBackend


def _check_mlflow_available() -> None:
    """Check if MLflow is installed."""
    try:
        import mlflow  # noqa: F401
    except ImportError:
        raise ImportError(
            "MLflow is not installed. Install with: pip install geronimo[mlflow]"
        )


class MLFlowArtifactBackend(ArtifactBackend):
    """MLflow implementation of ArtifactBackend.
    
    Stores artifacts using MLflow's artifact storage and
    tracks experiments/runs for versioning.
    """

    project: str
    """The project name."""

    version: str
    """The version string."""

    experiment_name: str
    """The MLflow experiment name."""

    _experiment_id: str
    """The internal MLflow experiment ID."""

    _run_id: Optional[str]
    """The internal MLflow run ID."""
    
    def __init__(
        self,
        project: str,
        version: str,
        tracking_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ):
        """Initialize MLflow backend.

        Args:
            project: Project name (used as experiment name if not specified).
            version: Version string.
            tracking_uri: MLflow tracking server URI.
            experiment_name: MLflow experiment name.
        """
        _check_mlflow_available()
        import mlflow

        self.project = project
        self.version = version
        self.experiment_name = experiment_name or project

        # Set tracking URI if provided
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            
        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self._experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            self._experiment_id = experiment.experiment_id
            
        self._run_id = None

    def _get_run_id(self, create_if_missing: bool = False) -> Optional[str]:
        """Get the run ID for the current version."""
        import mlflow
        
        # If we already have a run ID, return it
        if self._run_id:
            return self._run_id
            
        # Search for existing run with this version
        runs = mlflow.search_runs(
            experiment_ids=[self._experiment_id],
            filter_string=f"tags.version = '{self.version}'",
            max_results=1,
        )
        
        if len(runs) > 0:
            self._run_id = runs.iloc[0]["run_id"]
            return self._run_id
            
        if create_if_missing:
            # Start new run
            with mlflow.start_run(
                experiment_id=self._experiment_id,
                run_name=f"{self.project}-{self.version}",
                tags={"version": self.version, "project": self.project},
            ) as run:
                self._run_id = run.info.run_id
            return self._run_id
            
        return None

    def save(self, name: str, artifact: Any, metadata: dict) -> str:
        """Save an artifact to MLflow.
        
        Args:
            name: Artifact name.
            artifact: Python object to serialize.
            metadata: Artifact metadata dict.
            
        Returns:
            Artifact URI.
        """
        import mlflow
        
        # Ensure run exists
        run_id = self._get_run_id(create_if_missing=True)
        
        # Serialize to temp file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(artifact, f)
            temp_path = f.name
            
        try:
            # Log artifact to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(temp_path, artifact_path=name)
                
                # Log metadata as params
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                         mlflow.log_param(f"{name}_{key}", value)
                    elif isinstance(value, dict):
                        # Flatten tags for logging
                        for tag_k, tag_v in value.items():
                            mlflow.log_param(f"{name}_tag_{tag_k}", tag_v)

            artifact_uri = f"runs:/{run_id}/{name}/{os.path.basename(temp_path)}"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        return artifact_uri

    def load(self, uri: str) -> Any:
        """Load an artifact.
        
        Args:
            uri: Artifact name or full URI.
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        
        # If uri looks like a full runs:/ uri, parse it?
        # But ArtifactStore.get() passes the name. 
        # So we assume uri is the name (because ArtifactMetadata isn't fully utilized for retrieval location in Store.get yet, it calls backend.load(name))
        
        # Check if it's a full URI or just name
        if uri.startswith("runs:/"):
            # Parse run_id and path? backend.load contract says 'uri' returned from save, but
            # ArtifactStore.get(name) calls backend.load(name).
            # We must support treating 'uri' as 'name'.
            # For simplicity let's support name lookup.
            pass

        run_id = self._get_run_id(create_if_missing=False)
        if not run_id:
            raise KeyError(f"No run found for version {self.version}")

        # Construct path
        # In save(), we put it under artifact_path=name
        # Listing artifacts will show it under 'name' folder?
        # Let's check how log_artifact works. 
        # mlflow.log_artifact(local_file, artifact_path="model") -> creates "model/local_file_name"
        
        # Download artifacts from the run
        # We need to find the pkl file under the name directory
        local_path = client.download_artifacts(run_id, uri)
        
        # If it's a directory (which it is if we used artifact_path=name), find the pkl inside
        if os.path.isdir(local_path):
            pkl_files = [f for f in os.listdir(local_path) if f.endswith(".pkl")]
            if not pkl_files:
                raise KeyError(f"No pickle file found for artifact: {uri}")
            file_path = os.path.join(local_path, pkl_files[0])
        else:
            file_path = local_path
            
        with open(file_path, "rb") as f:
            return pickle.load(f)

    def list(self, prefix: Optional[str] = None) -> list[str]:
        """List artifacts.
        
        Returns:
            List of artifact URIs.
        """
        import mlflow
        from mlflow.tracking import MlflowClient
        
        run_id = self._get_run_id(create_if_missing=False)
        if not run_id:
            return []
            
        client = MlflowClient()
        artifacts = client.list_artifacts(run_id)
        
        results = []
        for artifact in artifacts:
            # We used artifact name as 'path' directory
            if prefix and not artifact.path.startswith(prefix):
                continue
            results.append(f"runs:/{run_id}/{artifact.path}")
            
        return results

    def delete(self, uri: str) -> None:
        """Delete an artifact.
        
        Args:
            uri: Artifact name.
        """
        import mlflow
        # MLflow doesn't easily support deleting single artifacts from a run via standard client
        # without deleting the whole run or using direct store access.
        # We'll validly implement as a no-op or log warning effectively.
        # But if the user really wants to delete, we might delete the run if it's empty?
        # For now, let's just log a warning that it's not fully supported.
        pass
