"""BatchPipeline base class for scheduled/triggered batch jobs."""

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from geronimo.batch.schedule import Schedule, Trigger
    from geronimo.models import Model


class BatchPipeline(ABC):
    """Base class for batch ML pipelines.

    Provides a standardized interface for batch jobs with
    schedule/trigger support and integrated artifact management.

    Example:
        ```python
        from geronimo.batch import BatchPipeline, Schedule
        from myproject.models import CreditRiskModel

        class DailyScoringPipeline(BatchPipeline):
            model_class = CreditRiskModel
            schedule = Schedule.daily(hour=6)

            def run(self):
                # Load data
                data = self.model.features.data_source.load()

                # Transform features
                X = self.model.features.transform(data)

                # Predict
                predictions = self.model.predict(X)

                # Save results
                self.save_results(predictions)

        # Execute
        pipeline = DailyScoringPipeline()
        pipeline.initialize()
        pipeline.execute()
        ```
    """

    # Override in subclass
    model_class: type["Model"] = None
    schedule: Optional["Schedule"] = None
    trigger: Optional["Trigger"] = None
    artifact_project: Optional[str] = None
    artifact_version: Optional[str] = None

    def __init__(self):
        """Initialize pipeline."""
        self.model: Optional["Model"] = None
        self._is_initialized: bool = False

    def initialize(
        self,
        project: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Initialize pipeline by loading model artifacts.

        Args:
            project: Artifact project name.
            version: Artifact version.
        """
        from geronimo.artifacts import ArtifactStore

        project = project or self.artifact_project or self.model_class.name
        version = version or self.artifact_version or self.model_class.version

        store = ArtifactStore.load(project=project, version=version)

        self.model = self.model_class()
        self.model.load(store)
        self._is_initialized = True

    def execute(self) -> Any:
        """Execute the pipeline.

        Returns:
            Pipeline result.
        """
        if not self._is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")

        return self.run()

    @abstractmethod
    def run(self) -> Any:
        """Main pipeline logic.

        Override this method with your batch processing logic.

        Returns:
            Pipeline result.
        """
        pass

    def save_results(
        self,
        results: Any,
        output_path: Optional[str] = None,
        use_artifact_store: bool = False,
        artifact_name: Optional[str] = None,
    ) -> str:
        """Save pipeline results.

        Args:
            results: Results to save (DataFrame, dict, etc.).
            output_path: Optional output path (for file-based storage).
            use_artifact_store: If True, save via ArtifactStore for consistency
                               with other artifacts. Default False for backward
                               compatibility.
            artifact_name: Name for artifact when using artifact store.
                          Defaults to pipeline class name.

        Returns:
            Path or URI where results were saved.
        """
        import pandas as pd
        from datetime import datetime
        from pathlib import Path

        # Use ArtifactStore if requested
        if use_artifact_store:
            from geronimo.artifacts import ArtifactStore
            
            project = self.artifact_project or (
                self.model_class.name if self.model_class else self.__class__.__name__
            )
            version = self.artifact_version or (
                self.model_class.version if self.model_class else "1.0.0"
            )
            
            store = ArtifactStore(project=project, version=version)
            name = artifact_name or f"{self.__class__.__name__}_results"
            return store.save(name, results, artifact_type="pipeline_result")

        # Default file-based storage (backward compatible)
        if output_path is None:
            output_path = f"output/{self.__class__.__name__}_{datetime.now().isoformat()}.parquet"

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(results, pd.DataFrame):
            results.to_parquet(path)
        else:
            import json

            path = path.with_suffix(".json")
            path.write_text(json.dumps(results, default=str))

        return str(path)

    @property
    def is_initialized(self) -> bool:
        """Check if pipeline is initialized."""
        return self._is_initialized

    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "not initialized"
        model_name = self.model_class.__name__ if self.model_class else "None"
        schedule_str = str(self.schedule) if self.schedule else "manual"
        return f"{self.__class__.__name__}(model={model_name}, schedule={schedule_str}, {status})"
