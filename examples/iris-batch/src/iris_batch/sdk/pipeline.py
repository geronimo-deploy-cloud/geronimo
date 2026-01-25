"""Pipeline definition for Iris batch scoring."""

from typing import Optional
import pandas as pd
from pathlib import Path

from geronimo.pipelines import BatchPipeline, Schedule
from geronimo.artifacts import ArtifactStore
from .model import IrisModel
from .data_sources import training_data


class ScoringPipeline(BatchPipeline):
    """Batch pipeline for scoring new data.
    
    Loads the pre-trained model from ArtifactStore and runs predictions
    on the input dataset.
    """
    
    # Schedule: Run daily at 6:00 AM
    schedule = Schedule.daily(hour=6)
    
    def initialize(self, project: str = "iris-batch", version: str = "1.0.0") -> None:
        """Initialize pipeline by loading model from ArtifactStore."""
        self.project = project
        self.version = version
        
        # Load artifacts
        print(f"Loading model artifacts for {project}@{version}...")
        self._store = ArtifactStore.load(project=project, version=version)
        self.model = IrisModel()
        self.model.load(self._store)
        self._is_initialized = True

    def run(self) -> None:
        """Execute the batch scoring job.
        
        1. Load data (in production this would be new data, here reuse training)
        2. Run predictions
        3. Save results
        """
        if not hasattr(self, "_is_initialized") or not self._is_initialized:
            self.initialize()
            
        print("Starting batch scoring job...")
        
        # 1. Load data to score
        # For this example, we'll re-score the training data as a demo
        print("Loading data...")
        df = training_data.load()
        
        # 2. Run predictions
        print(f"Scoring {len(df)} records...")
        probabilities = self.model.predict_proba(df)
        predictions = self.model.predict(df)
        
        # 3. Format results
        results = df.copy()
        results["predicted_species_idx"] = predictions
        results["predicted_species"] = [IrisModel.SPECIES[p] for p in predictions]
        results["max_probability"] = probabilities.max(axis=1)
        
        # 4. Save results (to CSV for demo)
        output_path = Path("iris_predictions.csv")
        results.to_csv(output_path, index=False)
        print(f"✅ Scoring complete. Results saved to {output_path}")
        print(results[["species_name", "predicted_species", "max_probability"]].head())


# Singleton
_pipeline: Optional[ScoringPipeline] = None

def get_pipeline() -> ScoringPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = ScoringPipeline()
    return _pipeline
