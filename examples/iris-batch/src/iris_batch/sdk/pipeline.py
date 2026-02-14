"""Pipeline definition - batch processing logic."""

from geronimo.batch import BatchPipeline
from geronimo.batch.schedule import Schedule
from .model import IrisBatchModel

# For this example, we'll just re-score the training data as a demo
from .data_sources import training_data

class ScoringPipeline(BatchPipeline):
    """Batch scoring pipeline.
    
    This is a working demo pipeline. Replace run() with your actual 
    implementation once you have a trained model.
    
    To train a model:
        uv run python -m iris_batch.train
    """
    
    # Point to the model class - this just helps with doc gen
    model_class = IrisBatchModel
    # Schedule: Run daily at 6:00 AM
    schedule = Schedule.daily(hour=6)

    def initialize(self):
        """Initialize pipeline.

        Parent class will load the model from ArtifactStore.
        """
        super().initialize()

    def execute(self):
        """Execute the pipeline.

        Parent class just calls run() by default.
        """
        return super().execute()

    def run(self):
        """Execute batch processing.
        
        Demo mode implementation - replace with your actual logic.
        
        Returns:
            Dict with execution results
        """
        print("Starting batch scoring job...")
        
        # 1. Load data to score
        print("Loading data...")
        df = training_data.load()

        # 2. Transform data using fitted feature transformers from training
        X = self.model.features.transform(df)
        
        # 3. Run predictions
        print(f"Scoring {len(X)} records...")
        probabilities = self.model.predict_proba(X)
        predictions = self.model.predict(X)
        
        # 4. Format results
        results = df.copy()
        results["predicted_species_idx"] = predictions
        results["max_probability"] = probabilities.max(axis=1)
        
        return results