"""Iris batch scoring pipeline using BatchPipeline.

This demonstrates a proper batch job that runs on a schedule,
not a real-time API endpoint.
"""

from geronimo.batch import BatchPipeline, Schedule
from iris_batch.model import IrisModel


class IrisScoringPipeline(BatchPipeline):
    """Daily batch scoring pipeline for Iris classification.
    
    Runs daily at 6 AM to score new iris samples and save predictions.
    
    Example:
        ```python
        pipeline = IrisScoringPipeline()
        pipeline.initialize()
        pipeline.execute()
        ```
    """
    
    name = "iris-scoring"
    model_class = IrisModel
    schedule = Schedule.daily(hour=6, minute=0)
    
    # Artifact configuration
    artifact_project = "iris-classifier"
    artifact_version = "1.0.0"
    
    def run(self):
        """Main batch processing logic.
        
        1. Load new data to score
        2. Apply feature transformations
        3. Generate predictions
        4. Save results to output
        """
        import pandas as pd
        from pathlib import Path
        
        # Load data to score
        data_path = Path("data/iris_to_score.csv")
        if not data_path.exists():
            # Use sample data if no input file
            data = self._generate_sample_data()
        else:
            data = pd.read_csv(data_path)
        
        print(f"Loaded {len(data)} samples to score")
        
        # Transform features
        X = self.model.features.transform(data)
        
        # Generate predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Create results DataFrame
        results = data.copy()
        results["prediction"] = predictions
        results["probability"] = probabilities.max(axis=1)
        results["scored_at"] = pd.Timestamp.now().isoformat()
        
        # Save results
        output_path = self.save_results(results)
        print(f"Saved {len(results)} predictions to {output_path}")
        
        return {
            "samples_scored": len(results),
            "output_path": output_path,
        }
    
    def _generate_sample_data(self) -> "pd.DataFrame":
        """Generate sample data for demonstration."""
        import pandas as pd
        import numpy as np
        
        np.random.seed(42)
        n_samples = 100
        
        return pd.DataFrame({
            "sepal_length": np.random.uniform(4.0, 8.0, n_samples),
            "sepal_width": np.random.uniform(2.0, 4.5, n_samples),
            "petal_length": np.random.uniform(1.0, 7.0, n_samples),
            "petal_width": np.random.uniform(0.1, 2.5, n_samples),
        })


if __name__ == "__main__":
    # Manual execution
    pipeline = IrisScoringPipeline()
    pipeline.initialize()
    result = pipeline.execute()
    print(f"Pipeline complete: {result}")
