"""Metaflow flow for Iris batch scoring.

This is the orchestration layer that can run locally or deploy to:
- AWS Step Functions (with AWS Batch compute)
- Kubernetes (with Argo)
- Airflow (via DAG export)

Run locally:
    python flows/scoring_flow.py run
    
Deploy to Step Functions:
    python flows/scoring_flow.py step-functions create
"""

from metaflow import FlowSpec, step, Parameter, schedule


@schedule(daily=True)
class IrisScoringFlow(FlowSpec):
    """Daily batch scoring flow for Iris classification.
    
    Steps:
        1. load_data - Load samples to score
        2. transform - Apply feature transformations
        3. predict - Generate predictions
        4. save_results - Write output to storage
    """
    
    # Parameters
    input_path = Parameter(
        "input_path",
        help="Path to input data (CSV)",
        default="data/iris_to_score.csv",
    )
    output_path = Parameter(
        "output_path",
        help="Path for output predictions",
        default="output/predictions.parquet",
    )
    
    @step
    def start(self):
        """Initialize the flow and load model artifacts."""
        from iris_batch.pipeline import IrisScoringPipeline
        
        self.pipeline = IrisScoringPipeline()
        self.pipeline.initialize()
        print(f"Initialized pipeline: {self.pipeline}")
        
        self.next(self.load_data)
    
    @step
    def load_data(self):
        """Load data to score."""
        import pandas as pd
        from pathlib import Path
        
        path = Path(self.input_path)
        if path.exists():
            self.data = pd.read_csv(path)
        else:
            # Generate sample data
            self.data = self.pipeline._generate_sample_data()
        
        print(f"Loaded {len(self.data)} samples")
        self.next(self.transform)
    
    @step
    def transform(self):
        """Apply feature transformations."""
        self.X = self.pipeline.model.features.transform(self.data)
        print(f"Transformed features: {self.X.shape}")
        self.next(self.predict)
    
    @step
    def predict(self):
        """Generate predictions."""
        import pandas as pd
        
        predictions = self.pipeline.model.predict(self.X)
        probabilities = self.pipeline.model.predict_proba(self.X)
        
        # Build results
        self.results = self.data.copy()
        self.results["prediction"] = predictions
        self.results["probability"] = probabilities.max(axis=1)
        self.results["scored_at"] = pd.Timestamp.now().isoformat()
        
        print(f"Generated {len(self.results)} predictions")
        self.next(self.save_results)
    
    @step
    def save_results(self):
        """Save predictions to storage."""
        from pathlib import Path
        
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.results.to_parquet(path, index=False)
        print(f"Saved results to {path}")
        
        self.next(self.end)
    
    @step
    def end(self):
        """Flow complete."""
        print(f"Scored {len(self.results)} samples")
        print(f"Output: {self.output_path}")


if __name__ == "__main__":
    IrisScoringFlow()
