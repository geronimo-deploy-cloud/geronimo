"""Pipeline definition - implement your batch processing logic."""

from geronimo.batch import BatchPipeline, Schedule
from .model import ProjectModel
from .data_sources import scoring_data  # Import scoring data source


class ScoringPipeline(BatchPipeline):
    """Batch scoring pipeline.
    
    Implement the run() method with your batch processing logic.
    The pipeline will be executed on the defined schedule.
    Uses scoring_data from data_sources.py to load input data.
    
    Example:
        def run(self):
            # Load data from configured source
            data = scoring_data.load()
            
            # Transform and predict
            X = self.model.features.transform(data)
            predictions = self.model.predict(X)
            
            # Save results
            results = data.assign(prediction=predictions)
            return self.save_results(results)
    """

    name = "test-batch-scoring"
    model_class = ProjectModel
    schedule = Schedule.daily(hour=6, minute=0)
    data_source = scoring_data  # Connect to scoring data source

    def run(self):
        """Execute batch processing.
        
        Returns:
            Dict with execution results
        """
        # TODO: Implement batch logic
        # Load data from configured source
        # data = self.data_source.load()
        # 
        # Transform features
        # X = self.model.features.transform(data)
        # 
        # Generate predictions
        # predictions = self.model.predict(X)
        # 
        # Save results
        # results = data.assign(prediction=predictions)
        # output_path = self.save_results(results)
        # return {"samples_scored": len(results), "output_path": output_path}
        raise NotImplementedError("Implement run()")
