"""Batch pipeline definition."""

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
