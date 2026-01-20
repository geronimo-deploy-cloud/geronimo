"""Metaflow flow - thin wrapper around SDK pipeline.

Run locally:
    python -m test_batch.flow run

Deploy to Step Functions:
    python -m test_batch.flow step-functions create
"""

from metaflow import FlowSpec, step, schedule
from test_batch.sdk.pipeline import ScoringPipeline


@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Batch scoring flow - wraps SDK pipeline."""

    @step
    def start(self):
        """Initialize pipeline and load model."""
        self.pipeline = ScoringPipeline()
        self.pipeline.initialize()
        print(f"Initialized: {self.pipeline}")
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        """Execute the SDK pipeline."""
        self.result = self.pipeline.execute()
        print(f"Result: {self.result}")
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Pipeline complete: {self.result}")


if __name__ == "__main__":
    ScoringFlow()
