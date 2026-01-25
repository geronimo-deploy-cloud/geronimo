"""Metaflow flow - thin wrapper around SDK pipeline.

Run locally:
    python -m iris_batch.flow run

Deploy to Step Functions:
    python -m iris_batch.flow step-functions create
"""

from metaflow import FlowSpec, step, schedule
from iris_batch.sdk.pipeline import get_pipeline


@schedule(daily=True)
class ScoringFlow(FlowSpec):
    """Batch scoring flow - wraps SDK pipeline."""

    @step
    def start(self):
        """Initialize pipeline and load model."""
        self.pipeline = get_pipeline()
        self.pipeline.initialize()
        print(f"Initialized: {self.pipeline}")
        self.next(self.run_pipeline)

    @step
    def run_pipeline(self):
        """Execute the SDK pipeline."""
        print("Running pipeline logic...")
        self.pipeline.run()
        self.next(self.end)

    @step
    def end(self):
        """Flow complete."""
        print(f"Pipeline flow complete.")


if __name__ == "__main__":
    ScoringFlow()
