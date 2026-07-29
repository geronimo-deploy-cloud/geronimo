"""Geronimo Batch Processing Module.

The batch module provides the framework for defining and scheduling recurring
machine learning workflows, such as batch inference or automated training jobs.

Key components:
- BatchPipeline: Defines a sequence of steps to execute for a batch job.
- Schedule: Specifies when the pipeline should run (e.g., cron expressions).
- Trigger: Defines events that can start a pipeline execution.

This module helps create robust, scheduled ML pipelines that can run on various
execution environments.
"""

from geronimo.batch.output_spec import OutputSpec
from geronimo.batch.pipeline import BatchPipeline
from geronimo.batch.schedule import Schedule, Trigger

__all__ = ["BatchPipeline", "OutputSpec", "Schedule", "Trigger"]

__docformat__ = "google"
