"""Geronimo Pipelines Module.

Exposes pipeline base classes for batch and stream processing.
"""

from geronimo.batch.pipeline import BatchPipeline
from geronimo.batch.schedule import Schedule

__all__ = ["BatchPipeline", "Schedule"]
