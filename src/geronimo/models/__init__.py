"""Geronimo Models Module.

The models module defines the base classes and interfaces for creating machine
learning models within Geronimo. It handles standardizing the interface for
training, evaluation, and inference.

Key components:
- Model: The base class that all user models must inherit from.
- HyperParams: A Pydantic model for defining and validating hyperparameters.

By subclassing `Model`, users can plug their custom logic (using PyTorch,
Scikit-Learn, etc.) into the Geronimo ecosystem.
"""

from geronimo.models.base import Model
from geronimo.models.objective import ObjectiveSpec
from geronimo.models.params import HyperParams

__all__ = ["Model", "ObjectiveSpec", "HyperParams"]

__docformat__ = "google"
