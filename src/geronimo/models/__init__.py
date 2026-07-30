"""Geronimo Models Module.

The models module defines the base classes and interfaces for creating machine
learning models within Geronimo. It handles standardizing the interface for
training, evaluation, and inference.

Key components:
- Model: The base class that all user models must inherit from.
- HyperParams: A Pydantic model for defining and validating hyperparameters.
- CalibrationSpec: Opt-in probability calibration specification.

By subclassing `Model`, users can plug their custom logic (using PyTorch,
Scikit-Learn, etc.) into the Geronimo ecosystem.
"""

from geronimo.models.base import Model
from geronimo.models.calibration import CalibrationSpec
from geronimo.models.params import HyperParams
from geronimo.models.objective import ObjectiveSpec

__all__ = [
    "CalibrationSpec",
    "HyperParams",
    "Model", 
    "ObjectiveSpec"
]

__docformat__ = "google"
