"""API dependencies."""

from typing import Optional
from iris_realtime.ml.predictor import ModelPredictor

# Global model instance
predictor: Optional[ModelPredictor] = None


def get_predictor() -> ModelPredictor:
    """Get the loaded model predictor."""
    if predictor is None:
        raise RuntimeError("Model not loaded")
    return predictor
