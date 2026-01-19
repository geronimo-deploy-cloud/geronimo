"""Model predictor for ML inference.

Handles model loading, caching, and prediction logic.
"""

import logging
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Default model path (relative to project root)
DEFAULT_MODEL_PATH = Path("models/model.joblib")


class ModelPredictor:
    """Handles model loading and predictions.

    Implements lazy loading and caching for efficient inference.
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        """Initialize the predictor.

        Args:
            model_path: Path to the model artifact. Uses default if not provided.
        """
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model: Any = None
        self._version: str = "1.0.0"

    @property
    def version(self) -> str:
        """Get the model version."""
        return self._version

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._model is not None

    def load(self) -> None:
        """Load the model from disk.

        Raises:
            FileNotFoundError: If model file doesn't exist.
            RuntimeError: If model loading fails.
        """
        if not self.model_path.exists():
            logger.warning(
                f"Model file not found at {self.model_path}. "
                "Using placeholder for development."
            )
            self._model = self._create_placeholder_model()
            return

        try:
            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
            logger.info("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _create_placeholder_model(self) -> Any:
        """Create a placeholder model for development/testing."""
        # Returns a simple function that echoes input
        return lambda x: 0.5

    def predict(self, features: dict[str, Any]) -> Any:
        """Generate predictions for input features.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Model prediction (type depends on model).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert features to model input format
        # This should be customized based on your model's requirements
        try:
            if callable(self._model):
                # Placeholder model
                return self._model(features)

            # For sklearn-style models with predict method
            import pandas as pd
            import numpy as np

            # Convert dict to DataFrame for sklearn compatibility
            df = pd.DataFrame([features])

            # Get prediction
            prediction = self._model.predict(df)

            # Return single value if single prediction
            if isinstance(prediction, np.ndarray) and len(prediction) == 1:
                return float(prediction[0])

            return prediction.tolist()

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
