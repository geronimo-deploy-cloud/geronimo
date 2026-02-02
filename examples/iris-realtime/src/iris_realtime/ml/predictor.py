"""Model predictor for ML inference.

Handles model loading from ArtifactStore and prediction logic.
Provides a transparent serving layer that automatically picks up
artifacts saved by the SDK Model class.
"""

import logging
from typing import Any, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from geronimo.models import Model

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Handles model loading and predictions from ArtifactStore.

    Implements lazy loading and caching for efficient inference.
    Automatically loads both the estimator and fitted features
    to ensure consistent preprocessing at serving time.

    Example:
        ```python
        from iris_realtime.ml.predictor import ModelPredictor
        from iris_realtime.sdk.model import IrisModel

        predictor = ModelPredictor(IrisModel)
        predictor.load()

        result = predictor.predict({
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        })
        ```
    """

    def __init__(
        self,
        model_class: type["Model"],
        project: str | None = None,
        version: str | None = None,
    ) -> None:
        """Initialize the predictor.

        Args:
            model_class: The Model class to use for inference.
                Project/version defaults are read from class attributes.
            project: Override project name for ArtifactStore lookup.
            version: Override version for ArtifactStore lookup.
        """
        self.model_class = model_class
        self.project = project or model_class.name
        self.version = version or model_class.version
        self._estimator: Any = None
        self._features: Any = None

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._estimator is not None

    def load(self) -> None:
        """Load the model and features from ArtifactStore.

        Raises:
            KeyError: If artifacts not found (model not trained).
            RuntimeError: If model loading fails.
        """
        from geronimo.artifacts import ArtifactStore

        try:
            logger.info(
                f"Loading artifacts from ArtifactStore "
                f"(project={self.project}, version={self.version})"
            )
            store = ArtifactStore.load(project=self.project, version=self.version)

            self._estimator = store.get("estimator")
            self._features = store.get("features")

            logger.info("Model and features loaded successfully")
        except KeyError as e:
            raise KeyError(
                f"Artifacts not found for {self.project}@{self.version}. "
                "Did you run training first?"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def predict(self, features: dict[str, Any]) -> Any:
        """Generate predictions for input features.

        Args:
            features: Dictionary of feature name to value.

        Returns:
            Model prediction (class label for classifiers).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert dict to DataFrame
        df = pd.DataFrame([features])

        # Apply fitted feature transformations
        X_transformed = self._features.transform(df)

        # Get prediction
        prediction = self._estimator.predict(X_transformed)

        # Return single value if single prediction
        if hasattr(prediction, "__len__") and len(prediction) == 1:
            return prediction[0]

        return prediction

    def predict_proba(self, features: dict[str, Any] | pd.DataFrame) -> Any:
        """Generate probability predictions for input features.

        Args:
            features: Dictionary of feature name to value, or DataFrame.

        Returns:
            Probability array of shape (n_samples, n_classes).

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Convert dict to DataFrame if needed
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features

        # Apply fitted feature transformations
        X_transformed = self._features.transform(df)

        # Get probabilities
        return self._estimator.predict_proba(X_transformed)
