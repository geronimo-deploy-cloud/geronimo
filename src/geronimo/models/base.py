"""Model base class for ML model definition."""

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, Optional

from geronimo.models.objective import ObjectiveSpec
from geronimo.models.params import HyperParams

if TYPE_CHECKING:
    from geronimo.artifacts import ArtifactStore
    from geronimo.features import FeatureSet


class Model(ABC):
    """Base class for Geronimo models.

    Provides a standardized interface for training and inference
    with integrated artifact management.

    Example:
        ```python
        from geronimo.models import Model, HyperParams
        from geronimo.features import FeatureSet
        from sklearn.ensemble import RandomForestClassifier

        class CreditRiskModel(Model):
            name = "credit-risk"
            version = "1.2.0"

            def train(self, X, y, params: HyperParams):
                self.estimator = RandomForestClassifier(
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                )
                self.estimator.fit(X, y)

            def predict(self, X):
                return self.estimator.predict_proba(X)
        ```
    """

    # Override in subclass
    name: str = "unnamed"
    """The name of the model."""

    version: str = "1.0.0"
    """The version of the model."""

    features: Optional["FeatureSet"] = None
    """The associated feature set."""

    objective: Optional[ObjectiveSpec] = None
    """The training objective (loss function) specification.

    If ``None``, no objective is declared and training proceeds with
    the estimator's default loss — preserving backward compatibility
    with existing models that don't declare an objective.
    """

    estimator: Any
    """The underlying model estimator (e.g., sklearn object)."""

    _is_fitted: bool
    """Internal flag tracking whether the model has been trained."""

    def __init__(self):
        """Initialize model."""
        self.estimator: Any = None
        self._is_fitted: bool = False

    @abstractmethod
    def train(self, X, y, params: HyperParams) -> None:
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target vector.
            params: Hyperparameters.
        """
        pass

    @abstractmethod
    def predict(self, X) -> Any:
        """Generate predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predictions.
        """
        pass

    def save(self, store: "ArtifactStore") -> None:
        """Save model to artifact store.

        Args:
            store: ArtifactStore instance.
        """
        if self.estimator is None:
            raise ValueError("No estimator to save. Train the model first.")

        store.save("model", self.estimator, artifact_type="estimator")

        # Save features if present
        if self.features is not None and hasattr(self.features, "save"):
            self.features.save(store)

    def load(self, store: "ArtifactStore") -> None:
        """Load model from artifact store.

        Args:
            store: ArtifactStore instance.
        """
        self.estimator = store.get("model")
        self._is_fitted = True

        # Load features if present
        if self.features is not None and hasattr(self.features, "load"):
            self.features.load(store)

    @property
    def is_fitted(self) -> bool:
        """Check if model has been trained."""
        return self._is_fitted

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return f"{self.__class__.__name__}({self.name}@{self.version}, {status})"
