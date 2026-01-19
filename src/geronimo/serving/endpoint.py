"""Endpoint base class for real-time serving."""

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from geronimo.models import Model


class Endpoint(ABC):
    """Base class for real-time prediction endpoints.

    Provides a standardized interface for pre/post processing
    with automatic model and feature loading.

    Example:
        ```python
        from geronimo.serving import Endpoint
        from myproject.models import CreditRiskModel

        class PredictEndpoint(Endpoint):
            model_class = CreditRiskModel

            def preprocess(self, request: dict) -> dict:
                # Transform request data
                return self.model.features.transform(request["data"])

            def postprocess(self, prediction) -> dict:
                return {
                    "score": float(prediction[0]),
                    "class": "approved" if prediction[0] > 0.5 else "denied",
                }

        # Create FastAPI route
        endpoint = PredictEndpoint()
        endpoint.initialize()  # Loads model artifacts

        @app.post("/predict")
        def predict(request: dict):
            return endpoint.handle(request)
        ```
    """

    # Override in subclass
    model_class: type["Model"] = None
    artifact_project: Optional[str] = None
    artifact_version: Optional[str] = None

    def __init__(self):
        """Initialize endpoint."""
        self.model: Optional["Model"] = None
        self._is_initialized: bool = False

    def initialize(
        self,
        project: Optional[str] = None,
        version: Optional[str] = None,
    ) -> None:
        """Initialize endpoint by loading model artifacts.

        Args:
            project: Artifact project name.
            version: Artifact version.
        """
        from geronimo.artifacts import ArtifactStore

        project = project or self.artifact_project or self.model_class.name
        version = version or self.artifact_version or self.model_class.version

        store = ArtifactStore.load(project=project, version=version)

        self.model = self.model_class()
        self.model.load(store)
        self._is_initialized = True

    def handle(self, request: dict) -> dict:
        """Handle prediction request.

        Args:
            request: Input request data.

        Returns:
            Response dictionary.
        """
        if not self._is_initialized:
            raise RuntimeError("Endpoint not initialized. Call initialize() first.")

        # Preprocess
        features = self.preprocess(request)

        # Predict
        prediction = self.model.predict(features)

        # Postprocess
        return self.postprocess(prediction)

    @abstractmethod
    def preprocess(self, request: dict) -> Any:
        """Preprocess request data.

        Args:
            request: Raw request data.

        Returns:
            Preprocessed features for model.
        """
        pass

    @abstractmethod
    def postprocess(self, prediction: Any) -> dict:
        """Postprocess model prediction.

        Args:
            prediction: Raw model output.

        Returns:
            Response dictionary.
        """
        pass

    @property
    def is_initialized(self) -> bool:
        """Check if endpoint is initialized."""
        return self._is_initialized

    def __repr__(self) -> str:
        status = "initialized" if self._is_initialized else "not initialized"
        model_name = self.model_class.__name__ if self.model_class else "None"
        return f"{self.__class__.__name__}(model={model_name}, {status})"
