"""Endpoint definition for Iris prediction API.

The endpoint loads artifacts from the ArtifactStore - no training occurs here.
Training must be done separately (via train.py) before the endpoint can serve.
"""

from typing import Optional
import numpy as np
import pandas as pd

from geronimo.serving import Endpoint
from geronimo.artifacts import ArtifactStore
from .model import IrisModel


class IrisEndpoint(Endpoint):
    """REST API endpoint for Iris species prediction.
    
    Loads a pre-trained model from ArtifactStore. Training should be done 
    separately using train.py before starting the endpoint.
    
    Example request:
        {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    
    Example response:
        {
            "prediction": "setosa",
            "confidence": 0.97,
            "probabilities": {
                "setosa": 0.97,
                "versicolor": 0.02,
                "virginica": 0.01
            }
        }
    """

    model_class = IrisModel

    def initialize(self, project: str = "iris-realtime", version: str = "1.0.0") -> None:
        """Initialize endpoint by loading model from ArtifactStore.
        
        The model and fitted features MUST exist in the artifact store.
        Run train.py first to create the artifacts.
        
        Args:
            project: Project name in ArtifactStore
            version: Model version to load
            
        Raises:
            KeyError: If artifacts not found (model not trained)
        """
        # Load artifacts from store
        self._store = ArtifactStore.load(project=project, version=version)
        
        # Initialize model and load from store
        self.model = IrisModel()
        self.model.load(self._store)
        
        # List loaded artifacts
        artifacts = self._store.list()
        artifact_names = [a.name for a in artifacts]
        print(f"Loaded artifacts from store: {artifact_names}")
        
        self._is_initialized = True

    def preprocess(self, request: dict) -> pd.DataFrame:
        """Transform request into DataFrame for model.
        
        The model will use its fitted IrisFeatures for transformation.
        
        Args:
            request: JSON request body with flower measurements
            
        Returns:
            DataFrame with feature columns
        """
        # Handle both flat and nested request formats
        if "features" in request:
            req = request["features"]
        else:
            req = request
            
        # Create DataFrame with proper column names
        df = pd.DataFrame([{
            "sepal_length": float(req.get("sepal_length", 0)),
            "sepal_width": float(req.get("sepal_width", 0)),
            "petal_length": float(req.get("petal_length", 0)),
            "petal_width": float(req.get("petal_width", 0)),
        }])
        
        return df

    def postprocess(self, probabilities: np.ndarray) -> dict:
        """Format model output as API response.
        
        Args:
            probabilities: Class probabilities from model
            
        Returns:
            Response dict with prediction and confidence
        """
        probs = probabilities[0]
        predicted_class = int(np.argmax(probs))
        species = IrisModel.SPECIES[predicted_class]
        confidence = float(probs[predicted_class])
        
        return {
            "prediction": species,
            "confidence": round(confidence, 4),
            "probabilities": {
                name: round(float(p), 4)
                for name, p in zip(IrisModel.SPECIES, probs)
            }
        }
    
    def handle(self, request: dict) -> dict:
        """Handle a prediction request.
        
        Preprocessing creates a DataFrame, the model's predict_proba
        uses the fitted IrisFeatures for transformation internally.
        
        Args:
            request: Input request with flower measurements
            
        Returns:
            Prediction response
        """
        if not self._is_initialized:
            raise RuntimeError(
                "Endpoint not initialized. Did you run train.py first? "
                "Call initialize() after training to load the model."
            )
        
        # Preprocess → Predict (model uses declarative features) → Postprocess
        features_df = self.preprocess(request)
        probabilities = self.model.predict_proba(features_df)
        return self.postprocess(probabilities)


# Singleton for FastAPI app
_endpoint: Optional[IrisEndpoint] = None


def get_endpoint() -> IrisEndpoint:
    """Get or create the endpoint singleton.
    
    Raises:
        KeyError: If model artifacts not found (need to run train.py first)
    """
    global _endpoint
    if _endpoint is None:
        _endpoint = IrisEndpoint()
        _endpoint.initialize()
    return _endpoint
