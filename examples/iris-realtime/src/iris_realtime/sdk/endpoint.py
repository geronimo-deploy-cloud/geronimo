"""Endpoint definition for Iris prediction API."""

from typing import Optional
import numpy as np
import pandas as pd

from geronimo.serving import Endpoint
from .model import IrisModel


class IrisEndpoint(Endpoint):
    """REST API endpoint for Iris species prediction.
    
    Accepts flower measurements and returns predicted species
    with confidence scores. Uses the declarative IrisFeatures
    for preprocessing via the IrisModel.
    
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

    def initialize(self, model_path: str = "models") -> None:
        """Initialize endpoint by loading or training the model.
        
        Args:
            model_path: Path to saved model artifacts
        """
        self.model = IrisModel()
        
        # Try to load existing model (includes fitted features)
        try:
            self.model.load(model_path)
            self._is_initialized = True
        except FileNotFoundError:
            # No saved model - train a new one
            print("No saved model found. Training new model...")
            metrics = self.model.train()
            print(f"Model trained: accuracy={metrics['accuracy']:.3f}")
            self.model.save(model_path)
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
            raise RuntimeError("Endpoint not initialized. Call initialize() first.")
        
        # Preprocess → Predict (model uses declarative features) → Postprocess
        features_df = self.preprocess(request)
        probabilities = self.model.predict_proba(features_df)
        return self.postprocess(probabilities)


# Singleton for FastAPI app
_endpoint: Optional[IrisEndpoint] = None


def get_endpoint() -> IrisEndpoint:
    """Get or create the endpoint singleton."""
    global _endpoint
    if _endpoint is None:
        _endpoint = IrisEndpoint()
        _endpoint.initialize()
    return _endpoint
