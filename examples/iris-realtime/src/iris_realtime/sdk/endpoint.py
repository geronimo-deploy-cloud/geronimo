"""Endpoint definition - handle incoming prediction requests."""

from geronimo.serving import Endpoint
from .model import IrisRealtimeModel


class IrisRealtimeEndpoint(Endpoint):
    """REST API endpoint for Iris species prediction.
    
    Uses IrisRealtimeModel for inference, which loads pre-trained
    artifacts from ArtifactStore. Training should be done 
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


    model_class = IrisRealtimeModel

    def preprocess(self, request: dict):
        """Transform incoming request to model input.
        
        Args:
            request: JSON request body with "features" key
            
        Returns:
            Feature matrix ready for model.predict()
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

    def postprocess(self, prediction):
        """Format model output for response.
        
        Args:
            prediction: Raw model output
            
        Returns:
            JSON-serializable response
        """
        return {"result": prediction}
    
    def initialize(self, project=None, version=None):
        """Initialize endpoint.
        
        Parent class handles loading of the fitted model from the artifact store.
        """
        super().initialize(project=project, version=version)
    
    def handle(self, request: dict) -> dict:
        """Handle prediction request.
        
        Parent class handles calling the model's predict method.
        """
        return super().handle(request)
