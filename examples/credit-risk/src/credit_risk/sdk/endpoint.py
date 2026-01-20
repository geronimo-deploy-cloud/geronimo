"""Endpoint definition for credit risk API."""

from geronimo.serving import Endpoint
from .model import ProjectModel


class PredictEndpoint(Endpoint):
    """Credit risk prediction endpoint.
    
    Transforms incoming loan application data, generates default probability,
    and returns risk assessment.
    """

    model_class = ProjectModel

    def preprocess(self, request: dict):
        """Transform loan application to model features.
        
        Args:
            request: JSON with loan application data
            
        Returns:
            Feature matrix for model.predict()
        """
        import pandas as pd
        
        # Extract features from request
        features = request.get("features", request)
        df = pd.DataFrame([features])
        
        # Apply feature transformations
        # return self.model.features.transform(df)
        return df

    def postprocess(self, prediction):
        """Format default probability as risk assessment.
        
        Args:
            prediction: Default probability (0-1)
            
        Returns:
            Risk assessment with probability and decision
        """
        prob = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
        
        # Risk categorization
        if prob < 0.2:
            risk_level = "LOW"
            decision = "APPROVE"
        elif prob < 0.5:
            risk_level = "MEDIUM"
            decision = "REVIEW"
        else:
            risk_level = "HIGH"
            decision = "DECLINE"
        
        return {
            "default_probability": round(prob, 4),
            "risk_level": risk_level,
            "recommendation": decision,
        }
