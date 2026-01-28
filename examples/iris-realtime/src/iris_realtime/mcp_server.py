"""MCP Server for Iris Classifier.

Run with: python -m iris_realtime.mcp_server
Register with gemini-cli: gemini mcp add iris-classifier python -m iris_realtime.mcp_server
"""

import json
from geronimo.mcp import MCPServer, Tool, ToolResult
from geronimo.artifacts import ArtifactStore


class IrisMCPServer(MCPServer):
    """MCP Server exposing Iris classification model."""
    
    name = "iris-classifier"
    version = "1.0.0"
    description = "Iris species classification model"
    
    def __init__(self):
        # Load model from artifacts
        self.store = ArtifactStore(
            project="iris-realtime",
            version="1.0.0",
        )
        self.model = None
        self.encoder = None
        self._load_model()
        
        super().__init__()
    
    def _load_model(self):
        """Load trained model and encoder from artifact store."""
        try:
            self.model = self.store.load("model")
            self.encoder = self.store.load("label_encoder")
        except FileNotFoundError:
            # Demo mode - model not trained yet
            pass
    
    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                name="predict",
                description="Predict iris species from flower measurements",
                input_schema={
                    "type": "object",
                    "properties": {
                        "sepal_length": {
                            "type": "number",
                            "description": "Sepal length in cm",
                        },
                        "sepal_width": {
                            "type": "number", 
                            "description": "Sepal width in cm",
                        },
                        "petal_length": {
                            "type": "number",
                            "description": "Petal length in cm",
                        },
                        "petal_width": {
                            "type": "number",
                            "description": "Petal width in cm",
                        },
                    },
                    "required": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
                },
                handler=self.predict,
            ),
            Tool(
                name="get_model_info",
                description="Get information about the iris classification model",
                input_schema={"type": "object", "properties": {}},
                handler=self.get_model_info,
            ),
            Tool(
                name="list_features",
                description="List the input features expected by the model",
                input_schema={"type": "object", "properties": {}},
                handler=self.list_features,
            ),
        ]
    
    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> ToolResult:
        """Run prediction on the iris model."""
        if self.model is None:
            return ToolResult.error(
                "Model not loaded. Run train.py first to train the model."
            )
        
        import numpy as np
        
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = self.model.predict(features)
        species = self.encoder.inverse_transform(prediction)[0]
        
        # Get probabilities if available
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0].tolist()
        
        result = {
            "species": species,
            "prediction": int(prediction[0]),
            "input": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
        }
        
        if proba:
            result["probabilities"] = {
                cls: round(p, 4) 
                for cls, p in zip(self.encoder.classes_, proba)
            }
        
        return ToolResult.json(result)
    
    def get_model_info(self) -> ToolResult:
        """Return model metadata."""
        info = {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "model_type": type(self.model).__name__ if self.model else "Not loaded",
            "classes": list(self.encoder.classes_) if self.encoder else [],
            "status": "ready" if self.model else "not_trained",
        }
        return ToolResult.json(info)
    
    def list_features(self) -> ToolResult:
        """List expected input features."""
        features = [
            {"name": "sepal_length", "type": "number", "unit": "cm", "description": "Length of sepal"},
            {"name": "sepal_width", "type": "number", "unit": "cm", "description": "Width of sepal"},
            {"name": "petal_length", "type": "number", "unit": "cm", "description": "Length of petal"},
            {"name": "petal_width", "type": "number", "unit": "cm", "description": "Width of petal"},
        ]
        return ToolResult.json({"features": features})


def main():
    """Run the MCP server."""
    server = IrisMCPServer()
    server.run_stdio()


if __name__ == "__main__":
    main()
