"""Endpoint definition for real-time serving."""

from geronimo.serving import Endpoint

# from .model import ProjectModel


class PredictEndpoint(Endpoint):
    """Prediction endpoint."""

    # model_class = ProjectModel

    def preprocess(self, request: dict):
        """Preprocess incoming request."""
        # df = pd.DataFrame([request["data"]])
        # return self.model.features.transform(df)
        raise NotImplementedError("Implement preprocess() method")

    def postprocess(self, prediction):
        """Postprocess model output."""
        # return {"score": float(prediction[0])}
        raise NotImplementedError("Implement postprocess() method")
