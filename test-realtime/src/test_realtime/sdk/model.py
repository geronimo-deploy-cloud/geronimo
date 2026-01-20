"""Model definition."""

from geronimo.models import Model, HyperParams

# from .features import ProjectFeatures


class ProjectModel(Model):
    """Main model class."""

    name = "test-realtime"
    version = "1.0.0"
    # features = ProjectFeatures()

    def train(self, X, y, params: HyperParams) -> None:
        """Train the model."""
        # self.estimator = YourModel(**params.to_dict())
        # self.estimator.fit(X, y)
        raise NotImplementedError("Implement train() method")

    def predict(self, X):
        """Generate predictions."""
        # return self.estimator.predict(X)
        raise NotImplementedError("Implement predict() method")
