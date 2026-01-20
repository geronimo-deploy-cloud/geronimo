"""Model definition - implement your ML model here."""

from geronimo.models import Model, HyperParams
from .features import ProjectFeatures
from .data_sources import training_data  # Import your data source


class ProjectModel(Model):
    """Main model class.
    
    Define your model's train and predict methods.
    The features attribute connects to your FeatureSet.
    The data_source attribute defines where training data comes from.
    
    Example:
        from sklearn.ensemble import RandomForestClassifier
        
        def train(self, X, y, params):
            self.estimator = RandomForestClassifier(**params.to_dict())
            self.estimator.fit(X, y)
    """

    name = "test-batch"
    version = "1.0.0"
    features = ProjectFeatures()
    data_source = training_data  # Connect to data source

    def train(self, X, y, params: HyperParams) -> None:
        """Train the model.
        
        Args:
            X: Feature matrix
            y: Target labels
            params: Hyperparameters from HyperParams
        """
        # TODO: Implement training logic
        # self.estimator = YourModel(**params.to_dict())
        # self.estimator.fit(X, y)
        raise NotImplementedError("Implement train() method")

    def predict(self, X):
        """Generate predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        # TODO: Implement prediction logic
        # return self.estimator.predict(X)
        raise NotImplementedError("Implement predict() method")
