"""Model definition - credit risk prediction model."""

from geronimo.models import Model, HyperParams
from .features import ProjectFeatures
from .data_sources import training_data


class ProjectModel(Model):
    """Credit risk prediction model.
    
    Predicts probability of loan default based on applicant features.
    """

    name = "credit-risk"
    version = "1.0.0"
    features = ProjectFeatures()
    data_source = training_data

    def train(self, X, y, params: HyperParams) -> None:
        """Train the credit risk model.
        
        Args:
            X: Feature matrix with credit features
            y: Binary target (1=default, 0=no default)
            params: Hyperparameters
        """
        from sklearn.ensemble import GradientBoostingClassifier
        
        self.estimator = GradientBoostingClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", 5),
            learning_rate=params.get("learning_rate", 0.1),
        )
        self.estimator.fit(X, y)

    def predict(self, X):
        """Generate default probability predictions."""
        return self.estimator.predict_proba(X)[:, 1]
